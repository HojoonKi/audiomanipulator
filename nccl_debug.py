#!/usr/bin/env python3
"""
NCCL 진단 및 디버깅 도구
분산 훈련 전에 NCCL 환경을 검사하고 문제를 진단합니다.
"""

import os
import subprocess
import socket
import torch
import torch.distributed as dist
from datetime import datetime


def check_gpu_availability():
    """GPU 가용성 확인"""
    print("🔍 GPU 환경 검사:")
    
    if not torch.cuda.is_available():
        print("❌ CUDA가 사용 불가능합니다.")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 사용 가능한 GPU 수: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    return gpu_count > 0


def check_nccl_version():
    """NCCL 버전 확인"""
    print("\n🔍 NCCL 버전 확인:")
    
    try:
        # PyTorch의 NCCL 버전 확인
        if hasattr(torch.cuda.nccl, 'version'):
            nccl_version = torch.cuda.nccl.version()
            print(f"✅ PyTorch NCCL 버전: {nccl_version}")
        else:
            print("⚠️ NCCL 버전 정보를 가져올 수 없습니다.")
        
        # 시스템 NCCL 라이브러리 확인
        try:
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
            if 'libnccl' in result.stdout:
                print("✅ 시스템에 NCCL 라이브러리가 설치되어 있습니다.")
            else:
                print("⚠️ 시스템에서 NCCL 라이브러리를 찾을 수 없습니다.")
        except Exception:
            print("⚠️ 시스템 라이브러리 검사 실패")
            
    except Exception as e:
        print(f"❌ NCCL 버전 확인 실패: {e}")


def check_network_interfaces():
    """네트워크 인터페이스 확인 - Docker 호환"""
    print("\n🔍 네트워크 인터페이스 확인:")
    
    try:
        # 방법 1: ip 명령어 시도
        try:
            result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True)
            if result.returncode == 0:
                interfaces = []
                for line in result.stdout.split('\n'):
                    if ': ' in line and 'state UP' in line:
                        iface = line.split(':')[1].strip().split('@')[0]
                        interfaces.append(iface)
                
                if interfaces:
                    print(f"✅ 활성 네트워크 인터페이스: {', '.join(interfaces)}")
                    
                    # 기본 라우트 확인
                    try:
                        result = subprocess.run(['ip', 'route', 'get', '8.8.8.8'], 
                                              capture_output=True, text=True)
                        if result.returncode == 0 and 'dev' in result.stdout:
                            import re
                            match = re.search(r'dev\s+(\w+)', result.stdout)
                            if match:
                                default_iface = match.group(1)
                                print(f"✅ 기본 네트워크 인터페이스: {default_iface}")
                                return default_iface
                    except:
                        pass
        except FileNotFoundError:
            print("⚠️ 'ip' 명령어를 찾을 수 없습니다 (Docker 환경에서 일반적)")
        
        # 방법 2: /sys/class/net 디렉토리 확인 (Linux 표준)
        try:
            import os
            net_path = '/sys/class/net'
            if os.path.exists(net_path):
                available_interfaces = [iface for iface in os.listdir(net_path) if iface != 'lo']
                if available_interfaces:
                    print(f"✅ /sys/class/net에서 발견된 인터페이스: {', '.join(available_interfaces)}")
                    
                    # /proc/net/route에서 기본 라우트 확인
                    try:
                        with open('/proc/net/route', 'r') as f:
                            for line in f:
                                fields = line.strip().split()
                                if len(fields) >= 2 and fields[1] == '00000000':  # 기본 라우트
                                    interface = fields[0]
                                    if interface != 'Iface' and interface in available_interfaces:
                                        print(f"✅ 기본 네트워크 인터페이스: {interface}")
                                        return interface
                    except:
                        pass
                    
                    # 첫 번째 비-loopback 인터페이스 반환
                    default_iface = available_interfaces[0]
                    print(f"✅ 기본 인터페이스로 사용: {default_iface}")
                    return default_iface
        except Exception as e:
            print(f"⚠️ /sys/class/net 확인 실패: {e}")
        
        # 방법 3: /proc/net/dev 파일 확인
        try:
            with open('/proc/net/dev', 'r') as f:
                interfaces = []
                for line in f:
                    if ':' in line:
                        interface = line.split(':')[0].strip()
                        if interface not in ['lo', 'Inter-|   face']:  # loopback과 헤더 제외
                            interfaces.append(interface)
                
                if interfaces:
                    print(f"✅ /proc/net/dev에서 발견된 인터페이스: {', '.join(interfaces)}")
                    default_iface = interfaces[0]
                    print(f"✅ 기본 인터페이스로 사용: {default_iface}")
                    return default_iface
        except Exception as e:
            print(f"⚠️ /proc/net/dev 확인 실패: {e}")
        
        print("❌ 네트워크 인터페이스를 찾을 수 없습니다.")
        
    except Exception as e:
        print(f"❌ 네트워크 인터페이스 확인 실패: {e}")
    
    return None


def check_port_availability(port):
    """포트 사용 가능성 확인"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False


def suggest_nccl_settings():
    """NCCL 설정 제안"""
    print("\n💡 권장 NCCL 환경변수 설정:")
    
    # 기본 네트워크 인터페이스 감지
    default_iface = check_network_interfaces()
    
    settings = {
        'NCCL_DEBUG': 'WARN',
        'NCCL_TIMEOUT': '1800',
        'NCCL_TREE_THRESHOLD': '0',
        'NCCL_IB_DISABLE': '1',  # InfiniBand 비활성화 (일반 환경)
        'NCCL_P2P_DISABLE': '0',  # P2P 활성화 (단일 노드)
        'NCCL_BLOCKING_WAIT': '1',
        'NCCL_ASYNC_ERROR_HANDLING': '1',
    }
    
    if default_iface:
        settings['NCCL_SOCKET_IFNAME'] = default_iface
    
    print("export 명령어:")
    for key, value in settings.items():
        print(f"export {key}={value}")
    
    print("\nDocker run 옵션:")
    for key, value in settings.items():
        print(f"--env {key}={value} \\")


def test_basic_nccl():
    """기본 NCCL 기능 테스트"""
    print("\n🧪 기본 NCCL 테스트:")
    
    if not torch.cuda.is_available():
        print("❌ CUDA가 없어 NCCL 테스트를 건너뜁니다.")
        return
    
    try:
        # 단일 GPU에서 NCCL 텐서 생성 테스트
        device = torch.device('cuda:0')
        test_tensor = torch.randn(10, device=device)
        
        # NCCL 백엔드가 사용 가능한지 확인
        if torch.distributed.is_nccl_available():
            print("✅ NCCL 백엔드가 사용 가능합니다.")
        else:
            print("❌ NCCL 백엔드가 사용 불가능합니다.")
            
    except Exception as e:
        print(f"❌ 기본 NCCL 테스트 실패: {e}")


def test_multi_gpu_setup():
    """멀티 GPU 설정 테스트"""
    print("\n🧪 멀티 GPU 설정 테스트:")
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"⚠️ GPU가 {gpu_count}개만 있어 멀티 GPU 테스트를 건너뜁니다.")
        return
    
    try:
        # 각 GPU에서 텐서 생성 테스트
        tensors = []
        for i in range(gpu_count):
            device = torch.device(f'cuda:{i}')
            tensor = torch.randn(10, device=device)
            tensors.append(tensor)
            print(f"✅ GPU {i}: 텐서 생성 성공")
        
        # GPU 간 메모리 복사 테스트
        if gpu_count >= 2:
            tensor_copy = tensors[0].to('cuda:1')
            print("✅ GPU 간 메모리 복사 성공")
            
    except Exception as e:
        print(f"❌ 멀티 GPU 테스트 실패: {e}")


def diagnose_common_issues():
    """일반적인 NCCL 문제 진단"""
    print("\n🔍 일반적인 NCCL 문제 진단:")
    
    issues_found = []
    
    # 1. 방화벽 확인
    try:
        result = subprocess.run(['systemctl', 'is-active', 'firewalld'], 
                              capture_output=True, text=True)
        if result.stdout.strip() == 'active':
            issues_found.append("방화벽이 활성화되어 있습니다. NCCL 통신을 차단할 수 있습니다.")
    except:
        pass
    
    # 2. SELinux 확인
    try:
        result = subprocess.run(['getenforce'], capture_output=True, text=True)
        if result.stdout.strip() == 'Enforcing':
            issues_found.append("SELinux가 Enforcing 모드입니다. NCCL 통신을 제한할 수 있습니다.")
    except:
        pass
    
    # 3. Docker 환경 확인
    if os.path.exists('/.dockerenv'):
        print("🐳 Docker 환경에서 실행 중입니다.")
        
        # --ipc=host 확인
        try:
            with open('/proc/self/cgroup', 'r') as f:
                cgroup_info = f.read()
                if 'docker' in cgroup_info:
                    issues_found.append("Docker 환경: --ipc=host 옵션이 필요할 수 있습니다.")
        except:
            pass
    
    # 4. 메모리 제한 확인
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            if 'MemAvailable' in meminfo:
                for line in meminfo.split('\n'):
                    if 'MemAvailable' in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / (1024 * 1024)
                        if mem_gb < 8:
                            issues_found.append(f"사용 가능한 메모리가 부족합니다: {mem_gb:.1f} GB")
                        break
    except:
        pass
    
    if issues_found:
        print("⚠️ 발견된 잠재적 문제:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ 일반적인 문제가 발견되지 않았습니다.")


def main():
    """메인 진단 함수"""
    print("🚀 NCCL 환경 진단 도구")
    print("=" * 50)
    print(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 기본 환경 확인
    check_gpu_availability()
    check_nccl_version()
    check_network_interfaces()
    
    # NCCL 테스트
    test_basic_nccl()
    test_multi_gpu_setup()
    
    # 설정 제안
    suggest_nccl_settings()
    
    # 문제 진단
    diagnose_common_issues()
    
    print("\n" + "=" * 50)
    print("🎯 진단 완료!")
    print("\n💡 문제가 지속되면 다음을 확인하세요:")
    print("   1. nvidia-smi 명령으로 GPU 상태 확인")
    print("   2. docker logs로 컨테이너 로그 확인")
    print("   3. NCCL_DEBUG=INFO로 상세 로그 활성화")
    print("   4. 네트워크 연결 및 방화벽 설정 확인")


if __name__ == "__main__":
    main()
