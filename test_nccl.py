#!/usr/bin/env python3
"""
간단한 NCCL 분산 훈련 테스트
실제 훈련 전에 NCCL 통신이 정상 작동하는지 확인합니다.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from datetime import datetime


def setup_distributed(rank, world_size, master_addr='localhost', master_port='12355'):
    """분산 환경 설정"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    print(f"[Rank {rank}] 분산 초기화 시작...")
    
    try:
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout
        )
        print(f"[Rank {rank}] ✅ NCCL 초기화 성공!")
        return True
    except Exception as e:
        print(f"[Rank {rank}] ❌ NCCL 초기화 실패: {e}")
        return False


def test_allreduce(rank, world_size):
    """AllReduce 통신 테스트"""
    device = torch.device(f'cuda:{rank}')
    
    # 테스트 텐서 생성
    tensor = torch.ones(10, device=device) * rank
    print(f"[Rank {rank}] 초기 텐서: {tensor.cpu().numpy()}")
    
    # AllReduce 실행
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # 예상 결과: 모든 rank의 합 (0 + 1 + ... + (world_size-1))
    expected_sum = sum(range(world_size))
    expected_tensor = torch.ones(10) * expected_sum
    
    print(f"[Rank {rank}] AllReduce 결과: {tensor.cpu().numpy()}")
    print(f"[Rank {rank}] 예상 결과: {expected_tensor.numpy()}")
    
    # 결과 검증
    if torch.allclose(tensor.cpu(), expected_tensor):
        print(f"[Rank {rank}] ✅ AllReduce 테스트 성공!")
        return True
    else:
        print(f"[Rank {rank}] ❌ AllReduce 테스트 실패!")
        return False


def test_broadcast(rank, world_size):
    """Broadcast 통신 테스트"""
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        # 마스터에서 브로드캐스트할 데이터
        tensor = torch.randn(5, device=device)
        master_data = tensor.clone()
    else:
        # 다른 프로세스는 빈 텐서
        tensor = torch.zeros(5, device=device)
        master_data = None
    
    print(f"[Rank {rank}] Broadcast 전 텐서: {tensor.cpu().numpy()}")
    
    # Broadcast 실행
    dist.broadcast(tensor, src=0)
    
    print(f"[Rank {rank}] Broadcast 후 텐서: {tensor.cpu().numpy()}")
    
    if rank == 0:
        print(f"[Rank {rank}] ✅ Broadcast 테스트 완료 (마스터)")
        return True
    else:
        print(f"[Rank {rank}] ✅ Broadcast 테스트 완료 (워커)")
        return True


def test_barrier(rank, world_size):
    """Barrier 동기화 테스트"""
    import time
    import random
    
    # 랜덤 지연 시뮬레이션
    delay = random.uniform(0.1, 1.0)
    print(f"[Rank {rank}] {delay:.2f}초 지연 후 barrier 대기...")
    time.sleep(delay)
    
    print(f"[Rank {rank}] Barrier 진입...")
    dist.barrier()
    print(f"[Rank {rank}] ✅ Barrier 통과!")
    
    return True


def worker_process(rank, world_size, args):
    """워커 프로세스 메인 함수"""
    try:
        print(f"\n[Rank {rank}] 워커 프로세스 시작")
        print(f"[Rank {rank}] GPU: {torch.cuda.get_device_name(rank)}")
        
        # 분산 환경 설정
        if not setup_distributed(rank, world_size, args.master_addr, args.master_port):
            return
        
        # 테스트 실행
        tests = [
            ("Barrier", test_barrier),
            ("AllReduce", test_allreduce),
            ("Broadcast", test_broadcast),
        ]
        
        for test_name, test_func in tests:
            print(f"\n[Rank {rank}] === {test_name} 테스트 시작 ===")
            try:
                success = test_func(rank, world_size)
                if success:
                    print(f"[Rank {rank}] ✅ {test_name} 테스트 성공")
                else:
                    print(f"[Rank {rank}] ❌ {test_name} 테스트 실패")
            except Exception as e:
                print(f"[Rank {rank}] ❌ {test_name} 테스트 예외: {e}")
        
        # 최종 동기화
        dist.barrier()
        if rank == 0:
            print(f"\n🎉 모든 NCCL 테스트 완료!")
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ 워커 프로세스 오류: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 정리
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"[Rank {rank}] 분산 프로세스 그룹 정리 완료")


def main():
    parser = argparse.ArgumentParser(description='NCCL 분산 훈련 테스트')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(),
                       help=f'프로세스 수 (기본값: {torch.cuda.device_count()})')
    parser.add_argument('--master_addr', type=str, default='localhost',
                       help='마스터 주소')
    parser.add_argument('--master_port', type=str, default='12355',
                       help='마스터 포트')
    
    args = parser.parse_args()
    
    print("🚀 NCCL 분산 훈련 테스트")
    print("=" * 50)
    print(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"World Size: {args.world_size}")
    print(f"Master: {args.master_addr}:{args.master_port}")
    
    # GPU 가용성 확인
    if not torch.cuda.is_available():
        print("❌ CUDA가 사용 불가능합니다.")
        return
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < args.world_size:
        print(f"❌ 요청된 world_size({args.world_size})가 사용 가능한 GPU 수({gpu_count})보다 큽니다.")
        return
    
    print(f"✅ 사용 가능한 GPU: {gpu_count}개")
    for i in range(min(args.world_size, gpu_count)):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 환경변수 설정 확인
    print(f"\n🔍 NCCL 환경변수:")
    nccl_vars = ['NCCL_DEBUG', 'NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE', 
                 'NCCL_P2P_DISABLE', 'NCCL_TIMEOUT']
    for var in nccl_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    # 멀티프로세스 시작
    print(f"\n🚀 {args.world_size}개 프로세스로 NCCL 테스트 시작...")
    try:
        mp.spawn(
            worker_process,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
        print("✅ NCCL 테스트 완료!")
    except Exception as e:
        print(f"❌ NCCL 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
