#!/usr/bin/env python3
"""
배치 크기에 따른 메인 훈련 시간 테스트
- 16, 32, 64, 128, 256 배치 크기로 테스트
- 실제 훈련 없이 forward pass만으로 시간 측정
- GPU 메모리 사용량도 함께 측정
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import traceback

# HuggingFace tokenizers parallelism 경고 해결
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 현재 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dynamic_pipeline_factory import DynamicPipelineFactory
from dataset import (
    PureDescriptionDataset,
    create_custom_collate_fn, 
    load_descriptions, 
    split_descriptions
)

def format_time(seconds):
    """시간을 읽기 쉬운 형태로 포맷"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_gpu_memory_info(device):
    """GPU 메모리 사용량 정보 반환"""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
    
    return {
        "allocated": allocated,
        "reserved": reserved, 
        "total": total,
        "allocated_pct": (allocated/total)*100,
        "reserved_pct": (reserved/total)*100
    }

class BatchSizeTester:
    def __init__(self, data_path='/workspace/AudioManipulator'):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.descriptions = None
        
        print(f"🚀 배치 크기 테스터 초기화")
        print(f"   - Device: {self.device}")
        print(f"   - Data path: {data_path}")
        
    def setup_model(self):
        """모델 초기화"""
        print("\n🔄 모델 초기화 중...")
        
        # CLAP 로딩 시 출력 억제
        from contextlib import redirect_stdout, redirect_stderr
        
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                factory = DynamicPipelineFactory()
                
                # 기본 설정으로 모델 생성
                self.model = factory.create_pipeline(
                    encoder_preset='sentence-transformer-large',
                    use_clap=True,
                    backbone_type='residual',
                    decoder_type='parallel',
                    sample_rate=44100,
                    target_params=500000
                )
        
        self.model = self.model.to(self.device)
        self.model.eval()  # 평가 모드로 설정 (dropout 등 비활성화)
        
        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ 모델 초기화 완료")
        print(f"   - 총 파라미터: {total_params:,}")
        print(f"   - 훈련 가능 파라미터: {trainable_params:,}")
        
        # 초기 GPU 메모리 사용량
        if torch.cuda.is_available():
            mem_info = get_gpu_memory_info(self.device)
            print(f"   - GPU 메모리 (모델 로드 후): {mem_info['allocated']:.2f}GB allocated, {mem_info['reserved']:.2f}GB reserved")
    
    def load_test_data(self):
        """테스트용 데이터 로드"""
        print("\n📚 테스트 데이터 로드 중...")
        
        # 빠른 테스트를 위해 적은 수의 descriptions만 사용
        self.descriptions = load_descriptions(
            data_path=self.data_path,
            use_sampled_descriptions=False,  # 기본 descriptions.txt 사용
            max_descriptions=1000  # 최대 1000개로 제한
        )
        
        if not self.descriptions:
            raise ValueError("❌ Description 로드 실패")
        
        # 훈련용으로만 사용 (검증 데이터는 불필요)
        train_descriptions, _ = split_descriptions(self.descriptions, train_ratio=0.9)
        self.train_descriptions = train_descriptions[:500]  # 500개로 제한
        
        print(f"✅ 테스트 데이터 로드 완료")
        print(f"   - 사용할 descriptions: {len(self.train_descriptions)}개")
        
    def create_test_dataset(self, batch_size):
        """특정 배치 크기로 테스트 데이터셋 생성"""
        dataset = PureDescriptionDataset(
            descriptions=self.train_descriptions,
            audio_dataset_path=os.path.join(self.data_path, 'audio_dataset'),
            sample_rate=44100,
            audio_length=5.0
        )
        
        custom_collate_fn = create_custom_collate_fn(include_guide_preset=False)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # 테스트에서는 셔플 안함 (일관성 위해)
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        return dataloader
    
    def test_batch_size(self, batch_size, num_test_batches=10):
        """특정 배치 크기로 시간 측정"""
        print(f"\n🔬 배치 크기 {batch_size} 테스트 시작...")
        
        try:
            # 데이터로더 생성
            dataloader = self.create_test_dataset(batch_size)
            
            # GPU 메모리 초기화
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # 워밍업 (첫 번째 배치는 제외)
            warmup_batch = next(iter(dataloader))
            descriptions = warmup_batch['description']
            audios = warmup_batch['audio'].to(self.device, non_blocking=True)
            
            if audios.dim() == 2:
                audios = audios.unsqueeze(1)
            
            # 워밍업 forward pass
            with torch.no_grad():
                _ = self.model(texts=descriptions, audio=audios, use_real_audio=False)
            
            # 메모리 정보 (워밍업 후)
            mem_after_warmup = get_gpu_memory_info(self.device) if torch.cuda.is_available() else {}
            
            # 실제 시간 측정
            batch_times = []
            total_samples = 0
            
            start_time = time.time()
            
            # 제한된 수의 배치만 테스트 (빠른 테스트를 위해)
            test_batches = min(num_test_batches, len(dataloader))
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= test_batches:
                    break
                
                batch_start = time.time()
                
                descriptions = batch['description']
                audios = batch['audio'].to(self.device, non_blocking=True)
                
                if audios.dim() == 2:
                    audios = audios.unsqueeze(1)
                
                # Forward pass (backward는 제외)
                with torch.no_grad():
                    outputs = self.model(
                        texts=descriptions,
                        audio=audios,
                        use_real_audio=False
                    )
                    
                    # CLAP loss 계산 시뮬레이션 (실제 계산은 안함)
                    if 'processed_audio' in outputs:
                        processed_audio = outputs['processed_audio']
                        # 간단한 연산만 수행 (실제 CLAP 계산 없이)
                        _ = torch.mean(processed_audio)
                
                batch_end = time.time()
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)
                total_samples += len(descriptions)
            
            total_time = time.time() - start_time
            
            # 통계 계산
            avg_batch_time = sum(batch_times) / len(batch_times)
            samples_per_second = total_samples / total_time
            
            # 전체 데이터셋 기준 epoch time 추정
            total_dataset_size = len(self.train_descriptions)
            batches_per_epoch = (total_dataset_size + batch_size - 1) // batch_size
            estimated_epoch_time = avg_batch_time * batches_per_epoch
            
            # 메모리 정보 (테스트 후)
            mem_after_test = get_gpu_memory_info(self.device) if torch.cuda.is_available() else {}
            peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0
            
            return {
                'batch_size': batch_size,
                'avg_batch_time': avg_batch_time,
                'samples_per_second': samples_per_second,
                'estimated_epoch_time': estimated_epoch_time,
                'total_batches_per_epoch': batches_per_epoch,
                'tested_batches': len(batch_times),
                'total_samples_tested': total_samples,
                'memory_after_warmup': mem_after_warmup,
                'memory_after_test': mem_after_test,
                'peak_memory_gb': peak_memory,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ 배치 크기 {batch_size} 테스트 실패: {e}")
            traceback.print_exc()
            return {
                'batch_size': batch_size,
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_test(self):
        """16부터 256까지 배치 크기 종합 테스트"""
        print("\n" + "="*80)
        print("🧪 배치 크기 종합 테스트 시작")
        print("="*80)
        
        # 테스트할 배치 크기들
        batch_sizes = [16, 32, 64, 128, 256]
        results = []
        
        for batch_size in batch_sizes:
            result = self.test_batch_size(batch_size, num_test_batches=10)
            results.append(result)
            
            if result['success']:
                print(f"✅ 배치 {batch_size:3d}: "
                      f"{result['avg_batch_time']:.3f}s/batch, "
                      f"{result['samples_per_second']:.1f} samples/s, "
                      f"예상 epoch: {format_time(result['estimated_epoch_time'])}")
                
                if torch.cuda.is_available():
                    print(f"      GPU 메모리: {result['peak_memory_gb']:.2f}GB peak, "
                          f"{result['memory_after_test']['allocated']:.2f}GB allocated")
            else:
                print(f"❌ 배치 {batch_size:3d}: 실패 - {result['error']}")
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 결과 요약
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """결과 요약 출력"""
        print("\n" + "="*80)
        print("📊 배치 크기 테스트 결과 요약")
        print("="*80)
        
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("❌ 성공한 테스트가 없습니다.")
            return
        
        print(f"{'배치크기':<8} {'배치시간':<10} {'샘플/초':<10} {'Epoch시간':<12} {'GPU메모리':<10} {'효율성':<8}")
        print("-" * 70)
        
        # 기준 성능 (첫 번째 성공한 결과)
        baseline = successful_results[0]
        baseline_efficiency = baseline['samples_per_second']
        
        for result in successful_results:
            if not result['success']:
                continue
                
            batch_size = result['batch_size']
            batch_time = result['avg_batch_time']
            samples_per_sec = result['samples_per_second']
            epoch_time = format_time(result['estimated_epoch_time'])
            
            # GPU 메모리
            if torch.cuda.is_available():
                gpu_mem = f"{result['peak_memory_gb']:.1f}GB"
            else:
                gpu_mem = "N/A"
            
            # 효율성 (baseline 대비)
            efficiency = (samples_per_sec / baseline_efficiency) * 100
            efficiency_str = f"{efficiency:.0f}%"
            
            print(f"{batch_size:<8} {batch_time:<10.3f} {samples_per_sec:<10.1f} {epoch_time:<12} {gpu_mem:<10} {efficiency_str:<8}")
        
        # 최적 배치 크기 추천
        best_result = max(successful_results, key=lambda x: x['samples_per_second'])
        print(f"\n🏆 최고 성능: 배치 크기 {best_result['batch_size']} "
              f"({best_result['samples_per_second']:.1f} samples/s, "
              f"예상 epoch: {format_time(best_result['estimated_epoch_time'])})")
        
        # 메모리 효율성 분석
        if torch.cuda.is_available():
            print(f"\n💾 GPU 메모리 분석:")
            for result in successful_results:
                batch_size = result['batch_size']
                peak_mem = result['peak_memory_gb']
                samples_per_gb = result['samples_per_second'] / peak_mem if peak_mem > 0 else 0
                print(f"   배치 {batch_size}: {peak_mem:.2f}GB -> {samples_per_gb:.1f} samples/s/GB")
        
        print("\n📈 권장사항:")
        print(f"   - 최고 처리량: 배치 크기 {best_result['batch_size']}")
        
        # 메모리 효율성 고려 권장
        if torch.cuda.is_available():
            memory_efficient = min(successful_results, 
                                 key=lambda x: x['peak_memory_gb'] / x['samples_per_second'])
            print(f"   - 메모리 효율성: 배치 크기 {memory_efficient['batch_size']}")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='배치 크기 성능 테스트')
    parser.add_argument('--data_path', type=str, default='/workspace/AudioManipulator',
                       help='데이터셋 루트 경로')
    parser.add_argument('--num_test_batches', type=int, default=10,
                       help='각 배치 크기당 테스트할 배치 수')
    
    args = parser.parse_args()
    
    try:
        # 테스터 초기화
        tester = BatchSizeTester(data_path=args.data_path)
        
        # 모델 및 데이터 설정
        tester.setup_model()
        tester.load_test_data()
        
        # 종합 테스트 실행
        results = tester.run_comprehensive_test()
        
        print(f"\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
