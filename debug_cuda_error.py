#!/usr/bin/env python3
"""
CUDA 인덱싱 에러 디버깅 스크립트
"""

import torch
import torch.distributed as dist
import os

def debug_tensor_shapes():
    """텐서 모양 및 인덱싱 문제 디버깅"""
    print("🔍 CUDA 텐서 디버깅 시작...")
    
    # 기본 GPU 설정
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA 사용 불가")
        return
    
    try:
        # 다양한 크기의 텐서로 인덱싱 테스트
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            print(f"\n📊 배치 크기 {batch_size} 테스트:")
            
            # 오디오 텐서 생성 (5초, 44.1kHz)
            audio_length = int(44100 * 5.0)
            audio_tensor = torch.randn(batch_size, audio_length, device=device)
            print(f"   오디오 텐서: {audio_tensor.shape}")
            
            # 텍스트 임베딩 텐서 생성
            text_dim = 512
            text_tensor = torch.randn(batch_size, text_dim, device=device)
            print(f"   텍스트 텐서: {text_tensor.shape}")
            
            # 인덱싱 테스트
            try:
                # 정상적인 인덱싱
                subset_audio = audio_tensor[:batch_size//2] if batch_size > 1 else audio_tensor
                subset_text = text_tensor[:batch_size//2] if batch_size > 1 else text_tensor
                print(f"   ✅ 인덱싱 성공: {subset_audio.shape}, {subset_text.shape}")
                
                # 임베딩 테이블 시뮬레이션 (일반적인 에러 원인)
                vocab_size = 1000
                embedding = torch.nn.Embedding(vocab_size, text_dim).to(device)
                
                # 안전한 인덱스 생성
                indices = torch.randint(0, vocab_size, (batch_size, 10), device=device)
                embedded = embedding(indices)
                print(f"   ✅ 임베딩 성공: {embedded.shape}")
                
            except Exception as e:
                print(f"   ❌ 인덱싱 실패: {e}")
                
    except Exception as e:
        print(f"❌ 전체 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def debug_distributed_tensors():
    """분산 환경에서 텐서 동기화 테스트"""
    print("\n🔍 분산 텐서 디버깅...")
    
    # 분산 환경 확인
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"   분산 환경: Rank {rank}/{world_size}")
        
        try:
            # 각 GPU에서 다른 크기의 텐서 생성 (문제 시뮬레이션)
            base_size = 32
            tensor_size = base_size + rank  # 각 GPU마다 다른 크기!
            
            device = torch.device(f'cuda:{rank}')
            test_tensor = torch.randn(tensor_size, 512, device=device)
            
            print(f"   Rank {rank} 텐서 크기: {test_tensor.shape}")
            
            # 크기 동기화 테스트
            size_tensor = torch.tensor(tensor_size, device=device)
            dist.all_reduce(size_tensor, op=dist.ReduceOp.MIN)
            min_size = size_tensor.item()
            
            print(f"   최소 크기로 조정: {tensor_size} → {min_size}")
            
            # 크기 맞추기
            adjusted_tensor = test_tensor[:min_size]
            print(f"   ✅ 조정된 텐서: {adjusted_tensor.shape}")
            
        except Exception as e:
            print(f"   ❌ 분산 텐서 테스트 실패: {e}")
    else:
        print("   단일 GPU 환경")

if __name__ == "__main__":
    # CUDA 에러 디버깅 활성화
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    debug_tensor_shapes()
    debug_distributed_tensors()
    
    print("\n✅ 디버깅 완료!")
