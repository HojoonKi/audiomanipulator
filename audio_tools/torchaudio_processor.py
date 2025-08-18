#!/usr/bin/env python3
# root/audio_tools/torchaudio_processor.py

import torch
import torch.nn as nn
import time
from typing import Dict, List

# --- 로컬 프로젝트 모듈에서 이펙트 함수 임포트 ---
from dasp_pytorch.functional import mean_distortion, differentiable_hybrid_eq, differentiable_flexible_eq, professional_reverb_optimized



class TorchAudioProcessor(nn.Module):
    """
    Main processor that applies all effects to the entire batch at once.
    This version is fully vectorized and calls the final optimized effect functions.
    """
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.training = True

    def forward(self, audio: torch.Tensor, preset: Dict) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        processed_audio = audio
        
        # 이펙터 순서: Distortion -> EQ -> Reverb (일반적인 시그널 체인)
        if "distortion" in preset:
            processed_audio = mean_distortion(processed_audio, preset["distortion"])
        
        if "equalizer" in preset:
            processed_audio = differentiable_hybrid_eq(
                processed_audio, self.sample_rate, preset["equalizer"], training=self.training
            )
            
        if "reverb" in preset:
            # Reverb 함수는 파라미터 딕셔너리를 키워드 인자로 받음
            processed_audio = professional_reverb_optimized(
                processed_audio, self.sample_rate, **preset["reverb"]
            )
            
        return processed_audio

# --- 테스트 코드 ---

if __name__ == '__main__':
    print("\n🧪 Testing Final Differentiable Audio Processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    sample_rate = 48000
    num_samples = sample_rate * 1 # 테스트를 위해 1초로 줄임

    print(f"Batch size: {batch_size}, Device: {device}, Audio length: 1s")

    test_audio = torch.randn(batch_size, 1, num_samples, device=device)
    
    # --- differentiable_hybrid_eq에 맞는 파라미터 생성 ---
    preset_template = [
        # Band 1: Low-Shelf Boost
        {'type_logits': [-5.0, 5.0], 'freq': 100.0, 'gain': 3.0, 'q': 0.7},
        # Band 2: Low-Mid Cut
        {'type_logits': [1.0], 'freq': 350.0, 'gain': -2.0, 'q': 1.5},
        # Band 3: Mid Scoop
        {'type_logits': [1.0], 'freq': 1000.0, 'gain': -6.0, 'q': 1.0},
        # Band 4: Upper-Mid Presence Boost
        {'type_logits': [1.0], 'freq': 3000.0, 'gain': 2.0, 'q': 1.5},
        # Band 5: High Sibilance Cut
        {'type_logits': [1.0], 'freq': 7000.0, 'gain': 8.0, 'q': 3.0},
        # Band 6: High-Shelf Air Boost
        {'type_logits': [5.0, -5.0], 'freq': 12000.0, 'gain': 4.0, 'q': 0.7},
    ]
    dummy_preset = {
        "equalizer": {
            f"band_{i+1}": {
                'filter_type': (torch.tensor(preset_template[i]['type_logits'], device=device)
                                .unsqueeze(0).expand(batch_size, -1) 
                                + torch.randn(batch_size, len(preset_template[i]['type_logits']), device=device) * 0.1
                               ).requires_grad_(),
                'center_freq': (torch.full((batch_size, 1), preset_template[i]['freq'], device=device)
                                + torch.randn(batch_size, 1, device=device) * 10
                               ).requires_grad_(),
                'gain_db': (torch.full((batch_size, 1), preset_template[i]['gain'], device=device)
                            + torch.randn(batch_size, 1, device=device) * 0.5
                           ).requires_grad_(),
                'q': (torch.full((batch_size, 1), preset_template[i]['q'], device=device)
                      + torch.rand(batch_size, 1, device=device) * 0.2 - 0.1
                     ).requires_grad_(),
            } for i in range(6)
        },
        "reverb": {
            # Reverb와 Distortion은 기존 랜덤 방식 유지
            'wet_gain': (torch.rand(batch_size, 1, device=device)*0.5).requires_grad_(),
            'room_size': (torch.rand(batch_size, 1, device=device) * 10.0).requires_grad_(),
            'damping': torch.rand(batch_size, 1, device=device).requires_grad_(),
            'diffusion': torch.rand(batch_size, 1, device=device).requires_grad_(),
            'pre_delay': (torch.rand(batch_size, 1, device=device) * 0.1).requires_grad_(),
            'decay_time': (torch.rand(batch_size, 1, device=device) * 4).requires_grad_(),
        },
        "distortion": {
            'gain': (torch.rand(batch_size, 1, device=device)*9+1).requires_grad_(),
            'color': torch.randn(batch_size, 1, device=device).requires_grad_(),
        },
    }
    
    processor = TorchAudioProcessor(sample_rate=sample_rate).to(device)
    processor.train() # EQ가 self.training 상태를 사용하므로 train 모드로 설정

    # --- 성능 테스트 ---
    if device == 'cuda': torch.cuda.synchronize()
    start_time = time.time()
    processed_audio = processor(test_audio, dummy_preset)
    if device == 'cuda': torch.cuda.synchronize()
    execution_time = time.time() - start_time
    print(f"\nForward pass execution time for batch of {batch_size}: {execution_time:.4f} seconds")

    # --- 그래디언트 테스트 ---
    print("\n🔄 Testing backward pass...")
    try:
        loss = processed_audio.mean()
        loss.backward()
        print("\n✅ Backward pass successful.")
        # (더 상세한 그래디언트 검사 로직은 이전 답변을 참고하여 추가 가능)
    except Exception as e:
        print("\n❌ An error occurred during the backward pass.")
        raise e

    print(f"\n✅ Processing successful!")
    print(f"   Input audio shape:  {test_audio.shape}")
    print(f"   Output audio shape: {processed_audio.shape}")
