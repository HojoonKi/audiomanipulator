#!/usr/bin/env python3
# root/audio_tools/torchaudio_processor.py

import torch
import torch.nn as nn
import time
from typing import Dict, List

# --- 로컬 프로젝트 모듈에서 이펙트 함수 임포트 ---
from dasp_pytorch.functional import flexible_five_band_eq, professional_reverb_optimized, simplified_reverb_optimized

class TorchAudioDistortion(nn.Module):
    """Fully vectorized distortion effect."""
    def forward(self, audio: torch.Tensor, params: Dict) -> torch.Tensor:
        # 파라미터를 (B, 1, 1) 형태로 바꿔 오디오 (B, C, T)와 브로드캐스팅
        gain = params['gain'].view(-1, 1, 1).clamp(1.0, 10.0)
        color = params['color'].view(-1, 1, 1).clamp(-1.0, 1.0)
        return torch.tanh((audio + color) * gain)

# --- 메인 프로세서 (for 루프 완전 제거) ---

class TorchAudioProcessor(nn.Module):
    """
    Main processor that applies all effects to the entire batch at once.
    This version is fully vectorized and does not use a Python for-loop.
    """
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.distortion = TorchAudioDistortion()
        # EQ와 Reverb는 stateless 함수이므로 __init__에서 생성할 필요 없음

    def forward(self, audio: torch.Tensor, preset: Dict) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        # --- for 루프 없이 전체 배치를 한 번에 처리 ---
        processed_audio = audio
        
        if "distortion" in preset:
            processed_audio = self.distortion(processed_audio, preset["distortion"])
        
        if "equalizer" in preset:
            # flexible_five_band_eq는 이미 배치 처리를 지원
            processed_audio = flexible_five_band_eq(
                processed_audio, self.sample_rate, preset["equalizer"]
            )
            
        if "reverb" in preset:
            # reverb 함수가 배치 파라미터를 받도록 **preset["reverb"]를 전달
            processed_audio = professional_reverb_optimized(
                processed_audio, self.sample_rate, **preset["reverb"]
            )
            
        return processed_audio

# --- 테스트 코드 ---

if __name__ == '__main__':
    print("\n🧪 Testing Differentiable Audio Processor (Fully Vectorized)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 배치 크기를 늘려 성능 향상 체감
    batch_size = 32 
    sample_rate = 48000
    num_samples = sample_rate * 5

    print(f"Batch size: {batch_size}, Device: {device}")

    test_audio = torch.randn(batch_size, 1, num_samples, device=device)
    
    # 파라미터 생성 (이전과 동일)
    eq_band_types = ["low_shelf", "peaking", "peaking", "peaking", "high_shelf"]
    dummy_preset = {
        "equalizer": [
            {
                'filter_type': eq_band_types[i],
                'cutoff_freq': (torch.rand(batch_size, 1, device=device)*19980+20).requires_grad_(),
                'gain_db': (torch.randn(batch_size, 1, device=device)*10).requires_grad_(),
                'q_factor': (torch.rand(batch_size, 1, device=device)*5+0.5).requires_grad_(),
            } for i in range(5)
        ],
        "reverb": {
            'wet_gain': (torch.rand(batch_size, 1, device=device)*0.5).requires_grad_(),
            'room_size': torch.rand(batch_size, 1, device=device).requires_grad_(),
            'damping': torch.rand(batch_size, 1, device=device).requires_grad_(),
            'diffusion': torch.rand(batch_size, 1, device=device).requires_grad_(),
            'pre_delay_ms': (torch.rand(batch_size, 1, device=device)*100).requires_grad_()
        },
        "distortion": {
            'gain': (torch.rand(batch_size, 1, device=device)*9+1).requires_grad_(),
            'color': torch.randn(batch_size, 1, device=device).requires_grad_(),
        },
    }
    
    processor = TorchAudioProcessor(sample_rate=sample_rate).to(device)
    
    # --- 성능 테스트 ---
    if device == 'cuda': torch.cuda.synchronize()
    start_time = time.time()
    processed_audio = processor(test_audio, dummy_preset)
    if device == 'cuda': torch.cuda.synchronize()
    execution_time = time.time() - start_time
    print(f"\nExecution time for batch of {batch_size}: {execution_time:.4f} seconds")

    # --- 그래디언트 테스트 ---
    print("\n🔄 Testing backward pass...")
    try:
        loss = processed_audio.mean()
        loss.backward()
        print("\n✅ Backward pass successful.")
    except Exception as e:
        print("\n❌ An error occurred during the backward pass.")
        raise e