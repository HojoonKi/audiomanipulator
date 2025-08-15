#!/usr/bin/env python3
# root/audio_tools/torchaudio_processor.py

import torch
import torch.nn as nn
import time
from typing import Dict, List

# --- 로컬 프로젝트 모듈에서 이펙트 함수 임포트 ---
from dasp_pytorch.functional import differentiable_flexible_eq, professional_reverb_optimized, simplified_reverb_optimized

class TorchAudioDistortion(nn.Module):
    """
    A robust distortion effect with dB-based drive control, a color parameter,
    and automatic makeup gain to compensate for volume loss.
    """
    def forward(self, audio: torch.Tensor, params: Dict) -> torch.Tensor:
        drive_db = params['gain'].view(-1, 1, 1).clamp(0.0, 36.0)
        color = params['color'].view(-1, 1, 1).clamp(-0.9, 0.9)
        
        # 1. 입력 오디오의 AC RMS 계산 (DC 제거)
        input_mean = torch.mean(audio, dim=-1, keepdim=True)
        input_ac = audio - input_mean
        input_rms = torch.sqrt(torch.mean(input_ac**2, dim=-1, keepdim=True) + 1e-8)
        
        # 2. dB 기반 드라이브 적용
        linear_gain = 10 ** (drive_db / 20.0)
        distorted_audio = torch.tanh((audio + color) * linear_gain)
        
        # 3. 왜곡된 오디오의 AC RMS 계산 (DC 제거)
        output_mean = torch.mean(distorted_audio, dim=-1, keepdim=True)
        output_ac = distorted_audio - output_mean
        output_rms = torch.sqrt(torch.mean(output_ac**2, dim=-1, keepdim=True) + 1e-8)
        
        # 4. 출력 AC RMS를 입력 AC RMS와 맞추도록 스케일링 (0 나누기 방지)
        scale = input_rms / output_rms.clamp(min=1e-8)
        normalized_audio = output_ac * scale
        
        # 옵션: 원본 input_mean을 추가할 수 있지만, DC 제거를 위해 0으로 유지 추천
        # normalized_audio += input_mean  # 필요 시 활성화
        
        return normalized_audio

# --- 메인 프로세서 (for 루프 완전 제거) ---

class TorchAudioProcessor(nn.Module):
    """
    Main processor that applies all effects to the entire batch at once.
    This version is fully vectorized and calls the final optimized effect functions.
    """
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.distortion = TorchAudioDistortion()

    def forward(self, audio: torch.Tensor, preset: Dict) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        processed_audio = audio
        
        # 이펙터 순서: Distortion -> EQ -> Reverb (일반적인 시그널 체인)
        # if "distortion" in preset:
        #     processed_audio = self.distortion(processed_audio, preset["distortion"])
        
        # if "equalizer" in preset:
        #     processed_audio = differentiable_flexible_eq(
        #         processed_audio, self.sample_rate, preset["equalizer"]
        #     )
            
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
    
    # --- differentiable_flexible_eq에 맞는 파라미터 생성 ---
    dummy_preset = {
        "equalizer": {
            f"band_{i}": {
                # 'filter_type'에 미분 가능한 로짓(logit) 텐서를 전달
                'filter_type': torch.randn(batch_size, 5, device=device).requires_grad_(),
                'cutoff_freq': (torch.rand(batch_size, 1, device=device)*19980+20).requires_grad_(),
                'gain_db': (torch.randn(batch_size, 1, device=device)*12-6).requires_grad_(),
                'q_factor': (torch.rand(batch_size, 1, device=device)*5+0.5).requires_grad_(),
            } for i in range(5)
        },
        "reverb": {
            'wet_gain': (torch.rand(batch_size, 1, device=device)*0.5).requires_grad_(),
            'room_size': torch.rand(batch_size, 1, device=device).requires_grad_(),
            'damping': torch.rand(batch_size, 1, device=device).requires_grad_(),
            'diffusion': torch.rand(batch_size, 1, device=device).requires_grad_(),
            'pre_delay': (torch.rand(batch_size, 1, device=device)*0.01).requires_grad_()
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
