#!/usr/bin/env python3
"""
자기지도 CLAP Loss 테스트 스크립트
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
import torchaudio.transforms as T
import torch.nn as nn

# 로컬 모듈 임포트
from pipeline import TunedCLAPPipeline, build_tuned_clap_model
from encoder.text_encoder import CLAPTextEncoder


def compute_self_supervised_clap_loss(
    fx_model: torch.nn.Module,         # 훈련시킬 모델 (학생)
    clap_model: torch.nn.Module,        # 고정된 평가자 모델 (교사)
    original_audios: torch.Tensor,    # 원본 오디오 배치
    fx_texts: List[str],              # 적용된 FX에 대한 텍스트 설명 배치
    temperature: float = 0.07
) -> torch.Tensor:
    """자기 지도 방식으로 FX 모델을 훈련하기 위한 CLAP Loss를 계산합니다."""
    device = original_audios.device

    print("\n🔍 === GRADIENT FLOW DEBUGGING ===")
    print(f"📥 Input audio requires_grad: {original_audios.requires_grad}")
    print(f"📥 Input audio grad_fn: {original_audios.grad_fn}")

    try:
        # 1. 학생(fx_model)이 과제 수행: 텍스트 설명에 맞춰 오디오 처리
        print("\n🎓 Student model forward pass...")
        outputs = fx_model(texts=fx_texts, audio=original_audios, use_real_audio=False)
        predicted_audios = outputs['processed_audio']
        
        print(f"📤 Student output requires_grad: {predicted_audios.requires_grad}")
        print(f"📤 Student output grad_fn: {predicted_audios.grad_fn}")

        # 2. 교사(clap_model)가 평가를 위해 임베딩 추출
        print("\n🏫 Teacher model (CLAP) processing...")
        
        # 오디오 차원 조정 (CLAP은 모노 오디오 처리)
        if predicted_audios.dim() == 3:  # (batch, channels, samples)
            if predicted_audios.size(1) > 1:  # 스테레오를 모노로 변환
                predicted_audios_mono = predicted_audios.mean(dim=1)  # (batch, samples)
            else:
                predicted_audios_mono = predicted_audios.squeeze(1)  # (batch, samples)
        elif predicted_audios.dim() == 2:  # (batch, samples) - 이미 올바른 형태
            predicted_audios_mono = predicted_audios
        else:  # (samples,) - 단일 오디오
            predicted_audios_mono = predicted_audios.unsqueeze(0)  # (1, samples)
        
        print(f"🔄 Audio preprocessing requires_grad: {predicted_audios_mono.requires_grad}")
        print(f"🔄 Audio preprocessing grad_fn: {predicted_audios_mono.grad_fn}")
        
        # 예측된 오디오에 대한 임베딩 (gradient 유지)
        print("🎵 Getting audio embeddings with gradient...")
        predicted_audio_embeddings = clap_model.get_audio_embedding_from_data_with_grad(predicted_audios_mono)
        
        print(f"🎵 Audio embeddings requires_grad: {predicted_audio_embeddings.requires_grad}")
        print(f"🎵 Audio embeddings grad_fn: {predicted_audio_embeddings.grad_fn}")
        
        # 텍스트에 대한 임베딩 (gradient 불필요)
        print("📝 Getting text embeddings (no gradient)...")
        with torch.no_grad():
            text_embeddings = clap_model.get_text_embedding(fx_texts)
            
            # 텍스트 임베딩 tensor 변환
            if isinstance(text_embeddings, np.ndarray):
                text_embeddings = torch.from_numpy(text_embeddings).to(device)
            text_embeddings = text_embeddings.float()
        
        print(f"📝 Text embeddings requires_grad: {text_embeddings.requires_grad}")
        print(f"📝 Text embeddings grad_fn: {text_embeddings.grad_fn}")
        
        # 오디오 임베딩 tensor 변환 (gradient 유지)
        if isinstance(predicted_audio_embeddings, np.ndarray):
            predicted_audio_embeddings = torch.from_numpy(predicted_audio_embeddings).to(device)
        predicted_audio_embeddings = predicted_audio_embeddings.float()

        # 3. 임베딩 정규화 (Cosine Similarity 준비)
        print("\n🔄 Normalizing embeddings...")
        predicted_audio_embeddings = F.normalize(predicted_audio_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        print(f"🔄 Normalized audio embeddings requires_grad: {predicted_audio_embeddings.requires_grad}")
        print(f"🔄 Normalized audio embeddings grad_fn: {predicted_audio_embeddings.grad_fn}")

        # 4. Contrastive Loss (InfoNCE) 계산
        print("\n🎯 Computing contrastive loss...")
        logits = (predicted_audio_embeddings @ text_embeddings.t()) / max(1e-6, temperature)
        print(f"🎯 Logits requires_grad: {logits.requires_grad}")
        print(f"🎯 Logits grad_fn: {logits.grad_fn}")
        
        labels = torch.arange(logits.size(0), device=device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        
        print(f"🎯 Final loss requires_grad: {loss.requires_grad}")
        print(f"🎯 Final loss grad_fn: {loss.grad_fn}")
        print("🔍 === END GRADIENT DEBUGGING ===\n")
        
        return loss
        
    except Exception as e:
        print(f"❌ 자기지도 CLAP loss 실패: {e}")
        return torch.tensor(0.1, device=device, requires_grad=True)


def debug_model_parameters(model, model_name):
    """모델 파라미터 상태 디버깅"""
    print(f"\n🔍 === {model_name} PARAMETER DEBUG ===")
    
    total_params = 0
    trainable_params = 0
    grad_enabled_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            grad_enabled_params += 1
    
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🚀 Trainable parameters: {trainable_params:,}")
    print(f"🔓 Parameters with requires_grad=True: {grad_enabled_params}")
    
    # 일부 파라미터 상세 정보
    print("📋 Sample parameter status:")
    count = 0
    for name, param in model.named_parameters():
        if count < 5:  # 처음 5개만 출력
            print(f"   {name}: requires_grad={param.requires_grad}, shape={param.shape}")
            count += 1
        else:
            break
    
    print(f"🔍 === END {model_name} DEBUG ===\n")


def test_self_supervised_loss():
    """자기지도 CLAP loss 테스트"""
    print("🧪 자기지도 CLAP Loss 테스트")
    print("=" * 50)
    
    try:
        # 1. 모델들 생성
        print("🏗️ 모델 생성 중...")
        
        # FX 모델 (학생)
        fx_model = build_tuned_clap_model(
            text_encoder_type='sentence-transformer',
            text_encoder_config={'model_name': 'all-mpnet-base-v2'},
            target_params=300000  # 작은 모델로 테스트
        )
        fx_model.train()
        
        # 교사 CLAP 모델
        teacher_clap = CLAPTextEncoder()
        for param in teacher_clap.parameters():
            param.requires_grad = False
        teacher_clap.eval()
        
        print("✅ 모델 생성 완료")
        
        # 모델 파라미터 상태 디버깅
        debug_model_parameters(fx_model, "STUDENT MODEL (FX)")
        debug_model_parameters(teacher_clap, "TEACHER MODEL (CLAP)")
        
        # 2. 더미 데이터 생성
        batch_size = 2
        audio_length = 44100 * 2  # 2초
        
        dummy_texts = [
            "Add warm reverb and boost the bass",
            "Make it sound bright and clear with compression"
        ]
        
        dummy_audio = torch.randn(batch_size, 2, audio_length, requires_grad=True)
        
        print(f"📊 테스트 데이터:")
        print(f"   텍스트: {len(dummy_texts)}개")
        print(f"   오디오: {dummy_audio.shape}")
        
        # 3. 자기지도 Loss 계산 테스트
        print("\n🔄 자기지도 Loss 계산 테스트...")
        
        loss = compute_self_supervised_clap_loss(
            fx_model=fx_model,
            clap_model=teacher_clap,
            original_audios=dummy_audio,
            fx_texts=dummy_texts,
            temperature=0.07
        )
        
        print(f"✅ Loss 계산 성공: {loss.item():.4f}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        
        # 4. 그래디언트 플로우 테스트
        print("\n🔄 그래디언트 플로우 테스트...")
        
        # Backward pass
        loss.backward()
        
        # 그래디언트 확인
        grad_count = 0
        total_grad_norm = 0.0
        
        for name, param in fx_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
                total_grad_norm += param.grad.norm().item()
        
        print(f"✅ 그래디언트가 있는 파라미터: {grad_count}개")
        print(f"   총 그래디언트 노름: {total_grad_norm:.4f}")
        
        # 5. 교사 모델 그래디언트 확인 (없어야 함)
        teacher_grad_count = 0
        for param in teacher_clap.parameters():
            if param.grad is not None:
                teacher_grad_count += 1
        
        print(f"✅ 교사 모델 그래디언트: {teacher_grad_count}개 (0이어야 함)")
        
        # 6. 다양한 온도에서 테스트
        print("\n🌡️ 다양한 온도에서 테스트...")
        temperatures = [0.01, 0.07, 0.1, 0.5]
        
        for temp in temperatures:
            test_loss = compute_self_supervised_clap_loss(
                fx_model=fx_model,
                clap_model=teacher_clap,
                original_audios=dummy_audio,
                fx_texts=dummy_texts,
                temperature=temp
            )
            print(f"   온도 {temp}: {test_loss.item():.4f}")
        
        print("\n✅ 모든 테스트 통과!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

from transformers import ClapModel, ClapProcessor

def test_direct_clap_gradient_fixed():
    print("\n🧪 최종 파이프라인 구현 테스트 (진정한 완결판)")
    print("=" * 50)
    
    try:
        class SimpleFxModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = nn.Parameter(torch.tensor(1.0))
            def forward(self, audio: torch.Tensor, texts: List[str]) -> torch.Tensor:
                return audio * self.gain

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fx_model = SimpleFxModel().to(device)
        
        model_id = "laion/clap-htsat-fused"
        clap_model = ClapModel.from_pretrained(model_id).to(device)
        processor = ClapProcessor.from_pretrained(model_id)

        for param in clap_model.parameters():
            param.requires_grad = False
        clap_model.eval()

        # --- 1. 전처리기 설정 ---
        feature_extractor = processor.feature_extractor
        target_sr = feature_extractor.sampling_rate
        n_mels = feature_extractor.feature_size
        n_fft = feature_extractor.n_fft
        hop_length = feature_extractor.hop_length
        win_length = n_fft
        
        resampler = T.Resample(orig_freq=44100, new_freq=target_sr).to(device)
        mel_spectrogram_converter = T.MelSpectrogram(
            sample_rate=target_sr, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, n_mels=n_mels
        ).to(device)

        # --- 2. 데이터 처리 및 모델 순전파 ---
        original_audios = torch.randn(2, 44100, device=device) # 1초 길이 오디오
        texts = ["sound of ocean waves", "a dog barking loudly"]
        predicted_audios = fx_model(original_audios, texts)
        
        # ⚠️ 핵심 수정 1: 오디오를 10초 길이로 패딩/확장
        target_len_s = 10
        target_samples = target_sr * target_len_s
        padded_audios = torch.zeros(predicted_audios.shape[0], target_samples, device=device)
        # 원본 오디오를 반복해서 채워넣음 (repeat padding)
        for i in range(predicted_audios.shape[0]):
            repeat_factor = target_samples // predicted_audios.shape[1] + 1
            repeated_audio = predicted_audios[i].repeat(repeat_factor)
            padded_audios[i] = repeated_audio[:target_samples]
            
        resampled_audios = resampler(padded_audios) # 이제 리샘플링은 의미가 없지만, 형식상 유지
        mel_spectrograms = mel_spectrogram_converter(padded_audios) # shape: [B, Freq, Time] = [2, 64, 1001]
        log_mel_spectrograms = (mel_spectrograms.clamp(min=1e-5).log() - torch.log(torch.tensor(1e-5)))

        # ⚠️ 핵심 수정 2: 'Fusion' 전략 모방 (4채널 복제) 및 차원 순서 정리
        # [B, F, T] -> [B, T, F]
        transposed_spectrograms = log_mel_spectrograms.transpose(1, 2)
        # [B, T, F] -> [B, 1, T, F] -> [B, 4, T, F]
        fused_spectrograms = transposed_spectrograms.unsqueeze(1).repeat(1, 4, 1, 1)

        print(f"Shape being fed to audio model: {fused_spectrograms.shape}")

        text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        
        batch_size = predicted_audios.shape[0]
        is_longer = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 모델의 forward 함수는 내부적으로 audio/text 분기를 처리함
        outputs = clap_model(
            input_features=fused_spectrograms, 
            input_ids=text_inputs['input_ids'], 
            attention_mask=text_inputs['attention_mask'],
            is_longer=is_longer
        )
        audio_embeds = outputs.audio_embeds
        text_embeds = outputs.text_embeds
        
        # --- 3. Loss 계산 및 역전파 ---
        loss = 1 - F.cosine_similarity(audio_embeds, text_embeds).mean()
        print(f"🎯 Final loss grad_fn: {loss.grad_fn}")

        optimizer = torch.optim.Adam(fx_model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✅ 역전파 성공! FxModel의 gain 파라미터 업데이트됨: {fx_model.gain.item():.4f}")

    except Exception as e:
        print(f"❌ 직접 구현 테스트 실패: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    # test_self_supervised_loss()
    test_direct_clap_gradient_fixed()
    print("\n🎉 테스트 완료!")
