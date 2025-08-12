#!/usr/bin/env python3
"""
Accelerate Training Script
- Hugging Face Accelerate 사용
- DDP의 복잡성 제거
- 자동 분산 처리
"""

import os
# 환경 변수는 최대한 이른 시점에 설정되어야 함 (Torch/Transformers 로드 전에)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import traceback
from datetime import datetime

# Accelerate 임포트
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs

# 로컬 모듈 임포트
sys.path.append('/app')
from pipeline import TunedCLAPPipeline, build_tuned_clap_model
from encoder.text_encoder import CLAPTextEncoder
from dataset import (
    PureDescriptionDataset,
    PretrainDataset,  # 사전 훈련 전용 데이터셋 추가
    custom_collate_no_guide,
    custom_collate_with_guide,
    load_descriptions, 
    split_descriptions
)
from typing import List


def create_model_and_optimizer(args):
    """모델 생성 - TunedCLAPPipeline + 별도 CLAP encoder"""
    # 결정론/시드 고정
    set_seed(getattr(args, 'seed', 42))
    
    # TunedCLAPPipeline 생성 - Factory 함수 사용
    text_encoder_type = getattr(args, 'text_encoder_type', 'sentence-transformer')
    text_encoder_config = {}
    if text_encoder_type == 'sentence-transformer':
        text_encoder_config = {'model_name': 'all-mpnet-base-v2'}
    
    model = build_tuned_clap_model(
        text_encoder_type=text_encoder_type,
        text_encoder_config=text_encoder_config,
        sample_rate=args.sample_rate,
        freeze_text_encoder=True,
        target_params=500000
    )
    
    # 별도의 교사 CLAP encoder 생성 (frozen)
    print("🎵 Loading teacher CLAP encoder...")
    teacher_clap = CLAPTextEncoder()
    # CLAP 모델 frozen 상태로 설정
    for param in teacher_clap.parameters():
        param.requires_grad = False
    teacher_clap.eval()  # 평가 모드로 설정
    
    # 어댑터만 훈련하도록 설정
    print("🔧 Setting up adapter-only training...")
    
    # 1. 모든 파라미터를 frozen으로 설정
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. 어댑터 관련 파라미터만 훈련 가능하도록 설정
    adapter_params = []
    for name, param in model.named_parameters():
        # TunedCLAPWithAdapters의 어댑터 관련 파라미터만 훈련
        # - backbone.adapters.*: CrossAttentionAdapter 모듈들
        # - backbone.final_norm.*: 최종 정규화 레이어
        if any(adapter_key in name for adapter_key in [
            'backbone.adapters',  # CrossAttentionAdapter들
            'backbone.final_norm',  # 최종 LayerNorm
            'decoder.',  # decoder는 항상 훈련
        ]):
            param.requires_grad = True
            adapter_params.append(param)
            print(f"  ✅ Training: {name}")
    
    print(f"📊 Total adapter parameters: {sum(p.numel() for p in adapter_params):,}")
    
    # 3. 어댑터 파라미터 초기화
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    
    # 4. 어댑터 파라미터만으로 옵티마이저 생성
    if not adapter_params:
        raise RuntimeError("No adapter parameters found for training!")
    
    optimizer = optim.AdamW(
        adapter_params,  # 어댑터 파라미터만 전달
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    return model, optimizer, teacher_clap


def create_datasets(args):
    """데이터셋 생성 - 분산 샘플러 불필요"""
    # 데이터 로드
    descriptions = load_descriptions(
        data_path=args.data_path,
        use_sampled_descriptions=False,
        max_descriptions=100000
    )
    
    if not descriptions:
        raise RuntimeError("Description 로드 실패")
    
    # 데이터 분할
    train_descriptions, val_descriptions = split_descriptions(descriptions, train_ratio=0.8)
    
    print(f"📚 데이터셋: 훈련={len(train_descriptions)}, 검증={len(val_descriptions)}")
    
    # 데이터셋 생성
    train_dataset = PureDescriptionDataset(
        descriptions=train_descriptions,
        audio_dataset_path=os.path.join(args.data_path, 'audio_dataset'),
        sample_rate=args.sample_rate,
        audio_length=args.audio_length
    )
    
    val_dataset = PureDescriptionDataset(
        descriptions=val_descriptions,
        audio_dataset_path=os.path.join(args.data_path, 'audio_dataset'),
        sample_rate=args.sample_rate,
        audio_length=args.audio_length
    )
    
    # 일반 데이터로더 - Accelerate가 자동으로 분산 처리
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Accelerate가 분산 환경에서 자동 처리
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_no_guide,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_no_guide,
        drop_last=True
    )
    
    return train_loader, val_loader


def compute_self_supervised_clap_loss(
    fx_model: torch.nn.Module,         # 훈련시킬 모델 (학생)
    clap_model: torch.nn.Module,        # 고정된 평가자 모델 (교사)
    original_audios: torch.Tensor,    # 원본 오디오 배치
    fx_texts: List[str],              # 적용된 FX에 대한 텍스트 설명 배치
    temperature: float = 0.07
) -> torch.Tensor:
    """
    자기 지도 방식으로 FX 모델을 훈련하기 위한 CLAP Loss를 계산합니다.
    업데이트된 CLAPTextEncoder의 새로운 메서드들을 사용합니다.
    
    Args:
        fx_model: 훈련 대상인 FX 적용 모델 (TunedCLAPPipeline).
        clap_model: 가중치가 고정된(frozen) 사전 훈련된 CLAP 모델.
        original_audios: 원본 오디오 텐서 배치.
        fx_texts: 각 오디오에 적용된 랜덤 FX를 설명하는 텍스트 리스트.
        temperature: Contrastive Loss의 온도 파라미터.
        
    Returns:
        torch.Tensor: FX 모델 업데이트를 위한 스칼라 손실 값.
    """
    device = original_audios.device

    try:
        # 1. 학생(fx_model)이 과제 수행: 텍스트 설명에 맞춰 오디오 처리
        with torch.cuda.amp.autocast(enabled=False):  # 안정성을 위해 FP32 유지
            outputs = fx_model(texts=fx_texts, audio=original_audios, use_real_audio=False)
            predicted_audios = outputs['processed_audio']

        # 2. 오디오 차원 조정 (CLAP은 모노 오디오 처리) - gradient 유지
        if predicted_audios.dim() == 3:  # (batch, channels, samples)
            if predicted_audios.size(1) > 1:  # 스테레오를 모노로 변환
                predicted_audios_mono = predicted_audios.mean(dim=1)  # (batch, samples)
            else:
                predicted_audios_mono = predicted_audios.squeeze(1)  # (batch, samples)
        elif predicted_audios.dim() == 2:  # (batch, samples) - 이미 올바른 형태
            predicted_audios_mono = predicted_audios
        else:  # (samples,) - 단일 오디오
            predicted_audios_mono = predicted_audios.unsqueeze(0)  # (1, samples)
        
        # 3. 업데이트된 CLAPTextEncoder의 compute_clap_loss 메서드 직접 사용
        # 이 메서드는 내부적으로 gradient flow가 보장된 방식으로 구현되어 있음
        loss = clap_model.compute_clap_loss(predicted_audios_mono, fx_texts)
        
        return loss
        
    except Exception as e:
        print(f"자기지도 CLAP loss 실패: {e}")
        import traceback
        traceback.print_exc()
        return torch.tensor(0.1, device=device, requires_grad=True)


def train_epoch(model, train_loader, optimizer, accelerator, epoch, teacher_clap, args):
    """훈련 에포크"""
    model.train()
    total_loss = 0.0
    
    # 메인 프로세스에서만 진행 표시줄 생성
    pbar = train_loader
    if accelerator.is_main_process:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            descriptions = batch['description']
            audios = batch['audio']
            
            # 차원 확인
            if audios.dim() == 2:
                audios = audios.unsqueeze(1)
            
            # 자기지도 CLAP 손실 계산
            # model(fx_model), teacher_clap, original_audio, descriptions 사용
            loss = compute_self_supervised_clap_loss(
                fx_model=model,
                clap_model=teacher_clap,
                original_audios=audios,
                fx_texts=descriptions,
                temperature=0.07
            )
            
            # Backward pass - accelerator 사용
            optimizer.zero_grad()
            accelerator.backward(loss)  # loss.backward() 대신
            
            # Gradient clipping
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # 진행 표시줄 업데이트 (메인 프로세스에서만)
            if accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'AvgLoss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
                # wandb logging (배치별) - 50번마다만 로깅
                if args.use_wandb and batch_idx % 50 == 0:
                    global_step = epoch * len(train_loader) + batch_idx
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/batch_idx': batch_idx,
                        'train/epoch': epoch + 1
                    }, step=global_step)
                
        except Exception as e:
            if accelerator.is_main_process:
                print(f"배치 {batch_idx} 에러: {e}")
            continue
    
    # 평균 손실 계산 - accelerator가 자동으로 분산 평균 처리
    avg_loss = total_loss / len(train_loader)
    
    # 모든 프로세스 간 평균 계산
    if accelerator.num_processes > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
        avg_loss = accelerator.gather(avg_loss_tensor).mean().item()
    
    return avg_loss


def validate(model, val_loader, accelerator, teacher_clap):
    """검증"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                descriptions = batch['description']
                audios = batch['audio']
                
                if audios.dim() == 2:
                    audios = audios.unsqueeze(1)
                
                # 자기지도 CLAP 손실 계산 (검증용)
                loss = compute_self_supervised_clap_loss(
                    fx_model=model,
                    clap_model=teacher_clap,
                    original_audios=audios,
                    fx_texts=descriptions,
                    temperature=0.07
                )
                total_loss += loss.item()
                
            except Exception as e:
                continue
    
    # 평균 손실 계산
    avg_loss = total_loss / len(val_loader)
    
    # 분산 평균
    if accelerator.num_processes > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
        avg_loss = accelerator.gather(avg_loss_tensor).mean().item()
    
    return avg_loss


def extract_guide_values(guide_preset):
    """Guide preset에서 28개 값 추출 - fined_presets_filtered.py 구조에 맞춤"""
    values = []
    
    # Filter type 매핑 (문자열 -> 숫자)
    filter_type_mapping = {
        "low-shelf": 0,
        "bell": 1, 
        "high-shelf": 2,
        "highpass": 3,
        "lowpass": 4,
        # 추가 호환성
        "notch": 1,  # notch는 bell과 유사
        "low_shelf": 0,  # 언더스코어 버전
        "high_shelf": 2,
        "low_pass": 4,
        "high_pass": 3
    }
    
    # EQ (20개): frequency, gain, q, filter_type × 5
    eq_section = guide_preset['Equalizer']  # 리스트 형태
    for eq_band in eq_section:  # 5개 밴드
        # filter_type을 숫자로 변환
        filter_type_str = eq_band['filter_type']
        filter_type_num = filter_type_mapping.get(filter_type_str, 1)  # 기본값: bell (1)
        
        values.extend([
            eq_band['frequency'],
            eq_band['gain'], 
            eq_band['q'],
            filter_type_num
        ])
    
    # Reverb (5개)
    reverb = guide_preset['Reverb']
    values.extend([
        reverb['room_size'],
        reverb['pre_delay'], 
        reverb['diffusion'],
        reverb['damping'],
        reverb['wet_gain']
    ])
    
    # Distortion (2개)
    dist = guide_preset['Distortion']
    values.extend([
        dist['gain'],
        dist['color']
    ])
    
    # Pitch (1개)
    pitch = guide_preset['Pitch']
    values.append(pitch['scale'])
    
    return values  # 28개 값 보장됨


def compute_guide_loss(model, generated_preset, guide_preset, device):
    """Guide preset과의 차이를 이용한 간단한 MSE loss"""
    try:
        # 디코더에서 _raw_params 직접 추출
        if isinstance(generated_preset, dict) and "_raw_params" in generated_preset:
            generated_tensor = generated_preset["_raw_params"].to(device)
            
            # Guide preset을 간단한 tensor로 변환
            guide_values = extract_guide_values(guide_preset)
            if guide_values is None:
                return torch.tensor(0.1, device=device, requires_grad=True)
            
            guide_tensor = torch.FloatTensor(guide_values).to(device)
            
            # 배치 차원 처리
            if generated_tensor.dim() == 2:
                generated_tensor = generated_tensor.squeeze(0)
            
            # 직접 MSE 계산
            mse_loss = nn.MSELoss()(generated_tensor, guide_tensor)
            
            return mse_loss
        else:
            return torch.tensor(0.1, device=device, requires_grad=True)
            
    except Exception as e:
        print(f"❌ Guide loss 실패: {e}")
        return torch.tensor(0.1, device=device, requires_grad=True)


def compute_batch_guide_loss(model, batch_generated_preset, batch_guide_presets, device):
    """배치 단위 Guide Loss 계산"""
    try:
        # 배치 단위로 _raw_params 추출
        if isinstance(batch_generated_preset, dict) and "_raw_params" in batch_generated_preset:
            generated_batch_tensor = batch_generated_preset["_raw_params"].to(device)
            
            if generated_batch_tensor.dim() == 3:
                generated_batch_tensor = generated_batch_tensor.squeeze(1)  # [batch_size, 28]
            
            # 모든 preset이 이미 검증됨, 직접 변환
            batch_guide_values = []
            
            for guide_preset in batch_guide_presets:
                guide_values = extract_guide_values(guide_preset)
                batch_guide_values.append(guide_values)
            
            # Guide values를 배치 텐서로 스택
            guide_batch_tensor = torch.FloatTensor(batch_guide_values).to(device)  # [batch_size, 28]
            
            # 배치 MSE loss 계산
            batch_mse_loss = nn.MSELoss()(generated_batch_tensor, guide_batch_tensor)
            
            return batch_mse_loss
            
        else:
            return torch.tensor(0.1, device=device, requires_grad=True)
            
    except Exception as e:
        print(f"❌ Batch guide loss 실패: {e}")
        return torch.tensor(0.1, device=device, requires_grad=True)


def simple_pretrain(model, optimizer, accelerator, args, teacher_clap):
    """간단한 사전 훈련 - Fine Preset만 사용"""
    if not args.enable_pretrain:
        return
    
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("🎯 사전 훈련 시작 (Fine Preset Only - No Descriptions)")
        print("="*60)
        print(f"   - 사전 훈련 에포크: {args.pretrain_epochs}")
        print(f"   - 사전 훈련 학습률: {args.pretrain_lr}")
        print(f"   - 모드: Fine Preset 파라미터 매칭만")
    
    # Fine preset 경로
    fine_preset_path = os.path.join(args.data_path, 'descriptions', 'fined_presets_filtered.py')
    
    # 기존 옵티마이저 백업
    original_lr = args.learning_rate
    
    # 사전 훈련용 옵티마이저 생성
    pretrain_params = [p for p in model.parameters() if p.requires_grad]
    pretrain_optimizer = optim.AdamW(
        pretrain_params,
        lr=args.pretrain_lr,
        weight_decay=0.01
    )
    
    # 사전 훈련용 스케줄러
    pretrain_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        pretrain_optimizer, T_max=args.pretrain_epochs
    )
    
    try:
        # 사전 훈련 전용 데이터셋 로드
        if accelerator.is_main_process:
            print(f"🔍 DEBUG: 사전 훈련 데이터셋 로딩 시작...")
            print(f"   - Fine preset 경로: {fine_preset_path}")
            print(f"   - Audio 경로: {os.path.join(args.data_path, 'audio_dataset')}")
        
        train_dataset = PretrainDataset(
            fine_preset_path=fine_preset_path,
            audio_dataset_path=os.path.join(args.data_path, 'audio_dataset'),
            sample_rate=args.sample_rate,
            audio_length=args.audio_length
        )
        

        
        # 검증용으로는 적은 수의 샘플만 사용
        val_size = min(100, len(train_dataset) // 10)
        val_indices = random.sample(range(len(train_dataset)), val_size)
        train_indices = [i for i in range(len(train_dataset)) if i not in val_indices]
        
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)
        

        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_subset,
            batch_size=args.pretrain_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=custom_collate_with_guide,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=args.pretrain_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=custom_collate_with_guide,
            drop_last=True
        )
        
        # Accelerate로 준비
        pretrain_optimizer, train_loader, val_loader = accelerator.prepare(
            pretrain_optimizer, train_loader, val_loader
        )
        
        if accelerator.is_main_process:
            print(f"🔍 DEBUG: DataLoader 생성 완료:")
            print(f"   - 훈련 배치 크기: {args.pretrain_batch_size}")
            print(f"   - 훈련 샘플: {len(train_subset)}")
            print(f"   - 검증 샘플: {len(val_subset)}")
            print(f"   - 예상 훈련 배치 수: {len(train_loader)}")
            print(f"   - 예상 검증 배치 수: {len(val_loader)}")
            print(f"   - drop_last=True로 설정됨")
        
        # 사전 훈련 루프
        best_pretrain_loss = float('inf')
        
        for epoch in range(args.pretrain_epochs):
            # 사전 훈련 에포크
            train_loss = pretrain_epoch(model, train_loader, pretrain_optimizer, accelerator, epoch, args)
            
            # 검증
            val_loss = pretrain_validate(model, val_loader, accelerator, args)
            
            # 스케줄러 업데이트
            pretrain_scheduler.step()
            
            if accelerator.is_main_process:
                current_lr = pretrain_scheduler.get_last_lr()[0]
                print(f"Pretrain {epoch+1}/{args.pretrain_epochs}: "
                      f"Loss={train_loss:.6f}, Val={val_loss:.6f}, LR={current_lr:.1e}")
                
                # Wandb 사전 훈련 로깅
                if args.use_wandb:
                    wandb.log({
                        'pretrain/epoch': epoch + 1,
                        'pretrain/train_loss': train_loss,
                        'pretrain/val_loss': val_loss,
                        'pretrain/learning_rate': current_lr,
                        'pretrain/phase': 'guide_preset_only'
                    }, step=epoch + 1)
            
            # 최고 성능 추적
            if val_loss < best_pretrain_loss:
                best_pretrain_loss = val_loss
        
        if accelerator.is_main_process:
            print(f"✅ 사전 훈련 완료! 최고 성능: {best_pretrain_loss:.6f}")
            print("="*60)
    
    except Exception as e:
        if accelerator.is_main_process:
            print(f"❌ 사전 훈련 실패: {e}")
            traceback.print_exc()
    
    finally:
        # 원래 설정 복원
        args.learning_rate = original_lr


def pretrain_epoch(model, train_loader, optimizer, accelerator, epoch, args):
    """사전 훈련 전용 에포크"""
    model.train()
    total_loss = 0.0
    
    
    pbar = train_loader
    if accelerator.is_main_process:
        pbar = tqdm(train_loader, desc=f"Pretrain E{epoch+1}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            descriptions = batch['description']
            audios = batch['audio']
            guide_presets = batch.get('guide_preset', [])
            

            
            # 차원 확인
            if audios.dim() == 2:
                audios = audios.unsqueeze(1)
            
            # 유효한 guide preset 필터링
            valid_indices = []
            valid_descriptions = []
            valid_audios = []
            valid_guide_presets = []
            
            for i, guide_preset in enumerate(guide_presets):
                if guide_preset:  # 빈 딕셔너리가 아니면 유효
                    valid_indices.append(i)
                    valid_descriptions.append(descriptions[i])
                    valid_audios.append(audios[i])
                    valid_guide_presets.append(guide_preset)
            
            
            if len(valid_indices) == 0:
                if batch_idx < 5 and accelerator.is_main_process:  # 처음 5개 배치만 경고
                    print(f"⚠️ 배치 {batch_idx}: 유효한 guide preset이 없음")
                continue
            
            # 배치 처리
            valid_audio_batch = torch.stack(valid_audios)
            
            outputs = model(
                texts=valid_descriptions,
                audio=valid_audio_batch,
                use_real_audio=False
            )
            
            if 'preset_params' not in outputs:
                continue
            
            # Guide Loss 계산
            batch_loss = compute_batch_guide_loss(
                model, outputs['preset_params'], valid_guide_presets, accelerator.device
            )
            
            # Backward pass
            optimizer.zero_grad()
            accelerator.backward(batch_loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            # Progress bar 업데이트
            if accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'Loss': f'{batch_loss.item():.4f}',
                    'AvgLoss': f'{total_loss/(batch_idx+1):.4f}',
                    'Valid': f'{len(valid_indices)}/{len(descriptions)}'
                })
                
        except Exception as e:
            if accelerator.is_main_process:
                print(f"❌ Pretrain batch {batch_idx} 실패: {e}")
            continue
    
    # 평균 손실 계산
    avg_loss = total_loss / len(train_loader)
    if accelerator.num_processes > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
        avg_loss = accelerator.gather(avg_loss_tensor).mean().item()
    
    return avg_loss


def pretrain_validate(model, val_loader, accelerator, args):
    """사전 훈련 전용 검증"""
    model.eval()
    total_loss = 0.0
    
    pbar = val_loader
    if accelerator.is_main_process:
        pbar = tqdm(val_loader, desc="Pretrain Val")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                descriptions = batch['description']
                audios = batch['audio']
                guide_presets = batch.get('guide_preset', [])
                
                if audios.dim() == 2:
                    audios = audios.unsqueeze(1)
                
                # 유효한 guide preset 필터링
                valid_indices = []
                valid_descriptions = []
                valid_audios = []
                valid_guide_presets = []
                
                for i, guide_preset in enumerate(guide_presets):
                    if guide_preset:
                        valid_indices.append(i)
                        valid_descriptions.append(descriptions[i])
                        valid_audios.append(audios[i])
                        valid_guide_presets.append(guide_preset)
                
                if len(valid_indices) == 0:
                    continue
                
                valid_audio_batch = torch.stack(valid_audios)
                
                outputs = model(
                    texts=valid_descriptions,
                    audio=valid_audio_batch,
                    use_real_audio=False
                )
                
                if 'preset_params' not in outputs:
                    continue
                
                batch_loss = compute_batch_guide_loss(
                    model, outputs['preset_params'], valid_guide_presets, accelerator.device
                )
                
                total_loss += batch_loss.item()
                
                if accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'ValLoss': f'{batch_loss.item():.4f}',
                        'AvgVal': f'{total_loss/(batch_idx+1):.4f}'
                    })
                    
            except Exception as e:
                continue
    
    # 평균 손실 계산
    avg_loss = total_loss / len(val_loader)
    if accelerator.num_processes > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
        avg_loss = accelerator.gather(avg_loss_tensor).mean().item()
    
    return avg_loss


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Accelerate Training")
    
    # 기본 설정
    parser.add_argument('--data_path', type=str, default='/app', help='데이터 경로')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=400, help='에포크 수')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='학습률')
    parser.add_argument('--sample_rate', type=int, default=44100, help='샘플링 레이트')
    parser.add_argument('--audio_length', type=float, default=5.0, help='오디오 길이')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--text_encoder_type', type=str, default='sentence-transformer', help='텍스트 인코더 타입(simple|sentence-transformer|e5-large|clap)')
    parser.add_argument('--logging_dir', type=str, default='/app/output/logs', help='TensorBoard 로그 디렉토리')
    
    # 사전 훈련 관련
    parser.add_argument('--enable_pretrain', action='store_true', default=True,
                       help='Guide preset으로 사전 훈련 활성화')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                       help='사전 훈련 에포크 수')
    parser.add_argument('--pretrain_batch_size', type=int, default=2,
                       help='사전 훈련 배치 크기')
    parser.add_argument('--pretrain_lr', type=float, default=1e-2,
                       help='사전 훈련 학습률')
    
    # 로깅 관련
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Weights & Biases 사용')
    parser.add_argument('--project_name', type=str, default='audiomanipulator',
                       help='W&B 프로젝트 이름')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='체크포인트 저장 디렉토리')
    parser.add_argument('--save_every', type=int, default=10,
                       help='체크포인트 저장 주기')
    
    args = parser.parse_args()
    
    # Accelerator 초기화 먼저 수행
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=args.logging_dir,
        rng_types=["cuda", "torch"],
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Weights & Biases 초기화
    if args.use_wandb:
        try:
            if accelerator.is_main_process:
                print("🚀 Wandb 초기화 중...")
                wandb.init(
                    project=args.project_name, 
                    config=vars(args),
                    name=f"accelerate-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags=['accelerate', 'clap-loss', 'audio-processing'],
                    settings=wandb.Settings(
                        _disable_stats=True,
                        _disable_meta=True,
                        console="off",
                        code_dir=None,
                    )
                )
                print("✅ Wandb 초기화 완료")
        except Exception as e:
            print(f"⚠️ Wandb 초기화 실패: {e}")
            args.use_wandb = False
    
    # 로그 디렉토리 보장
    try:
        os.makedirs(args.logging_dir, exist_ok=True)
    except Exception:
        pass
    
    if not accelerator.is_main_process:
        import builtins
        # 내장 print 함수를 아무 작업도 하지 않는 함수로 덮어쓰기
        builtins.print = lambda *args, **kwargs: None
    
    # 시드 설정 - 모든 프로세스에서 일관성 보장
    set_seed(args.seed)
    
    # 메인 프로세스에서만 출력
    if accelerator.is_main_process:
        print("🎵 Accelerate Training")
        print("=" * 40)
        print(f"📁 Data path: {args.data_path}")
        print(f"🔢 Batch size: {args.batch_size}")
        print(f"📚 Epochs: {args.num_epochs}")
        print(f"🖥️ Device: {accelerator.device}")
        print(f"🔄 Processes: {accelerator.num_processes}")
    
    # 모델 및 옵티마이저 생성
    model, optimizer, teacher_clap = create_model_and_optimizer(args)
    

    
    train_loader, val_loader = create_datasets(args)
    

    
    # Accelerate로 준비하기 전에 전체 파이프라인 워밍업으로 lazy-init 제거
    try:
        model.eval()
        with torch.no_grad():
            dummy_texts = ["initialization text", "initialization text 2"]
            warmup_secs = min(float(getattr(args, 'audio_length', 5.0)), 2.0)
            dummy_len = int(args.sample_rate * warmup_secs)
            dummy_audio = torch.randn(2, 1, dummy_len)
            _ = model(texts=dummy_texts, audio=dummy_audio, use_real_audio=False)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if accelerator.is_main_process:
            print(f"✅ 워밍업 완료 - 총 파라미터: {total:,}, 학습 파라미터: {trainable:,}")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"⚠️ 워밍업 중 경고(계속 진행): {e}")
    finally:
        model.train()

    # Accelerate로 모든 것을 준비 - teacher_clap도 포함
    model, optimizer, train_loader, val_loader, teacher_clap = accelerator.prepare(
        model, optimizer, train_loader, val_loader, teacher_clap
    )
    
    # 모델 워밍업 (옵션) - 모든 프로세스에서 짧게 수행 후 캐시 정리로 메모리 균형화
    try:
        model.eval()
        with torch.no_grad():
            dummy_texts = ["dummy text", "dummy text 2"]
            warmup_secs = min(float(getattr(args, 'audio_length', 5.0)), 1.0)
            dummy_len = int(args.sample_rate * warmup_secs)
            dummy_audio = torch.randn(2, 1, dummy_len, device=accelerator.device)
            _ = model(texts=dummy_texts, audio=dummy_audio, use_real_audio=False)
        model.train()
    except Exception as e:
        if accelerator.is_main_process:
            print(f"⚠️ 워밍업 중 경고: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ 모델 준비 완료 - 총 파라미터: {total_params:,}, 학습 파라미터: {trainable_params:,}")
    
    # 메인 프로세스에서만 정보 출력
    if accelerator.is_main_process:
        print(f"📊 훈련 배치: {len(train_loader)}")
        print(f"📊 검증 배치: {len(val_loader)}")
        
        # 훈련 전략 출력
        if args.enable_pretrain:
            print(f"🎯 사전 훈련 + 일반 훈련")
            print(f"   - 사전 훈련: {args.pretrain_epochs} epochs, lr={args.pretrain_lr}")
            print(f"   - 일반 훈련: {args.num_epochs} epochs, lr={args.learning_rate}")
        else:
            print(f"📝 일반 훈련만 (Pure Description)")
        
        print("🚀 훈련 시작!")
    
    # 1. 사전 훈련 실행 (활성화된 경우)
    if args.enable_pretrain:
        simple_pretrain(model, optimizer, accelerator, args, teacher_clap)
        
        if accelerator.is_main_process:
            print("🔄 사전 훈련 완료 - 일반 훈련 시작")
    
    # 훈련 루프
    for epoch in range(args.num_epochs):
        # 훈련
        train_loss = train_epoch(model, train_loader, optimizer, accelerator, epoch, teacher_clap, args)
        
        # 검증
        val_loss = validate(model, val_loader, accelerator, teacher_clap)
        
        # 결과 출력 (메인 프로세스에서만)
        if accelerator.is_main_process:
            phase_tag = " [Post-Pretrain]" if args.enable_pretrain else ""
            print(f"Epoch {epoch+1}/{args.num_epochs}: "
                  f"Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}{phase_tag}")
            
            # wandb logging (에포크별)
            if args.use_wandb:
                epoch_step = (epoch + 1) * len(train_loader)
                wandb.log({
                    'epoch/train_loss': train_loss,
                    'epoch/val_loss': val_loss,
                    'epoch/epoch_num': epoch + 1,
                    'epoch/pretrain_enabled': args.enable_pretrain
                }, step=epoch_step)
    
    if accelerator.is_main_process:
        print("✅ 훈련 완료!")
        
        # Wandb 종료
        if args.use_wandb:
            try:
                wandb.finish()
            except:
                pass


if __name__ == "__main__":
    main()
