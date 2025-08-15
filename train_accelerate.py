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
# HuggingFace 모델 다운로드 관련 설정 (캐시 우선 사용)
# os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 오프라인 모드는 비활성화
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # 필요시 온라인 다운로드 허용

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
from loss import (
    compute_batch_hybrid_guide_loss,
    compute_self_supervised_clap_loss,
    compute_batch_guide_loss_normalized_l1,
    extract_guide_values,
    _normalize_parameters_for_loss,
    compute_adversarial_training_loss,
)
from discriminator import create_discriminator
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
    
    # 5. 학습률 스케줄러 생성 (본훈련용)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.1
    )
    
    # 6. Discriminator 생성 (적대적 학습용)
    discriminator = None
    discriminator_optimizer = None
    discriminator_scheduler = None
    
    if getattr(args, 'use_adversarial', False):
        print("🎯 적대적 학습용 Discriminator 생성...")
        discriminator = create_discriminator({
            'input_dim': 27,
            'hidden_dims': [128, 64, 32],
            'dropout_rate': 0.3
        })
        
        # Discriminator 옵티마이저 (Generator보다 약간 낮은 학습률)
        discriminator_optimizer = optim.AdamW(
            discriminator.parameters(),
            lr=args.learning_rate * 0.5,  # Generator의 절반 학습률
            weight_decay=0.01
        )
        
        # Discriminator 스케줄러
        discriminator_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            discriminator_optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.05
        )
        
        print(f"✅ Discriminator 파라미터: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    return model, optimizer, teacher_clap, scheduler, discriminator, discriminator_optimizer, discriminator_scheduler


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





def train_epoch(model, train_loader, optimizer, accelerator, epoch, teacher_clap, args, scheduler=None):
    """훈련 에포크"""
    model.train()
    total_loss = 0.0
    metrics_logged_this_epoch = False  # 에포크당 한 번만 메트릭 로깅
    
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
            
            # 모델 예측값 분석 (에포크당 한 번만)
            if not metrics_logged_this_epoch and accelerator.is_main_process:
                try:
                    # 모델 예측 수행
                    with torch.no_grad():
                        outputs = model(texts=descriptions, audio=audios, use_real_audio=False)
                        if 'preset_params' in outputs and '_raw_params' in outputs['preset_params']:
                            _log_prediction_metrics(outputs['preset_params'], epoch, len(train_loader), args)
                            metrics_logged_this_epoch = True
                except Exception as e:
                    pass  # 메트릭 로깅 실패해도 훈련은 계속
            
            # 진행 표시줄 업데이트 (메인 프로세스에서만)
            if accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
                current_lr = scheduler.get_last_lr()[0] if scheduler else args.learning_rate
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'AvgLoss': f'{total_loss/(batch_idx+1):.4f}',
                    'LR': f'{current_lr:.1e}'
                })
                
                # wandb logging (배치별) - 50번마다만 로깅
                if args.use_wandb and batch_idx % 50 == 0:
                    # Main training step 계산 (pretrain step 이후부터 시작)
                    pretrain_offset = args.pretrain_epochs if args.enable_pretrain else 0
                    main_training_step = pretrain_offset + epoch * len(train_loader) + batch_idx + 1
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/batch_idx': batch_idx,
                        'train/epoch': epoch + 1,
                        'train/learning_rate': current_lr,
                        'train/phase': 'main_training'
                    }, step=main_training_step)
                
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


def save_checkpoint(model, optimizer, scheduler, accelerator, epoch, train_loss, val_loss, args, is_best=False):
    """체크포인트 저장 - 메인 프로세스에서만"""
    if not accelerator.is_main_process:
        return
    
    try:
        # 저장 디렉토리 생성
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Accelerate에서는 unwrap_model로 원본 모델 접근
        base_model = accelerator.unwrap_model(model)
        full_state = base_model.state_dict()
        
        # 어댑터 파라미터만 필터링 (훈련 가능한 파라미터만)
        adapter_state = {}
        frozen_params = []
        
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                # 어댑터 관련 파라미터만 저장
                if any(adapter_key in name for adapter_key in [
                    'backbone.adapters',     # CrossAttentionAdapter들
                    'backbone.final_norm',   # 최종 LayerNorm
                    'decoder.',             # decoder는 항상 훈련
                ]):
                    adapter_state[name] = full_state[name]
            else:
                frozen_params.append(name)
        
        print(f"💾 어댑터 파라미터만 저장: {len(adapter_state)}/{len(full_state)} layers")
        if accelerator.is_main_process and epoch == 0:  # 첫 번째 저장시에만 상세 정보
            print(f"   - 저장되는 어댑터: {list(adapter_state.keys())[:3]}...")
            print(f"   - Frozen 파라미터 수: {len(frozen_params)}")
        
        model_state = adapter_state
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,  # 어댑터 파라미터만
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': args,
            'adapter_info': {
                'total_params': len(full_state),
                'adapter_params': len(adapter_state),
                'frozen_params': len(frozen_params),
                'adapter_keys': list(adapter_state.keys()),
                'training_mode': 'adapter_only'
            }
        }
        
        # 일반 체크포인트 저장
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 체크포인트 저장: {checkpoint_path}")
        
        # 최고 성능 체크포인트 저장
        if is_best:
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"🏆 최고 성능 모델 저장: {best_path}")
            
    except Exception as e:
        print(f"❌ 체크포인트 저장 실패: {e}")











def _log_generated_parameters(generated_preset, guide_preset, epoch):
    """생성된 파라미터와 가이드 파라미터 비교 출력"""
    try:
        print(f"\n🔍 [Epoch {epoch+1}] 생성된 파라미터 vs 가이드 파라미터:")
        print("=" * 70)
        
        # Raw params 추출
        if "_raw_params" in generated_preset:
            generated_raw = generated_preset["_raw_params"]
            if generated_raw.dim() > 1:
                generated_raw = generated_raw[0]  # 첫 번째 배치 아이템
            generated_values = generated_raw.detach().cpu().numpy()
        else:
            print("❌ _raw_params가 없습니다.")
            return
        
        # Guide values 추출
        guide_values = extract_guide_values(guide_preset)
        
        # 파라미터 이름 정의 (pitch 제외)
        param_names = []
        
        # EQ (20개)
        for i in range(5):
            param_names.extend([
                f"EQ{i+1}_freq", f"EQ{i+1}_gain", f"EQ{i+1}_q", f"EQ{i+1}_type"
            ])
        
        # Reverb (5개)
        param_names.extend([
            "Rev_room", "Rev_delay", "Rev_diff", "Rev_damp", "Rev_wet"
        ])
        
        # Distortion (2개)
        param_names.extend([
            "Dist_gain", "Dist_color"
        ])
        
        # Pitch 제외
        
        # 파라미터 비교 출력 (처음 10개만) - 길이 정렬
        print("📊 주요 파라미터 비교 (Generated vs Guide):")
        gen_vec = list(generated_values)[:27]
        guide_vec = list(guide_values)[:27]
        for i in range(min(10, len(gen_vec))):
            gen_val = gen_vec[i]
            guide_val = guide_vec[i] if i < len(guide_vec) else 0
            diff = abs(gen_val - guide_val)
            
            print(f"  {param_names[i]:12}: {gen_val:8.4f} vs {guide_val:8.4f} (diff: {diff:6.4f})")
        
        # if len(generated_values) > 10:
        #     print(f"  ... (총 {len(generated_values)}개 파라미터 중 10개만 표시)")
        
        # 전체 MSE 계산 (원본값)
        raw_mse = sum((g - t)**2 for g, t in zip(gen_vec, guide_vec)) / max(1, len(gen_vec))
        
        # 정규화된 MSE 계산
        device = generated_preset["_raw_params"].device if hasattr(generated_preset["_raw_params"], 'device') else torch.device('cpu')
        generated_tensor = torch.tensor(gen_vec, dtype=torch.float32, device=device).unsqueeze(0)
        guide_tensor = torch.tensor(guide_vec, dtype=torch.float32, device=device).unsqueeze(0)
        normalized_gen, normalized_guide = _normalize_parameters_for_loss(generated_tensor, guide_tensor)
        normalized_mse = torch.mean((normalized_gen - normalized_guide) ** 2).item()
        
        print(f"📈 원본 MSE: {raw_mse:.6f}")
        print(f"📈 정규화 MSE: {normalized_mse:.6f} (실제 loss에 사용)")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ 파라미터 로깅 실패: {e}")


def _log_prediction_metrics(generated_preset, epoch, loader_len, args):
    """본훈련 시 예측값 분포 및 다양한 메트릭 로깅"""
    try:
        if "_raw_params" in generated_preset and args.use_wandb:
            raw_params = generated_preset["_raw_params"]
            if raw_params.dim() > 1:
                raw_params = raw_params[0]  # 첫 번째 배치 아이템
            
            values = raw_params.detach().cpu().numpy()
            
            # Main training step 계산 (에포크 시작 시점 기준으로 설정하여 step 역행 방지)
            pretrain_offset = args.pretrain_epochs if hasattr(args, 'enable_pretrain') and args.enable_pretrain else 0
            current_step = pretrain_offset + epoch * loader_len + 1
            
            # 기본 통계
            stats = {
                'predictions/mean': float(np.mean(values)),
                'predictions/std': float(np.std(values)),
                'predictions/min': float(np.min(values)),
                'predictions/max': float(np.max(values)),
                'predictions/median': float(np.median(values)),
            }
            
            # 파라미터 그룹별 분석
            # EQ parameters (0-19)
            eq_params = values[:20]
            stats.update({
                'eq/mean': float(np.mean(eq_params)),
                'eq/std': float(np.std(eq_params)),
                'eq/range': float(np.max(eq_params) - np.min(eq_params)),
            })
            
            # Reverb parameters (20-24)
            reverb_params = values[20:25]
            stats.update({
                'reverb/mean': float(np.mean(reverb_params)),
                'reverb/std': float(np.std(reverb_params)),
                'reverb/range': float(np.max(reverb_params) - np.min(reverb_params)),
            })
            
            # Distortion parameters (25-26)
            dist_params = values[25:27]
            stats.update({
                'distortion/mean': float(np.mean(dist_params)),
                'distortion/std': float(np.std(dist_params)),
                'distortion/gain': float(dist_params[0]),
                'distortion/color': float(dist_params[1]),
            })
            
            # Pitch parameter (27)
            pitch_param = values[27]
            stats.update({
                'pitch/scale': float(pitch_param),
            })
            
            # 특정 값 범위 분석
            stats.update({
                'analysis/zero_count': int(np.sum(np.abs(values) < 0.01)),
                'analysis/extreme_count': int(np.sum(np.abs(values) > 10.0)),
                'analysis/negative_count': int(np.sum(values < 0)),
                'analysis/positive_count': int(np.sum(values > 0)),
            })
            
            # Wandb에 로깅 (옵션 사용 시에만)
            if getattr(args, 'use_wandb', False):
                wandb.log(stats, step=current_step)
            
            # 콘솔에도 간단히 출력
            print(f"📊 [Epoch {epoch+1}] 예측값 분포: "
                  f"평균={stats['predictions/mean']:.3f}, "
                  f"표준편차={stats['predictions/std']:.3f}, "
                  f"범위=[{stats['predictions/min']:.3f}, {stats['predictions/max']:.3f}]")
            
    except Exception as e:
        print(f"❌ 예측값 메트릭 로깅 실패: {e}")


def simple_pretrain(model, optimizer, accelerator, args, teacher_clap, discriminator=None, discriminator_optimizer=None):
    """간단한 사전 훈련 - Fine Preset만 사용"""
    if not args.enable_pretrain:
        return 0  # 사전 훈련 없으면 step 0 반환
    
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
    
    # 사전훈련용 Discriminator Optimizer (적대적 학습 사용시)
    pretrain_discriminator_optimizer = None
    if discriminator is not None:
        pretrain_discriminator_optimizer = optim.AdamW(
            discriminator.parameters(),
            lr=args.pretrain_lr * 0.5,  # Generator의 절반 학습률
            weight_decay=0.01
        )
        if accelerator.is_main_process:
            print(f"🎯 사전훈련용 Discriminator Optimizer 생성됨 (LR: {args.pretrain_lr * 0.5:.1e})")
    
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
        
        # Diversity loss 활성화 (frequency 다양성 장려)
        try:
            if hasattr(model, 'module'):  # DDP wrapped model
                if hasattr(model.module, 'decoder') and hasattr(model.module.decoder, 'enable_diversity_loss'):
                    model.module.decoder.enable_diversity_loss()
                    if accelerator.is_main_process:
                        print("🎯 사전훈련용 Frequency diversity loss 활성화됨")
            else:
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'enable_diversity_loss'):
                    model.decoder.enable_diversity_loss()
                    if accelerator.is_main_process:
                        print("🎯 사전훈련용 Frequency diversity loss 활성화됨")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"⚠️ Diversity loss 활성화 실패: {e}")
        
        if accelerator.is_main_process:
            if discriminator is not None:
                print("📋 사전훈련 Loss = Guide(0.4x) + Adversarial(1.2x) + CLAP(0.3x) + Diversity(0.3x) + Regularization(0.2x)")
                print("⚔️ 적대적 학습 활성화 → 모드 붕괴 방지, 다양성 증진")
                print("🎯 Discriminator → 평균값 출력 감지 및 페널티")
                print("🎯 Generator → 실제와 구별 불가능한 다양한 preset 생성")
            else:
                print("📋 사전훈련 Loss = Normalized L1 Guide(0.8x) + CLAP(0.5x) + Diversity(0.5x) + Regularization(0.3x)")
                print("🎯 정규화된 L1 Loss → frequency 스케일 문제 해결, 평균값 함정 방지")
            print("🎯 CLAP Loss 추가 → 음향적 품질 향상")
            print("🎯 SiLU 활성화 → 부드럽고 다양한 출력")
            print("🎯 특정 값 타겟 제거 → 모델이 데이터로부터 자유롭게 학습")
        
        # Accelerate로 준비 (discriminator 포함 여부에 따라 다르게 처리)
        if discriminator is not None and pretrain_discriminator_optimizer is not None:
            pretrain_optimizer, train_loader, val_loader, discriminator, pretrain_discriminator_optimizer = accelerator.prepare(
                pretrain_optimizer, train_loader, val_loader, discriminator, pretrain_discriminator_optimizer
            )
        else:
            pretrain_optimizer, train_loader, val_loader = accelerator.prepare(
                pretrain_optimizer, train_loader, val_loader
            )
        
        
        # 사전 훈련 루프
        best_pretrain_loss = float('inf')
        
        for epoch in range(args.pretrain_epochs):
            # 사전 훈련 에포크 (적대적 학습 포함 가능)
            train_loss = pretrain_epoch(model, train_loader, pretrain_optimizer, accelerator, epoch, args, teacher_clap, discriminator, pretrain_discriminator_optimizer)
            
            # 검증
            val_loss = pretrain_validate(model, val_loader, accelerator, args, teacher_clap)
            
            # 스케줄러 업데이트
            pretrain_scheduler.step()
            
            if accelerator.is_main_process:
                current_lr = pretrain_scheduler.get_last_lr()[0]
                print(f"Pretrain {epoch+1}/{args.pretrain_epochs}: "
                      f"Loss={train_loss:.6f}, Val={val_loss:.6f}, LR={current_lr:.1e}")
                
                # Wandb 사전 훈련 로깅 (별도 step 시퀀스)
                if args.use_wandb:
                    pretrain_step = epoch + 1  # 사전훈련은 1부터 시작
                    wandb.log({
                        'pretrain/epoch': epoch + 1,
                        'pretrain/train_loss': train_loss,
                        'pretrain/val_loss': val_loss,
                        'pretrain/learning_rate': current_lr,
                        'pretrain/phase': 'guide_preset_only'
                    }, step=pretrain_step)
            
            # 최고 성능 추적
            if val_loss < best_pretrain_loss:
                best_pretrain_loss = val_loss
        
        if accelerator.is_main_process:
            print(f"✅ 사전 훈련 완료! 최고 성능: {best_pretrain_loss:.6f}")
            print("="*60)
        
        # 사전훈련 완료 후 diversity loss 비활성화 (본훈련에서는 사용 안 함)
        try:
            if hasattr(model, 'module'):  # DDP wrapped model
                if hasattr(model.module, 'decoder') and hasattr(model.module.decoder, 'disable_diversity_loss'):
                    model.module.decoder.disable_diversity_loss()
                    if accelerator.is_main_process:
                        print("🔄 사전훈련 완료 → Frequency diversity loss 비활성화됨")
            else:
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'disable_diversity_loss'):
                    model.decoder.disable_diversity_loss()
                    if accelerator.is_main_process:
                        print("🔄 사전훈련 완료 → Frequency diversity loss 비활성화됨")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"⚠️ Diversity loss 비활성화 실패: {e}")
        
        if accelerator.is_main_process:
            print("📋 본훈련에서는 CLAP loss만 사용됩니다.")
        
        # 사전 훈련에서 사용한 마지막 step 반환
        return args.pretrain_epochs
    
    except Exception as e:
        if accelerator.is_main_process:
            print(f"❌ 사전 훈련 실패: {e}")
            traceback.print_exc()
        return 0
    
    finally:
        # 원래 설정 복원
        args.learning_rate = original_lr


def pretrain_epoch(model, train_loader, optimizer, accelerator, epoch, args, teacher_clap, discriminator=None, discriminator_optimizer=None):
    """사전 훈련 전용 에포크"""
    model.train()
    total_loss = 0.0
    param_logged_this_epoch = False  # 에포크당 한 번만 파라미터 출력
    
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
            
            # 1. CLAP Loss 계산 (사전훈련에서도 CLAP 활용)
            clap_loss = compute_self_supervised_clap_loss(
                fx_model=model,
                clap_model=teacher_clap,
                original_audios=valid_audio_batch,
                fx_texts=valid_descriptions,
                temperature=0.07
            )
            
            # 2. Guide + Adversarial (사전훈련 전용) - 공용 함수 사용
            #    use_adversarial=False이면 내부에서 guide만 계산하게 할 수도 있지만,
            #    여기서는 discriminator 존재시에만 adversarial 스텝 수행
            guide_weight = getattr(args, 'guide_weight', 6.0)
            adv_weight = getattr(args, 'adversarial_weight', 4.0)
            clap_weight = 0.4  # 고정 가중치 (필요시 args로 빼기)

            if discriminator is not None and discriminator_optimizer is not None:
                # G step 손실 조합 (guide 하이브리드 + adversarial)
                adv_dict = compute_adversarial_training_loss(
                    model=model,
                    discriminator=discriminator,
                    batch_generated_preset=outputs['preset_params'],
                    batch_guide_presets=valid_guide_presets,
                    device=accelerator.device,
                    adversarial_weight=adv_weight,
                    guide_weight=guide_weight,
                    use_feature_matching=args.use_feature_matching,
                    guide_mode='hybrid',
                    lambda_regression=0.3,
                    use_gated_offset=True,
                    feature_matching_weight=0.1,
                    discriminator_optimizer=discriminator_optimizer,
                    accelerator=accelerator,
                )
                batch_loss = adv_dict['total_loss'] + clap_weight * clap_loss
            else:
                # Adversarial 미사용: 하이브리드 guide만 사용
                hybrid = compute_batch_hybrid_guide_loss(
                    batch_generated_preset=outputs['preset_params'],
                    batch_guide_presets=valid_guide_presets,
                    device=accelerator.device,
                    lambda_regression=0.3,
                    use_gated_offset=True,
                )
                batch_loss = guide_weight * hybrid['total_loss'] + clap_weight * clap_loss
            
            # 첫 번째 배치에서만 파라미터 출력 (에포크당 한 번)
            if not param_logged_this_epoch and accelerator.is_main_process and len(valid_guide_presets) > 0:
                _log_generated_parameters(outputs['preset_params'], valid_guide_presets[0], epoch)
                param_logged_this_epoch = True
            
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


def pretrain_validate(model, val_loader, accelerator, args, teacher_clap):
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
                
                batch_loss = compute_batch_guide_loss_normalized_l1(
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
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=400, help='에포크 수')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='학습률')
    parser.add_argument('--sample_rate', type=int, default=44100, help='샘플링 레이트')
    parser.add_argument('--audio_length', type=float, default=5.0, help='오디오 길이')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--text_encoder_type', type=str, default='sentence-transformer', help='텍스트 인코더 타입(simple|sentence-transformer|e5-large|clap)')
    parser.add_argument('--logging_dir', type=str, default='/app/output/logs', help='TensorBoard 로그 디렉토리')
    
    # 사전 훈련 관련
    parser.add_argument('--enable_pretrain', action='store_true', default=False,
                       help='Guide preset으로 사전 훈련 활성화')
    parser.add_argument('--pretrain_epochs', type=int, default=40,
                       help='사전 훈련 에포크 수')
    parser.add_argument('--pretrain_batch_size', type=int, default=2,
                       help='사전 훈련 배치 크기')
    parser.add_argument('--pretrain_lr', type=float, default=2e-3,
                       help='사전 훈련 학습률')
    
    # 적대적 학습 관련
    parser.add_argument('--use_adversarial', action='store_true', default=True,
                       help='적대적 학습(GAN) 사용')
    parser.add_argument('--adversarial_weight', type=float, default=0.4,
                       help='적대적 손실 가중치')
    parser.add_argument('--guide_weight', type=float, default=0.6,
                       help='가이드 손실 가중치')
    parser.add_argument('--use_feature_matching', action='store_true', default=True,
                       help='Feature matching 손실 사용')
    
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
    model, optimizer, teacher_clap, scheduler, discriminator, discriminator_optimizer, discriminator_scheduler = create_model_and_optimizer(args)
    

    
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
            traceback.print_exc()
    finally:
        model.train()

    # Accelerate로 모든 것을 준비 - teacher_clap도 포함
    model, optimizer, scheduler, train_loader, val_loader, teacher_clap = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader, teacher_clap
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
        
        if args.use_adversarial:
            print(f"⚔️ 적대적 학습 활성화됨! (사전훈련에서 적용)")
            print(f"   - Adversarial Weight: {args.adversarial_weight}")
            print(f"   - Guide Weight: {args.guide_weight}")
            print(f"   - Feature Matching: {args.use_feature_matching}")
            print(f"   - 본훈련에서는 CLAP Loss만 사용")
        
        print("🚀 훈련 시작!")
    
    # 1. 사전 훈련 실행 (활성화된 경우)
    if args.enable_pretrain:
        # 사전훈련에서는 자체적으로 discriminator optimizer를 생성하므로 None 전달
        simple_pretrain(model, optimizer, accelerator, args, teacher_clap, discriminator, discriminator_optimizer)
        
        if accelerator.is_main_process:
            print("🔄 사전 훈련 완료 - 일반 훈련 시작")
            print(f"   - 사전 훈련 step: 1~{args.pretrain_epochs}")
            print(f"   - 일반 훈련 step: {args.pretrain_epochs + 1}부터 시작")
    
    # 훈련 루프
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # 훈련
        train_loss = train_epoch(model, train_loader, optimizer, accelerator, epoch, teacher_clap, args, scheduler)
        
        # 검증
        val_loss = validate(model, val_loader, accelerator, teacher_clap)
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 최고 성능 추적
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # 결과 출력 (메인 프로세스에서만)
        if accelerator.is_main_process:
            current_lr = scheduler.get_last_lr()[0]
            phase_tag = " [Post-Pretrain]" if args.enable_pretrain else ""
            best_tag = " 🏆" if is_best else ""
            print(f"Epoch {epoch+1}/{args.num_epochs}: "
                  f"Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}, "
                  f"LR = {current_lr:.1e}{phase_tag}{best_tag}")
            
            # wandb logging (에포크별)
            if args.use_wandb:
                # Main training epoch step (pretrain step 이후부터 시작)
                pretrain_offset = args.pretrain_epochs if args.enable_pretrain else 0
                main_epoch_step = pretrain_offset + (epoch + 1) * len(train_loader)
                wandb.log({
                    'epoch/train_loss': train_loss,
                    'epoch/val_loss': val_loss,
                    'epoch/learning_rate': current_lr,
                    'epoch/epoch_num': epoch + 1,
                    'epoch/pretrain_enabled': args.enable_pretrain,
                    'epoch/best_val_loss': best_val_loss,
                    'epoch/phase': 'main_training'
                }, step=main_epoch_step)
        
        # 체크포인트 저장
        if (epoch + 1) % args.save_every == 0 or is_best:
            save_checkpoint(model, optimizer, scheduler, accelerator, epoch, train_loss, val_loss, args, is_best)
    
    if accelerator.is_main_process:
        print("✅ 훈련 완료!")
        print(f"🏆 최고 검증 성능: {best_val_loss:.6f}")
        print(f"💾 체크포인트 저장 위치: {args.save_dir}")
        print(f"   - 최고 성능 모델: {os.path.join(args.save_dir, 'best_model.pt')}")
        print(f"   - 마지막 에포크: {os.path.join(args.save_dir, f'checkpoint_epoch_{args.num_epochs}.pt')}")
        
        # Wandb 종료
        if args.use_wandb:
            try:
                wandb.finish()
            except:
                pass


if __name__ == "__main__":
    main()
