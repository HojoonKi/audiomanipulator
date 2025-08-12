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
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Accelerate 임포트
from accelerate import Accelerator
from accelerate.utils import set_seed

# 로컬 모듈 임포트
sys.path.append('/app')
from pipeline import TextToAudioProcessingPipeline
from dataset import (
    PureDescriptionDataset, 
    custom_collate_no_guide,
    load_descriptions, 
    split_descriptions
)


def create_model_and_optimizer(args):
    """모델 생성 - Accelerate에서는 device 지정 불필요"""
    # 결정론/시드 고정
    set_seed(getattr(args, 'seed', 42))
    # Factory 없이 pipeline 직접 생성 (옵션화)
    text_encoder_type = getattr(args, 'text_encoder_type', 'sentence-transformer')
    use_clap = True
    model = TextToAudioProcessingPipeline(
        text_encoder_type=text_encoder_type,
        text_encoder_config={'model_name': 'all-mpnet-base-v2'} if text_encoder_type == 'sentence-transformer' else {},
        use_clap=use_clap,
        backbone_type='dual_embedding',
        decoder_type='parallel',
        sample_rate=args.sample_rate,
        freeze_text_encoder=True,
        target_params=500000
    )
    
    # 모델 초기화(가중치/바이어스 분리)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    return model, optimizer


def create_datasets(args):
    """데이터셋 생성 - 분산 샘플러 불필요"""
    # 데이터 로드
    descriptions = load_descriptions(
        data_path=args.data_path,
        use_sampled_descriptions=False,
        max_descriptions=10000
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


def compute_clap_loss(model, processed_audio, descriptions, accelerator):
    """CLAP 손실 계산"""
    try:
        # Accelerate에서는 model.module 체크 불필요
        clap_module = getattr(model, 'clap_encoder', None)
        
        if clap_module is None:
            return torch.tensor(0.1, device=accelerator.device, requires_grad=True)
        
        return clap_module.compute_clap_loss(processed_audio, descriptions)
        
    except Exception as e:
        print(f"CLAP loss 실패: {e}")
        return torch.tensor(0.1, device=accelerator.device, requires_grad=True)


def train_epoch(model, train_loader, optimizer, accelerator, epoch):
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
            
            # Forward pass
            outputs = model(
                texts=descriptions,
                audio=audios,
                use_real_audio=False
            )
            
            # 처리된 오디오 추출
            if isinstance(outputs, dict) and 'processed_audio' in outputs:
                processed_audio = outputs['processed_audio']
            else:
                processed_audio = audios
            
            # 손실 계산
            loss = compute_clap_loss(model, processed_audio, descriptions, accelerator)
            
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
                
        except Exception as e:
            if accelerator.is_main_process:
                print(f"배치 {batch_idx} 에러: {e}")
            continue
    
    # 평균 손실 계산 - accelerator가 자동으로 분산 평균 처리
    avg_loss = total_loss / len(train_loader)
    
    # 모든 프로세스 간 평균 계산
    if accelerator.num_processes > 1:
        avg_loss = accelerator.gather(torch.tensor(avg_loss)).mean().item()
    
    return avg_loss


def validate(model, val_loader, accelerator):
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
                
                outputs = model(
                    texts=descriptions,
                    audio=audios,
                    use_real_audio=False
                )
                
                if isinstance(outputs, dict) and 'processed_audio' in outputs:
                    processed_audio = outputs['processed_audio']
                else:
                    processed_audio = audios
                
                loss = compute_clap_loss(model, processed_audio, descriptions, accelerator)
                total_loss += loss.item()
                
            except Exception as e:
                continue
    
    # 평균 손실 계산
    avg_loss = total_loss / len(val_loader)
    
    # 분산 평균
    if accelerator.num_processes > 1:
        avg_loss = accelerator.gather(torch.tensor(avg_loss)).mean().item()
    
    return avg_loss


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Accelerate Training")
    
    # 기본 설정
    parser.add_argument('--data_path', type=str, default='/app', help='데이터 경로')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=5, help='에포크 수')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='학습률')
    parser.add_argument('--sample_rate', type=int, default=44100, help='샘플링 레이트')
    parser.add_argument('--audio_length', type=float, default=5.0, help='오디오 길이')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--text_encoder_type', type=str, default='sentence-transformer', help='텍스트 인코더 타입(simple|sentence-transformer|e5-large|clap)')
    parser.add_argument('--logging_dir', type=str, default='/app/output/logs', help='TensorBoard 로그 디렉토리')
    
    args = parser.parse_args()
    
    # Accelerator 초기화
    # 로그 디렉토리 보장
    try:
        os.makedirs(args.logging_dir, exist_ok=True)
    except Exception:
        pass

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="tensorboard",  # 옵션: 로깅
        project_dir=args.logging_dir,
        rng_types=["cuda", "torch"],  # numpy 동기화 제외하여 mt19937 문제 회피
    )
    
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
    model, optimizer = create_model_and_optimizer(args)
    
    # 데이터셋 생성
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

    # Accelerate로 모든 것을 준비 - 이것이 핵심!
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
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
        print("🚀 훈련 시작!")
    
    # 훈련 루프
    for epoch in range(args.num_epochs):
        # 훈련
        train_loss = train_epoch(model, train_loader, optimizer, accelerator, epoch)
        
        # 검증
        val_loss = validate(model, val_loader, accelerator)
        
        # 결과 출력 (메인 프로세스에서만)
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{args.num_epochs}: "
                  f"Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}")
    
    if accelerator.is_main_process:
        print("✅ 훈련 완료!")


if __name__ == "__main__":
    main()
