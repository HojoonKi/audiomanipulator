#!/usr/bin/env python3
"""
Simple DDP Training Script
- 최소한의 DDP 구현
- 동일한 데이터셋과 모델 사용
- 부가 기능 제거 (WandB, 사전훈련, 복잡한 에러 처리 등)
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from tqdm import tqdm
from datetime import timedelta

# 로컬 모듈 임포트
sys.path.append('/app')
from dynamic_pipeline_factory import DynamicPipelineFactory
from dataset import (
    PureDescriptionDataset, 
    custom_collate_no_guide,
    load_descriptions, 
    split_descriptions
)


def setup_distributed(rank, world_size):
    """기본적인 DDP 설정"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # 기본 NCCL 설정
    os.environ['NCCL_TIMEOUT'] = os.environ.get('NCCL_TIMEOUT', '1800')
    os.environ['NCCL_DEBUG'] = os.environ.get('NCCL_DEBUG', 'WARN')
    # 단일 노드/컨테이너 환경 안전 설정
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
    # 로컬 실행시 인터페이스 지정(필요 시 변경)
    os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')
    # 디버깅
    os.environ.setdefault('TORCH_DISTRIBUTED_DEBUG', 'DETAIL')
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        init_method='env://',
        timeout=timedelta(minutes=30)
    )
    
    if rank == 0:
        print(f"✅ DDP 초기화 완료 (World Size: {world_size})")


def cleanup_distributed():
    """DDP 정리"""
    dist.destroy_process_group()


def create_model_and_optimizer(args, device):
    """모델 생성"""
    # 결정론적 동작을 위한 시드 고정
    seed = getattr(args, 'seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    factory = DynamicPipelineFactory()
    
    model = factory.create_pipeline(
        encoder_preset='sentence-transformer-large',
        use_clap=True,
        sample_rate=args.sample_rate,
        backbone_type=getattr(args, 'backbone_type', 'simple')  # 백본 타입 지원
    )
    
    model = model.to(device)

    # 필요 시 학습 파라미터 초기화 일관화
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() >= 2:
            torch.nn.init.xavier_uniform_(param)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    return model, optimizer


def create_datasets(args, rank, world_size):
    """데이터셋 생성"""
    # 데이터 로드
    descriptions = load_descriptions(
        data_path=args.data_path,
        use_sampled_descriptions=False,
        max_descriptions=10000  # 테스트용으로 줄임
    )
    
    if not descriptions:
        raise RuntimeError("Description 로드 실패")
    
    # 데이터 분할
    train_descriptions, val_descriptions = split_descriptions(descriptions, train_ratio=0.8)
    
    if rank == 0:
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
    
    # 분산 샘플러
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True
    )
    
    # 데이터로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_no_guide,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_no_guide,
        drop_last=True
    )
    
    return train_loader, val_loader


def compute_clap_loss(model, processed_audio, descriptions, device):
    """간단한 CLAP 손실 계산"""
    try:
        # DDP 래핑 해제
        base_model = model.module if hasattr(model, 'module') else model
        
        # TunedCLAPWithAdapters 백본 사용 시
        if hasattr(base_model, 'backbone') and hasattr(base_model.backbone, 'compute_contrastive_loss'):
            # 백본 내부의 frozen CLAP 모델을 사용하여 contrastive loss 계산
            loss = base_model.backbone.compute_contrastive_loss(
                fused_audio=processed_audio,
                texts=descriptions,
                temperature=0.07
            )
            return loss
        
        # 일반적인 CLAP encoder 사용 시
        clap_module = getattr(base_model, 'clap_encoder', None)
        if clap_module is not None:
            return clap_module.compute_clap_loss(processed_audio, descriptions)
        
        # CLAP이 없으면 기본 MSE 손실
        target = torch.zeros_like(processed_audio)
        loss = torch.nn.functional.mse_loss(processed_audio, target)
        return loss
        
    except Exception as e:
        print(f"CLAP loss 실패: {e}")
        return torch.tensor(0.1, device=device, requires_grad=True)


def train_epoch(model, train_loader, optimizer, epoch, rank, device):
    """훈련 에포크"""
    model.train()
    total_loss = 0.0
    
    # 샘플러 에포크 설정
    train_loader.sampler.set_epoch(epoch)
    
    pbar = train_loader
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            descriptions = batch['description']
            audios = batch['audio'].to(device, non_blocking=True)
            
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
            elif isinstance(outputs, dict) and 'backbone_features' in outputs:
                # 백본 출력을 사용 (TunedCLAPWithAdapters의 경우)
                processed_audio = outputs['backbone_features']
            else:
                processed_audio = audios
            
            # 손실 계산
            loss = compute_clap_loss(model, processed_audio, descriptions, device)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Progress bar 업데이트
            if rank == 0 and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'AvgLoss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
        except Exception as e:
            if rank == 0:
                print(f"배치 {batch_idx} 에러: {e}")
            continue
    
    # 분산 평균
    if dist.is_initialized():
        avg_loss = torch.tensor(total_loss / len(train_loader), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / dist.get_world_size()
        return avg_loss.item()
    else:
        return total_loss / len(train_loader)


def validate(model, val_loader, rank, device):
    """검증"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                descriptions = batch['description']
                audios = batch['audio'].to(device, non_blocking=True)
                
                if audios.dim() == 2:
                    audios = audios.unsqueeze(1)
                
                outputs = model(
                    texts=descriptions,
                    audio=audios,
                    use_real_audio=False
                )
                
                if isinstance(outputs, dict) and 'processed_audio' in outputs:
                    processed_audio = outputs['processed_audio']
                elif isinstance(outputs, dict) and 'backbone_features' in outputs:
                    processed_audio = outputs['backbone_features']
                else:
                    processed_audio = audios
                
                loss = compute_clap_loss(model, processed_audio, descriptions, device)
                total_loss += loss.item()
                
            except Exception as e:
                continue
    
    # 분산 평균
    if dist.is_initialized():
        avg_loss = torch.tensor(total_loss / len(val_loader), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / dist.get_world_size()
        return avg_loss.item()
    else:
        return total_loss / len(val_loader)


def train_worker(rank, world_size, args):
    """DDP 워커 함수"""
    try:
        # DDP 설정
        setup_distributed(rank, world_size)
        
        # GPU 설정
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
        
        if rank == 0:
            print(f"🚀 Simple DDP 훈련 시작 (Rank {rank})")
        
        # 모델 생성
        model, optimizer = create_model_and_optimizer(args, device)

        # 전체 파이프라인 워밍업: 더미 입력으로 모든 레이어 강제 생성
        try:
            model.eval()
            with torch.no_grad():
                dummy_texts = ["dummy text for initialization", "dummy text for initialization 2"]
                # 2초 길이 더미 오디오(모노)
                dummy_audio = torch.randn(2, 1, int(args.sample_rate * 2), device=device)
                _ = model(texts=dummy_texts, audio=dummy_audio, use_real_audio=False)
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                total = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"✅ 모델 워밍업 완료 - 총 파라미터: {total}, 학습 파라미터: {trainable}")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ 워밍업 중 경고: {e}")
        finally:
            model.train()

        # 진단: 랭크별 학습 파라미터 개수/샘플 shape 확인
        try:
            trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            local_count = sum(p.numel() for _, p in trainable_params)
            sample_shapes = [(n, tuple(p.shape)) for n, p in trainable_params[:10]]
            if dist.is_initialized():
                obj_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj_list, (rank, local_count, sample_shapes))
                if rank == 0:
                    print("🧪 랭크별 학습 파라미터 수:")
                    for r, cnt, _sh in sorted(obj_list, key=lambda x: x[0]):
                        print(f"   - rank {r}: {cnt}")
                    counts = [cnt for _, cnt, _ in obj_list]
                    if len(set(counts)) != 1:
                        print("❌ 랭크 간 파라미터 수 불일치 감지. 상위 파라미터 shape 샘플:")
                        for r, cnt, sh in sorted(obj_list, key=lambda x: x[0]):
                            print(f"   [rank {r}] count={cnt}, samples={sh}")
                        raise RuntimeError("DDP 전 모델 파라미터 구성이 랭크 간 다릅니다. 위 샘플 로그를 확인하세요.")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ 파라미터 진단 중 경고: {e}")
        
        # DDP 래핑
        model = DDP(
            model,
            device_ids=[rank],
            find_unused_parameters=True,
            broadcast_buffers=False
        )
        
        # 데이터셋 생성
        train_loader, val_loader = create_datasets(args, rank, world_size)
        
        if rank == 0:
            print(f"📊 배치 크기: {args.batch_size}")
            print(f"📊 훈련 배치: {len(train_loader)}")
            print(f"📊 검증 배치: {len(val_loader)}")
        
        # 훈련 루프
        for epoch in range(args.num_epochs):
            # 훈련
            train_loss = train_epoch(model, train_loader, optimizer, epoch, rank, device)
            
            # 검증
            val_loss = validate(model, val_loader, rank, device)
            
            # 결과 출력 (rank 0에서만)
            if rank == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs}: "
                      f"Train Loss = {train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}")
        
        if rank == 0:
            print("✅ 훈련 완료!")
            
    except Exception as e:
        print(f"Rank {rank} 에러: {e}")
        raise
    finally:
        if dist.is_initialized():
            cleanup_distributed()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Simple DDP Training")
    
    # 기본 설정
    parser.add_argument('--data_path', type=str, default='/app', help='데이터 경로')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=5, help='에포크 수')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='학습률')
    parser.add_argument('--sample_rate', type=int, default=44100, help='샘플링 레이트')
    parser.add_argument('--audio_length', type=float, default=5.0, help='오디오 길이')
    parser.add_argument('--num_gpus', type=int, default=2, help='GPU 수')
    parser.add_argument('--backbone_type', type=str, default='simple',
                       choices=['simple', 'transformer', 'tuned_clap_adapters'],
                       help='백본 타입')
    
    args = parser.parse_args()
    
    # GPU 확인
    if args.num_gpus > torch.cuda.device_count():
        print(f"사용 가능한 GPU: {torch.cuda.device_count()}")
        args.num_gpus = torch.cuda.device_count()
    
    print("🎵 Simple DDP Training")
    print("=" * 40)
    print(f"📁 Data path: {args.data_path}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"📚 Epochs: {args.num_epochs}")
    print(f"🖥️ GPUs: {args.num_gpus}")
    
    # 메모리 안전 설정
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    if args.num_gpus > 1:
        print(f"🚀 DDP 시작: {args.num_gpus} GPUs")
        mp.spawn(train_worker, args=(args.num_gpus, args), nprocs=args.num_gpus, join=True)
    else:
        print("🚀 단일 GPU 훈련")
        # 단일 GPU는 간단하게 처리
        device = torch.device('cuda:0')
        model, optimizer = create_model_and_optimizer(args, device)
        # ... 단일 GPU 훈련 코드 (필요시 구현)


if __name__ == "__main__":
    main()
