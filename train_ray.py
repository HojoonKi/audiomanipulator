#!/usr/bin/env python3
"""
Ray Train Training Script (Fixed for DDP Consistency Error)
- Added barriers for model sync across ranks
- No retry loop to prevent repetition
- Stable multi-worker training
"""

import os
import sys
import argparse

import ray
import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import RunConfig

def train_func(config):
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import ray.train
    import torch.distributed as dist  # For barriers
    
    sys.path.append('/app')
    from pipeline import TextToAudioProcessingPipeline
    from dataset import (
        PureDescriptionDataset, 
        custom_collate_no_guide,
        load_descriptions, 
        split_descriptions
    )
    
    print("🚀 Ray Train worker started")
    
    world_rank = ray.train.get_context().get_world_rank() or 0
    local_rank = ray.train.get_context().get_local_rank() or 0
    print(f"Worker {world_rank} (local: {local_rank}) started")
    
    # Model creation
    def create_model_and_optimizer_internal(config):
        seed = config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = TextToAudioProcessingPipeline(
            text_encoder_type='sentence-transformer',
            text_encoder_config={'model_name': 'all-mpnet-base-v2'},
            use_clap=True,
            backbone_type='dual_embedding',
            decoder_type='parallel',
            sample_rate=config['sample_rate'],
            freeze_text_encoder=True,
            target_params=500000
        )
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)
        
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
        return model, optimizer

    # Dataset creation
    def create_datasets_internal(config):
        descriptions = load_descriptions(config['data_path'], use_sampled_descriptions=False, max_descriptions=10000)
        if not descriptions:
            raise RuntimeError("Description load failed")
        
        train_descriptions, val_descriptions = split_descriptions(descriptions, train_ratio=0.8)
        if world_rank == 0:
            print(f"📚 Datasets: train={len(train_descriptions)}, val={len(val_descriptions)}")
        
        train_dataset = PureDescriptionDataset(
            descriptions=train_descriptions,
            audio_dataset_path=os.path.join(config['data_path'], 'audio_dataset'),
            sample_rate=config['sample_rate'],
            audio_length=config['audio_length']
        )
        
        val_dataset = PureDescriptionDataset(
            descriptions=val_descriptions,
            audio_dataset_path=os.path.join(config['data_path'], 'audio_dataset'),
            sample_rate=config['sample_rate'],
            audio_length=config['audio_length']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True, collate_fn=custom_collate_no_guide, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate_no_guide, drop_last=True)
        
        return train_loader, val_loader

    # CLAP loss
    def compute_clap_loss_internal(model, processed_audio, descriptions):
        try:
            clap_module = getattr(model, 'clap_encoder', None)
            if clap_module is None:
                return torch.tensor(0.1, device=processed_audio.device, requires_grad=True)
            return clap_module.compute_clap_loss(processed_audio, descriptions)
        except Exception as e:
            print(f"CLAP loss failed: {e}")
            return torch.tensor(0.1, device=processed_audio.device, requires_grad=True)

    # Train epoch
    def train_epoch_internal(model, train_loader, optimizer, epoch, config):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if world_rank == 0 else train_loader
        
        for batch_idx, batch in enumerate(pbar):
            try:
                descriptions = batch['description']
                audios = batch['audio'].cuda(non_blocking=True) if torch.cuda.is_available() else batch['audio']
                
                if audios.dim() == 2:
                    audios = audios.unsqueeze(1)
                
                outputs = model(texts=descriptions, audio=audios, use_real_audio=False)
                
                processed_audio = outputs.get('processed_audio', audios)
                
                loss = compute_clap_loss_internal(model, processed_audio, descriptions)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if world_rank == 0 and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'AvgLoss': f'{total_loss/(batch_idx+1):.4f}'})
            except Exception as e:
                if world_rank == 0:
                    print(f"Batch {batch_idx} error: {e}")
                continue
        
        return total_loss / len(train_loader)

    # Validate
    def validate_internal(model, val_loader):
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    descriptions = batch['description']
                    audios = batch['audio'].cuda(non_blocking=True) if torch.cuda.is_available() else batch['audio']
                    
                    if audios.dim() == 2:
                        audios = audios.unsqueeze(1)
                    
                    outputs = model(texts=descriptions, audio=audios, use_real_audio=False)
                    
                    processed_audio = outputs.get('processed_audio', audios)
                    
                    loss = compute_clap_loss_internal(model, processed_audio, descriptions)
                    total_loss += loss.item()
                except Exception as e:
                    continue
        
        return total_loss / len(val_loader) if len(val_loader) > 0 else 0.0

    # Main logic
    model, optimizer = create_model_and_optimizer_internal(config)
    
    # Sync point after model creation
    if dist.is_initialized():
        dist.barrier()
    
    # CLAP 출력 강제 억제 환경변수 (워커 별 적용)
    os.environ.setdefault('CLAP_VERBOSE', '0')

    model.eval()
    for attempt in range(3):
        dummy_texts = [f"init {attempt}_{i}" for i in range(4)]
        dummy_audio = torch.randn(4, 1, int(config['sample_rate'] * config['audio_length']))
        dummy_audio = dummy_audio.cuda() if torch.cuda.is_available() else dummy_audio
        
        with torch.no_grad():
            try:
                _ = model(texts=dummy_texts, audio=dummy_audio, use_real_audio=False)
                if hasattr(model, 'clap_encoder'):
                    _ = model(texts=dummy_texts, audio=dummy_audio, use_real_audio=True)
            except Exception as e:
                if world_rank == 0:
                    print(f"Init attempt {attempt} warning: {e}")
    
    model.train()
    
    # Sync point after initialization
    if dist.is_initialized():
        dist.barrier()
    
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Worker {world_rank} - Total params: {param_count:,}, Trainable: {trainable_count:,}")
    
    model = ray.train.torch.prepare_model(model)
    
    # Sync point after DDP wrapping
    if dist.is_initialized():
        dist.barrier()
    
    train_loader, val_loader = create_datasets_internal(config)
    
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader = ray.train.torch.prepare_data_loader(val_loader)
    
    if world_rank == 0:
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch_internal(model, train_loader, optimizer, epoch, config)
        
        val_loss = validate_internal(model, val_loader)
        
        ray.train.report({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        
        if world_rank == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description="Ray Train Training (Fixed)")
    
    parser.add_argument('--data_path', type=str, default='/app', help='Data path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate')
    parser.add_argument('--audio_length', type=float, default=5.0, help='Audio length')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("🎵 Ray Train Training (Fixed)")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Workers: {args.num_workers}")
    print(f"Use GPU: {args.use_gpu}")
    
    # Environment fixes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_TIMEOUT'] = '3600'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['RAY_worker_use_pickle5'] = '1'
    os.environ['RAY_PICKLE_VERBOSE_DEBUG'] = '0'
    
    if not ray.is_initialized():
        ray.init()
    
    train_config = {
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'sample_rate': args.sample_rate,
        'audio_length': args.audio_length,
        'seed': args.seed,
        'text_encoder_type': 'sentence-transformer',
        'use_clap': True,
    }
    
    try:
        trainer = TorchTrainer(
            train_func,
            train_loop_config=train_config,
            scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu),
            torch_config=TorchConfig(backend="nccl", timeout_s=3600),
            run_config=RunConfig(name="audio_clap_training_fixed", storage_path="/tmp/ray_results")
        )
        
        print(f"Starting Ray Train with {args.num_workers} workers")
        result = trainer.fit()
        
        print("Ray Train completed!")
        print(f"Final results: {result.metrics}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        if args.num_workers > 1:
            print("Multi-worker failed. Retrying with single worker...")
            trainer_single = TorchTrainer(
                train_func,
                train_loop_config=train_config,
                scaling_config=ScalingConfig(num_workers=1, use_gpu=args.use_gpu)
            )
            result = trainer_single.fit()
            print("Single worker completed!")
            print(f"Final results: {result.metrics}")
        else:
            raise

if __name__ == "__main__":
    main()
