#!/usr/bin/env python3

# HuggingFace tokenizers parallelism 경고 해결
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import random
import wandb
import traceback
import os

# HuggingFace tokenizers parallelism 경고 해결
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb
from datetime import datetime

# Import our custom modules
from dynamic_pipeline_factory import DynamicPipelineFactory
from dataset import (
    PresetDataset, 
    PureDescriptionDataset, 
    PretrainDataset,  # 사전 훈련 전용 데이터셋 추가
    create_custom_collate_fn, 
    load_descriptions, 
    split_descriptions
)

class TrainingManager:
    """훈련 관리자 - 멀티GPU 지원"""
    
    def __init__(self, args, rank=0, world_size=1):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # 분산 훈련 초기화
        if world_size > 1:
            self.setup_distributed()
        
        # CLAP 로딩 시 출력 억제
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        import os
        
        if self.rank == 0:
            print("🔄 모델 초기화 중... (CLAP 로딩)")
        
        # stdout/stderr 임시 억제 (CLAP verbose 출력 방지)
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                # DynamicPipelineFactory를 사용한 모델 초기화
                factory = DynamicPipelineFactory()
                
                # args에서 text_encoder_type 가져오기 (기본값: sentence-transformer-large)
                text_encoder_type = getattr(args, 'text_encoder_type', 'sentence-transformer-large')
                
                # 동적 파이프라인 생성 - 차원은 자동으로 결정됨
                self.model = factory.create_pipeline(
                    encoder_preset=text_encoder_type,
                    use_clap=True,
                    backbone_type='residual',
                    decoder_type='parallel',
                    sample_rate=args.sample_rate,
                    target_params=500000
                )
                
                # 실제 텍스트 임베딩 차원 업데이트 (동적으로 결정된 값)
                if hasattr(self.model, 'text_encoder'):
                    actual_text_dim = self.model.text_encoder.get_embedding_dim()
                    args.text_embed_dim = actual_text_dim  # args 업데이트
                    if self.rank == 0:
                        print(f"🔧 실제 텍스트 임베딩 차원: {actual_text_dim}")
                        print(f"   선택된 인코더: {text_encoder_type}")
                
                # 백본 모델의 실제 구성 확인
                if hasattr(self.model, 'backbone'):
                    if hasattr(self.model.backbone, 'text_dim'):
                        backbone_text_dim = self.model.backbone.text_dim
                        if self.rank == 0:
                            print(f"🔧 백본 텍스트 입력 차원: {backbone_text_dim}")
                    if hasattr(self.model.backbone, 'clap_dim'):
                        backbone_clap_dim = self.model.backbone.clap_dim
                        if self.rank == 0:
                            print(f"🔧 백본 CLAP 입력 차원: {backbone_clap_dim}")
        
        # 명시적으로 device로 이동 (verbose 출력을 위해)
        if self.rank == 0:
            print(f"🚀 모델을 {self.device}로 이동 중...")
        self.model = self.model.to(self.device)
        
        # Pipeline 내부의 CLAP 모듈에 접근 (중복 로드 방지)
        self.clap_module = self.model.clap_encoder if hasattr(self.model, 'clap_encoder') else None
        
        # 멀티GPU용 모델 래핑
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank)
            # CLAP은 이미 model 내부에 있으므로 별도로 래핑하지 않음
        elif torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            # CLAP은 이미 model 내부에 있으므로 별도로 래핑하지 않음
            
        # Optimizer - 오직 model parameters만 (CLAP은 frozen)
        model_params = list(self.model.parameters())
        trainable_params = [p for p in model_params if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.num_epochs
        )
        
        # Alternative: ReduceLROnPlateau for adaptive learning rate
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        # )
        
        # Loss function
        self.criterion = nn.CosineEmbeddingLoss()
        
        if self.rank == 0:  # 메인 프로세스에서만 출력
            model_params = sum(p.numel() for p in self.model.parameters())
            model_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_trainable = model_trainable
            
            print(f"🚀 훈련 매니저 초기화 완료")
            print(f"   - Device: {self.device}")
            print(f"   - World size: {world_size}")
            print(f"   - Model parameters: {model_params:,} ({model_trainable:,} trainable)")
            print(f"   - Total trainable: {total_trainable:,}")
            print(f"   - CLAP 사용: {'✅ Enabled (내장)' if self.clap_module else '❌ Disabled'}")
            
            
            # GPU 메모리 사용량 확인
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
                max_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
                
                print(f"\n🖥️ GPU 메모리 사용량:")
                print(f"   - Allocated: {allocated:.2f} GB")
                print(f"   - Reserved: {reserved:.2f} GB") 
                print(f"   - Total GPU Memory: {max_memory:.2f} GB")
                print(f"   - Usage: {(allocated/max_memory)*100:.1f}% allocated, {(reserved/max_memory)*100:.1f}% reserved")
                
                # 모델이 실제로 GPU에 있는지 확인
                model_on_gpu = next(self.model.parameters()).device
                print(f"   - Model Device: {model_on_gpu}")
            
            print("=" * 60)
    

    
    def setup_distributed(self):
        """분산 훈련 설정"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            backend='nccl',
            rank=self.rank,
            world_size=self.world_size
        )
    
    def load_datasets(self):
        """데이터셋 로드 - 분산 훈련 지원"""
        # dataset.py의 함수들을 사용하여 descriptions 로드
        descriptions = load_descriptions(
            data_path=self.args.data_path,
            use_sampled_descriptions=self.args.use_sampled_descriptions,
            max_descriptions=self.args.max_descriptions
        )
        
        if not descriptions:
            if self.rank == 0:
                print("❌ Description 로드 실패")
            return None, None
        
        # 데이터셋 분할
        train_descriptions, val_descriptions = split_descriptions(descriptions, train_ratio=0.8)
        
        if self.rank == 0:
            print(f"📚 데이터셋 분할 완료:")
            print(f"   - 훈련: {len(train_descriptions)}개")
            print(f"   - 검증: {len(val_descriptions)}개")
        
        # Fine preset 경로
        fine_preset_path = os.path.join(self.args.data_path, 'descriptions', 'fined_presets_filtered.py')
        
        # Custom collate function 생성 (dataset.py 함수 사용)
        custom_collate_fn = create_custom_collate_fn(include_guide_preset=self.args.use_guide_presets)
        
        # 데이터셋 생성 - 사전 훈련 여부와 가이드 프리셋 사용 여부에 따라 선택
        if self.args.use_guide_presets:
            # Guide preset 사용 (PresetDataset)
            train_dataset = PresetDataset(
                descriptions=train_descriptions,
                audio_dataset_path=os.path.join(self.args.data_path, 'audio_dataset'),
                use_fine_tuned_presets=True,
                fine_preset_path=fine_preset_path,
                sample_rate=self.args.sample_rate,
                audio_length=self.args.audio_length
            )
            
            val_dataset = PresetDataset(
                descriptions=val_descriptions,
                audio_dataset_path=os.path.join(self.args.data_path, 'audio_dataset'),
                use_fine_tuned_presets=False,  # 검증에서는 guide preset 사용 안함
                sample_rate=self.args.sample_rate,
                audio_length=self.args.audio_length
            )
        else:
            # Pure Description 전용 (PureDescriptionDataset)
            train_dataset = PureDescriptionDataset(
                descriptions=train_descriptions,
                audio_dataset_path=os.path.join(self.args.data_path, 'audio_dataset'),
                sample_rate=self.args.sample_rate,
                audio_length=self.args.audio_length
            )
            
            val_dataset = PureDescriptionDataset(
                descriptions=val_descriptions,
                audio_dataset_path=os.path.join(self.args.data_path, 'audio_dataset'),
                sample_rate=self.args.sample_rate,
                audio_length=self.args.audio_length
            )
        
        # 분산 샘플러 설정
        train_sampler = None
        val_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn  # Use custom collate function
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn  # Use custom collate function
        )
        
        return train_loader, val_loader
    
    def compute_clap_loss_batch_wrapper(self, processed_audio_batch, target_descriptions):
        """배치 처리를 위한 래퍼 - 기존 compute_clap_loss 사용"""
        try:
            # 기존 compute_clap_loss는 이미 배치를 지원함
            return self.compute_clap_loss(processed_audio_batch, target_descriptions)
        except Exception as e:
            print(f"❌ CLAP batch wrapper 실패: {e}")
            # fallback
            dummy_param = next(self.model.parameters())
            return torch.mean(dummy_param * 0.0) + 0.1

    def compute_clap_loss(self, processed_audio, target_description):
        """CLAP 기반 오디오-텍스트 유사도 loss 계산 (gradient 유지)"""
        try:
            # Pipeline 내부의 CLAP 모듈 접근 (DDP 고려)
            if hasattr(self.model, 'module'):
                # DDP wrapped model
                clap_module = getattr(self.model.module, 'clap_encoder', None)
            else:
                # Normal model
                clap_module = getattr(self.model, 'clap_encoder', None)
            
            if clap_module is None:
                print("⚠️  CLAP 모듈을 찾을 수 없음, fallback loss 사용")
                return torch.tensor(1.0, device=self.device, requires_grad=True)
            
            # 단일 description 처리
            if isinstance(target_description, list) and len(target_description) == 1:
                target_description = target_description[0]
            
            # Processed audio가 list인 경우 첫 번째 요소 선택하거나 텐서로 변환
            if isinstance(processed_audio, list):
                if len(processed_audio) > 0:
                    processed_audio = processed_audio[0]  # 첫 번째 요소 사용
                else:
                    print("⚠️  Processed audio list가 비어있음")
                    return torch.tensor(1.0, device=self.device, requires_grad=True)
            
            # 텐서가 아닌 경우 텐서로 변환
            if not isinstance(processed_audio, torch.Tensor):
                print(f"⚠️  Processed audio가 텐서가 아님: {type(processed_audio)}")
                return torch.tensor(1.0, device=self.device, requires_grad=True)
            
            # Processed audio 차원 처리
            if processed_audio.dim() == 3:  # [batch, channels, samples]
                processed_audio = processed_audio.squeeze(1)  # [batch, samples]
            elif processed_audio.dim() == 1:  # [samples]
                processed_audio = processed_audio.unsqueeze(0)  # [1, samples]
            
            # CLAP loss 계산 (단순히 embedding 비교)
            clap_loss = clap_module.compute_clap_loss(processed_audio, target_description)
            
            # CLAP loss가 gradient를 가지는지 확인
            if not clap_loss.requires_grad:
                print(f"⚠️ CLAP loss가 gradient를 가지지 않음, 더미 loss 생성")
                # gradient가 있는 더미 loss 생성 (모델 파라미터 사용)
                dummy_param = next(self.model.parameters())
                clap_loss = torch.mean(dummy_param * 0.0) + 0.1  # 모델과 연결된 더미 loss
            
            return clap_loss
            
        except Exception as e:
            print(f"❌ CLAP loss 계산 실패: {e}")
            # gradient가 있는 더미 loss 생성
            try:
                dummy_param = next(self.model.parameters())
                fallback_loss = torch.mean(dummy_param * 0.0) + 0.1
                return fallback_loss
            except:
                return torch.tensor(1.0, device=self.device, requires_grad=True)
    
    
    def train_epoch(self, train_loader, epoch):
        """한 에포크 훈련 - Pure Description Training Only"""
        self.model.train()
        total_loss = 0.0
        
        # 훈련 모드: 항상 Pure Description
        training_mode = "Text Descriptions"
        
        # 분산 샘플러 에포크 설정
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        if self.rank == 0:  # 메인 프로세스에서만 progress bar 표시
            desc = f"E{epoch+1}/{self.args.num_epochs} (Desc)"
            pbar = tqdm(train_loader, desc=desc, leave=False)  # leave=False로 progress bar 지우기
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            try:
                descriptions = batch['description']
                audios = batch['audio'].to(self.device, non_blocking=True)
                
                # Audio tensor 차원 확인 및 수정
                if audios.dim() == 2:  # [batch_size, samples]
                    audios = audios.unsqueeze(1)  # [batch_size, 1, samples] -> mono 채널 추가
                
                # Pure Description Training: 모든 항목을 description으로 처리
                try:
                    # 전체 배치를 한번에 처리
                    self.model.train()
                    outputs = self.model(
                        texts=descriptions,
                        audio=audios,
                        use_real_audio=False
                    )
                    
                    # 배치 전체에 대해 한 번에 CLAP loss 계산
                    if 'processed_audio' in outputs:
                        processed_audio_batch = outputs['processed_audio']
                    else:
                        processed_audio_batch = audios
                    
                    # CLAP loss를 배치 단위로 계산
                    batch_loss = self.compute_clap_loss_batch_wrapper(processed_audio_batch, descriptions)
                    valid_items = len(descriptions)
                
                except Exception as e:
                    print(f"❌ Description batch 처리 실패: {e}")
                    # gradient가 있는 더미 loss
                    dummy_param = next(self.model.parameters())
                    batch_loss = torch.mean(dummy_param * 0.0) + 0.1
                    valid_items = 1  # 더미 항목 1개로 설정
                
                # Valid items가 없으면 건너뛰기
                if valid_items == 0:
                    if self.rank == 0:
                        print(f"⚠️ 배치 {batch_idx}: 처리할 항목 없음")
                    continue
                
                # 배치 처리에서는 이미 평균 loss가 계산됨
                # batch_loss = batch_loss / valid_items  # 제거
                
                # batch_loss가 gradient를 가지는지 확인
                if not batch_loss.requires_grad:
                    if self.rank == 0:
                        print(f"⚠️ 배치 {batch_idx}: Loss가 gradient를 가지지 않음, 건너뛰기")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                
                # Progress bar 업데이트 (메인 프로세스만)
                if self.rank == 0 and hasattr(pbar, 'set_postfix'):
                    # 현재 학습률 가져오기
                    current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.args.learning_rate
                    
                    pbar.set_postfix({
                        'Loss': f'{batch_loss.item():.4f}',
                        'AvgLoss': f'{total_loss/(batch_idx+1):.4f}',
                        'Items': f'{valid_items}/{len(descriptions)}',
                        'LR': f'{current_lr:.1e}',
                        'Mode': training_mode[:5]  # "Guide" or "Text "
                    })
                    
                    # wandb logging (배치별) - 50번마다만 로깅
                    if self.args.use_wandb and batch_idx % 50 == 0:
                        # 배치별 로깅은 글로벌 step으로 계산
                        global_step = epoch * len(train_loader) + batch_idx
                        wandb.log({
                            'batch_loss': batch_loss.item(),
                            'batch_idx': batch_idx,
                            'epoch': epoch + 1,
                            'training_mode': training_mode,
                            'learning_rate': self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.args.learning_rate
                        }, step=global_step)
                    
            except Exception as e:
                if self.rank == 0:
                    print(f"❌ 배치 {batch_idx} 처리 실패: {e}")
                    import traceback
                    traceback.print_exc()
                # 에러가 발생해도 계속 진행
                continue
        
        # 분산 훈련에서 loss 평균 계산
        if self.world_size > 1:
            avg_loss = torch.tensor(total_loss / len(train_loader)).to(self.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss / self.world_size
            final_loss = avg_loss.item()
        else:
            final_loss = total_loss / max(1, len(train_loader))
        
        # wandb logging (에포크별)
        if self.rank == 0 and self.args.use_wandb:
            wandb.log({
                'train_loss_epoch': final_loss,
                'training_mode': training_mode
            }, step=epoch)
        
        return final_loss
    
    def compute_guide_loss(self, generated_preset, guide_preset):
        """Guide preset과의 차이를 이용한 간단한 MSE loss"""
        try:
            # 디코더에서 _raw_params 직접 추출
            if isinstance(generated_preset, dict) and "_raw_params" in generated_preset:
                generated_tensor = generated_preset["_raw_params"].to(self.device)
                
                # Guide preset을 간단한 tensor로 변환
                guide_values = self.extract_guide_values(guide_preset)
                if guide_values is None:
                    return self.create_dummy_loss()
                
                guide_tensor = torch.FloatTensor(guide_values).to(self.device)
                
                # 배치 차원 처리
                if generated_tensor.dim() == 2:
                    generated_tensor = generated_tensor.squeeze(0)
                
                # 직접 MSE 계산
                mse_loss = nn.MSELoss()(generated_tensor, guide_tensor)
                
                # 첫 번째 디버깅
                if self.rank == 0 and hasattr(self, '_first_guide_loss'):
                    print(f"📊 Guide Loss: {mse_loss.item():.6f}")
                    print(f"   Generated: {generated_tensor.shape} [{generated_tensor.min():.3f}, {generated_tensor.max():.3f}]")
                    print(f"   Guide: {guide_tensor.shape} [{guide_tensor.min():.3f}, {guide_tensor.max():.3f}]")
                    del self._first_guide_loss
                
                return mse_loss
            else:
                if self.rank == 0:
                    print(f"⚠️ _raw_params 없음")
                return self.create_dummy_loss()
                
        except Exception as e:
            if self.rank == 0:
                print(f"❌ Guide loss 실패: {e}")
            return self.create_dummy_loss()
    
    def compute_batch_guide_loss(self, batch_generated_preset, batch_guide_presets):
        """배치 단위 Guide Loss 계산 - 사전 검증된 preset 사용"""
        try:
            # 배치 단위로 _raw_params 추출
            if isinstance(batch_generated_preset, dict) and "_raw_params" in batch_generated_preset:
                generated_batch_tensor = batch_generated_preset["_raw_params"].to(self.device)
                # generated_batch_tensor: [batch_size, 28] 또는 [batch_size, 1, 28]
                
                if generated_batch_tensor.dim() == 3:
                    generated_batch_tensor = generated_batch_tensor.squeeze(1)  # [batch_size, 28]
                
                # 🔥 간소화: 모든 preset이 이미 검증됨, 직접 변환
                batch_guide_values = []
                
                for guide_preset in batch_guide_presets:
                    guide_values = self.extract_guide_values(guide_preset)
                    batch_guide_values.append(guide_values)
                
                # Guide values를 배치 텐서로 스택
                guide_batch_tensor = torch.FloatTensor(batch_guide_values).to(self.device)  # [batch_size, 28]
                
                # 배치 MSE loss 계산
                batch_mse_loss = nn.MSELoss()(generated_batch_tensor, guide_batch_tensor)
                
                # 첫 번째 배치 디버깅
                if self.rank == 0 and hasattr(self, '_first_guide_loss'):
                    print(f"📊 Batch Guide Loss: {batch_mse_loss.item():.6f}")
                    print(f"   Generated Batch: {generated_batch_tensor.shape} [{generated_batch_tensor.min():.3f}, {generated_batch_tensor.max():.3f}]")
                    print(f"   Guide Batch: {guide_batch_tensor.shape} [{guide_batch_tensor.min():.3f}, {guide_batch_tensor.max():.3f}]")
                    print(f"   All presets pre-validated ✅")
                    del self._first_guide_loss
                
                return batch_mse_loss
                
            else:
                
                if self.rank == 0:
                    print(f"⚠️ 배치에서 _raw_params 없음")
                return self.create_dummy_loss()
                
        except Exception as e:
            if self.rank == 0:
                print(f"❌ Batch guide loss 실패: {e}")
            return self.create_dummy_loss()
    
    def fallback_individual_guide_processing(self, descriptions, audios, guide_presets):
        """배치 처리 실패시 개별 처리 fallback - 사전 검증된 preset 사용"""
        try:
            total_loss = 0.0
            
            for desc, audio, guide_preset in zip(descriptions, audios, guide_presets):
                try:
                    single_audio = audio.unsqueeze(0)  # [1, channels, samples]
                    
                    outputs = self.model(
                        texts=[desc],
                        audio=single_audio,
                        use_real_audio=False
                    )
                    
                    if 'preset_params' not in outputs:
                        continue
                        
                    predicted_params = outputs['preset_params']
                    guide_loss = self.compute_guide_loss(predicted_params, guide_preset)
                    
                    # 사전 검증된 preset이므로 간단한 검증만
                    if guide_loss.requires_grad:
                        total_loss += guide_loss
                        
                except Exception:
                    continue
            
            # 모든 preset이 유효하므로 valid_items = len(guide_presets)
            return total_loss / len(guide_presets) if len(guide_presets) > 0 else self.create_dummy_loss()
            
        except Exception as e:
            if self.rank == 0:
                print(f"❌ Fallback 처리 실패: {e}")
            return self.create_dummy_loss()
    
    def extract_guide_values(self, guide_preset):
        """Guide preset에서 28개 값 추출 - 사전 검증된 preset 사용"""
        # 데이터셋에서 이미 검증됨 - 바로 값 추출
        values = []
        
        # EQ (20개): center_freq, gain_db, q, filter_type × 5
        eq_section = guide_preset['eq']
        for i in range(1, 6):  # band_1 ~ band_5
            band = eq_section[f'band_{i}']
            values.extend([
                band['center_freq'],
                band['gain_db'], 
                band['q'],
                band['filter_type']
            ])
        
        # Reverb (5개)
        reverb = guide_preset['reverb']
        values.extend([
            reverb['room_size'],
            reverb['pre_delay'], 
            reverb['diffusion'],
            reverb['damping'],
            reverb['wet_gain']
        ])
        
        # Distortion (2개)
        dist = guide_preset['distortion']
        values.extend([
            dist['gain'],
            dist['color']
        ])
        
        # Pitch (1개)
        pitch = guide_preset['pitch']
        values.append(pitch['scale'])
        
        return values  # 28개 값 보장됨
    
    def create_dummy_loss(self):
        """간단한 더미 loss"""
        dummy_param = next(self.model.parameters())
        return torch.mean(dummy_param * 0.0) + 0.001
    
    def validate(self, val_loader):
        """검증 - 분산 훈련 지원"""
        self.model.eval()
        total_loss = 0.0
        
        pbar = val_loader
        if self.rank == 0:
            pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                descriptions = batch['description']
                audios = batch['audio'].to(self.device, non_blocking=True)
                
                # Forward pass through pipeline (eval mode로 빠르게)
                self.model.eval()  # 검증에서는 eval 모드 사용
                outputs = self.model(
                    texts=descriptions,
                    audio=audios,
                    use_real_audio=False
                )
                
                # 배치 전체에 대해 한 번에 CLAP loss 계산
                if 'processed_audio' in outputs:
                    processed_audio_batch = outputs['processed_audio']
                else:
                    processed_audio_batch = audios
                
                # CLAP loss를 배치 단위로 계산 (기존 메서드 사용)
                batch_loss = self.compute_clap_loss(processed_audio_batch, descriptions)
                total_loss += batch_loss.item()
                
                # Progress bar 업데이트 (validation)
                if self.rank == 0 and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'ValLoss': f'{batch_loss.item():.4f}',
                        'AvgValLoss': f'{total_loss/(batch_idx+1):.4f}',
                        'Samples': len(descriptions)
                    })
        
        # 분산 훈련에서 validation loss 평균 계산
        if self.world_size > 1:
            avg_loss = torch.tensor(total_loss / len(val_loader)).to(self.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss / self.world_size
            final_val_loss = avg_loss.item()
        else:
            final_val_loss = total_loss / len(val_loader)
        
        return final_val_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """체크포인트 저장 - 메인 프로세스에서만"""
        if self.rank != 0:  # 메인 프로세스가 아니면 저장하지 않음
            return
            
        # DDP 모델의 경우 module 접근 필요
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'args': self.args
        }
        
        # 일반 체크포인트
        checkpoint_path = os.path.join(self.args.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 체크포인트
        if is_best:
            best_path = os.path.join(self.args.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def cleanup_distributed(self):
        """분산 훈련 정리"""
        if self.world_size > 1:
            dist.destroy_process_group()
    
    def simple_pretrain(self):
        """간단한 사전 훈련 - Fine Preset만 사용 (Description 없음)"""
        if not self.args.enable_pretrain:
            return
        
        if self.rank == 0:
            print("\n" + "="*60)
            print("🎯 사전 훈련 시작 (Fine Preset Only - No Descriptions)")
            print("="*60)
            print(f"   - 사전 훈련 에포크: {self.args.pretrain_epochs}")
            print(f"   - 사전 훈련 학습률: {self.args.pretrain_lr}")
            print(f"   - 모드: Fine Preset 파라미터 매칭만")
        
        # Fine preset 경로
        fine_preset_path = os.path.join(self.args.data_path, 'descriptions', 'fined_presets_filtered.py')
        
        # 기존 옵티마이저 백업  
        original_optimizer = self.optimizer
        original_lr = self.args.learning_rate
        
        # 사전 훈련용 옵티마이저 생성
        pretrain_optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.pretrain_lr,
            weight_decay=self.args.weight_decay
        )
        self.optimizer = pretrain_optimizer
        self.args.learning_rate = self.args.pretrain_lr
        
        try:
            # 사전 훈련 전용 데이터셋 로드 (Fine Preset만)
            train_dataset = PretrainDataset(
                fine_preset_path=fine_preset_path,
                audio_dataset_path=os.path.join(self.args.data_path, 'audio_dataset'),
                sample_rate=self.args.sample_rate,
                audio_length=self.args.audio_length
            )
            
            # 검증용으로는 적은 수의 샘플만 사용
            val_size = min(100, len(train_dataset) // 10)  # 10%와 100개 중 작은 것
            val_indices = random.sample(range(len(train_dataset)), val_size)
            train_indices = [i for i in range(len(train_dataset)) if i not in val_indices]
            
            train_subset = torch.utils.data.Subset(train_dataset, train_indices)
            val_subset = torch.utils.data.Subset(train_dataset, val_indices)
            
            # 분산 샘플러 설정
            train_sampler = None
            val_sampler = None
            if self.world_size > 1:
                from torch.utils.data.distributed import DistributedSampler
                train_sampler = DistributedSampler(
                    train_subset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True
                )
                val_sampler = DistributedSampler(
                    val_subset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False
                )
            
            # Custom collate function (guide preset 포함)
            custom_collate_fn = create_custom_collate_fn(include_guide_preset=True)
            
            # 사전 훈련 전용 배치 크기 사용 (설정되지 않으면 일반 배치 크기 사용)
            pretrain_batch_size = self.args.pretrain_batch_size if self.args.pretrain_batch_size is not None else self.args.batch_size
            
            train_loader = DataLoader(
                train_subset,
                batch_size=pretrain_batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=self.args.num_workers,
                pin_memory=True,
                collate_fn=custom_collate_fn
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=pretrain_batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.args.num_workers,
                pin_memory=True,
                collate_fn=custom_collate_fn
            )
            
            if self.rank == 0:
                print(f"🎯 사전 훈련 데이터:")
                print(f"   - 훈련 샘플: {len(train_subset)}")
                print(f"   - 검증 샘플: {len(val_subset)}")
                print(f"   - 사전 훈련 배치 크기: {pretrain_batch_size}")
                print(f"   - 일반 훈련 배치 크기: {self.args.batch_size}")
            
            # 사전 훈련 루프 - Guide Preset 파라미터 매칭만 수행
            best_pretrain_loss = float('inf')
            
            for epoch in range(self.args.pretrain_epochs):
                # 분산 샘플러 에포크 설정
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                # 사전 훈련 에포크 (Guide Preset 매칭만)
                train_loss = self.pretrain_epoch(train_loader, epoch)
                
                # 간단한 검증
                val_loss = self.pretrain_validate(val_loader)
                
                if self.rank == 0:
                    print(f"Pretrain {epoch+1}/{self.args.pretrain_epochs}: "
                          f"Loss={train_loss:.6f}, Val={val_loss:.6f} (Guide Preset Only)")
                    
                    # Wandb 사전 훈련 로깅
                    if self.args.use_wandb:
                        wandb.log({
                            'pretrain_epoch': epoch + 1,
                            'pretrain_train_loss': train_loss,
                            'pretrain_val_loss': val_loss,
                            'pretrain_lr': self.args.pretrain_lr,
                            'phase': 'pretrain_guide_only'
                        }, step=epoch)
                
                # 최고 성능 추적
                if val_loss < best_pretrain_loss:
                    best_pretrain_loss = val_loss
                    
                    # 최고 성능 모델 저장
                    if self.rank == 0:
                        pretrain_dir = os.path.join(self.args.save_dir, 'pretrain')
                        os.makedirs(pretrain_dir, exist_ok=True)
                        
                        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                        torch.save({
                            'model_state_dict': model_state,
                            'epoch': epoch,
                            'pretrain_loss': val_loss,
                            'args': self.args
                        }, os.path.join(pretrain_dir, 'best_pretrain.pt'))
                        
                        # Wandb에 최고 성능 기록
                        if self.args.use_wandb:
                            wandb.log({
                                'pretrain_best_loss': best_pretrain_loss,
                                'pretrain_best_epoch': epoch + 1
                            }, step=epoch)
            
            if self.rank == 0:
                print(f"✅ 사전 훈련 완료! 최고 성능: {best_pretrain_loss:.6f}")
                print("   사전 훈련: Fine Preset 파라미터 매칭만 수행됨")
                print("="*60)
        
        except Exception as e:
            if self.rank == 0:
                print(f"❌ 사전 훈련 실패: {e}")
                import traceback
                traceback.print_exc()
        
        finally:
            # 원래 설정 복원
            self.args.learning_rate = original_lr
            self.optimizer = original_optimizer

    def pretrain_epoch(self, train_loader, epoch):
        """사전 훈련 전용 에포크 - Guide Preset 파라미터 매칭만 (배치 최적화)"""
        self.model.train()
        total_loss = 0.0
        
        # 디버깅 플래그 설정 (첫 번째 에포크에서만)
        if epoch == 0:
            self._first_preset_conversion = True
            self._first_guide_loss = True
        
        if self.rank == 0:
            desc = f"Pretrain E{epoch+1}/{self.args.pretrain_epochs} (Guide Only)"
            pbar = tqdm(train_loader, desc=desc, leave=False)
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            try:
                descriptions = batch['description']  # 더미 description
                audios = batch['audio'].to(self.device, non_blocking=True)
                guide_presets = batch.get('guide_preset', [])
                
                # Audio tensor 차원 확인 및 수정
                if audios.dim() == 2:
                    audios = audios.unsqueeze(1)
                
                # 🔥 배치 최적화: 유효한 guide preset이 있는 항목만 필터링 (간소화)
                valid_indices = []
                valid_descriptions = []
                valid_audios = []
                valid_guide_presets = []
                
                for i, guide_preset in enumerate(guide_presets):
                    # 데이터셋에서 이미 검증됨 - 빈 딕셔너리만 확인
                    if guide_preset:  # 빈 딕셔너리가 아니면 유효
                        valid_indices.append(i)
                        valid_descriptions.append(descriptions[i])
                        valid_audios.append(audios[i])
                        valid_guide_presets.append(guide_preset)
                
                if len(valid_indices) == 0:
                    # 유효한 guide preset이 없으면 건너뛰기
                    continue
                
                # 🚀 배치 처리: 유효한 항목들을 한 번에 처리
                try:
                    # 유효한 오디오들을 배치 텐서로 스택
                    valid_audio_batch = torch.stack(valid_audios)  # [valid_batch_size, channels, samples]
                    
                    # 모델 forward pass - 배치 단위로 처리
                    outputs = self.model(
                        texts=valid_descriptions,
                        audio=valid_audio_batch,
                        use_real_audio=False
                    )
                    
                    if 'preset_params' not in outputs:
                        if self.rank == 0:
                            print(f"⚠️ 배치 {batch_idx}: preset_params가 출력에 없음")
                        continue
                    
                    # 🎯 배치 Guide Loss 계산
                    batch_loss = self.compute_batch_guide_loss(outputs['preset_params'], valid_guide_presets)
                    
                    # Loss 유효성 검증
                    if not batch_loss.requires_grad or torch.isnan(batch_loss) or torch.isinf(batch_loss):
                        if self.rank == 0:
                            print(f"⚠️ 배치 {batch_idx}: 무효한 batch guide loss")
                        continue
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    total_loss += batch_loss.item()
                    
                    # Progress bar 업데이트
                    if self.rank == 0 and hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({
                            'GuideLoss': f'{batch_loss.item():.4f}',
                            'AvgLoss': f'{total_loss/(batch_idx+1):.4f}',
                            'ValidItems': f'{len(valid_indices)}/{len(descriptions)}',
                            'BatchEff': f'{len(valid_indices)}/{len(descriptions)*100:.0f}%',
                            'Mode': 'Guide'
                        })
                
                except Exception as e:
                    if self.rank == 0:
                        print(f"❌ 배치 forward pass 실패: {e}")
                    # Fallback: 개별 처리
                    batch_loss = self.fallback_individual_guide_processing(
                        valid_descriptions, valid_audios, valid_guide_presets
                    )
                    if batch_loss is not None and batch_loss.requires_grad:
                        self.optimizer.zero_grad()
                        batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        total_loss += batch_loss.item()
                    
            except Exception as e:
                if self.rank == 0:
                    print(f"❌ 사전 훈련 배치 {batch_idx} 실패: {e}")
                continue
        
        # 분산 훈련에서 loss 평균 계산
        if self.world_size > 1:
            avg_loss = torch.tensor(total_loss / len(train_loader)).to(self.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss / self.world_size
            final_loss = avg_loss.item()
        else:
            final_loss = total_loss / max(1, len(train_loader))
        
        return final_loss

    def pretrain_validate(self, val_loader):
        """사전 훈련 전용 검증 - Guide Preset 매칭만 (배치 최적화)"""
        self.model.eval()
        total_loss = 0.0
        
        pbar = val_loader
        if self.rank == 0:
            pbar = tqdm(val_loader, desc="Pretrain Validation", leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                try:
                    descriptions = batch['description']
                    audios = batch['audio'].to(self.device, non_blocking=True)
                    guide_presets = batch.get('guide_preset', [])
                    
                    if audios.dim() == 2:
                        audios = audios.unsqueeze(1)
                    
                    # 🔥 배치 최적화: 유효한 guide preset 필터링 (간소화)
                    valid_indices = []
                    valid_descriptions = []
                    valid_audios = []
                    valid_guide_presets = []
                    
                    for i, guide_preset in enumerate(guide_presets):
                        # 데이터셋에서 이미 검증됨 - 빈 딕셔너리만 확인
                        if guide_preset:  # 빈 딕셔너리가 아니면 유효
                            valid_indices.append(i)
                            valid_descriptions.append(descriptions[i])
                            valid_audios.append(audios[i])
                            valid_guide_presets.append(guide_preset)
                    
                    if len(valid_indices) == 0:
                        continue
                    
                    # 🚀 배치 처리: 유효한 항목들을 한 번에 검증
                    try:
                        valid_audio_batch = torch.stack(valid_audios)
                        
                        outputs = self.model(
                            texts=valid_descriptions,
                            audio=valid_audio_batch,
                            use_real_audio=False
                        )
                        
                        if 'preset_params' not in outputs:
                            continue
                            
                        predicted_params = outputs['preset_params']
                        batch_loss = self.compute_batch_guide_loss(predicted_params, valid_guide_presets)
                        
                        total_loss += batch_loss.item()
                        
                        # Progress bar 업데이트
                        if self.rank == 0 and hasattr(pbar, 'set_postfix'):
                            pbar.set_postfix({
                                'ValGuideLoss': f'{batch_loss.item():.4f}',
                                'AvgValLoss': f'{total_loss/(batch_idx+1):.4f}',
                                'ValidItems': f'{len(valid_indices)}/{len(descriptions)}',
                                'BatchEff': f'{len(valid_indices)/len(descriptions)*100:.0f}%'
                            })
                    
                    except Exception as e:
                        # Fallback: 개별 검증
                        batch_loss = 0.0
                        valid_items = 0
                        
                        for desc, audio, guide_preset in zip(valid_descriptions, valid_audios, valid_guide_presets):
                            try:
                                single_audio = audio.unsqueeze(0)
                                
                                outputs = self.model(
                                    texts=[desc],
                                    audio=single_audio,
                                    use_real_audio=False
                                )
                                
                                if 'preset_params' not in outputs:
                                    continue
                                    
                                predicted_params = outputs['preset_params']
                                guide_loss = self.compute_guide_loss(predicted_params, guide_preset)
                                
                                batch_loss += guide_loss.item()
                                valid_items += 1
                                
                            except Exception:
                                continue
                        
                        if valid_items > 0:
                            batch_loss = batch_loss / valid_items
                            total_loss += batch_loss
                        
                        # Progress bar 업데이트 (fallback)
                        if self.rank == 0 and hasattr(pbar, 'set_postfix'):
                            pbar.set_postfix({
                                'ValGuideLoss': f'{batch_loss:.4f}' if valid_items > 0 else 'N/A',
                                'ValidItems': f'{valid_items}/{len(descriptions)}',
                                'Mode': 'Fallback'
                            })
                        
                except Exception as e:
                    if self.rank == 0:
                        print(f"❌ 검증 배치 {batch_idx} 실패: {e}")
                    continue
        
        # 분산 훈련에서 validation loss 평균 계산
        if self.world_size > 1:
            avg_loss = torch.tensor(total_loss / len(val_loader)).to(self.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss / self.world_size
            final_val_loss = avg_loss.item()
        else:
            final_val_loss = total_loss / max(1, len(val_loader))
        
        return final_val_loss

    def train(self):
        """전체 훈련 프로세스 - 사전 훈련 + 일반 훈련"""
        if self.rank == 0:
            print("🎯 훈련 프로세스 시작")
        
        # 1. 사전 훈련 실행 (활성화된 경우)
        if self.args.enable_pretrain:
            self.simple_pretrain()
            
            # 사전 훈련 완료 후, 일반 훈련에서는 guide preset 비활성화
            if self.rank == 0:
                print("🔄 사전 훈련 완료 - 일반 훈련에서 Guide Preset 비활성화")
            self.args.use_guide_presets = False
        
        # 2. 일반 훈련 시작
        if self.rank == 0:
            if self.args.enable_pretrain:
                print("\n🎓 일반 훈련 시작 (Pure Description Training)")
            else:
                print("\n🎓 일반 훈련 시작")
        
        # 데이터셋 로드
        train_loader, val_loader = self.load_datasets()
        if train_loader is None:
            if self.rank == 0:
                print("❌ 데이터셋 로드 실패")
            return
        
        # 저장 디렉토리 생성 (메인 프로세스만)
        if self.rank == 0:
            os.makedirs(self.args.save_dir, exist_ok=True)
        
        # 모든 프로세스 동기화
        if self.world_size > 1:
            dist.barrier()
        
        # 훈련 루프
        best_val_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            # 훈련
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 검증
            val_loss = self.validate(val_loader)
            
            # 스케줄러 업데이트
            self.scheduler.step()
            
            # 로깅 (메인 프로세스만)
            if self.rank == 0:
                training_mode = "Text Descriptions"
                if self.args.use_guide_presets:
                    training_mode = "Guide Presets" if epoch < self.args.guide_epochs else "Text Descriptions"
                
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.args.learning_rate
                
                # 학습 진행 상황 더 자세히 출력
                improvement = ""
                if epoch > 0:
                    if hasattr(self, 'prev_train_loss'):
                        train_diff = train_loss - self.prev_train_loss
                        val_diff = val_loss - self.prev_val_loss
                        improvement = f" (ΔTrain: {train_diff:+.6f}, ΔVal: {val_diff:+.6f})"
                
                # 현재 에포크 결과 출력
                phase_tag = " [Post-Pretrain]" if self.args.enable_pretrain else ""
                print(f"Epoch {epoch+1}/{self.args.num_epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={current_lr:.1e} ({training_mode}){improvement}{phase_tag}")
                
                # 이전 loss 저장
                self.prev_train_loss = train_loss
                self.prev_val_loss = val_loss
                
                # 학습 정체 감지
                if epoch > 10:  # 10 에포크 후부터 확인
                    if abs(train_diff) < 1e-6 and abs(val_diff) < 1e-6:
                        print("⚠️  학습이 정체된 것 같습니다. Learning rate 조정을 고려해보세요.")
                
                if self.args.use_wandb:
                    # wandb logging - 글로벌 step 사용 (배치 단위)
                    global_step = epoch * len(train_loader) + len(train_loader)  # 에포크 끝의 step
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss_epoch': train_loss,
                        'val_loss_epoch': val_loss,
                        'learning_rate_epoch': self.scheduler.get_last_lr()[0],
                        'training_mode_epoch': training_mode,
                        'is_guide_epoch': self.args.use_guide_presets and epoch < self.args.guide_epochs,
                        'use_guide_presets': self.args.use_guide_presets,
                        'guide_epochs': self.args.guide_epochs if self.args.use_guide_presets else 0,
                        'pretrain_enabled': self.args.enable_pretrain
                    }, step=global_step)
            
            # 체크포인트 저장
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if (epoch + 1) % self.args.save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        if self.rank == 0:
            final_msg = f"✅ 전체 훈련 완료! Best validation loss: {best_val_loss:.4f}"
            if self.args.enable_pretrain:
                final_msg += " (사전 훈련 + 일반 훈련)"
            print(final_msg)
            
            # Wandb 종료
            if self.args.use_wandb:
                try:
                    wandb.finish()
                except:
                    pass


def train_worker(rank, world_size, args):
    """멀티GPU 훈련 워커 함수"""
    try:
        # 훈련 시작
        trainer = TrainingManager(args, rank, world_size)
        trainer.train()
    finally:
        # 정리
        if world_size > 1:
            trainer.cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Audio Effect Preset Generation Training')
    
    # 데이터 관련
    parser.add_argument('--data_path', type=str, default='/workspace/AudioManipulator',
                       help='데이터셋 루트 경로')
    parser.add_argument('--sample_rate', type=int, default=44100,
                       help='오디오 샘플링 레이트')
    parser.add_argument('--audio_length', type=float, default=5.0,
                       help='오디오 길이 (초)')
    parser.add_argument('--max_descriptions', type=int, default=50000,
                       help='사용할 최대 description 수 (0=모두 사용)')
    parser.add_argument('--use_sampled_descriptions', action='store_true', default=False,
                       help='500개 샘플 description 파일 사용')
    
    # 모델 관련
    parser.add_argument('--text_encoder_type', type=str, default='sentence-transformer-large',
                       choices=['sentence-transformer-mini', 'sentence-transformer-large', 'e5-large', 'clap'],
                       help='사용할 텍스트 인코더 타입')
    parser.add_argument('--text_embed_dim', type=int, default=512,
                       help='텍스트 임베딩 차원')
    parser.add_argument('--audio_embed_dim', type=int, default=1024,
                       help='오디오 임베딩 차원')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='히든 레이어 차원')
    
    # 훈련 관련
    parser.add_argument('--batch_size', type=int, default=128,
                       help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='전체 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                       help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # 가이드 프리셋 관련 (사전 훈련 비활성화시에만 사용)
    parser.add_argument('--use_guide_presets', action='store_true', default=False,
                       help='Fine-tuned guide preset 사용 여부 (enable_pretrain과 함께 사용 불가)')
    parser.add_argument('--guide_epochs', type=int, default=20,
                       help='Guide preset을 사용할 에포크 수 (enable_pretrain 비활성화시에만)')
    parser.add_argument('--guide_weight', type=float, default=0.5,
                       help='Guide loss의 가중치 (enable_pretrain 비활성화시에만)')
    
    # 사전 훈련 관련 (권장: 사전 훈련 사용시 guide preset 옵션들은 무시됨)
    parser.add_argument('--enable_pretrain', action='store_true', default=False,
                       help='Guide preset으로 사전 훈련 활성화 (일반 훈련은 Pure Description)')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                       help='사전 훈련 에포크 수')
    parser.add_argument('--pretrain_batch_size', type=int, default=16,
                       help='사전 훈련 배치 크기 (기본값: 일반 훈련과 동일)')
    parser.add_argument('--pretrain_lr', type=float, default=3e-4,
                       help='사전 훈련 학습률')
    
    # 시스템 관련
    parser.add_argument('--num_workers', type=int, default=4,
                       help='데이터로더 워커 수')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='체크포인트 저장 디렉토리')
    parser.add_argument('--save_every', type=int, default=10,
                       help='체크포인트 저장 주기')
    
    # 로깅
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Weights & Biases 사용')
    parser.add_argument('--project_name', type=str, default='audiomanipulator',
                       help='W&B 프로젝트 이름')
    
    # 멀티GPU 관련
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='사용할 GPU 수 (0=CPU only)')
    parser.add_argument('--distributed', action='store_true',
                       help='분산 훈련 사용 (DistributedDataParallel)')
    
    args = parser.parse_args()
    
    # GPU 설정 확인
    if args.num_gpus > 0 and not torch.cuda.is_available():
        print("⚠️  CUDA가 사용 불가능합니다. CPU 모드로 전환합니다.")
        args.num_gpus = 0
    
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"⚠️  요청된 GPU 수({args.num_gpus})가 사용 가능한 GPU 수({available_gpus})보다 많습니다.")
        args.num_gpus = available_gpus
    
    # Argument validation 및 충돌 해결
    if args.enable_pretrain and args.use_guide_presets:
        print("⚠️  사전 훈련 활성화시 일반 훈련에서는 Guide Preset이 자동 비활성화됩니다.")
        print("   Guide 관련 arguments (--guide_epochs, --guide_weight)는 무시됩니다.")
        print("   사전 훈련: Guide Preset → 일반 훈련: Pure Description")
    
    # Weights & Biases 초기화 (최적화된 설정)
    if args.use_wandb:
        try:
            print("🚀 Wandb 초기화 중...")
            wandb.init(
                project=args.project_name, 
                config=vars(args),  # args를 dict로 변환
                name=f"preset-gen-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=['audio-processing', 'clap-loss', 'preset-generation'],
                # 업로드 최적화 설정
                settings=wandb.Settings(
                    _disable_stats=True,  # 시스템 통계 비활성화
                    _disable_meta=True,   # 메타데이터 업로드 비활성화
                    console="off",        # 콘솔 로그 업로드 비활성화
                    code_dir=None,        # 코드 업로드 비활성화
                )
            )
            print("✅ Wandb 초기화 완료")
            print(f"📊 Project: {args.project_name}")
            print(f"🔗 Run URL: {wandb.run.url if wandb.run else 'N/A'}")
                
        except Exception as e:
            print(f"⚠️  Wandb 초기화 실패: {e}")
            print("   로깅 없이 계속 진행합니다.")
            args.use_wandb = False
    
    print("🎵 Audio Effect Preset Generation Training")
    print("=" * 50)
    print(f"📁 Data path: {args.data_path}")
    print(f"🤖 Text Encoder: {args.text_encoder_type}")
    print(f"🎛️  Model dimensions: text={args.text_embed_dim}, audio={args.audio_embed_dim}, hidden={args.hidden_dim}")
    print(f"🚀 Training: {args.num_epochs} epochs, batch_size={args.batch_size}, lr={args.learning_rate}")
    
    # 훈련 전략 출력
    if args.enable_pretrain:
        print(f"🎯 사전 훈련 + Pure Description Training")
        print(f"   - 사전 훈련: {args.pretrain_epochs} epochs (Guide Preset), lr={args.pretrain_lr}")
        print(f"   - 일반 훈련: {args.num_epochs} epochs (Pure Description), lr={args.learning_rate}")
    elif args.use_guide_presets:
        print(f"🎯 Guide Presets + Description Training")
        print(f"   - Guide epochs: {args.guide_epochs}")
        print(f"   - Guide weight: {args.guide_weight}")
        print(f"   - Description epochs: {args.num_epochs - args.guide_epochs}")
    else:
        print(f"📝 Pure Description Training (no guide presets)")
    
    print(f"🖥️  GPU 설정: {args.num_gpus} GPUs, distributed={args.distributed}")
    
    # 훈련 시작
    if args.num_gpus > 1 and args.distributed:
        # 분산 훈련
        print(f"🚀 분산 훈련 시작: {args.num_gpus} GPUs")
        mp.spawn(train_worker, args=(args.num_gpus, args), nprocs=args.num_gpus, join=True)
    else:
        # 단일 GPU 또는 DataParallel 훈련
        if args.num_gpus > 1:
            print(f"🚀 DataParallel 훈련 시작: {args.num_gpus} GPUs")
        elif args.num_gpus == 1:
            print("🚀 단일 GPU 훈련 시작")
        else:
            print("🚀 CPU 훈련 시작")
        
        trainer = TrainingManager(args, rank=0, world_size=1)
        trainer.train()
        
        # 프로그램 종료 전 정리
        if args.use_wandb:
            try:
                wandb.finish()
            except:
                pass


if __name__ == "__main__":
    main()
