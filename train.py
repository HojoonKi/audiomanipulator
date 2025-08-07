#!/usr/bin/env python3

# HuggingFace tokenizers parallelism 경고 해결
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import json
import random
import re
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import wandb
from datetime import datetime

# Import our custom modules
from pipeline import TextToAudioProcessingPipeline
from dynamic_pipeline_factory import DynamicPipelineFactory
from encoder.text_encoder import CLAPTextEncoder  # CLAP 모듈 import

class PresetDataset(Dataset):
    """오디오 이펙트 프리셋 데이터셋"""
    
    def __init__(self, descriptions, audio_dataset_path, use_fine_tuned_presets=False, 
                 fine_preset_path=None, sample_rate=44100, audio_length=5.0):
        self.descriptions = descriptions
        self.audio_dataset_path = audio_dataset_path
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.use_fine_tuned_presets = use_fine_tuned_presets
        
        # 악기와 폴더 매핑
        self.instrument_mapping = {
            'electro guitar': 'Electro_Guitar',
            'acoustic guitar': 'Acoustic_Guitar', 
            'bass guitar': 'Bass_Guitar',
            'piano': 'Piano',
            'keyboard': 'Keyboard',
            'organ': 'Organ',
            'drum set': 'Drum_set',
            'violin': 'Violin',
            'flute': 'flute',
            'saxophone': 'Saxophone',
            'trumpet': 'Trumpet'
        }
        
        # Fine-tuned presets 로드 (초기 가이드용)
        self.fine_presets = []
        if use_fine_tuned_presets and fine_preset_path and os.path.exists(fine_preset_path):
            self.fine_presets = self._load_fine_presets(fine_preset_path)
        
        print(f"📊 데이터셋 초기화 완료:")
        print(f"   - Description 수: {len(self.descriptions)}")
        print(f"   - Fine presets 수: {len(self.fine_presets)}")
        print(f"   - 오디오 길이: {audio_length}초")
        
    def _load_fine_presets(self, preset_path):
        """Fine-tuned presets 파일 로드 (Python eval 방식)"""
        presets = []
        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"📖 Fine preset 파일 읽기 시작: {preset_path}")
            
            # Python dictionary 형태로 파싱 - 중괄호 블록 추출
            blocks = []
            i = 0
            while i < len(content):
                if content[i] == '{':
                    # 시작점 찾음
                    start = i
                    brace_count = 1
                    i += 1
                    
                    while i < len(content) and brace_count > 0:
                        if content[i] == '{':
                            brace_count += 1
                        elif content[i] == '}':
                            brace_count -= 1
                        i += 1
                    
                    if brace_count == 0:
                        block = content[start:i]
                        # "prompt"가 포함된 블록만 선택
                        if '"prompt"' in block or "'prompt'" in block:
                            blocks.append(block)
                else:
                    i += 1
            
            # 각 블록을 Python dictionary로 파싱
            for i, block in enumerate(blocks):
                try:
                    # 주석 제거 함수 (문자열 내부는 제외)
                    def remove_python_comments(text):
                        lines = text.split('\n')
                        cleaned_lines = []
                        for line in lines:
                            # # 이후 모든 내용 제거 (문자열 내부가 아닌 경우)
                            in_string = False
                            quote_char = None
                            comment_pos = -1
                            
                            for j, char in enumerate(line):
                                if char in ['"', "'"] and (j == 0 or line[j-1] != '\\'):
                                    if not in_string:
                                        in_string = True
                                        quote_char = char
                                    elif char == quote_char:
                                        in_string = False
                                        quote_char = None
                                elif char == '#' and not in_string:
                                    comment_pos = j
                                    break
                            
                            if comment_pos >= 0:
                                line = line[:comment_pos]
                            
                            line = line.rstrip()
                            if line:
                                cleaned_lines.append(line)
                        
                        return '\n'.join(cleaned_lines)
                    
                    # 주석 제거
                    cleaned_block = remove_python_comments(block)
                    
                    # Python의 eval로 dictionary 파싱 (안전하게)
                    preset = eval(cleaned_block)
                    
                    if isinstance(preset, dict) and 'prompt' in preset:
                        presets.append(preset)
                    
                except Exception as e:
                    if i < 5:  # 처음 5개 실패만 출력
                        print(f"❌ Failed parsing preset {i+1}: {str(e)[:100]}")
                        print(f"   Block preview: {block[:200]}...")
                    continue
                    
        except Exception as e:
            print(f"❌ Failed loading fine presets: {e}")
            import traceback
            traceback.print_exc()
            
        print(f"🎯 성공적으로 로드된 preset 수: {len(presets)}")
        
        return presets
    
    def _extract_keywords_from_description(self, description):
        """Description에서 중요한 키워드 추출"""
        # 일반적인 오디오 효과 관련 키워드들
        effect_keywords = [
            'reverb', 'echo', 'delay', 'room', 'hall', 'large', 'small', 'cave', 'cathedral',
            'distortion', 'overdrive', 'fuzz', 'saturation', 'warm', 'bright', 'dark',
            'chorus', 'flanger', 'phaser', 'tremolo', 'vibrato', 'modulation',
            'bass', 'treble', 'mid', 'low', 'high', 'frequency',
            'whisper', 'shout', 'loud', 'quiet', 'soft', 'harsh', 'smooth',
            'monster', 'robot', 'alien', 'fairy', 'giant', 'tiny', 'deep', 'thin',
            'male', 'female', 'child', 'old', 'young',
            'guitar', 'piano', 'drum', 'vocal', 'singing', 'speaking'
        ]
        
        # Description을 소문자로 변환하고 단어 분할
        words = description.lower().split()
        
        # 키워드 추출
        keywords = []
        for word in words:
            # 구두점 제거
            clean_word = word.strip('.,!?()[]{}";:')
            if clean_word in effect_keywords:
                keywords.append(clean_word)
            
            # 부분 매칭도 확인
            for keyword in effect_keywords:
                if keyword in clean_word and len(keyword) > 3:
                    keywords.append(keyword)
        
        return list(set(keywords))  # 중복 제거
    
    def _extract_subject_from_description(self, description):
        """Description에서 주체(악기/사람) 추출"""
        description_lower = description.lower()
        
        # 사람 주체 확인
        if any(word in description_lower for word in ['male', 'female', 'speaks', 'whispers', 'singing', 'says']):
            if 'male' in description_lower:
                return 'male', 'speech'
            elif 'female' in description_lower:
                return 'female', 'speech'
            else:
                return 'male', 'speech'  # 기본값
        
        # 악기 주체 확인
        for instrument, folder in self.instrument_mapping.items():
            if instrument in description_lower:
                return instrument, 'instrumental'
        
        # 기본값: 피아노
        return 'piano', 'instrumental'
    
    def _get_random_audio_file(self, subject, audio_type):
        """주체에 맞는 랜덤 오디오 파일 선택"""
        if audio_type == 'speech':
            folder_path = os.path.join(self.audio_dataset_path, 'speech', subject)
        else:  # instrumental
            folder_name = self.instrument_mapping.get(subject, 'Piano')
            folder_path = os.path.join(self.audio_dataset_path, 'instrumentals', folder_name)
        
        if not os.path.exists(folder_path):
            print(f"⚠️  폴더를 찾을 수 없음: {folder_path}")
            # 대체 폴더 사용
            folder_path = os.path.join(self.audio_dataset_path, 'instrumentals', 'Piano')
        
        # WAV, FLAC 파일들 찾기 (다양한 오디오 형식 지원)
        audio_files = []
        for ext in ['*.wav', '*.flac', '*.mp3', '*.m4a']:
            audio_files.extend(list(Path(folder_path).glob(ext)))
        
        if not audio_files:
            print(f"⚠️  오디오 파일을 찾을 수 없음: {folder_path}")
            return None
        
        return str(random.choice(audio_files))
    
    def _load_audio(self, audio_path):
        """오디오 파일 로드 및 전처리"""
        try:
            # 오디오 로드
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.audio_length)
            
            # 길이 맞추기
            target_length = int(self.sample_rate * self.audio_length)
            if len(audio) < target_length:
                # 패딩
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                # 자르기
                audio = audio[:target_length]
            
            return torch.FloatTensor(audio)
        
        except Exception as e:
            print(f"❌ 오디오 로드 실패 {audio_path}: {e}")
            # 더미 오디오 반환
            target_length = int(self.sample_rate * self.audio_length)
            return torch.zeros(target_length)
    
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        description = self.descriptions[idx]
        
        # 주체 추출
        subject, audio_type = self._extract_subject_from_description(description)
        
        # 오디오 파일 선택 및 로드
        audio_path = self._get_random_audio_file(subject, audio_type)
        if audio_path is None:
            # 더미 오디오
            target_length = int(self.sample_rate * self.audio_length)
            audio = torch.zeros(target_length)
        else:
            audio = self._load_audio(audio_path)
        
        # Fine preset 가이드 (스마트 할당)
        guide_preset = None
        if self.use_fine_tuned_presets and self.fine_presets:
            # 1단계: Description과 정확히 매칭되는 preset 찾기
            for preset in self.fine_presets:
                preset_prompt = preset.get('prompt', '').strip().lower()
                if preset_prompt and (preset_prompt in description.lower() or description.lower() in preset_prompt):
                    guide_preset = preset
                    break
            
            # 2단계: 정확한 매칭이 없으면 선택적 할당 (확률 기반)
            if guide_preset is None and self.fine_presets:
                import random
                # 가이드 프리셋 사용 확률 (30%만 가이드 사용, 70%는 순수 description 학습)
                guide_usage_probability = 0.3
                
                if random.random() < guide_usage_probability:
                    # 유사한 키워드 기반 매칭 시도
                    description_keywords = self._extract_keywords_from_description(description.lower())
                    best_match_preset = None
                    best_score = 0
                    
                    for preset in self.fine_presets:
                        preset_prompt = preset.get('prompt', '').strip().lower()
                        preset_keywords = self._extract_keywords_from_description(preset_prompt)
                        
                        # 키워드 매칭 점수 계산
                        common_keywords = set(description_keywords) & set(preset_keywords)
                        if common_keywords:
                            score = len(common_keywords) / max(len(description_keywords), len(preset_keywords))
                            if score > best_score:
                                best_score = score
                                best_match_preset = preset
                    
                    # 최소 점수 이상이면 사용, 아니면 랜덤
                    if best_match_preset and best_score > 0.1:
                        guide_preset = best_match_preset
                        if idx < 3:  # 처음 3개만 출력
                            print(f"🎯 Item {idx}: 키워드 매칭 guide preset (score: {best_score:.2f})")
                            print(f"   - Description: {description[:50]}...")
                            print(f"   - Guide prompt: {best_match_preset.get('prompt', 'No prompt')[:50]}...")
                    else:
                        # 낮은 확률로 랜덤 할당 (10%)
                        if random.random() < 0.1:
                            guide_preset = random.choice(self.fine_presets)
                            if idx < 2:  # 처음 2개만 출력
                                print(f"🎯 Item {idx}: 랜덤 guide preset 할당")
                # else: guide_preset은 None으로 유지 (순수 description 학습)
        
        return {
            'description': description,
            'audio': audio,
            'subject': subject,
            'audio_type': audio_type,
            'guide_preset': guide_preset if guide_preset is not None else {}  # None 대신 빈 dict 반환
        }
    



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
                    use_differentiable_audio=True,
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
            
            # 더 상세한 모델 구성 및 GPU 메모리 사용량 출력
            print(f"\n📊 모델 구성 상세:")
            
            # 각 구성요소별 파라미터 수 및 디바이스 위치 계산
            if hasattr(self.model, 'text_encoder'):
                text_encoder_params = sum(p.numel() for p in self.model.text_encoder.parameters())
                # Text Encoder의 실제 디바이스 확인
                te_devices = set()
                for name, param in self.model.text_encoder.named_parameters():
                    te_devices.add(str(param.device))
                    if len(te_devices) <= 3:  # 처음 몇 개만 상세 출력
                        print(f"      - {name[:30]}... : {param.device}")
                print(f"   📝 Text Encoder: {text_encoder_params:,} parameters")
                print(f"      Devices: {list(te_devices)}")
            
            if hasattr(self.model, 'clap_encoder') and self.model.clap_encoder:
                clap_params = sum(p.numel() for p in self.model.clap_encoder.parameters())
                clap_trainable = sum(p.numel() for p in self.model.clap_encoder.parameters() if p.requires_grad)
                
                # CLAP 모델의 디바이스들 확인
                clap_devices = set()
                for name, param in self.model.clap_encoder.named_parameters():
                    clap_devices.add(str(param.device))
                    if len(clap_devices) <= 3:  # 처음 몇 개만 상세 출력
                        print(f"      - {name[:30]}... : {param.device}")
                
                print(f"   🎵 CLAP Encoder: {clap_params:,} parameters ({clap_trainable:,} trainable)")
                print(f"      Devices: {list(clap_devices)}")
                
                # CLAP 내부 모델 구조 확인
                if hasattr(self.model.clap_encoder, 'clap_model'):
                    inner_clap_devices = set()
                    for name, param in self.model.clap_encoder.clap_model.named_parameters():
                        inner_clap_devices.add(str(param.device))
                    print(f"      Inner CLAP Model Devices: {list(inner_clap_devices)}")
            
            if hasattr(self.model, 'backbone'):
                backbone_params = sum(p.numel() for p in self.model.backbone.parameters())
                backbone_devices = set()
                for name, param in self.model.backbone.named_parameters():
                    backbone_devices.add(str(param.device))
                print(f"   🧠 Backbone: {backbone_params:,} parameters")
                print(f"      Devices: {list(backbone_devices)}")
            
            if hasattr(self.model, 'decoder'):
                decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
                decoder_devices = set()
                for name, param in self.model.decoder.named_parameters():
                    decoder_devices.add(str(param.device))
                print(f"   🎛️ Decoder: {decoder_params:,} parameters")
                print(f"      Devices: {list(decoder_devices)}")
            
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
        # Descriptions 로드
        descriptions = []
        
        if self.args.use_sampled_descriptions:
            # 500개 샘플 파일 사용
            desc_file = os.path.join(self.args.data_path, 'descriptions', '500_sampled_descriptions.txt')
        else:
            # 전체 descriptions 파일 사용
            desc_file = os.path.join(self.args.data_path, 'descriptions', 'descriptions.txt')
        print(f"📂 Description 파일 경로: {desc_file}")
        if os.path.exists(desc_file):
            with open(desc_file, 'r', encoding='utf-8') as f:
                descriptions = [line.strip() for line in f if line.strip()]
        else:
            if self.rank == 0:
                print(f"❌ Description 파일을 찾을 수 없음: {desc_file}")
            return None, None
        
        # 최대 description 수 제한
        if self.args.max_descriptions > 0 and len(descriptions) > self.args.max_descriptions:
            if self.rank == 0:
                print(f"📊 Description 수 제한: {len(descriptions)} → {self.args.max_descriptions}")
            random.seed(42)
            descriptions = random.sample(descriptions, self.args.max_descriptions)
        
        if self.rank == 0:
            print(f"📚 {len(descriptions)}개 description 로드됨")
            if len(descriptions) > 100000:
                print(f"⚠️  대용량 데이터셋! 한 에포크에 약 {len(descriptions):,}개 description 사용")
                print(f"   --max_descriptions 옵션으로 제한 권장 (예: --max_descriptions 50000)")
        
        # 데이터셋 분할 (모든 프로세스에서 동일한 시드 사용)
        random.seed(42)
        random.shuffle(descriptions)
        split_idx = int(len(descriptions) * 0.8)
        train_descriptions = descriptions[:split_idx]
        val_descriptions = descriptions[split_idx:]
        
        # Fine preset 경로
        fine_preset_path = os.path.join(self.args.data_path, 'descriptions', 'fined_presets.txt')
        
        # Custom collate function for handling guide_presets
        def custom_collate_fn(batch):
            """Custom collate function to properly handle guide_presets"""
            descriptions = [item['description'] for item in batch]
            audios = torch.stack([item['audio'] for item in batch])
            subjects = [item['subject'] for item in batch]
            audio_types = [item['audio_type'] for item in batch]
            guide_presets = [item['guide_preset'] for item in batch]  # List of dicts or empty dicts
            
            return {
                'description': descriptions,
                'audio': audios,
                'subject': subjects,
                'audio_type': audio_types,
                'guide_preset': guide_presets  # Keep as list
            }
        
        # 훈련 데이터셋 (가이드 프리셋 사용 여부에 따라)
        train_dataset = PresetDataset(
            descriptions=train_descriptions,
            audio_dataset_path=os.path.join(self.args.data_path, 'audio_dataset'),
            use_fine_tuned_presets=self.args.use_guide_presets,  # argument에 따라 결정
            fine_preset_path=fine_preset_path if self.args.use_guide_presets else None,
            sample_rate=self.args.sample_rate,
            audio_length=self.args.audio_length
        )
        
        # 검증 데이터셋
        val_dataset = PresetDataset(
            descriptions=val_descriptions,
            audio_dataset_path=os.path.join(self.args.data_path, 'audio_dataset'),
            use_fine_tuned_presets=False,
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
    
    def train_with_guide_preset(self, description, audio, guide_preset):
        """Guide preset을 사용한 개별 item 훈련"""
        try:
            # Guide preset이 빈 딕셔너리인 경우 건너뛰기
            if not guide_preset or not isinstance(guide_preset, dict):
                # gradient가 있는 더미 loss 생성
                dummy_param = next(self.model.parameters())
                return torch.mean(dummy_param * 0.0) + 0.01
            
            # 모델 forward pass로 예측된 파라미터 얻기
            self.model.train()
            outputs = self.model(
                texts=[description],
                audio=audio,
                use_real_audio=False
            )
            
            if 'preset_params' not in outputs:
                # Preset params가 없으면 CLAP loss로 대체
                processed_audio = outputs.get('processed_audio', audio)
                return self.compute_clap_loss(processed_audio, description)
            
            # 예측된 파라미터와 guide preset 간의 MSE loss
            predicted_params = outputs['preset_params']
            guide_loss = self.compute_guide_loss(predicted_params, guide_preset)
            
            # Guide preset으로 실제 오디오 처리해서 CLAP loss도 추가
            try:
                # Guide preset 파라미터로 오디오 처리 (만약 가능하다면)
                if 'processed_audio' in outputs:
                    processed_with_guide = outputs['processed_audio']
                    clap_loss = self.compute_clap_loss(processed_with_guide, description)
                    
                    # Guide loss (파라미터 매칭) + CLAP loss (결과 품질)
                    total_loss = self.args.guide_weight * guide_loss + (1 - self.args.guide_weight) * clap_loss
                    return total_loss
                else:
                    return guide_loss
                    
            except Exception:
                # Guide loss만 사용
                return guide_loss
            
        except Exception as e:
            print(f"❌ Guide preset 훈련 실패: {e}")
            # gradient가 있는 더미 loss 생성
            try:
                dummy_param = next(self.model.parameters())
                fallback_loss = torch.mean(dummy_param * 0.0) + 0.1
                return fallback_loss
            except:
                return torch.tensor(0.1, device=self.device, requires_grad=True)
    
    def train_epoch(self, train_loader, epoch):
        """한 에포크 훈련 - Guide vs Description 에포크 분리"""
        self.model.train()
        total_loss = 0.0
        
        # 훈련 모드 결정
        is_guide_epoch = self.args.use_guide_presets and epoch < self.args.guide_epochs
        if self.args.use_guide_presets:
            training_mode = "Guide Presets" if is_guide_epoch else "Text Descriptions"
        else:
            training_mode = "Text Descriptions"
        
        # 분산 샘플러 에포크 설정
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        if self.rank == 0:  # 메인 프로세스에서만 progress bar 표시
            desc = f"E{epoch+1}/{self.args.num_epochs} ({training_mode[:4]})"
            pbar = tqdm(train_loader, desc=desc, leave=False)  # leave=False로 progress bar 지우기
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            try:
                descriptions = batch['description']
                audios = batch['audio'].to(self.device, non_blocking=True)
                guide_presets = batch['guide_preset']
                
                # Audio tensor 차원 확인 및 수정
                if audios.dim() == 2:  # [batch_size, samples]
                    audios = audios.unsqueeze(1)  # [batch_size, 1, samples] -> mono 채널 추가
                
                batch_loss = 0.0
                valid_items = 0
                
                if is_guide_epoch:
                    # Guide Preset 에포크: guide가 있는 항목과 없는 항목을 균형있게 처리
                    guide_descriptions = []
                    guide_audios = []
                    valid_guide_presets = []
                    
                    # Non-guide descriptions (순수 description 학습용)
                    non_guide_descriptions = []
                    non_guide_audios = []
                    
                    # Guide preset 여부에 따라 분류
                    for i in range(len(descriptions)):
                        guide_preset = guide_presets[i] if isinstance(guide_presets, list) and i < len(guide_presets) else None
                        has_guide = isinstance(guide_presets, list) and i < len(guide_presets) and guide_preset is not None and bool(guide_preset)
                        
                        if has_guide:
                            guide_descriptions.append(descriptions[i])
                            guide_audios.append(audios[i])
                            valid_guide_presets.append(guide_preset)
                        else:
                            non_guide_descriptions.append(descriptions[i])
                            non_guide_audios.append(audios[i])
                    
                    # 1) Guide preset 항목들 처리 (Guide Loss)
                    guide_loss_total = 0.0
                    guide_items = 0
                    
                    if guide_descriptions:
                        try:
                            # 오디오 텐서들을 안전하게 스택
                            guide_audios_tensors = []
                            for audio_tensor in guide_audios:
                                if isinstance(audio_tensor, torch.Tensor):
                                    guide_audios_tensors.append(audio_tensor)
                                else:
                                    print(f"⚠️  오디오가 텐서가 아님: {type(audio_tensor)}")
                                    continue
                            
                            if guide_audios_tensors:
                                guide_audios_batch = torch.stack(guide_audios_tensors)  # [valid_items, channels, samples]
                                
                                # 모델에 배치로 전달
                                self.model.train()
                                outputs = self.model(
                                    texts=guide_descriptions,
                                    audio=guide_audios_batch,
                                    use_real_audio=False
                                )
                                
                                # Loss 누적을 위한 리스트
                                individual_losses = []
                                
                                # 각 항목별로 guide loss 계산
                                for i in range(len(guide_descriptions)):
                                    if 'preset_params' in outputs:
                                        # 예측된 파라미터와 guide preset 간의 loss
                                        predicted_params = outputs['preset_params']
                                        if isinstance(predicted_params, list):
                                            predicted_params = predicted_params[i] if i < len(predicted_params) else predicted_params[0]
                                        elif isinstance(predicted_params, torch.Tensor) and predicted_params.dim() > 1:
                                            predicted_params = predicted_params[i:i+1]
                                        
                                        guide_loss = self.compute_guide_loss(predicted_params, valid_guide_presets[i])
                                        individual_losses.append(guide_loss)
                                    else:
                                        # Preset params가 없으면 배치 CLAP loss 사용
                                        processed_audio = outputs.get('processed_audio', guide_audios_batch)
                                        
                                        # 해당 인덱스의 오디오와 description만 추출
                                        if isinstance(processed_audio, torch.Tensor):
                                            processed_audio_i = processed_audio[i:i+1]
                                        else:
                                            processed_audio_i = guide_audios_batch[i:i+1]
                                        
                                        clap_loss = self.compute_clap_loss(processed_audio_i, [guide_descriptions[i]])
                                        individual_losses.append(clap_loss)
                                    
                                    guide_items += 1
                                
                                # Guide loss들을 안전하게 합산
                                if individual_losses:
                                    guide_loss_total = sum(individual_losses)
                                
                        except Exception as e:
                            if self.rank == 0:
                                print(f"⚠️ Guide batch 처리 실패: {e}")
                                # 개별 처리로 폴백
                                fallback_losses = []
                                for i in range(len(guide_descriptions)):
                                    try:
                                        audio_tensor = guide_audios[i]
                                        if not isinstance(audio_tensor, torch.Tensor):
                                            continue
                                        
                                        if audio_tensor.dim() == 2:
                                            audio_tensor = audio_tensor.unsqueeze(0)
                                        elif audio_tensor.dim() == 1:
                                            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                                        
                                        item_loss = self.train_with_guide_preset(
                                            guide_descriptions[i], audio_tensor, valid_guide_presets[i]
                                        )
                                        fallback_losses.append(item_loss)
                                        guide_items += 1
                                    except Exception as e2:
                                        continue
                                
                                if fallback_losses:
                                    guide_loss_total = sum(fallback_losses)
                    
                    # 2) Non-guide preset 항목들 처리 (CLAP Loss) - Guide epoch에서도 일부 처리
                    clap_loss_total = 0.0
                    clap_items = 0
                    
                    # Guide epoch에서도 30% 정도는 순수 description 학습
                    if non_guide_descriptions:
                        import random
                        sample_size = min(len(non_guide_descriptions), max(1, len(guide_descriptions) // 2))
                        sampled_indices = random.sample(range(len(non_guide_descriptions)), sample_size)
                        
                        sampled_descriptions = [non_guide_descriptions[i] for i in sampled_indices]
                        sampled_audios = [non_guide_audios[i] for i in sampled_indices]
                        
                        try:
                            sampled_audios_tensors = []
                            for audio_tensor in sampled_audios:
                                if isinstance(audio_tensor, torch.Tensor):
                                    sampled_audios_tensors.append(audio_tensor)
                            
                            if sampled_audios_tensors:
                                sampled_audios_batch = torch.stack(sampled_audios_tensors)
                                
                                self.model.train()
                                outputs = self.model(
                                    texts=sampled_descriptions,
                                    audio=sampled_audios_batch,
                                    use_real_audio=False
                                )
                                
                                individual_clap_losses = []
                                for i in range(len(sampled_descriptions)):
                                    if 'processed_audio' in outputs:
                                        processed_audio_i = outputs['processed_audio'][i:i+1] if isinstance(outputs['processed_audio'], torch.Tensor) else [outputs['processed_audio'][i]]
                                    else:
                                        processed_audio_i = sampled_audios_batch[i:i+1]
                                    
                                    # 단일 아이템을 배치 형태로 처리
                                    if isinstance(processed_audio_i, list):
                                        processed_audio_i = torch.stack(processed_audio_i) if len(processed_audio_i) > 0 else sampled_audios_batch[i:i+1]
                                    
                                    clap_loss = self.compute_clap_loss(processed_audio_i, [sampled_descriptions[i]])
                                    individual_clap_losses.append(clap_loss)
                                    clap_items += 1
                                
                                if individual_clap_losses:
                                    clap_loss_total = sum(individual_clap_losses)
                        
                        except Exception as e:
                            if self.rank == 0:
                                print(f"⚠️ Non-guide batch 처리 실패: {e}")
                    
                    # 3) 최종 배치 loss 계산 (Guide + CLAP 조합)
                    total_items = guide_items + clap_items
                    if total_items > 0:
                        if guide_loss_total != 0.0 and clap_loss_total != 0.0:
                            # Guide loss와 CLAP loss를 가중 평균
                            guide_weight = self.args.guide_weight
                            batch_loss = (guide_weight * guide_loss_total / max(1, guide_items) + 
                                         (1 - guide_weight) * clap_loss_total / max(1, clap_items))
                            valid_items = total_items
                        elif guide_loss_total != 0.0:
                            batch_loss = guide_loss_total / max(1, guide_items)
                            valid_items = guide_items
                        elif clap_loss_total != 0.0:
                            batch_loss = clap_loss_total / max(1, clap_items)
                            valid_items = clap_items
                        else:
                            # Fallback loss
                            dummy_param = next(self.model.parameters())
                            batch_loss = torch.mean(dummy_param * 0.0) + 0.1
                            valid_items = 1
                    else:
                        # Fallback loss
                        dummy_param = next(self.model.parameters())
                        batch_loss = torch.mean(dummy_param * 0.0) + 0.1
                        valid_items = 1
                    
                    if self.rank == 0 and batch_idx == 0:
                        print(f"📊 Guide epoch 배치 결과: Guide={guide_items}, CLAP={clap_items}, Total={total_items}/{len(descriptions)}")
                else:
                    # Description 에포크: 모든 항목을 description으로 처리
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
                        
                        # CLAP loss를 배치 단위로 계산 (기존 메서드 사용)
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
        """Guide preset과의 차이를 이용한 보조 loss"""
        # 간단한 MSE loss로 구현
        try:
            # Preset을 tensor로 변환 (구현 필요)
            guide_tensor = self.preset_to_tensor(guide_preset)
            
            if guide_tensor is not None and isinstance(generated_preset, torch.Tensor):
                guide_tensor = guide_tensor.to(self.device)
                # 차원 맞추기
                if generated_preset.dim() != guide_tensor.dim():
                    if generated_preset.dim() == 2 and guide_tensor.dim() == 1:
                        guide_tensor = guide_tensor.unsqueeze(0)
                    elif generated_preset.dim() == 1 and guide_tensor.dim() == 2:
                        generated_preset = generated_preset.unsqueeze(0)
                
                # 크기 맞추기 (더 작은 쪽에 맞춤)
                min_size = min(generated_preset.shape[-1], guide_tensor.shape[-1])
                generated_preset_trimmed = generated_preset[..., :min_size]
                guide_tensor_trimmed = guide_tensor[..., :min_size]
                
                return nn.MSELoss()(generated_preset_trimmed, guide_tensor_trimmed)
            else:
                # fallback: gradient가 있는 더미 loss
                return torch.tensor(0.1, device=self.device, requires_grad=True)
        except Exception as e:
            print(f"⚠️ Guide loss 계산 실패: {e}")
            return torch.tensor(0.1, device=self.device, requires_grad=True)
    
    def preset_to_tensor(self, preset_dict):
        """Preset dictionary를 tensor로 변환"""
        # 이 부분은 실제 preset 구조에 맞게 구현 필요
        # 예시 구현
        try:
            values = []
            
            # Equalizer parameters
            if 'Equalizer' in preset_dict:
                for eq in preset_dict['Equalizer']:
                    values.extend([eq.get('frequency', 0), eq.get('gain', 0), eq.get('q', 0)])
            
            # Reverb parameters  
            if 'Reverb' in preset_dict:
                reverb = preset_dict['Reverb']
                values.extend([
                    reverb.get('room_size', 0),
                    reverb.get('pre_delay', 0),
                    reverb.get('diffusion', 0),
                    reverb.get('damping', 0),
                    reverb.get('wet_gain', 0)
                ])
            
            # Distortion parameters
            if 'Distortion' in preset_dict:
                dist = preset_dict['Distortion']
                values.extend([
                    dist.get('gain', 0),
                    dist.get('color', 0)
                ])
            
            # Pitch parameters
            if 'Pitch' in preset_dict:
                pitch = preset_dict['Pitch']
                values.extend([pitch.get('scale', 0)])
            
            return torch.FloatTensor(values) if values else None
            
        except Exception as e:
            return None
    
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
    
    def train(self):
        """전체 훈련 프로세스"""
        if self.rank == 0:
            print("🎯 훈련 시작")
        
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
                print(f"Epoch {epoch+1}/{self.args.num_epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={current_lr:.1e} ({training_mode}){improvement}")
                
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
                        'guide_epochs': self.args.guide_epochs if self.args.use_guide_presets else 0
                    }, step=global_step)
            
            # 체크포인트 저장
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if (epoch + 1) % self.args.save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        if self.rank == 0:
            print(f"✅ 훈련 완료! Best validation loss: {best_val_loss:.4f}")
            
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
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='전체 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                       help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # 가이드 프리셋 관련
    parser.add_argument('--use_guide_presets', action='store_true', default=False,
                       help='Fine-tuned guide preset 사용 여부')
    parser.add_argument('--guide_epochs', type=int, default=20,
                       help='Guide preset을 사용할 에포크 수 (use_guide_presets가 True일 때만)')
    parser.add_argument('--guide_weight', type=float, default=0.5,
                       help='Guide loss의 가중치 (use_guide_presets가 True일 때만)')
    
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
    
    if args.use_guide_presets:
        print(f"🎯 Guide Presets: ENABLED")
        print(f"   - Guide epochs: {args.guide_epochs}")
        print(f"   - Guide weight: {args.guide_weight}")
    else:
        print(f"📝 Pure Description Training: ENABLED (no guide presets)")
    
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
