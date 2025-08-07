#!/usr/bin/env python3

import os
import sys
import random
import torch
import librosa
import numpy as np
import importlib.util
import traceback
from pathlib import Path
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """사전 훈련 전용 데이터셋 - Fine Preset만 사용"""
    
    def __init__(self, fine_preset_path, audio_dataset_path, sample_rate=44100, audio_length=5.0):
        self.audio_dataset_path = audio_dataset_path
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        
        # Fine-tuned presets만 로드 (description 없음)
        self.fine_presets = []
        if fine_preset_path and os.path.exists(fine_preset_path):
            self.fine_presets = self._load_fine_presets(fine_preset_path)
        
        if not self.fine_presets:
            raise ValueError("❌ 사전 훈련용 Fine Preset을 로드할 수 없습니다")
        
        print(f"🎯 사전 훈련 데이터셋 초기화:")
        print(f"   - Fine presets 수: {len(self.fine_presets)}")
        print(f"   - 오디오 길이: {audio_length}초")
        print(f"   - 모드: Fine Preset Only (No Descriptions)")
    
    def _load_fine_presets(self, fine_preset_path):
        """Fine-tuned presets 로드"""
        try:
            spec = importlib.util.spec_from_file_location("fine_presets", fine_preset_path)
            fine_presets_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fine_presets_module)
            
            fine_presets = getattr(fine_presets_module, 'fined_presets', [])
            print(f"✅ {len(fine_presets)}개 fine preset 로드 완료")
            return fine_presets
            
        except Exception as e:
            print(f"❌ Fine presets 로드 실패: {e}")
            return []
    
    def _load_audio(self, audio_path):
        """오디오 파일 로드"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            target_length = int(self.sample_rate * self.audio_length)
            
            if len(audio) > target_length:
                start_idx = random.randint(0, len(audio) - target_length)
                audio = audio[start_idx:start_idx + target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            
            return torch.FloatTensor(audio)
        except Exception as e:
            print(f"⚠️ 오디오 로드 실패 {audio_path}: {e}")
            return torch.zeros(int(self.sample_rate * self.audio_length))
    
    def __len__(self):
        return len(self.fine_presets)
    
    def __getitem__(self, idx):
        """Fine preset 아이템 반환 (description 없음)"""
        try:
            preset = self.fine_presets[idx]
            
            # 랜덤 오디오 선택 (instrumentals 폴더에서)
            instrumentals_path = os.path.join(self.audio_dataset_path, 'instrumentals')
            audio_files = []
            
            for root, dirs, files in os.walk(instrumentals_path):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac')):
                        audio_files.append(os.path.join(root, file))
            
            if not audio_files:
                print("⚠️ 오디오 파일을 찾을 수 없습니다")
                audio = torch.zeros(int(self.sample_rate * self.audio_length))
            else:
                audio_path = random.choice(audio_files)
                audio = self._load_audio(audio_path)
            
            return {
                'description': f"preset_{idx}",  # 더미 description 
                'audio': audio,
                'guide_preset': preset,
                'is_guide': True,  # 사전 훈련 플래그
                'subject': 'instrumental',
                'audio_type': 'instrumental'
            }
            
        except Exception as e:
            print(f"❌ 사전 훈련 아이템 로드 실패 (idx: {idx}): {e}")
            return {
                'description': f"preset_{idx}",
                'audio': torch.zeros(int(self.sample_rate * self.audio_length)),
                'guide_preset': {},
                'is_guide': False,
                'subject': 'instrumental',
                'audio_type': 'instrumental'
            }


class PresetDataset(Dataset):
    """오디오 이펙트 프리셋 데이터셋 - Description과 Guide Preset 지원"""
    
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
        
        # Fine-tuned presets 로드 (가이드용)
        self.fine_presets = []
        if use_fine_tuned_presets and fine_preset_path and os.path.exists(fine_preset_path):
            self.fine_presets = self._load_fine_presets(fine_preset_path)
        
        print(f"📊 데이터셋 초기화 완료:")
        print(f"   - Description 수: {len(self.descriptions)}")
        print(f"   - Fine presets 수: {len(self.fine_presets)}")
        print(f"   - 오디오 길이: {audio_length}초")
        print(f"   - Guide Preset 사용: {'✅' if use_fine_tuned_presets else '❌'}")
    
    def _load_fine_presets(self, preset_path):
        """Fine-tuned presets 파일 로드"""
        presets = []
        try:
            print(f"📖 Fine preset 파일 읽기 시작: {preset_path}")
            
            # Python 파일을 모듈로 동적 로드
            spec = importlib.util.spec_from_file_location("fined_presets", preset_path)
            if spec is None:
                print(f"❌ 모듈 스펙 생성 실패: {preset_path}")
                return presets
            
            # 모듈 생성 및 로드
            module = importlib.util.module_from_spec(spec)
            
            # sys.modules에 추가 (중복 로드 방지)
            module_name = f"fined_presets_{id(module)}"
            sys.modules[module_name] = module
            
            # 실행
            spec.loader.exec_module(module)
            
            # fined_presets 리스트 가져오기
            if hasattr(module, 'fined_presets'):
                presets = module.fined_presets
                print(f"✅ {len(presets)}개 fine preset 로드 완료")
            else:
                print(f"❌ 'fined_presets' 속성을 찾을 수 없음")
                
        except Exception as e:
            print(f"❌ Fine preset 로드 실패: {e}")
            traceback.print_exc()
            
        return presets
    
    
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
                return 'neutral', 'speech'
        
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
        
        # WAV, FLAC 파일들 찾기
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
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
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
        guide_preset = {}
        if self.use_fine_tuned_presets and self.fine_presets:
            # 1단계: Description과 정확히 매칭되는 preset 찾기
            best_match_score = 0.0
            best_preset = None
            
            for preset in self.fine_presets:
                match_score = self._match_preset_to_description(description, preset)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_preset = preset
            
            # 2단계: 매칭도가 임계값 이상이면 사용, 아니면 확률적 할당
            if best_preset and best_match_score > 0.3:
                guide_preset = best_preset
            elif self.fine_presets and random.random() < 0.2:  # 20% 확률로 랜덤 할당
                guide_preset = random.choice(self.fine_presets)
        
        return {
            'description': description,
            'audio': audio,
            'subject': subject,
            'audio_type': audio_type,
            'guide_preset': guide_preset
        }


class PureDescriptionDataset(Dataset):
    """Pure Description 전용 데이터셋 (Guide Preset 없음)"""
    
    def __init__(self, descriptions, audio_dataset_path, sample_rate=44100, audio_length=5.0):
        self.descriptions = descriptions
        self.audio_dataset_path = audio_dataset_path
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        
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
        
        print(f"📊 Pure Description 데이터셋 초기화 완료:")
        print(f"   - Description 수: {len(self.descriptions)}")
        print(f"   - 오디오 길이: {audio_length}초")
        print(f"   - Guide Preset: ❌ 비활성화 (Pure Description Only)")
    
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
                return 'neutral', 'speech'
        
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
            # 대체 폴더 사용
            folder_path = os.path.join(self.audio_dataset_path, 'instrumentals', 'Piano')
        
        # WAV, FLAC 파일들 찾기
        audio_files = []
        for ext in ['*.wav', '*.flac', '*.mp3', '*.m4a']:
            audio_files.extend(list(Path(folder_path).glob(ext)))
        
        if not audio_files:
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
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
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
        
        return {
            'description': description,
            'audio': audio,
            'subject': subject,
            'audio_type': audio_type
        }


def create_custom_collate_fn(include_guide_preset=True):
    """커스텀 collate function 생성"""
    
    def collate_fn(batch):
        descriptions = [item['description'] for item in batch]
        audios = torch.stack([item['audio'] for item in batch])
        subjects = [item['subject'] for item in batch]
        audio_types = [item['audio_type'] for item in batch]
        
        result = {
            'description': descriptions,
            'audio': audios,
            'subject': subjects,
            'audio_type': audio_types
        }
        
        if include_guide_preset:
            guide_presets = [item.get('guide_preset', {}) for item in batch]
            result['guide_preset'] = guide_presets
        
        return result
    
    return collate_fn


def load_descriptions(data_path, use_sampled_descriptions=False, max_descriptions=0):
    """Description 파일 로드"""
    descriptions = []
    
    if use_sampled_descriptions:
        desc_file = os.path.join(data_path, 'descriptions', '500_sampled_descriptions.txt')
    else:
        desc_file = os.path.join(data_path, 'descriptions', 'descriptions.txt')
    
    print(f"📂 Description 파일 경로: {desc_file}")
    
    if os.path.exists(desc_file):
        with open(desc_file, 'r', encoding='utf-8') as f:
            descriptions = [line.strip() for line in f if line.strip()]
    else:
        print(f"❌ Description 파일을 찾을 수 없음: {desc_file}")
        return []
    
    # 최대 description 수 제한
    if max_descriptions > 0 and len(descriptions) > max_descriptions:
        print(f"📊 Description 수 제한: {len(descriptions)} → {max_descriptions}")
        random.seed(42)
        descriptions = random.sample(descriptions, max_descriptions)
    
    print(f"📚 {len(descriptions)}개 description 로드됨")
    if len(descriptions) > 100000:
        print(f"⚠️  대용량 데이터셋! 한 에포크에 약 {len(descriptions):,}개 description 사용")
        print(f"   --max_descriptions 옵션으로 제한 권장 (예: --max_descriptions 50000)")
    
    return descriptions


def split_descriptions(descriptions, train_ratio=0.8):
    """Description을 train/validation으로 분할"""
    random.seed(42)
    random.shuffle(descriptions)
    split_idx = int(len(descriptions) * train_ratio)
    train_descriptions = descriptions[:split_idx]
    val_descriptions = descriptions[split_idx:]
    
    return train_descriptions, val_descriptions
