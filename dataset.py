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
        print(f"   - 유효한 Fine presets 수: {len(self.fine_presets)}")
        print(f"   - 오디오 길이: {audio_length}초")
        print(f"   - 모드: Fine Preset Only (No Descriptions)")
        print(f"   - 데이터 포인트 수: {len(self.fine_presets)} (유효한 preset 개수와 동일)")
    
    def _load_fine_presets(self, fine_preset_path):
        """Fine-tuned presets 로드 및 유효성 검증"""
        try:
            spec = importlib.util.spec_from_file_location("fine_presets", fine_preset_path)
            fine_presets_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fine_presets_module)
            
            raw_presets = getattr(fine_presets_module, 'fined_presets', [])
            print(f"📥 {len(raw_presets)}개 raw preset 로드됨")
            
            # 🔍 유효한 preset만 필터링
            valid_presets = []
            invalid_count = 0
            
            for i, preset in enumerate(raw_presets):
                if self._is_valid_preset(preset):
                    valid_presets.append(preset)
                else:
                    invalid_count += 1
                    if invalid_count <= 5:  # 처음 5개 에러만 출력
                        print(f"⚠️ 무효한 preset (idx {i}): {type(preset)} - {str(preset)[:100]}")
            
            print(f"✅ {len(valid_presets)}개 유효한 preset (무효: {invalid_count}개)")
            return valid_presets
            
        except Exception as e:
            print(f"❌ Fine presets 로드 실패: {e}")
            return []
    
    def _is_valid_preset(self, preset):
        """Preset 유효성 검증"""
        if not preset or not isinstance(preset, dict):
            return False
        
        # 필수 키들이 있는지 확인
        required_sections = ['eq', 'reverb', 'distortion', 'pitch']
        for section in required_sections:
            if section not in preset:
                return False
            
            section_data = preset[section]
            if not isinstance(section_data, dict):
                return False
        
        # EQ 섹션 상세 검증
        eq_section = preset['eq']
        expected_eq_keys = ['band_1', 'band_2', 'band_3', 'band_4', 'band_5']
        for band in expected_eq_keys:
            if band not in eq_section:
                return False
            band_data = eq_section[band]
            if not isinstance(band_data, dict):
                return False
            # EQ 밴드 필수 파라미터 체크
            band_required = ['center_freq', 'gain_db', 'q', 'filter_type']
            for param in band_required:
                if param not in band_data:
                    return False
        
        # Reverb 섹션 검증
        reverb_section = preset['reverb']
        reverb_required = ['room_size', 'pre_delay', 'diffusion', 'damping', 'wet_gain']
        for param in reverb_required:
            if param not in reverb_section:
                return False
        
        # Distortion 섹션 검증
        dist_section = preset['distortion']
        dist_required = ['gain', 'color']
        for param in dist_required:
            if param not in dist_section:
                return False
        
        # Pitch 섹션 검증
        pitch_section = preset['pitch']
        if 'scale' not in pitch_section:
            return False
        
        return True
    
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
        """Fine preset 아이템 반환 (description 없음) - 사전 검증된 preset 사용"""
        try:
            preset = self.fine_presets[idx]  # 이미 유효성 검증된 preset
            
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
                'guide_preset': preset,  # 이미 검증된 preset
                'is_guide': True,  # 사전 훈련 플래그
                'subject': 'instrumental',
                'audio_type': 'instrumental'
            }
            
        except Exception as e:
            print(f"❌ 사전 훈련 아이템 로드 실패 (idx: {idx}): {e}")
            # 빈 preset이 아닌 기본 preset 구조 반환
            default_preset = {
                'eq': {
                    'band_1': {'center_freq': 100, 'gain_db': 0.0, 'q': 1.0, 'filter_type': 'high_pass'},
                    'band_2': {'center_freq': 500, 'gain_db': 0.0, 'q': 1.0, 'filter_type': 'bell'},
                    'band_3': {'center_freq': 2000, 'gain_db': 0.0, 'q': 1.0, 'filter_type': 'bell'},
                    'band_4': {'center_freq': 8000, 'gain_db': 0.0, 'q': 1.0, 'filter_type': 'bell'},
                    'band_5': {'center_freq': 15000, 'gain_db': 0.0, 'q': 1.0, 'filter_type': 'low_pass'}
                },
                'reverb': {'room_size': 5.0, 'pre_delay': 20.0, 'diffusion': 0.7, 'damping': 0.5, 'wet_gain': 0.3},
                'distortion': {'gain': 10.0, 'color': 0.6},
                'pitch': {'scale': 1.0}
            }
            
            return {
                'description': f"preset_{idx}_default",
                'audio': torch.zeros(int(self.sample_rate * self.audio_length)),
                'guide_preset': default_preset,
                'is_guide': True,
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
        """Fine-tuned presets 파일 로드 및 유효성 검증"""
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
                raw_presets = module.fined_presets
                print(f"📥 {len(raw_presets)}개 raw preset 로드됨")
                
                # 🔍 유효한 preset만 필터링
                valid_presets = []
                invalid_count = 0
                
                for i, preset in enumerate(raw_presets):
                    if self._is_valid_preset(preset):
                        valid_presets.append(preset)
                    else:
                        invalid_count += 1
                        if invalid_count <= 3:  # 처음 3개 에러만 출력
                            print(f"⚠️ 무효한 preset (idx {i}): {type(preset)}")
                
                presets = valid_presets
                print(f"✅ {len(presets)}개 유효한 preset (무효: {invalid_count}개)")
            else:
                print(f"❌ 'fined_presets' 속성을 찾을 수 없음")
                
        except Exception as e:
            print(f"❌ Fine preset 로드 실패: {e}")
            traceback.print_exc()
            
        return presets
    
    def _is_valid_preset(self, preset):
        """Preset 유효성 검증 (PresetDataset용)"""
        if not preset or not isinstance(preset, dict):
            return False
        
        # 필수 키들이 있는지 확인
        required_sections = ['eq', 'reverb', 'distortion', 'pitch']
        for section in required_sections:
            if section not in preset:
                return False
            
            section_data = preset[section]
            if not isinstance(section_data, dict):
                return False
        
        # EQ 섹션 상세 검증
        eq_section = preset['eq']
        expected_eq_keys = ['band_1', 'band_2', 'band_3', 'band_4', 'band_5']
        for band in expected_eq_keys:
            if band not in eq_section:
                return False
            band_data = eq_section[band]
            if not isinstance(band_data, dict):
                return False
            # EQ 밴드 필수 파라미터 체크
            band_required = ['center_freq', 'gain_db', 'q', 'filter_type']
            for param in band_required:
                if param not in band_data:
                    return False
        
        # Reverb 섹션 검증
        reverb_section = preset['reverb']
        reverb_required = ['room_size', 'pre_delay', 'diffusion', 'damping', 'wet_gain']
        for param in reverb_required:
            if param not in reverb_section:
                return False
        
        # Distortion 섹션 검증
        dist_section = preset['distortion']
        dist_required = ['gain', 'color']
        for param in dist_required:
            if param not in dist_section:
                return False
        
        # Pitch 섹션 검증
        pitch_section = preset['pitch']
        if 'scale' not in pitch_section:
            return False
        
        return True
    
    
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
        
        # Fine preset 가이드 (간단한 할당 - 이미 검증된 preset만 사용)
        guide_preset = {}
        if self.use_fine_tuned_presets and self.fine_presets:
            # 간단한 랜덤 할당 (모든 preset이 이미 검증됨)
            if random.random() < 0.3:  # 30% 확률로 guide preset 할당
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
    """커스텀 collate function 생성 - 사전 검증된 preset 사용"""
    
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
            # 모든 preset은 이미 dataset에서 검증됨 - 추가 검증 불필요
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
