#!/usr/bin/env python3
"""
Audio Effect Preset Generation Model Testing Script
훈련된 모델로 오디오 이펙트를 적용하는 테스트 스크립트
"""

import os
import sys
import torch
import torchaudio
import argparse
import numpy as np
from pathlib import Path
import glob
import json
import re
from datetime import datetime

# 환경 변수 설정
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 프로젝트 모듈 import
try:
    from pipeline import TextToAudioProcessingPipeline
    from dynamic_pipeline_factory import DynamicPipelineFactory
    from encoder.text_encoder import CLAPTextEncoder
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}")
    print("현재 디렉토리가 AudioManipulator 루트인지 확인하세요.")
    sys.exit(1)

class AudioEffectTester:
    """훈련된 모델을 사용한 오디오 이펙트 테스터"""
    
    def __init__(self, checkpoint_dir="./checkpoints", device=None):
        """
        Args:
            checkpoint_dir: 체크포인트 디렉토리 경로
            device: 사용할 디바이스 (None이면 자동 선택)
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.args = None
        
        print(f"🎵 Audio Effect Tester 초기화")
        print(f"   - Device: {self.device}")
        print(f"   - Checkpoint dir: {checkpoint_dir}")
    
    def find_latest_checkpoint(self):
        """가장 최신 체크포인트 찾기"""
        if not os.path.exists(self.checkpoint_dir):
            raise FileNotFoundError(f"체크포인트 디렉토리를 찾을 수 없습니다: {self.checkpoint_dir}")
        
        # 모든 체크포인트 파일 찾기
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_pattern}")
        
        # 에포크 번호로 정렬해서 가장 최신 것 선택
        def extract_epoch(filepath):
            try:
                filename = os.path.basename(filepath)
                # checkpoint_epoch_XX.pt에서 XX 추출
                epoch_str = filename.replace('checkpoint_epoch_', '').replace('.pt', '')
                return int(epoch_str)
            except:
                return -1
        
        latest_checkpoint = max(checkpoint_files, key=extract_epoch)
        latest_epoch = extract_epoch(latest_checkpoint)
        
        print(f"📁 최신 체크포인트 발견: {os.path.basename(latest_checkpoint)} (Epoch {latest_epoch})")
        return latest_checkpoint
    
    def load_model(self, checkpoint_path=None):
        """모델 로드"""
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
        
        print(f"🔄 체크포인트 로드 중: {checkpoint_path}")
        
        try:
            # 체크포인트 로드
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.args = checkpoint['args']
            
            print(f"✅ 체크포인트 정보:")
            print(f"   - Epoch: {checkpoint['epoch']}")
            print(f"   - Validation Loss: {checkpoint.get('val_loss', 'N/A')}")
            print(f"   - Text Encoder: {self.args.text_encoder_type}")
            print(f"   - Sample Rate: {self.args.sample_rate}")
            
            # 모델 재구성
            print("🔄 모델 재구성 중...")
            
            # DynamicPipelineFactory로 모델 생성 (훈련 시와 동일한 설정)
            factory = DynamicPipelineFactory()
            
            self.model = factory.create_pipeline(
                encoder_preset=self.args.text_encoder_type,
                use_clap=True,
                backbone_type='residual',
                decoder_type='parallel',
                sample_rate=self.args.sample_rate,
                use_differentiable_audio=True,
                target_params=500000
            )
            
            # 모델을 디바이스로 이동
            self.model = self.model.to(self.device)
            
            # 상태 딕셔너리 로드
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ 모델 상태 딕셔너리 로드 완료")
            except Exception as e:
                print(f"⚠️  모델 상태 딕셔너리 로드 중 오류: {e}")
                # 부분적 로드 시도
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                 if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                print(f"✅ 부분적 모델 로드 완료 ({len(pretrained_dict)}/{len(checkpoint['model_state_dict'])} layers)")
            
            # 평가 모드로 설정
            self.model.eval()
            
            print("✅ 모델 로드 완료!")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_audio(self, audio_path, target_length=None):
        """오디오 파일 로드 및 전처리"""
        print(f"🎵 오디오 로드: {audio_path}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        
        try:
            # torchaudio로 로드
            waveform, sample_rate = torchaudio.load(audio_path)
            original_length = waveform.shape[1] / sample_rate  # 원본 길이 계산
            
            # 모노로 변환 (채널이 여러 개인 경우)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 샘플링 레이트 변환
            if sample_rate != self.args.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.args.sample_rate
                )
                waveform = resampler(waveform)
                print(f"🔄 샘플링 레이트 변환: {sample_rate}Hz → {self.args.sample_rate}Hz")
            
            # 길이 조정 (target_length가 지정된 경우에만)
            if target_length:
                target_samples = int(self.args.sample_rate * target_length)
                current_samples = waveform.shape[1]
                
                if current_samples < target_samples:
                    # 패딩
                    padding = target_samples - current_samples
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                    print(f"🔄 오디오 패딩: {current_samples} → {target_samples} samples")
                elif current_samples > target_samples:
                    # 자르기
                    waveform = waveform[:, :target_samples]
                    print(f"🔄 오디오 자르기: {current_samples} → {target_samples} samples")
            
            print(f"✅ 오디오 로드 완료: {waveform.shape}, {self.args.sample_rate}Hz")
            return waveform.squeeze(0), original_length  # [length] 형태로 반환, 원본 길이도 반환
            
        except Exception as e:
            print(f"❌ 오디오 로드 실패: {e}")
            raise
    
    def process_audio(self, audio_tensor, text_prompt):
        """오디오에 텍스트 프롬프트 기반 이펙트 적용"""
        print(f"🎛️  오디오 프로세싱 시작...")
        print(f"   - Text prompt: {text_prompt}")
        print(f"   - Audio shape: {audio_tensor.shape}")
        
        try:
            with torch.no_grad():
                # 텐서를 디바이스로 이동
                audio_tensor = audio_tensor.to(self.device).unsqueeze(0)  # [1, length]
                
                # 텍스트 프롬프트를 리스트로 변환 (모델이 List[str]을 기대함)
                if isinstance(text_prompt, str):
                    text_prompts = [text_prompt]
                else:
                    text_prompts = text_prompt
                
                # 모델에 입력 (올바른 순서: texts 먼저, audio 두 번째)
                result = self.model(texts=text_prompts, audio=audio_tensor, use_real_audio=True)
                
                # 결과가 딕셔너리인 경우 processed_audio 키 사용
                if isinstance(result, dict):
                    preset_params = result.get('preset_params', None)
                    if 'processed_audio' in result:
                        processed_audio = result['processed_audio']
                    elif 'audio' in result:
                        processed_audio = result['audio']
                    else:
                        # 딕셔너리의 첫 번째 텐서 값 사용
                        processed_audio = next(iter(result.values()))
                else:
                    processed_audio = result
                    preset_params = None
                
                # 결과가 튜플인 경우 첫 번째 요소 사용
                if isinstance(processed_audio, tuple):
                    processed_audio = processed_audio[0]
                
                # 배치 차원 제거
                if processed_audio.dim() > 1:
                    processed_audio = processed_audio.squeeze(0)
                
                # CPU로 이동
                processed_audio = processed_audio.cpu()
                
                print(f"✅ 프로세싱 완료: {processed_audio.shape}")
                return processed_audio, preset_params
                
        except Exception as e:
            print(f"❌ 오디오 프로세싱 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_audio(self, audio_tensor, output_path, sample_rate=None):
        """처리된 오디오 저장"""
        if sample_rate is None:
            sample_rate = self.args.sample_rate
        
        print(f"💾 오디오 저장: {output_path}")
        
        try:
            # 오디오 정규화 (클리핑 방지)
            if audio_tensor.abs().max() > 1.0:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
                print("🔄 오디오 정규화 적용")
            
            # 스테레오로 변환 (채널 차원 추가)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # [1, length]
            
            # 출력 디렉토리 생성
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 저장
            torchaudio.save(output_path, audio_tensor, sample_rate)
            print(f"✅ 오디오 저장 완료: {output_path}")
            
        except Exception as e:
            print(f"❌ 오디오 저장 실패: {e}")
            raise
    
    def sanitize_filename(self, text):
        """텍스트를 파일명으로 사용 가능하도록 정리"""
        # 특수문자 제거 및 공백을 언더스코어로 변경
        filename = re.sub(r'[^\w\s-]', '', text.strip())
        filename = re.sub(r'[-\s]+', '_', filename)
        # 길이 제한 (50자)
        if len(filename) > 50:
            filename = filename[:50]
        return filename.lower()
    
    def save_preset_json(self, preset_params, text_prompt, output_dir, base_filename):
        """생성된 preset을 JSON 형태로 저장"""
        if preset_params is None:
            print("⚠️  Preset 파라미터가 없어 JSON 저장을 건너뜁니다.")
            return None
        
        try:
            # preset_params를 JSON 호환 딕셔너리로 변환
            def convert_to_serializable(obj):
                """재귀적으로 텐서를 리스트로 변환"""
                if isinstance(obj, torch.Tensor):
                    return obj.cpu().detach().numpy().tolist()
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                else:
                    # 알 수 없는 타입은 문자열로 변환
                    return str(obj)
            
            if isinstance(preset_params, torch.Tensor):
                # 텐서인 경우 numpy로 변환 후 딕셔너리로 파싱
                preset_array = preset_params.cpu().detach().numpy()
                preset_dict = self._parse_preset_array(preset_array)
            else:
                # 딕셔너리나 다른 타입은 재귀적으로 변환
                preset_dict = convert_to_serializable(preset_params)
            
            # 메타데이터 추가
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "text_prompt": text_prompt,
                "model_info": {
                    "text_encoder": self.args.text_encoder_type,
                    "sample_rate": self.args.sample_rate,
                    "backbone_type": "residual",
                    "decoder_type": "parallel"
                },
                "preset_parameters": preset_dict
            }
            
            # JSON 파일 저장
            json_path = os.path.join(output_dir, f"{base_filename}_preset.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Preset JSON 저장 완료: {json_path}")
            return json_path
            
        except Exception as e:
            print(f"❌ Preset JSON 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_preset_array(self, preset_array):
        """Preset 배열을 의미있는 딕셔너리로 파싱"""
        preset_dict = {}
        
        if len(preset_array.shape) == 1:
            # 1D 배열인 경우
            params = preset_array.tolist()
            
            # 파라미터 개수에 따라 다르게 파싱
            if len(params) >= 25:  # 충분한 파라미터가 있는 경우
                idx = 0
                
                # EQ 파라미터 (5 밴드 * 3 파라미터 = 15개)
                eq_params = {}
                for band in range(1, 6):
                    if idx + 2 < len(params):
                        eq_params[f"band_{band}"] = {
                            "center_freq": params[idx],
                            "gain_db": params[idx + 1], 
                            "q": params[idx + 2],
                            "filter_type": "bell" if band in [2, 3, 4] else ("high_pass" if band == 1 else "low_pass")
                        }
                        idx += 3
                preset_dict["equalizer"] = eq_params
                
                # Reverb 파라미터 (6개)
                if idx + 5 < len(params):
                    preset_dict["reverb"] = {
                        "room_size": params[idx],
                        "pre_delay": params[idx + 1],
                        "diffusion": params[idx + 2],
                        "damping": params[idx + 3],
                        "wet_gain": params[idx + 4],
                        "dry_gain": params[idx + 5]
                    }
                    idx += 6
                
                # Distortion 파라미터 (4개)
                if idx + 3 < len(params):
                    preset_dict["distortion"] = {
                        "gain": params[idx],
                        "bias": params[idx + 1],
                        "tone": params[idx + 2],
                        "mix": params[idx + 3]
                    }
                    idx += 4
                
                # Pitch 파라미터 (3개)
                if idx + 2 < len(params):
                    preset_dict["pitch"] = {
                        "pitch_shift": params[idx],
                        "formant_shift": params[idx + 1],
                        "mix": params[idx + 2]
                    }
                    idx += 3
                
                # 나머지 파라미터들
                for i, param in enumerate(params[idx:], idx):
                    preset_dict[f"extra_param_{i}"] = param
                    
            else:
                # 파라미터가 적은 경우 단순히 번호로 매핑
                for i, param in enumerate(params):
                    preset_dict[f"param_{i}"] = param
        else:
            # 다차원 배열인 경우
            preset_dict["raw_parameters"] = preset_array.tolist()
        
        return preset_dict
    
    def test_audio_effect(self, audio_path, text_prompt, output_path=None, audio_length=None):
        """오디오 이펙트 테스트 (전체 파이프라인)"""
        print(f"\n{'='*60}")
        print(f"🎯 오디오 이펙트 테스트 시작")
        print(f"{'='*60}")
        
        # 출력 디렉토리 및 파일명 설정
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_prompt = self.sanitize_filename(text_prompt)
            output_dir = os.path.join("./output", f"{timestamp}_{sanitized_prompt}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 오디오 파일명도 텍스트 기반으로 생성
            audio_filename = f"{sanitized_prompt}.wav"
            audio_output_path = os.path.join(output_dir, audio_filename)
        else:
            audio_output_path = output_path
            output_dir = os.path.dirname(audio_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            sanitized_prompt = self.sanitize_filename(text_prompt)
        
        try:
            # 1. 오디오 로드 (원본 길이 유지)
            audio_tensor, original_length = self.load_audio(audio_path, target_length=audio_length)
            
            # 2. 이펙트 적용
            processed_audio, preset_params = self.process_audio(audio_tensor, text_prompt)
            
            # 3. 결과 저장
            self.save_audio(processed_audio, audio_output_path)
            
            # 4. Preset JSON 저장
            base_filename = sanitized_prompt
            json_path = self.save_preset_json(preset_params, text_prompt, output_dir, base_filename)
            
            # 5. 메타데이터 파일 생성
            metadata_path = os.path.join(output_dir, "metadata.txt")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"Audio Effect Generation Report\n")
                f.write(f"=" * 40 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Input Audio: {audio_path}\n")
                f.write(f"Text Prompt: {text_prompt}\n")
                f.write(f"Original Duration: {original_length:.2f}s\n")
                f.write(f"Sample Rate: {self.args.sample_rate}Hz\n")
                f.write(f"Model: {self.args.text_encoder_type}\n\n")
                f.write(f"Output Files:\n")
                f.write(f"- Audio: {os.path.basename(audio_output_path)}\n")
                if json_path:
                    f.write(f"- Preset: {os.path.basename(json_path)}\n")
                f.write(f"- Metadata: {os.path.basename(metadata_path)}\n")
            
            # 6. 결과 요약
            print(f"\n🎉 테스트 완료!")
            print(f"   - Input: {audio_path}")
            print(f"   - Prompt: {text_prompt}")
            print(f"   - Output Dir: {output_dir}")
            print(f"   - Audio: {os.path.basename(audio_output_path)}")
            if json_path:
                print(f"   - Preset: {os.path.basename(json_path)}")
            print(f"   - Duration: {original_length:.2f}s (original)")
            
            return audio_output_path
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Audio Effect Model Tester')
    
    # 모델 관련
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='체크포인트 디렉토리 경로')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='특정 체크포인트 파일 경로 (지정하지 않으면 최신 사용)')
    
    # 입력 관련
    parser.add_argument('--input_audio', type=str, required=True,
                       help='입력 오디오 파일 경로')
    parser.add_argument('--text_prompt', type=str, required=True,
                       help='텍스트 프롬프트 (적용할 이펙트 설명)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='출력 오디오 파일 경로 (지정하지 않으면 자동 생성)')
    
    # 오디오 관련
    parser.add_argument('--audio_length', type=float, default=None,
                       help='처리할 오디오 길이 (초, 지정하지 않으면 원본 길이 사용)')
    
    # 시스템 관련
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', 'auto'],
                       help='사용할 디바이스 (auto는 자동 선택)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'auto' or args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🎵 Audio Effect Model Tester")
    print("=" * 50)
    print(f"📁 Input audio: {args.input_audio}")
    print(f"📝 Text prompt: {args.text_prompt}")
    print(f"💻 Device: {device}")
    if args.audio_length:
        print(f"⏱️  Audio length: {args.audio_length}s")
    else:
        print(f"⏱️  Audio length: original duration")
    
    try:
        # 테스터 초기화
        tester = AudioEffectTester(
            checkpoint_dir=args.checkpoint_dir,
            device=device
        )
        
        # 모델 로드
        tester.load_model(args.checkpoint_path)
        
        # 오디오 이펙트 테스트
        output_path = tester.test_audio_effect(
            audio_path=args.input_audio,
            text_prompt=args.text_prompt,
            output_path=args.output_path,
            audio_length=args.audio_length
        )
        
        print(f"\n🎉 모든 처리가 완료되었습니다!")
        print(f"결과 파일: {output_path}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()