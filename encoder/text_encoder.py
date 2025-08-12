#!/usr/bin/env python3

import os
import time
import io
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Optional
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import warnings
from abc import ABC, abstractmethod
import torchaudio.transforms as T


# Removed CLAPAudioEmbeddingFunction - using direct Hugging Face CLAP model instead

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import transformers
    from transformers import AutoTokenizer, AutoModel
    from transformers import logging as hf_logging
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Install with: pip install transformers")

try:
    from transformers import ClapModel, ClapProcessor
    CLAP_AVAILABLE = True
except Exception as e:
    CLAP_AVAILABLE = False
    warnings.warn(f"transformers CLAP not available or failed to import ({e}). Install with: pip install transformers")

@contextmanager
def _suppress_output(quiet: bool):
    """강력한 출력 억제: Python stdout/stderr + 로깅 + C/C++ FDs(1,2)까지 무음화"""
    if not quiet:
        yield
        return
    # 백업 로깅 레벨
    previous_level = logging.root.manager.disable
    # 파이썬 stdout/stderr 리디렉션
    buf = io.StringIO()
    # 파일 디스크립터 백업
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        # 모든 로깅 비활성화
        logging.disable(logging.CRITICAL)
        # C/C++ 레벨 출력 무음화
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        with redirect_stdout(buf), redirect_stderr(buf):
            yield
    finally:
        # 원복
        try:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
        finally:
            os.close(stdout_fd)
            os.close(stderr_fd)
            os.close(devnull_fd)
        logging.disable(previous_level)

# DDP 유틸 (있으면 사용)
try:
    import torch.distributed as dist
    DIST_AVAILABLE = True
except Exception:
    DIST_AVAILABLE = False


class BaseTextEncoder(nn.Module, ABC):
    """
    Base class for all text encoders
    Provides unified interface for different encoder types
    """
    
    @abstractmethod
    def encode_text(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text prompts to embeddings
        
        Args:
            text_prompts: Single string or list of strings
            
        Returns:
            embeddings: (batch_size, embedding_dim) tensor
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        pass
    
    def forward(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """Forward pass - calls encode_text for consistency"""
        return self.encode_text(text_prompts)


class CLAPTextEncoder(nn.Module):
    """
    Hugging Face CLAP 기반 인코더.
    자기 지도 학습을 위한 미분 가능한 오디오 임베딩 추출 기능을 포함합니다.
    """
    def __init__(self, model_name='laion/clap-htsat-unfused', freeze_model=True):
        super().__init__()
        # 중복 초기화를 방지하기 위한 간단한 플래그
        if hasattr(self, '_initialized'):
            return
        
        print(f"🎵 Initializing new CLAP instance: {model_name}")
        self.model_name = model_name
        self.freeze_model = freeze_model
        
        # --- 1. 모델 및 프로세서 로드 ---
        self.clap_model = ClapModel.from_pretrained(model_name, use_safetensors=True)
        self.clap_processor = ClapProcessor.from_pretrained(model_name)
        
        # --- 2. 모델 고정(Freeze) 및 평가 모드(eval) 설정 ---
        if self.freeze_model:
            for param in self.clap_model.parameters():
                param.requires_grad = False
                print("🔒 CLAP model is frozen for embedding extraction.")
        
        # ⚠️ BatchNorm 에러 해결: 교사 모델은 항상 eval() 모드여야 합니다.
        self.clap_model.eval()

        # --- 3. 미분 가능한 오디오 전처리기 생성 (torchaudio) ---
        # 이 변환기들은 한 번만 생성하여 재사용합니다.
        feature_extractor = self.clap_processor.feature_extractor
        self.target_sr = feature_extractor.sampling_rate
        n_mels = feature_extractor.feature_size
        n_fft = feature_extractor.n_fft
        hop_length = feature_extractor.hop_length
        win_length = n_fft  # 일반적인 관례
        
        self.resampler = T.Resample(orig_freq=48000, new_freq=self.target_sr) # 입력 오디오 SR을 48k로 가정
        self.mel_spectrogram_converter = T.MelSpectrogram(
            sample_rate=self.target_sr, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, n_mels=n_mels
        )

        self._initialized = True
        print(f"✅ CLAP Encoder initialized: {self.model_name}")

    def to(self, device):
        """모델과 전처리기들을 지정된 장치로 이동시키는 메서드"""
        super().to(device)
        self.clap_model.to(device)
        self.resampler.to(device)
        self.mel_spectrogram_converter.to(device)
        return self

    def get_text_embedding(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """텍스트 임베딩 추출"""
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        device = next(self.clap_model.parameters()).device
        inputs = self.clap_processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
        
        # 텍스트 임베딩은 그래디언트가 필요 없으므로 no_grad 사용
        with torch.no_grad():
            text_embeds = self.clap_model.get_text_features(**inputs)
        return text_embeds

    def get_audio_embedding_with_grad(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        그래디언트가 흐르는 오디오 임베딩 추출 (자기 지도 학습용)
        """
        import pdb
        
        # print(f"🔍 DEBUG: 입력 오디오 형태")
        # print(f"   Shape: {audio_tensor.shape}")
        # print(f"   Dtype: {audio_tensor.dtype}")
        # print(f"   Device: {audio_tensor.device}")
        # print(f"   Requires_grad: {audio_tensor.requires_grad}")
        # print(f"   Min/Max values: {audio_tensor.min().item():.4f} / {audio_tensor.max().item():.4f}")
        
        # pdb.set_trace()  # 디버깅 중단점
        
        device = audio_tensor.device
        self.to(device) # 모든 구성요소를 올바른 장치로 이동

        # --- 1. 오디오 길이 패딩 (10초) ---
        target_len_s = 10
        target_samples = self.target_sr * target_len_s
        
        # 입력 오디오 차원 확인 및 정규화
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # (samples,) -> (1, samples)
        elif audio_tensor.dim() == 3:  # (batch, channels, samples)
            if audio_tensor.size(1) > 1:
                # 스테레오를 모노로 변환
                audio_tensor = audio_tensor.mean(dim=1)  # (batch, samples)
            else:
                audio_tensor = audio_tensor.squeeze(1)  # (batch, samples)
        # 이제 audio_tensor는 (batch, samples) 형태
        
        batch_size, audio_len = audio_tensor.shape
        padded_audios = torch.zeros(batch_size, target_samples, device=device)
        
        for i in range(batch_size):
            if audio_len > 0:
                repeat_factor = (target_samples + audio_len - 1) // audio_len
                # 1차원 텐서에 대해 repeat 적용
                repeated_audio = audio_tensor[i].repeat(repeat_factor)
                padded_audios[i] = repeated_audio[:target_samples]
            else:
                # 빈 오디오인 경우 0으로 채움
                padded_audios[i] = torch.zeros(target_samples, device=device)

        # --- 2. 미분 가능한 전처리 수행 ---
        resampled_audios = self.resampler(padded_audios) # 입력 오디오 SR이 48k가 아닐 경우를 대비
        mel_spectrograms = self.mel_spectrogram_converter(resampled_audios)
        log_mel_spectrograms = (mel_spectrograms.clamp(min=1e-5).log() - torch.log(torch.tensor(1e-5)))

        # --- 3. 직접 구현  ---
        
        # 차원 변환
        transposed_spectrograms = log_mel_spectrograms.transpose(1, 2)  # [B, T, F]
        
        # --- 4. Direct 방식 사용 (gradient flow 유지) ---
        # Processor 출력 형태를 따라함: [B, 1, T, F] = [8, 1, 1001, 64]
        # Direct: [B, T, F] -> [B, 1, T, F]
        input_spectrograms = transposed_spectrograms.unsqueeze(1)  # [B, 1, T, F]
        # CLAP 모델 forward
        audio_embeds = self.clap_model.audio_model(input_spectrograms).pooler_output
        final_audio_embeds = self.clap_model.audio_projection(audio_embeds) # shape: [B, 512]
        return final_audio_embeds

    def compute_clap_loss(self, predicted_audios: torch.Tensor, text_prompts: List[str]) -> torch.Tensor:
        """
        학생 모델의 출력(predicted_audios)을 받아 CLAP Loss를 계산합니다.
        """
        # 1. 그래디언트가 흐르는 오디오 임베딩 추출
        audio_embeds = self.get_audio_embedding_with_grad(predicted_audios)

        # 2. 그래디언트가 필요 없는 텍스트 임베딩 추출
        text_embeds = self.get_text_embedding(text_prompts)

        # 3. Contrastive Loss (InfoNCE) 계산
        audio_embeds_norm = F.normalize(audio_embeds, p=2, dim=-1)
        text_embeds_norm = F.normalize(text_embeds, p=2, dim=-1)
        
        logits = (audio_embeds_norm @ text_embeds_norm.t()) / 0.07
        device = audio_embeds.device
        labels = torch.arange(logits.size(0), device=device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            
        return loss
            
    def forward(self, text_prompts: Union[str, List[str]]):
        """기본 forward는 텍스트 인코딩을 수행"""
        return self.get_text_embedding(text_prompts)


class SentenceTransformerEncoder(BaseTextEncoder):
    """
    SentenceTransformer-based text encoder
    
    Good general-purpose option with many pre-trained models available.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__()
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        self.model_name = model_name
        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        cache_folder = os.getenv('SENTENCE_TRANSFORMERS_HOME')
        if cache_folder:
            os.makedirs(cache_folder, exist_ok=True)
        try:
            if hf_token:
                self.model = SentenceTransformer(model_name, use_auth_token=hf_token, cache_folder=cache_folder)
            else:
                self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        except Exception as e:
            print("⚠️ SentenceTransformer 로드 실패, 토큰 없이 재시도합니다.")
            self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode_text(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text prompts to embeddings"""
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        # 안전성 검사 및 정리
        safe_prompts = []
        for prompt in text_prompts:
            if isinstance(prompt, (tuple, list)):
                # 튜플/리스트인 경우 첫 번째 요소 사용
                prompt = str(prompt[0]) if len(prompt) > 0 else "Apply audio effect"
            elif not isinstance(prompt, str):
                prompt = str(prompt)
            
            # 빈 문자열 처리
            prompt = prompt.strip()
            if not prompt:
                prompt = "Apply audio effect"
            
            safe_prompts.append(prompt)
        
        try:
            # sentence-transformer는 기본적으로 inference mode에서 실행됨
            # 따라서 no_grad() 컨텍스트에서 실행하고, 나중에 gradient가 필요한 곳에서 처리
            with torch.no_grad():
                embeddings = self.model.encode(safe_prompts, convert_to_tensor=True, show_progress_bar=False)
            
            # inference mode 밖에서 새로운 텐서 생성 (gradient 가능)
            # 이렇게 하면 원본 텐서의 값은 복사하되 gradient graph는 새로 시작됨
            embeddings = embeddings.clone().detach()
            embeddings.requires_grad_(True)
            
        except Exception as e:
            print(f"❌ sentence-transformers 인코딩 실패: {e}")
            print(f"   입력 프롬프트: {safe_prompts}")
            print(f"   프롬프트 타입: {[type(p) for p in safe_prompts]}")
            raise
        
        return embeddings.float()
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


## Removed: HuggingFaceTextEncoder (not used)


class E5TextEncoder(BaseTextEncoder):
    """
    최적화된 E5 Text Encoder - 배치 처리 효율성 극대화
    """
    
    def __init__(self, model_name='intfloat/e5-large-v2', device='cuda'):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        self.device = device
        self.model_name = model_name
        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        local_only = os.getenv('HF_LOCAL_ONLY', '0') in ('1', 'true', 'True') or os.getenv('TRANSFORMERS_OFFLINE', '0') in ('1', 'true', 'True')
        if local_only:
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        try:
            if hf_token:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token, local_files_only=local_only)
                self.model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token, local_files_only=local_only).to(device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
                self.model = AutoModel.from_pretrained(model_name, local_files_only=local_only).to(device)
        except Exception:
            print("⚠️ E5 토크나이저/모델 로드 실패, 토큰 없이 재시도합니다.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=local_only).to(device)
        
        # 메모리 최적화: 모델 파라미터 동결
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 최적 배치 크기 (하드웨어별 조정 필요)
        self.optimal_batch_size = self._find_optimal_batch_size()
    
    def encode_text(self, text_prompts: Union[str, List[str]], 
                   batch_size: Optional[int] = None) -> torch.Tensor:
        """통일된 인터페이스: 텍스트를 임베딩으로 인코딩"""
        return self.forward(text_prompts, batch_size)
    
    def get_embedding_dim(self) -> int:
        """임베딩 차원 반환"""
        return self.model.config.hidden_size
    
    def _find_optimal_batch_size(self) -> int:
        """하드웨어에 맞는 최적 배치 크기 찾기"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # GPU 메모리에 따른 권장 배치 크기
            if gpu_memory > 20e9:  # 20GB+
                return 128
            elif gpu_memory > 12e9:  # 12GB+
                return 64
            elif gpu_memory > 8e9:   # 8GB+
                return 32
            else:
                return 16
        return 8  # CPU
    
    def _sort_by_length(self, texts: List[str]) -> tuple:
        """길이별로 텍스트 정렬 (패딩 최소화)"""
        indexed_texts = [(i, text, len(text.split())) for i, text in enumerate(texts)]
        indexed_texts.sort(key=lambda x: x[2])
        
        sorted_indices = [x[0] for x in indexed_texts]
        sorted_texts = [x[1] for x in indexed_texts]
        reverse_indices = [0] * len(texts)
        
        for new_idx, orig_idx in enumerate(sorted_indices):
            reverse_indices[orig_idx] = new_idx
            
        return sorted_texts, reverse_indices
    
    def _dynamic_batching(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[str]]:
        """동적 배치 생성 - 길이가 비슷한 텍스트끼리 그룹화"""
        if batch_size is None:
            batch_size = self.optimal_batch_size
        
        # 길이별 정렬
        sorted_texts, _ = self._sort_by_length(texts)
        
        batches = []
        for i in range(0, len(sorted_texts), batch_size):
            batch = sorted_texts[i:i + batch_size]
            batches.append(batch)
        
        return batches, _
    
    def _encode_batch_optimized(self, texts: List[str]) -> torch.Tensor:
        """최적화된 배치 인코딩"""
        # E5 prefix 추가
        prefixed_texts = [f"query: {text}" for text in texts]
        
        # 적응적 토큰화: 배치 내 최대 길이만큼만 패딩
        encoded = self.tokenizer(
            prefixed_texts,
            padding='longest',  # 배치 내 최대 길이로만 패딩
            truncation=True,
            return_tensors='pt',
            max_length=512,
            return_attention_mask=True
        )
        
        # GPU로 효율적 이동
        encoded = {k: v.to(self.device, non_blocking=True) 
                  for k, v in encoded.items()}
        
        # 추론 모드로 빠른 처리
        with torch.no_grad(), torch.cuda.amp.autocast():  # Mixed precision
            outputs = self.model(**encoded)
            
            # Mean pooling (최적화된 버전)
            embeddings = self._fast_mean_pooling(
                outputs.last_hidden_state, 
                encoded['attention_mask']
            )
            
            # L2 정규화
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.float()
    
    def _fast_mean_pooling(self, token_embeddings: torch.Tensor, 
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """최적화된 mean pooling"""
        # Attention mask를 float로 변환하고 차원 확장
        mask_expanded = attention_mask.unsqueeze(-1).float()
        
        # 마스킹된 토큰 임베딩 합계
        masked_embeddings = token_embeddings * mask_expanded
        sum_embeddings = masked_embeddings.sum(dim=1)
        
        # 실제 토큰 수로 나누기 (패딩 제외)
        sum_mask = mask_expanded.sum(dim=1)
        mean_embeddings = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        
        return mean_embeddings
    
    def forward(self, text_prompts: Union[str, List[str]], 
                batch_size: Optional[int] = None) -> torch.Tensor:
        """최적화된 포워드 패스"""
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        if batch_size is None:
            batch_size = min(self.optimal_batch_size, len(text_prompts))
        
        # 단일 배치면 바로 처리
        if len(text_prompts) <= batch_size:
            return self._encode_batch_optimized(text_prompts)
        
        # 대용량 입력: 동적 배치 처리
        batches, reverse_indices = self._dynamic_batching(text_prompts, batch_size)
        
        embeddings_list = []
        for batch in batches:
            batch_embeddings = self._encode_batch_optimized(batch)
            embeddings_list.append(batch_embeddings)
        
        # 모든 배치 결과 합치기
        all_embeddings = torch.cat(embeddings_list, dim=0)
        
        # 원래 순서로 복구
        restored_embeddings = torch.zeros_like(all_embeddings)
        for i, orig_idx in enumerate(reverse_indices):
            restored_embeddings[orig_idx] = all_embeddings[i]
        
        return restored_embeddings
    
    @property
    def embedding_dim(self):
        return self.model.config.hidden_size



## Removed: BGETextEncoder (not used)



## Removed: SimpleTextEncoder (not used)
    """
    Simple text encoder using basic word embeddings
    
    Fallback option when other libraries are not available.
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=512, max_length=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Simple word-to-index mapping (in practice, use a proper tokenizer)
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.next_idx = 1  # 0 reserved for padding
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_encoding = nn.Embedding(max_length, embedding_dim)
        
        # Simple transformer-like processing
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def tokenize(self, text: str) -> List[int]:
        """Simple tokenization"""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            if word not in self.word_to_idx:
                if self.next_idx < self.vocab_size:
                    self.word_to_idx[word] = self.next_idx
                    self.idx_to_word[self.next_idx] = word
                    self.next_idx += 1
                else:
                    word = '<UNK>'  # Unknown token
                    if '<UNK>' not in self.word_to_idx:
                        self.word_to_idx['<UNK>'] = self.vocab_size - 1
            
            tokens.append(self.word_to_idx.get(word, self.vocab_size - 1))
        
        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([0] * (self.max_length - len(tokens)))
        
        return tokens
    
    def forward(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text prompts"""
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        # Tokenize all texts
        batch_tokens = []
        for text in text_prompts:
            tokens = self.tokenize(text)
            batch_tokens.append(tokens)
        
        tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long)
        batch_size, seq_len = tokens_tensor.shape
        
        # Move to same device as model
        device = next(self.parameters()).device
        tokens_tensor = tokens_tensor.to(device)
        
        # Get embeddings
        word_embeddings = self.embedding(tokens_tensor)
        
        # Add position encodings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.position_encoding(positions)
        
        embeddings = word_embeddings + pos_embeddings
        
        # Apply attention
        attended, _ = self.attention(embeddings, embeddings, embeddings)
        attended = self.norm(attended + embeddings)
        
        # Global pooling to get sentence embedding
        sentence_embeddings = self.global_pool(attended.transpose(1, 2)).squeeze(-1)
        
        return sentence_embeddings
    
    def get_embedding_dim(self):
        """Get embedding dimension"""
        return self.embedding_dim


class TextEncoderFactory:
    """Factory to create text encoders: E5, SentenceTransformer, or CLAP only"""

    @staticmethod
    def create_encoder(encoder_type='auto', **kwargs):
        if encoder_type == 'auto':
            if TRANSFORMERS_AVAILABLE:
                print("Using E5-large encoder (open source, strong performance)")
                return E5TextEncoder(model_name='intfloat/e5-large-v2', **kwargs)
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                print("Using SentenceTransformer encoder")
                return SentenceTransformerEncoder(model_name='all-mpnet-base-v2', **kwargs)
            if CLAP_AVAILABLE:
                print("Using CLAP text encoder (audio-text specialized)")
                return CLAPTextEncoder(**kwargs)
            raise ImportError("No supported text encoder available. Install transformers, sentence-transformers or laion_clap.")

        if encoder_type == 'e5':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Install with: pip install transformers")
            return E5TextEncoder(**kwargs)

        if encoder_type == 'sentence_transformer':
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("SentenceTransformers not available. Install with: pip install sentence-transformers")
            return SentenceTransformerEncoder(**kwargs)

        if encoder_type == 'clap':
            if not CLAP_AVAILABLE:
                raise ImportError("CLAP not available. Install with: pip install laion_clap")
            return CLAPTextEncoder(**kwargs)

        raise ValueError(f"Unknown encoder type: {encoder_type}")


# Example usage and recommendations
def recommend_text_encoder():
    """Provide recommendations for text encoders (simplified)"""
    recommendations = {
        "🚀 High Performance (Recommended)": {
            "encoder": "E5",
            "model": "intfloat/e5-large-v2",
            "install": "pip install transformers",
        },
        "📝 General Purpose": {
            "encoder": "SentenceTransformers",
            "model": "all-mpnet-base-v2",
            "install": "pip install sentence-transformers",
        },
        "🎵 Audio-Text Specialized": {
            "encoder": "CLAP (Hugging Face)",
            "model": "laion/larger_clap_music_and_speech",
            "install": "pip install transformers",
        },
    }
    print("📋 TEXT ENCODER RECOMMENDATIONS")
    print("=" * 50)
    for use_case, info in recommendations.items():
        print(f"\n{use_case}")
        print(f"  Model: {info['encoder']} - {info['model']}")
        print(f"  Install: {info['install']}")


if __name__ == "__main__":
    print("🔤 Text Encoder for Audio Processing")
    print("=" * 50)
    recommend_text_encoder()
    print("\n🧪 Testing Available Encoders...")
    test_prompt = "deep monster voice with heavy reverb"
    try:
        encoder = TextEncoderFactory.create_encoder('auto')
        embeddings = encoder([test_prompt])
        print(f"✅ Created encoder: {type(encoder).__name__}")
        if hasattr(encoder, 'get_embedding_dim'):
            print(f"   Embedding dim: {encoder.get_embedding_dim()}")
    except Exception as e:
        print(f"❌ Error creating encoder: {e}")


def get_text_encoder(encoder_type: str = 'e5-large', **kwargs):
    """Factory function to get text encoder (E5, SentenceTransformer, or CLAP)"""
    encoder_mapping = {
        'e5-large': 'e5',
        'clap': 'clap',
        'sentence-transformer': 'sentence_transformer',
        'auto': 'auto',
    }
    factory_type = encoder_mapping.get(encoder_type, encoder_type)
    encoder = TextEncoderFactory.create_encoder(factory_type, **kwargs)
    if not hasattr(encoder, 'encode'):
        def encode_method(texts):
            if isinstance(texts, str):
                texts = [texts]
            return encoder(texts)
        encoder.encode = encode_method
    return encoder
