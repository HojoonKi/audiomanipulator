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
    import laion_clap
    CLAP_AVAILABLE = True
except Exception as e:
    CLAP_AVAILABLE = False
    warnings.warn(f"laion_clap not available or failed to import ({e}). Install with: pip install laion_clap")

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
    CLAP-based text encoder - Best for audio-text tasks
    
    CLAP (Contrastive Language-Audio Pre-training) is specifically designed
    for audio-text alignment, making it ideal for our use case.
    
    Implements singleton-like behavior to avoid duplicate weight loading.
    """
    
    _instances = {}  # 모델별 인스턴스 캐시
    
    def __new__(cls, model_name='630k-audioset-best', freeze_audio_branch=False):
        # 동일한 모델명으로 이미 생성된 인스턴스가 있으면 재사용
        if model_name in cls._instances:
            print(f"🔄 기존 CLAP 인스턴스 재사용: {model_name}")
            return cls._instances[model_name]
        
        # 새 인스턴스 생성
        instance = super().__new__(cls)
        cls._instances[model_name] = instance
        return instance
    
    def __init__(self, model_name='630k-audioset-best', freeze_audio_branch=False, quiet: Optional[bool] = None):
        # 이미 초기화된 인스턴스는 스킵
        if hasattr(self, '_initialized'):
            return
        
        super().__init__()
        
        if not CLAP_AVAILABLE:
            raise ImportError("laion_clap not installed. Run: pip install laion_clap")
        
        self.model_name = model_name
        # 조용한 모드 설정 (환경변수 CLAP_VERBOSE=0/false 로 제어)
        if quiet is None:
            quiet_env = os.getenv('CLAP_VERBOSE', '1')
            self._quiet = str(quiet_env).lower() in ('0', 'false', 'no')
        else:
            self._quiet = bool(quiet)

        def _log(*args, **kwargs):
            if not self._quiet:
                print(*args, **kwargs)
        self._log = _log

        # 모듈 생성 시 출력 억제
        with _suppress_output(self._quiet):
            self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        
        # 다운로드/오프라인 제어 및 캐시 확인
        skip_download = os.getenv('CLAP_SKIP_DOWNLOAD', '0') in ('1', 'true', 'True')
        ckpt_path = os.getenv('CLAP_CKPT_PATH')
        ckpt_file = os.getenv('CLAP_CKPT_FILE')  # 명시적 파일 경로(최우선)
        loaded = False
        
        def list_candidate_cache_dirs():
            custom_cache = os.getenv('CLAP_CACHE_DIR')
            candidates = [
                custom_cache if custom_cache else None,
                os.path.expanduser('~/.cache/laion_clap'),
                os.path.expanduser('~/.cache/clip'),
                '/tmp/laion_clap_cache',
                './clap_cache'
            ]
            return [d for d in candidates if d]

        def find_cached_ckpt_in_dir(cache_dir: str):
            try:
                if not os.path.isdir(cache_dir):
                    return None
                hints = ('clap', '630k', 'audioset', 'music_audioset', 'esc')
                files = [
                    f for f in os.listdir(cache_dir)
                    if f.lower().endswith(('.pt', '.pth', '.bin')) and any(h in f.lower() for h in hints)
                ]
                if not files:
                    return None
                # 모델명과 유사도가 높은 파일 우선
                def score(name: str) -> int:
                    name_l = name.lower()
                    s = 0
                    if 'clap' in name_l:
                        s += 2
                    if '630k' in name_l or 'audioset' in name_l:
                        s += 2
                    if self.model_name.replace('-', '') in name_l.replace('-', ''):
                        s += 1
                    return s
                files.sort(key=score, reverse=True)
                return os.path.join(cache_dir, files[0])
            except Exception:
                return None

        def find_any_cached_ckpt():
            for d in list_candidate_cache_dirs():
                path = find_cached_ckpt_in_dir(d)
                if path:
                    return path
            return None
        
        # 0) 명시적 파일 경로가 지정된 경우 최우선 사용
        if ckpt_file and os.path.isfile(ckpt_file):
            try:
                with _suppress_output(self._quiet):
                    self.clap_model.load_ckpt(ckpt_file)
                loaded = True
                self._log(f"📦 CLAP 체크포인트 사용(명시 파일): {ckpt_file}")
            except Exception as e:
                print(f"⚠️ 명시 파일 로드 실패: {e}")

        # 1) 명시적 로컬 경로 우선 (파일 또는 디렉토리 모두 허용)
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                target = ckpt_path
                if os.path.isdir(ckpt_path):
                    candidate = find_cached_ckpt_in_dir(ckpt_path)
                    if candidate:
                        target = candidate
                with _suppress_output(self._quiet):
                    self.clap_model.load_ckpt(target)
                loaded = True
                self._log(f"📦 CLAP 로컬 체크포인트 사용: {target}")
            except Exception as e:
                print(f"⚠️ 로컬 CLAP 체크포인트 로드 실패: {e}")
        
        # 2) 캐시된 모델 확인 (여러 경로에서 탐색)
        if not loaded:
            cached = find_any_cached_ckpt()
            if cached:
                try:
                    with _suppress_output(self._quiet):
                        self.clap_model.load_ckpt(cached)
                    loaded = True
                    self._log(f"📦 CLAP 캐시된 체크포인트 사용: {cached}")
                except Exception:
                    pass
        
        # 2.5) 패키지 내 기본 체크포인트 경로 시도 (모든 랭크에서 동일 절대경로)
        if not loaded:
            try:
                pkg_dir = os.path.dirname(laion_clap.__file__)
                built_in = os.path.join(pkg_dir, '630k-audioset-best.pt')
                if os.path.isfile(built_in):
                    with _suppress_output(self._quiet):
                        self.clap_model.load_ckpt(built_in)
                    loaded = True
                    self._log(f"📦 CLAP 패키지 내 체크포인트 사용: {built_in}")
            except Exception:
                pass
        
        # 3) 원격 다운로드 허용 시에만 시도 (DDP 환경이면 rank 0 전용)
        if not loaded and not skip_download:
            is_ddp = DIST_AVAILABLE and dist.is_available() and dist.is_initialized()
            is_rank0 = False
            if is_ddp:
                try:
                    is_rank0 = (dist.get_rank() == 0)
                except Exception:
                    is_rank0 = False

            if not is_ddp or is_rank0:
                self._log("🌐 CLAP weight 다운로드 시도 중...")
                try:
                    with _suppress_output(self._quiet):
                        self.clap_model.load_ckpt()
                    loaded = True
                    self._log("✅ CLAP weight 다운로드 완료")
                except Exception as e:
                    print(f"⚠️ 기본 CLAP 체크포인트 로드 실패: {e}")
                    for alt in ('630k-best', 'music_audioset_epoch_15_esc_90.14.pt'):
                        try:
                            self._log(f"🔄 대안 체크포인트 시도: {alt}")
                            with _suppress_output(self._quiet):
                                self.clap_model.load_ckpt(alt)
                            loaded = True
                            self._log(f"✅ 대안 CLAP weight 로드 성공: {alt}")
                            break
                        except Exception as e2:
                            print(f"⚠️ {alt} 로드 실패: {e2}")
                            continue
                # 다운로드 후 캐시 경로 안내
                if loaded:
                    cached = find_any_cached_ckpt()
                    if cached:
                        self._log(f"📁 CLAP 가중치 캐시 경로 탐지: {cached}")
                        self._log(f"   다음 실행부터 캐시 재사용을 강제하려면: export CLAP_CKPT_FILE={cached}")
            else:
                self._log("⏳ Rank≠0: CLAP 가중치 캐시 대기 중...")
                wait_secs = int(os.getenv('CLAP_DDP_WAIT_SECS', '300'))
                start = time.time()
                while time.time() - start < wait_secs:
                    cached = find_any_cached_ckpt()
                    if cached:
                        try:
                            with _suppress_output(self._quiet):
                                self.clap_model.load_ckpt(cached)
                            loaded = True
                            self._log(f"📦 Rank≠0: 캐시 체크포인트 사용: {cached}")
                            break
                        except Exception:
                            pass
                    time.sleep(1.0)
                try:
                    dist.barrier()
                except Exception:
                    pass
        
        if not loaded:
            print("⚠️ CLAP 체크포인트 로드 생략 - 로컬 캐시 없음 또는 다운로드 비활성화")
            print("   환경변수 설정 예시:")
            print("   export CLAP_CKPT_PATH=/path/to/your/clap_model.pt")
            print("   export CLAP_SKIP_DOWNLOAD=0  # 다운로드 허용")
        
        # CLAP 모델 구조 확인 및 gradient 설정
        if not self._quiet:
            print(f"🔍 CLAP 모델 구조 확인:")
            for name, module in self.clap_model.named_children():
                print(f"   - {name}: {type(module)}")
        
        # CLAP 모델은 frozen하되 gradient computation은 허용
        for param in self.clap_model.parameters():
            param.requires_grad = False
        if not self._quiet:
            print("🔒 CLAP 모델 frozen (embedding 추출용, gradient flow는 허용)")
        
        # 학습 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in self.clap_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.clap_model.parameters())
        if not self._quiet:
            print(f"📊 CLAP 파라미터: {total_params:,} total, {trainable_params:,} trainable")
        
        # 초기화 완료 표시
        self._initialized = True
        if not self._quiet:
            print(f"✅ CLAP 인코더 초기화 완료: {model_name}")
    
    def get_text_embedding(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """
        Get text embeddings (frozen)
        
        Args:
            text_prompts: Single string or list of strings
            
        Returns:
            embeddings: (batch_size, embedding_dim) tensor
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        # Get CLAP text embeddings (always no_grad for text)
        with torch.no_grad():
            text_embeddings = self.clap_model.get_text_embedding(text_prompts)
        
        return torch.from_numpy(text_embeddings).float()
    
    def get_audio_embedding_from_data(self, audio_data: Union[np.ndarray, torch.Tensor], use_tensor=True) -> torch.Tensor:
        """
        Get audio embeddings from CLAP model (frozen, embedding only)
        
        Args:
            audio_data: Audio waveform data (batch_size, audio_length) or (audio_length,)
            use_tensor: Whether to return tensor (for consistency)
            
        Returns:
            embeddings: (batch_size, embedding_dim) tensor 
        """
        try:
            # Convert to numpy for CLAP (always expects numpy)
            if isinstance(audio_data, torch.Tensor):
                audio_np = audio_data.detach().cpu().numpy()
                device = audio_data.device
            else:
                audio_np = audio_data
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Ensure batch dimension
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]
                
            # Get CLAP embeddings (frozen model, no gradient)
            with torch.no_grad():
                embeddings = self.clap_model.get_audio_embedding_from_data(x=audio_np, use_tensor=False)
            
            # Convert to tensor
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings).float()
            
            embeddings = embeddings.to(device)
            
            return embeddings
            
        except Exception as e:
            print(f"❌ CLAP audio embedding 실패: {e}")
            
            # Safe fallback
            device = audio_data.device if hasattr(audio_data, 'device') else torch.device('cpu')
            batch_size = 1
            if hasattr(audio_data, 'shape') and len(audio_data.shape) > 1:
                batch_size = audio_data.shape[0]
                
            return torch.zeros(batch_size, 512, device=device)
    
    def compute_similarity(self, audio_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute simple cosine similarity between audio and text embeddings
        
        Args:
            audio_embeddings: Audio embeddings [batch, dim]
            text_embeddings: Text embeddings [batch, dim]
            
        Returns:
            similarities: Cosine similarities [-1, 1]
        """
        # Normalize embeddings
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(audio_embeddings, text_embeddings, dim=-1)
        
        return cosine_sim
    
    def compute_clap_loss(self, audio_data: Union[np.ndarray, torch.Tensor], text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """
        Compute simple CLAP loss - embedding 비교만!
        
        Args:
            audio_data: Audio waveform data
            text_prompts: Text descriptions
            
        Returns:
            loss: Simple contrastive loss (1 - cosine_similarity)
        """
        try:
            # Handle input formats - 배치 처리 개선
            if isinstance(text_prompts, str):
                text_prompts = [text_prompts]
            # List는 그대로 사용 (배치 처리)
                
            # Convert to tensor if needed
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data).float()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                audio_tensor = audio_tensor.to(device)
            else:
                audio_tensor = audio_data
                device = audio_tensor.device
            
            # Ensure batch dimension
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3:  # [batch, channels, samples]
                audio_tensor = audio_tensor.squeeze(1)  # Remove channel dim for mono
                
            # Get embeddings (both frozen)
            text_embeddings = self.get_text_embedding(text_prompts).to(device)
            audio_embeddings = self.get_audio_embedding_from_data(audio_tensor)
            
            # Ensure same batch size
            if audio_embeddings.shape[0] != text_embeddings.shape[0]:
                if text_embeddings.shape[0] == 1:
                    text_embeddings = text_embeddings.expand(audio_embeddings.shape[0], -1)
                elif audio_embeddings.shape[0] == 1:
                    audio_embeddings = audio_embeddings.expand(text_embeddings.shape[0], -1)
            
            # Simple cosine similarity
            similarities = self.compute_similarity(audio_embeddings, text_embeddings)
            
            # 더 직관적인 loss: similarity를 [0, 1] 범위로 정규화 후 1에서 빼기
            # similarity: [-1, 1] → normalized: [0, 1] → loss: [0, 1] 
            normalized_sim = (similarities.mean() + 1.0) / 2.0  # [-1,1] → [0,1]
            loss = 1.0 - normalized_sim  # [0,1] → [1,0]
            
            # Ensure gradient flow (loss should require grad even if embeddings don't)
            if not loss.requires_grad:
                loss = loss.clone().requires_grad_(True)
            
            return loss
            
        except Exception as e:
            print(f"❌ CLAP loss 실패: {e}")
            device = audio_data.device if hasattr(audio_data, 'device') else torch.device('cpu')
            return torch.tensor(1.0, device=device, requires_grad=True)
    
    def forward(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """Forward pass for text encoding (backward compatibility)"""
        return self.get_text_embedding(text_prompts)
    
    def get_embedding_dim(self):
        """Get the dimension of text embeddings"""
        return 512  # CLAP text embedding dimension


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
            with torch.no_grad():
                embeddings = self.model.encode(safe_prompts, convert_to_tensor=True, show_progress_bar=False)
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
            "encoder": "CLAP",
            "model": "630k-audioset-best",
            "install": "pip install laion_clap",
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
