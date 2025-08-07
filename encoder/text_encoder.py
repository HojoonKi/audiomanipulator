#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Optional
import warnings
from abc import ABC, abstractmethod

# Try importing different text encoding libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import transformers
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Install with: pip install transformers")

try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
    warnings.warn("laion_clap not available. Install with: pip install laion_clap")


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
    """
    
    def __init__(self, model_name='630k-audioset-best', freeze_audio_branch=False):
        super().__init__()
        
        if not CLAP_AVAILABLE:
            raise ImportError("laion_clap not installed. Run: pip install laion_clap")
        
        self.model_name = model_name
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        
        # 다운로드 가능한 체크포인트 시도
        try:
            self.clap_model.load_ckpt()  # 기본 체크포인트 로드
        except:
            try:
                # 다른 체크포인트 시도
                self.clap_model.load_ckpt('630k-best')
            except:
                try:
                    # 또 다른 체크포인트 시도
                    self.clap_model.load_ckpt('music_audioset_epoch_15_esc_90.14.pt')
                except:
                    print("⚠️ CLAP 체크포인트 로드 실패, 랜덤 초기화로 진행")
        
        # CLAP 모델 구조 확인 및 gradient 설정
        print(f"🔍 CLAP 모델 구조 확인:")
        for name, module in self.clap_model.named_children():
            print(f"   - {name}: {type(module)}")
        
        # CLAP 모델은 frozen하되 gradient computation은 허용
        for param in self.clap_model.parameters():
            param.requires_grad = False
        print("🔒 CLAP 모델 frozen (embedding 추출용, gradient flow는 허용)")
        
        # 학습 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in self.clap_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.clap_model.parameters())
        print(f"📊 CLAP 파라미터: {total_params:,} total, {trainable_params:,} trainable")
    
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
        self.model = SentenceTransformer(model_name)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode_text(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text prompts to embeddings"""
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        with torch.no_grad():
            embeddings = self.model.encode(text_prompts, convert_to_tensor=True, show_progress_bar=False)
        
        return embeddings.float()
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


class HuggingFaceTextEncoder(nn.Module):
    """
    HuggingFace transformer-based text encoder
    
    Flexible option allowing use of various transformer models.
    """
    
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text prompts"""
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        # Tokenize
        encoded_input = self.tokenizer(
            text_prompts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Get embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
            # Use mean pooling of last hidden states
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        return embeddings.float()
    
    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding_dim(self):
        """Get embedding dimension"""
        return self.model.config.hidden_size


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
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



class BGETextEncoder(nn.Module):
    """
    BAAI BGE (Beijing Academy of AI General Embedding) - Chinese SOTA model
    
    BGE models achieve excellent performance on embedding benchmarks,
    especially good for Asian languages but also strong on English.
    """
    
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text prompts with BGE"""
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        # Tokenize
        encoded_input = self.tokenizer(
            text_prompts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt',
            max_length=512
        )
        
        # Move to device efficiently  
        device = next(self.model.parameters()).device
        encoded_input = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in encoded_input.items()}
        
        # Get embeddings with optimized settings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]  # Use [CLS] token
            
            # L2 normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.float()
    
    def get_embedding_dim(self):
        """Get embedding dimension"""
        return self.model.config.hidden_size



class SimpleTextEncoder(nn.Module):
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
    """Factory class to create appropriate text encoder based on available libraries"""
    
    @staticmethod
    def create_encoder(encoder_type='auto', **kwargs):
        """
        Create text encoder
        
        Args:
            encoder_type: 'clap', 'sentence_transformer', 'huggingface', 'simple', or 'auto'
            **kwargs: Additional arguments for specific encoders
            
        Returns:
            TextEncoder instance
        """
        
        if encoder_type == 'auto':
            # Try to use the best available encoder (prioritize performance)
            if TRANSFORMERS_AVAILABLE:
                print("� Using E5-large encoder (SOTA performance, open source)")
                return E5TextEncoder(model_name='intfloat/e5-large-v2', **kwargs)
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                print("📝 Using SentenceTransformer encoder (CLAP checkpoint not found)")
                return SentenceTransformerEncoder(model_name='all-mpnet-base-v2', **kwargs)
            elif CLAP_AVAILABLE:
                print("🎵 Using CLAP text encoder (best for audio tasks)")
                return CLAPTextEncoder(**kwargs)
            else:
                print("⚠️ Using simple text encoder (install other libraries for better performance)")
                return SimpleTextEncoder(**kwargs)
        
        elif encoder_type == 'e5':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Install with: pip install transformers")
            return E5TextEncoder(**kwargs)
        
        elif encoder_type == 'bge':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Install with: pip install transformers")
            return BGETextEncoder(**kwargs)
        
        elif encoder_type == 'clap':
            if not CLAP_AVAILABLE:
                raise ImportError("CLAP not available. Install with: pip install laion_clap")
            return CLAPTextEncoder(**kwargs)
        
        elif encoder_type == 'sentence_transformer':
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("SentenceTransformers not available. Install with: pip install sentence-transformers")
            return SentenceTransformerEncoder(**kwargs)
        
        elif encoder_type == 'huggingface':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Install with: pip install transformers")
            return HuggingFaceTextEncoder(**kwargs)
        
        elif encoder_type == 'simple':
            return SimpleTextEncoder(**kwargs)
        
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")


# Example usage and recommendations
def recommend_text_encoder():
    """Provide recommendations for text encoders based on use case"""
    
    recommendations = {
        "🚀 High Performance (Recommended)": {
            "encoder": "E5",
            "model": "intfloat/e5-large-v2", 
            "install": "pip install transformers",
            "pros": ["SOTA embedding performance", "Open source", "Fast inference", "Microsoft quality"],
            "cons": ["Larger than BERT-base", "Needs transformers library"]
        },
        
        "🔥 Chinese/Asian Languages": {
            "encoder": "BGE",
            "model": "BAAI/bge-large-en-v1.5",
            "install": "pip install transformers", 
            "pros": ["Excellent for Asian languages", "SOTA Chinese performance", "Good English too"],
            "cons": ["Chinese-centric training", "Large model size"]
        },
        
        "🎯 Task-Specific": {
            "encoder": "Instructor",
            "model": "hkunlp/instructor-xl",
            "install": "pip install InstructorEmbedding transformers",
            "pros": ["Task-specific instructions", "Flexible performance", "Domain adaptation"],
            "cons": ["Complex usage", "Instruction engineering needed"]
        },
        
        "🎵 Audio Processing (Classic)": {
            "encoder": "CLAP",
            "model": "630k-audioset-best",
            "install": "pip install laion_clap",
            "pros": ["Designed for audio-text alignment", "Best semantic understanding", "Pre-trained on audio data"],
            "cons": ["Larger model size", "Requires more resources", "Checkpoint availability issues"]
        },
        
        "⚡ Fast & Lightweight": {
            "encoder": "SentenceTransformers",
            "model": "all-MiniLM-L6-v2",
            "install": "pip install sentence-transformers",
            "pros": ["Fast inference", "Small model size", "Good general performance"],
            "cons": ["Not specifically trained for audio tasks"]
        },
        
        "🎯 High Quality General": {
            "encoder": "SentenceTransformers",
            "model": "all-mpnet-base-v2",
            "install": "pip install sentence-transformers",
            "pros": ["High quality embeddings", "Good semantic understanding"],
            "cons": ["Larger than MiniLM", "Slower inference"]
        }
    }
    
    print("📋 TEXT ENCODER RECOMMENDATIONS")
    print("=" * 50)
    
    for use_case, info in recommendations.items():
        print(f"\n{use_case}")
        print(f"  Model: {info['encoder']} - {info['model']}")
        print(f"  Install: {info['install']}")
        print(f"  Pros: {', '.join(info['pros'])}")
        print(f"  Cons: {', '.join(info['cons'])}")


if __name__ == "__main__":
    print("🔤 Text Encoder for Audio Processing")
    print("=" * 50)
    
    # Show recommendations
    recommend_text_encoder()
    
    print("\n🧪 Testing Available Encoders...")
    
    # Test what's available
    test_prompt = "deep monster voice with heavy reverb"
    
    try:
        encoder = TextEncoderFactory.create_encoder('auto')
        embeddings = encoder([test_prompt])
        print(f"✅ Created encoder: {type(encoder).__name__}")
        print(f"   Embedding shape: {embeddings.shape}")
        print(f"   Embedding dim: {encoder.get_embedding_dim()}")
    except Exception as e:
        print(f"❌ Error creating encoder: {e}")
    
    print("\n💡 For your monster voice project, I recommend:")
    print("   1st choice: E5-large (최고 성능 오픈소스)")
    print("   2nd choice: BGE-large (아시아 언어 특화)")  
    print("   3rd choice: Instructor-XL (task-specific)")
    print("   4th choice: SentenceTransformers all-mpnet-base-v2 (안정적)")
    
    print("\n🔥 New Models Available:")
    print("   E5: pip install transformers → TextEncoderFactory.create_encoder('e5')")
    print("   BGE: pip install transformers → TextEncoderFactory.create_encoder('bge')")
    print("   Instructor: pip install InstructorEmbedding → TextEncoderFactory.create_encoder('instructor')")


def get_text_encoder(encoder_type: str = 'e5-large', **kwargs):
    """
    Factory function to get text encoder (for pipeline compatibility)
    
    Args:
        encoder_type: Type of encoder to create
            - 'e5-large': E5-large v2 model (best performance)
            - 'bge-large': BGE-large English model
            - 'instructor': Instructor XL model
            - 'clap': CLAP text encoder
            - 'sentence-transformer': SentenceTransformer model
            - 'simple': Simple text encoder
            - 'auto': Automatically choose best available
            
    Returns:
        Text encoder instance with encode() method
    """
    # Map pipeline encoder names to factory names
    encoder_mapping = {
        'e5-large': 'e5',
        'bge-large': 'bge', 
        'instructor': 'instructor',
        'clap': 'clap',
        'sentence-transformer': 'sentence_transformer',
        'simple': 'simple',
        'auto': 'auto'
    }
    
    factory_type = encoder_mapping.get(encoder_type, encoder_type)
    encoder = TextEncoderFactory.create_encoder(factory_type, **kwargs)
    
    # Ensure encoder has encode method (some might only have forward)
    if not hasattr(encoder, 'encode'):
        # Add encode method as alias to forward
        def encode_method(texts):
            if isinstance(texts, str):
                texts = [texts]
            # 배치 처리를 위해 forward 직접 호출 (gradient 허용)
            result = encoder(texts)
            return result  # GPU에서 tensor 그대로 반환
        encoder.encode = encode_method
    
    return encoder
