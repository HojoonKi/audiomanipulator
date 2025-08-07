#!/usr/bin/env python3
"""
Complete Text-to-Audio-Processing Pipeline

This module provides a complete pipeline that connects:
1. Text Encoder (E5-large, BGE, etc.)
2. Backbone Model (Simple, Transformer, etc.)
3. Decoder Heads (Parallel, Diffusion, Transformer)
4. Audio Tools (with differentiable bypass for training)

Key Features:
- End-to-end trainable architecture
- Differentiable audio processing bypass for gradient flow
- Multiple backbone and decoder architectures
- Flexible text encoder selection
- Training and inference modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Import our components
from encoder.text_encoder import get_text_encoder, CLAPTextEncoder
from model.backbone_model import create_backbone
from decoder.decoder import ParallelPresetDecoder
from audio_tools.torchaudio_processor import TorchAudioProcessor
from utils.parameter_mapper import ParameterMapper

# CLAP for audio-text embeddings
try:
    import laion_clap  
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
    print("⚠️ CLAP not available. Install with: pip install laion-clap")
from utils.parameter_mapper import ParameterMapper

class DualEmbeddingBackbone(nn.Module):
    """
    Lightweight MLP backbone that processes both text encoder and CLAP embeddings
    Designed for ~500K parameters total
    """
    
    def __init__(self, 
                 text_dim: int = 1024,  # E5-large dimension
                 clap_dim: int = 512,   # CLAP dimension
                 hidden_dim: int = 256,  # Reduced for parameter efficiency
                 num_layers: int = 3,    # Reduced layers
                 dropout: float = 0.1):
        super().__init__()
        
        self.text_dim = text_dim
        self.clap_dim = clap_dim
        self.hidden_dim = hidden_dim
        
        # Input projections to common dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.clap_proj = nn.Linear(clap_dim, hidden_dim)
        
        # Fusion layer to combine both embeddings
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # MLP layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Residual connections for better gradient flow
        self.use_residual = True
        
        self.output_dim = hidden_dim
        
        # Calculate parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ DualEmbeddingBackbone initialized:")
        print(f"   Text dim: {text_dim} -> {hidden_dim}")
        print(f"   CLAP dim: {clap_dim} -> {hidden_dim}")
        print(f"   Hidden layers: {num_layers}")
        print(f"   Total parameters: {total_params:,}")
    
    def forward(self, text_emb: torch.Tensor, clap_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dual embeddings
        
        Args:
            text_emb: Text encoder embeddings (batch_size, text_dim)
            clap_emb: CLAP embeddings (batch_size, clap_dim)
            
        Returns:
            features: Fused features (batch_size, hidden_dim)
        """
        # Project to common dimension
        text_feat = self.text_proj(text_emb)  # (batch_size, hidden_dim)
        clap_feat = self.clap_proj(clap_emb)  # (batch_size, hidden_dim)
        
        # Concatenate and fuse
        combined = torch.cat([text_feat, clap_feat], dim=-1)  # (batch_size, hidden_dim * 2)
        x = self.fusion(combined)  # (batch_size, hidden_dim)
        
        # Pass through MLP layers with residual connections
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        
        return x


class CompactMLP(nn.Module):
    """
    Ultra-compact MLP backbone for parameter efficiency
    Target: ~200K parameters for the backbone alone
    """
    
    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Normalization
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout (except last layer)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
        # Calculate parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ CompactMLP initialized:")
        print(f"   Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))}")
        print(f"   Parameters: {total_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TextToAudioProcessingPipeline(nn.Module):
    """
    Complete end-to-end pipeline for text-guided audio processing
    
    Architecture:
    Text -> Text Encoder -> Backbone -> Decoder -> Audio Processor -> Output
    
    Key Features:
    - Multiple text encoder options
    - Multiple backbone architectures
    - Multiple decoder architectures
    - Differentiable audio processing for training
    - Real audio processing for inference
    """
    
    def __init__(self,
                 # Text encoder config - 동적 구성 지원
                 text_encoder_type: str = 'sentence-transformer',  # 기본값을 SentenceTransformer로 유지
                 text_encoder_config: Dict = None,  # 추가: 인코더별 세부 설정
                 use_clap: bool = True,  # New: Use CLAP embeddings
                 
                 # Backbone config - 동적 dimension 처리
                 backbone_type: str = 'dual_embedding',  # New: dual_embedding, compact_mlp, transformer
                 backbone_config: Dict = None,
                 
                 # Decoder config
                 decoder_type: str = 'parallel',  # parallel, diffusion, transformer
                 decoder_config: Dict = None,
                 
                 # Audio processing config
                 sample_rate: int = 44100,
                 
                 # Training config
                 freeze_text_encoder: bool = True,
                 target_params: int = 500000):  # Target parameter count
        super().__init__()
        
        print("🏗️ Building TextToAudioProcessingPipeline...")
        
        # Store config
        self.text_encoder_type = text_encoder_type
        self.use_clap = use_clap
        self.backbone_type = backbone_type
        self.decoder_type = decoder_type
        self.sample_rate = sample_rate
        self.target_params = target_params
        
        # Set default configs
        if backbone_config is None:
            backbone_config = self._get_default_backbone_config(backbone_type, target_params)
        if decoder_config is None:
            decoder_config = self._get_default_decoder_config(decoder_type, target_params)
        
        # 1. Text Encoder - 동적 생성
        print(f"📝 Loading text encoder: {text_encoder_type}")
        self.text_encoder = self._create_text_encoder(text_encoder_type, text_encoder_config or {})
        if freeze_text_encoder:
            print("   🔒 Freezing text encoder parameters")
            self._freeze_text_encoder()
        
        # Get text embedding dimension dynamically
        self.text_dim = self.text_encoder.get_embedding_dim()
        print(f"   📐 Detected embedding dim: {self.text_dim}")
        print(f"   Text embedding dim: {self.text_dim}")
        
        # 2. CLAP Encoder (if enabled)
        if use_clap:
            print("🎵 Loading CLAP text encoder...")
            self.clap_encoder = CLAPTextEncoder()
            self.clap_dim = 512  # CLAP embedding dimension
            print(f"   CLAP embedding dim: {self.clap_dim}")
        else:
            self.clap_encoder = None
            self.clap_dim = 0
        
        # 3. Backbone - 동적 dimension 설정
        print(f"🧠 Building backbone: {backbone_type}")
        # 백본 설정에 실제 dimension 업데이트
        backbone_config = self._update_backbone_config(backbone_config, backbone_type)
        self.backbone = self._create_backbone(backbone_type, backbone_config)
        
        print(f"   Backbone output dim: {self.backbone.output_dim}")
        
        # 4. Decoder
        print(f"🎛️ Building decoder: {decoder_type}")
        decoder_config['text_embedding_dim'] = self.backbone.output_dim
        decoder_config['output_format'] = 'differentiable'  # Direct differentiable output
        if 'input_dim' in decoder_config:
            del decoder_config['input_dim']  # Remove incorrect parameter
        self.decoder = ParallelPresetDecoder(**decoder_config)
        
        # 4. Audio Processor
        print("🎵 Building audio processor")
        
        # Use TorchAudioProcessor directly (simpler and more reliable)
        self.audio_processor = TorchAudioProcessor(sample_rate)
        
        print("✅ Pipeline built successfully!")
        self._print_model_summary()
    
    def _get_default_backbone_config(self, backbone_type: str, target_params: int = 500000) -> Dict:
        """Get default configuration for backbone with parameter budget"""
        
        # Calculate optimal hidden dimensions based on parameter budget
        if backbone_type == 'dual_embedding':
            # For dual embedding: text_proj + clap_proj + fusion + MLP layers
            # Rough calculation: ~80% of params for MLP layers
            mlp_params = int(target_params * 0.6)  # Leave room for decoder
            hidden_dim = int(np.sqrt(mlp_params / 6))  # Rough estimate for 3 layers
            hidden_dim = min(256, max(128, hidden_dim))  # Clamp to reasonable range
            
            return {
                'hidden_dim': hidden_dim,
                'num_layers': 3,
                'dropout': 0.1
            }

        elif backbone_type == 'compact_mlp':
            # For compact MLP: calculate hidden dims based on parameter budget
            budget = int(target_params * 0.6)  # Leave room for decoder
            # Simple heuristic for 3-layer MLP
            h1 = min(512, int(budget * 0.5 / 1024))  # First layer: input_dim * h1
            h2 = min(256, int(h1 * 0.6))
            h3 = min(128, int(h2 * 0.8))
            
            return {
                'hidden_dims': [h1, h2, h3],
                'dropout': 0.1,
                'use_layer_norm': True
            }
        
        # Original configs (reduced for parameter efficiency)
        configs = {
            'simple': {
                'hidden_dims': [256, 128],  # Reduced from [512, 256]
                'dropout_rate': 0.1,
                'activation': 'gelu'
            },
            'transformer': {
                'hidden_dim': 256,  # Reduced from 512
                'num_layers': 4,    # Reduced from 6
                'num_heads': 8,
                'dim_head': 32,     # Reduced from 64
                'dropout': 0.1
            },
            'hierarchical': {
                'hidden_dims': [128, 256, 192],  # Reduced
                'num_layers_per_stage': [2, 2, 2],
                'num_heads': 4,  # Reduced from 8
                'dropout': 0.1
            },
            'crossmodal': {
                'text_dim': 1024,  # Will be updated
                'hidden_dim': 256,  # Reduced from 512
                'num_layers': 3,    # Reduced from 4
                'num_heads': 4,     # Reduced from 8
                'dropout': 0.1
            },
            'perceiver': {
                'latent_dim': 256,  # Reduced from 512
                'num_latents': 32,  # Reduced from 64
                'num_cross_layers': 2,
                'num_self_layers': 4,  # Reduced from 6
                'num_heads': 4,        # Reduced from 8
                'dropout': 0.1
            },
            'residual': {
                'hidden_dim': 256,  # Reduced from 512
                'num_blocks': 3,    # Reduced from 4
                'dropout_rate': 0.1
            }
        }
        return configs.get(backbone_type, configs['simple'])
    
    def _get_default_decoder_config(self, decoder_type: str, target_params: int = 500000) -> Dict:
        """Get default configuration for decoder with parameter budget"""
        # Allocate ~40% of parameter budget to decoder
        budget = int(target_params * 0.4)
        
        # Estimate dimensions based on budget - ensure divisible by 8 (common head count)
        shared_dim = min(256, max(128, int(budget * 0.3 / 256)))
        shared_dim = (shared_dim // 8) * 8  # Make divisible by 8
        
        decoder_dim = min(128, max(64, int(budget * 0.7 / (256 * 5))))
        decoder_dim = (decoder_dim // 8) * 8  # Make divisible by 8
        
        return {
            'shared_hidden_dim': shared_dim if shared_dim > 0 else 128,
            'decoder_hidden_dim': decoder_dim if decoder_dim > 0 else 64,
            'num_decoder_layers': 2,  # Reduced from 3
            'dropout': 0.1
        }
    
    def _get_text_dim(self, text_encoder_type: str) -> int:
        """Get text embedding dimension for encoder type"""
        dims = {
            'e5-large': 1024,
            'bge-large': 1024,
            'instructor': 768,
            'clap': 512,
            'sentence-transformer': 384,  # all-MiniLM-L6-v2 default
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768,
            'all-distilroberta-v1': 768
        }
        
        # Try to get actual dimension from encoder if available
        if hasattr(self.text_encoder, 'get_embedding_dim'):
            actual_dim = self.text_encoder.get_embedding_dim()
            print(f"   📐 Detected embedding dim: {actual_dim}")
            return actual_dim
        
        return dims.get(text_encoder_type, 768)  # Default to 768
    
    def _create_text_encoder(self, encoder_type: str, config: Dict):
        """동적으로 텍스트 인코더 생성"""
        from encoder.text_encoder import (
            SentenceTransformerEncoder, 
            E5TextEncoder, 
            CLAPTextEncoder,
            get_text_encoder
        )
        
        if encoder_type == 'sentence-transformer':
            model_name = config.get('model_name', 'all-mpnet-base-v2')  # 기본값을 768D 모델로 변경
            return SentenceTransformerEncoder(model_name=model_name)
        elif encoder_type == 'e5-large':
            model_name = config.get('model_name', 'intfloat/e5-large-v2')
            device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            return E5TextEncoder(model_name=model_name, device=device)
        elif encoder_type == 'clap':
            model_name = config.get('model_name', '630k-audioset-best')
            return CLAPTextEncoder(model_name=model_name)
        else:
            # 기존 get_text_encoder 함수 사용 (backward compatibility)
            return get_text_encoder(encoder_type, **config)
    
    def _freeze_text_encoder(self):
        """텍스트 인코더 파라미터 동결"""
        if hasattr(self.text_encoder, 'model'):
            for param in self.text_encoder.model.parameters():
                param.requires_grad = False
        else:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
    
    def _update_backbone_config(self, backbone_config: Dict, backbone_type: str) -> Dict:
        """백본 설정에 실제 dimension 업데이트"""
        updated_config = backbone_config.copy()
        
        if backbone_type == 'dual_embedding':
            updated_config['text_dim'] = self.text_dim
            updated_config['clap_dim'] = self.clap_dim
        elif backbone_type in ['compact_mlp', 'transformer', 'simple']:
            # simple, compact_mlp, transformer는 모두 결합된 임베딩을 입력으로 받음
            combined_dim = self.text_dim + (self.clap_dim if self.use_clap else 0)
            updated_config['input_dim'] = combined_dim
            print(f"   📐 Combined input dim: text({self.text_dim}) + clap({self.clap_dim if self.use_clap else 0}) = {combined_dim}")
        elif backbone_type in ['residual', 'dynamic', 'dynamic_transformer']:
            # DynamicBackbone 계열은 개별 차원을 전달
            updated_config['text_dim'] = self.text_dim
            updated_config['clap_dim'] = self.clap_dim if self.use_clap else 0
            print(f"   📐 Dynamic backbone dims: text({self.text_dim}), clap({self.clap_dim if self.use_clap else 0})")
        
        return updated_config
    
    def _create_backbone(self, backbone_type: str, backbone_config: Dict):
        """동적으로 백본 생성"""
        if backbone_type == 'dual_embedding':
            return DualEmbeddingBackbone(**backbone_config)
        elif backbone_type == 'compact_mlp':
            return CompactMLP(**backbone_config)
        else:
            # 기존 create_backbone 함수 사용
            return create_backbone(backbone_type, **backbone_config)
    
    def _print_model_summary(self):
        """Print model architecture summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate parameter breakdown
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        audio_processor_params = sum(p.numel() for p in self.audio_processor.parameters()) if hasattr(self.audio_processor, 'parameters') else 0
        
        print(f"\n📊 MODEL SUMMARY")
        print(f"=" * 60)
        print(f"Text Encoder: {self.text_encoder_type} ({self.text_dim}D)")
        print(f"CLAP Encoder: {'Enabled' if self.use_clap else 'Disabled'} ({self.clap_dim}D)")
        print(f"Backbone: {self.backbone_type} -> {self.backbone.output_dim}D")
        print(f"  └─ Parameters: {backbone_params:,}")
        print(f"Decoder: {self.decoder_type}")
        print(f"  └─ Parameters: {decoder_params:,}")
        print("Audio Processor: TorchAudio")
        print(f"  └─ Parameters: {audio_processor_params:,}")
        print(f"")
        print(f"🎯 Parameter Budget: {self.target_params:,}")
        print(f"📊 Total Parameters: {total_params:,}")
        print(f"🚀 Trainable Parameters: {trainable_params:,}")
        
        # Parameter efficiency check
        efficiency = (trainable_params / self.target_params) * 100 if self.target_params > 0 else 0
        status = "✅" if efficiency <= 100 else "⚠️"
        print(f"{status} Parameter Efficiency: {efficiency:.1f}% of target")
        print(f"=" * 60)
    
    def to(self, device):
        """Override to method to ensure all submodules are moved to device"""
        # Move main model
        super().to(device)
        
        # Move text encoder if it has a model attribute
        if hasattr(self.text_encoder, 'model') and hasattr(self.text_encoder.model, 'to'):
            self.text_encoder.model.to(device)
            print(f"📝 Text encoder moved to {device}")
        
        # Move CLAP encoder if available
        if self.clap_encoder is not None:
            if hasattr(self.clap_encoder, 'clap_model') and hasattr(self.clap_encoder.clap_model, 'to'):
                self.clap_encoder.clap_model.to(device)
                print(f"🎵 CLAP encoder moved to {device}")
            # Also move the CLAPTextEncoder itself if it's a nn.Module
            if isinstance(self.clap_encoder, nn.Module):
                self.clap_encoder.to(device)
        
        # Move audio processor if it has learnable parameters
        if hasattr(self.audio_processor, 'parameters'):
            self.audio_processor.to(device)
            print(f"🎛️ Audio processor moved to {device}")
        
        return self
    
    def encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode text inputs to embeddings (both regular and CLAP if enabled)
        
        Args:
            texts: List of text descriptions
            
        Returns:
            text_embeddings: Text embeddings (batch_size, text_dim)
            clap_embeddings: CLAP embeddings (batch_size, clap_dim) or None
        """
        # Regular text embeddings - 통일된 인터페이스 사용
        if hasattr(self.text_encoder, 'encode_text'):
            embeddings = self.text_encoder.encode_text(texts)
        else:
            # Backward compatibility
            embeddings = self.text_encoder(texts)
        
        # Handle different encoder output formats
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        
        # Convert to tensor if numpy
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        
        # Ensure proper device and dtype
        embeddings = embeddings.float()
        if next(self.parameters()).is_cuda:
            embeddings = embeddings.cuda()
        
        # Clone to detach from inference mode and enable gradients for downstream processing
        embeddings = embeddings.clone().detach().requires_grad_(True)
        
        # CLAP embeddings (if enabled) - 한 번에 배치 처리 (이미 효율적)
        clap_embeddings = None
        if self.use_clap and self.clap_encoder is not None:
            clap_embeddings = self.clap_encoder.get_text_embedding(texts)
            
            # Ensure proper device and dtype
            if isinstance(clap_embeddings, np.ndarray):
                clap_embeddings = torch.from_numpy(clap_embeddings)
            clap_embeddings = clap_embeddings.float()
            if next(self.parameters()).is_cuda:
                clap_embeddings = clap_embeddings.cuda()
            
            # Clone to detach from inference mode and enable gradients for downstream processing
            clap_embeddings = clap_embeddings.clone().detach().requires_grad_(True)
        
        return embeddings, clap_embeddings
    
    def forward(self, 
                texts: List[str],
                audio: Optional[torch.Tensor] = None,
                use_real_audio: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete pipeline
        
        Args:
            texts: List of text descriptions
            audio: Optional input audio (batch_size, channels, samples)
            use_real_audio: Whether to use real audio processing
            
        Returns:
            outputs: Dictionary containing:
                - text_embeddings: Encoded text
                - clap_embeddings: CLAP embeddings (if enabled)
                - backbone_features: Backbone output
                - preset_params: Decoded parameters
                - processed_audio: Final processed audio (if input audio provided)
        """
        batch_size = len(texts)
        
        # 1. Text encoding (both regular and CLAP)
        text_embeddings, clap_embeddings = self.encode_text(texts)
        
        # 2. Backbone processing
        if self.backbone_type == 'dual_embedding' and self.use_clap:
            # Dual embedding backbone
            backbone_features = self.backbone(text_embeddings, clap_embeddings)
        elif hasattr(self.backbone, 'fusion_type') and self.use_clap and clap_embeddings is not None:
            # DynamicBackbone with cross-attention or other fusion
            backbone_features = self.backbone(text_emb=text_embeddings, clap_emb=clap_embeddings)
        elif self.use_clap and clap_embeddings is not None:
            # Legacy: Concatenate embeddings for other backbone types
            combined_embeddings = torch.cat([text_embeddings, clap_embeddings], dim=-1)
            backbone_features = self.backbone(combined_embeddings)
        else:
            # Single embedding - check if it's DynamicBackbone
            if hasattr(self.backbone, 'fusion_type'):
                backbone_features = self.backbone(text_emb=text_embeddings, clap_emb=None)
            else:
                backbone_features = self.backbone(text_embeddings)
        
        # 3. Decode to preset parameters
        preset_params = self.decoder(backbone_features)
        
        outputs = {
            'text_embeddings': text_embeddings,
            'backbone_features': backbone_features,
            'preset_params': preset_params
        }
        
        # Add CLAP embeddings to output if available
        if clap_embeddings is not None:
            outputs['clap_embeddings'] = clap_embeddings
        
        # 4. Audio processing (if audio provided)
        if audio is not None:
            # Use TorchAudioProcessor directly
            processed_audio = self.audio_processor(audio, preset_params)
            
            outputs['processed_audio'] = processed_audio
        
        return outputs
    
    def process_audio_from_text(self, 
                               texts: List[str],
                               audio_files: List[str],
                               output_files: Optional[List[str]] = None,
                               use_real_audio: bool = True) -> List[np.ndarray]:
        """
        Process audio files guided by text descriptions
        
        Args:
            texts: Text descriptions for each audio file
            audio_files: Paths to input audio files
            output_files: Optional paths for output files
            use_real_audio: Whether to use real audio processing
            
        Returns:
            processed_audio_list: List of processed audio arrays
        """
        self.eval()
        
        processed_audio_list = []
        
        with torch.no_grad():
            for i, (text, audio_file) in enumerate(zip(texts, audio_files)):
                # Load audio using audio processor
                audio_data, sr = self.audio_processor.load_audio(audio_file)
                
                # Convert to tensor and resample if needed
                if sr != self.sample_rate:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
                
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()
                if len(audio_tensor.shape) == 2:  # Add channel dimension if mono
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                if next(self.parameters()).is_cuda:
                    audio_tensor = audio_tensor.cuda()
                
                # Process
                outputs = self.forward([text], audio_tensor, use_real_audio=use_real_audio)
                processed_audio = outputs['processed_audio']
                
                # Convert back to numpy
                processed_np = processed_audio.squeeze(0).cpu().numpy()
                processed_audio_list.append(processed_np)
                
                # Save if output path provided
                if output_files and i < len(output_files):
                    self.audio_processor.save_audio(processed_np, output_files[i], self.sample_rate)
                    print(f"✅ Saved processed audio: {output_files[i]}")
        
        return processed_audio_list
    
    def train_mode(self):
        """Set pipeline to training mode"""
        self.train()
        return self
    
    def eval_mode(self):
        """Set pipeline to evaluation mode"""
        self.eval()
        return self


# ===============================================
# Usage Examples and Comparisons
# ===============================================

# ===============================================
# Usage Examples
# ===============================================

def usage_examples():
    """Show how to use the simplified pipeline"""
    
    print("\n�️ USAGE EXAMPLES")
    print("=" * 40)
    
    print("\n✨ Simplified TorchAudio Pipeline:")
    print("```python")
    print("# Create pipeline with E5-large + CLAP dual embedding")
    print("pipeline = TextToAudioProcessingPipeline(")
    print("    text_encoder_type='e5-large',")
    print("    use_clap=True,")
    print("    backbone_type='dual_embedding',")
    print("    target_params=500000")
    print(")")
    print("")
    print("# Training and inference use the same TorchAudio processor")
    print("outputs = pipeline(")
    print("    texts=['Deep bass with warm reverb'],")
    print("    audio=input_audio")
    print(")")
    print("loss = mse_loss(outputs['processed_audio'], target_audio)")
    print("```")
    
    print("\n🎯 KEY FEATURES:")
    print("✅ End-to-end gradient flow")
    print("✅ Dual embedding (E5-large + CLAP)")  
    print("✅ Parameter efficient (~500K)")
    print("✅ TorchAudio processing (differentiable)")
    print("✅ Simplified architecture")


if __name__ == "__main__":
    print("🏗️ Text-to-Audio Processing Pipeline")
    print("=" * 50)
    
    print("\n🎯 SIMPLIFIED ARCHITECTURE:")
    print("Text → E5-large + CLAP → DualEmbedding → Decoder → TorchAudio")
    
    print("\n📊 PIPELINE COMPONENTS:")
    print("┌─────────────────────┬──────────────────┬─────────────────────┐")
    print("│ Component           │ Role             │ Differentiable      │")
    print("├─────────────────────┼──────────────────┼─────────────────────┤")
    print("│ E5-large + CLAP     │ Text encoding    │ 🔒 Frozen           │")
    print("│ DualEmbedding       │ Feature fusion   │ ✅ Yes              │") 
    print("│ ParallelDecoder     │ Preset params    │ ✅ Yes              │")
    print("│ TorchAudioProcessor │ Audio effects    │ ✅ Yes              │")
    print("└─────────────────────┴──────────────────┴─────────────────────┘")
    
    # Show usage examples
    usage_examples()
    
    print("\n💡 DESIGN DECISION:")
    print("✅ TorchAudioProcessor 직접 사용 (더 간단하고 안정적)")
    print("✅ End-to-end gradient flow로 직접 텍스트-오디오 학습") 
    print("✅ Parameter efficient: ~500K parameters")
    print("✅ 완전 differentiable audio processing")
    print("✅ 복잡한 wrapper 클래스 제거")
    
    print("\n🚀 USAGE:")
    print("   동일한 TorchAudioProcessor로 training과 inference 모두 처리")
    print("   더 이상 모드 전환 필요 없음")
    
    # Example instantiation
    print("\n🧪 EXAMPLE USAGE:")
    try:
        pipeline = TextToAudioProcessingPipeline(
            text_encoder_type='e5-large',
            use_clap=True,
            backbone_type='dual_embedding',
            target_params=500000
        )
        print("✅ Simplified pipeline created successfully!")
        
    except Exception as e:
        print(f"⚠️  Pipeline creation failed: {e}")
        print("   (This is expected if dependencies are missing)")
    
    print("\n🎉 Simplified pipeline ready! Clean TorchAudio architecture!")


def build_model(text_encoder_type: str = 'e5-large',
                backbone_type: str = 'transformer',
                decoder_type: str = 'parallel',
                **kwargs) -> TextToAudioProcessingPipeline:
    """
    Factory function to build a complete text-to-audio processing model
    
    Args:
        text_encoder_type: Type of text encoder ('e5-large', 'bge-large', etc.)
        backbone_type: Type of backbone ('simple', 'transformer', 'hierarchical', etc.)
        decoder_type: Type of decoder ('parallel', 'diffusion', 'transformer')
        **kwargs: Additional configuration parameters
        
    Returns:
        model: Complete pipeline model
    """
    return TextToAudioProcessingPipeline(
        text_encoder_type=text_encoder_type,
        backbone_type=backbone_type,
        decoder_type=decoder_type,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    print("🎵 TEXT-TO-AUDIO PROCESSING PIPELINE")
    print("=" * 50)
    
    # Build different model configurations
    models_to_test = [
        {
            'name': 'Simple Model',
            'config': {
                'text_encoder_type': 'e5-large',
                'backbone_type': 'simple',
                'decoder_type': 'parallel'
            }
        },
        {
            'name': 'Transformer Model',
            'config': {
                'text_encoder_type': 'e5-large',
                'backbone_type': 'transformer',
                'decoder_type': 'parallel'
            }
        }
    ]
    
    for model_config in models_to_test:
        print(f"\n🔧 Testing {model_config['name']}:")
        
        try:
            # Build model
            model = build_model(**model_config['config'])
            
            # Test forward pass
            sample_texts = [
                "Add warm reverb and boost the bass",
                "Make it sound distorted and aggressive",
                "Clean bright sound with sparkle"
            ]
            
            # Create dummy audio
            dummy_audio = torch.randn(3, 2, 44100)  # 3 samples, stereo, 1 second
            
            # Forward pass
            outputs = model(sample_texts, dummy_audio)
            
            print(f"✅ Forward pass successful!")
            print(f"   Text embeddings: {outputs['text_embeddings'].shape}")
            print(f"   Backbone features: {outputs['backbone_features'].shape}")
            print(f"   Processed audio: {outputs['processed_audio'].shape}")
            
            # Test parameter extraction
            preset_params = outputs['preset_params']
            print(f"   Preset parameters:")
            for key, value in preset_params.items():
                if isinstance(value, dict):
                    print(f"     {key}: (nested dict with {len(value)} parameters)")
                    for subkey, subvalue in value.items():
                        if hasattr(subvalue, 'shape'):
                            print(f"       {subkey}: {subvalue.shape}")
                        else:
                            print(f"       {subkey}: {type(subvalue)}")
                elif hasattr(value, 'shape'):
                    print(f"     {key}: {value.shape}")
                else:
                    print(f"     {key}: {type(value)}")
                
        except Exception as e:
            print(f"❌ Error testing {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Pipeline testing complete!")
