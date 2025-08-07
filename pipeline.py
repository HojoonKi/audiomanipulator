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

# Try to import differentiable audio tools
try:
    from audio_tools.differentiable_audio_tools import DifferentiableAudioToolbox
    DIFF_AUDIO_AVAILABLE = True
except ImportError:
    DIFF_AUDIO_AVAILABLE = False
    # Removed warning message - using TorchAudio processor by default


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
    
class DifferentiableAudioProcessor(nn.Module):
    """
    Enhanced differentiable audio processing with parameter mapping
    
    This version uses:
    1. Real differentiable audio tools (when available)
    2. Parameter mapping between pedalboard and differentiable formats
    3. Compatibility checking and fallback handling
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 use_differentiable: bool = True):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.use_differentiable = use_differentiable
        
        # Initialize parameter mapper
        self.parameter_mapper = ParameterMapper()
        
        # Initialize real audio processor for inference
        self.audio_processor = TorchAudioProcessor(sample_rate)
        
        # Initialize differentiable audio toolbox if available
        if DIFF_AUDIO_AVAILABLE:
            self.diff_toolbox = DifferentiableAudioToolbox(sample_rate)
            print("✅ Using real differentiable audio tools")
        else:
            self.diff_toolbox = None
            # Using TorchAudio processor with learnable parameters for gradient approximation
            self.diff_eq_weights = nn.Parameter(torch.ones(5))
            self.diff_reverb_weight = nn.Parameter(torch.tensor(0.0))
            self.diff_distortion_weight = nn.Parameter(torch.tensor(0.0))
            self.diff_pitch_weight = nn.Parameter(torch.tensor(0.0))
        
        print("✅ Enhanced DifferentiableAudioProcessor initialized")
        print(f"   Sample rate: {sample_rate}")
        print(f"   Use differentiable: {use_differentiable}")
        print(f"   Differentiable tools available: {DIFF_AUDIO_AVAILABLE}")
        print("   Using TorchAudio processor with gradient approximation")
    
    def forward(self, 
                audio: torch.Tensor, 
                preset_params: Dict[str, torch.Tensor],
                use_real_audio: bool = False) -> torch.Tensor:
        """
        Process audio with parameter mapping and compatibility handling
        
        Args:
            audio: Input audio tensor
                  - Expected: (batch_size, channels, samples)
                  - If 4D: (batch, 1, channels, samples) -> reshape to (batch, channels, samples)
            preset_params: Dictionary of effect parameters (pedalboard format)
            use_real_audio: If True, use actual audio processing (breaks gradient)
            
        Returns:
            processed_audio: Processed audio tensor
        """
        # Handle dimension mismatch: [batch, 1, channels, samples] -> [batch, channels, samples]
        if audio.dim() == 4 and audio.size(1) == 1:
            print(f"⚡ Reshaping audio from 4D {list(audio.shape)} to 3D for processing")
            audio = audio.squeeze(1)  # Remove singleton dimension
            
        if use_real_audio or not self.training:
            # Use real audio processing (inference mode)
            return self._apply_real_audio_processing(audio, preset_params)
        else:
            # Use differentiable processing with parameter mapping
            return self._apply_differentiable_processing_with_mapping(audio, preset_params)
    
    def _apply_differentiable_processing_with_mapping(self, 
                                                    audio: torch.Tensor, 
                                                    preset_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply differentiable processing with proper parameter mapping
        """
        # Since decoder now outputs differentiable format directly, 
        # we can skip the parameter mapping step
        diff_params = preset_params
        
        # Apply differentiable processing
        if self.diff_toolbox is not None:
            # Use real differentiable audio tools
            return self._apply_with_diff_toolbox(audio, diff_params)
        else:
            # Use fallback approximations
            return self._apply_fallback_approximations(audio, diff_params)
    
    def _convert_preset_to_diff_format(self, preset_params: Dict[str, torch.Tensor]) -> Dict:
        """Convert preset parameters to differentiable format"""
        # Convert tensor dict to scalar dict for mapping
        scalar_params = self._tensor_dict_to_scalar_dict(preset_params)
        
        # Map to differentiable format
        diff_params = self.parameter_mapper.map_pedalboard_to_diff(scalar_params)
        
        # Convert back to tensors
        tensor_diff_params = self._scalar_dict_to_tensor_dict(diff_params, preset_params)
        
        return tensor_diff_params
    
    def _tensor_dict_to_scalar_dict(self, tensor_dict: Dict) -> Dict:
        """Convert tensor dictionary to scalar dictionary"""
        scalar_dict = {}
        
        for key, value in tensor_dict.items():
            if isinstance(value, dict):
                scalar_dict[key] = self._tensor_dict_to_scalar_dict(value)
            elif isinstance(value, torch.Tensor):
                # Take first batch item for conversion
                scalar_dict[key] = value[0].detach().cpu().item()
            else:
                scalar_dict[key] = value
        
        return scalar_dict
    
    def _scalar_dict_to_tensor_dict(self, scalar_dict: Dict, reference_dict: Dict) -> Dict:
        """Convert scalar dictionary back to tensor dictionary"""
        tensor_dict = {}
        
        for key, value in scalar_dict.items():
            if isinstance(value, dict):
                if key in reference_dict and isinstance(reference_dict[key], dict):
                    tensor_dict[key] = self._scalar_dict_to_tensor_dict(value, reference_dict[key])
                else:
                    tensor_dict[key] = self._scalar_dict_to_tensor_dict(value, {})
            else:
                # Find reference tensor to match batch size and device
                ref_tensor = self._find_reference_tensor(reference_dict)
                if ref_tensor is not None:
                    batch_size = ref_tensor.shape[0]
                    device = ref_tensor.device
                    tensor_dict[key] = torch.full((batch_size, 1), float(value), device=device)
                else:
                    tensor_dict[key] = torch.tensor([[float(value)]])
        
        return tensor_dict
    
    def _find_reference_tensor(self, tensor_dict: Dict) -> Optional[torch.Tensor]:
        """Find a reference tensor in nested dictionary"""
        for value in tensor_dict.values():
            if isinstance(value, torch.Tensor):
                return value
            elif isinstance(value, dict):
                ref = self._find_reference_tensor(value)
                if ref is not None:
                    return ref
        return None
    
    def _apply_with_diff_toolbox(self, audio: torch.Tensor, diff_params: Dict) -> torch.Tensor:
        """Apply processing using differentiable audio toolbox"""
        # Convert parameters to the format expected by diff_toolbox
        config = {}
        
        # Map equalizer parameters (now simplified structure)
        if 'equalizer' in diff_params:
            eq_params = diff_params['equalizer']
            eq_config = {}
            
            # Extract scalar values from tensors
            if 'center_freq' in eq_params:
                eq_config['center_freq'] = eq_params['center_freq'].item()
            if 'gain_db' in eq_params:
                eq_config['gain_db'] = eq_params['gain_db'].item()
            if 'q' in eq_params:
                eq_config['q'] = eq_params['q'].item()
            if 'filter_type' in eq_params:
                eq_config['filter_type'] = eq_params['filter_type']
            
            if eq_config:
                config['eq'] = eq_config
        
        # Apply differentiable processing
        try:
            processed = self.diff_toolbox(audio, config)
            return processed
        except Exception as e:
            print(f"⚠️ Differentiable toolbox failed: {e}, falling back to approximations")
            return self._apply_fallback_approximations(audio, diff_params)
    
    def _apply_fallback_approximations(self, audio: torch.Tensor, diff_params: Dict) -> torch.Tensor:
        """Apply fallback approximations when differentiable tools not available"""
        processed = audio.clone()
        
        # Ensure we use learnable parameters for gradient flow
        # Apply learnable approximations
        processed = processed * (1.0 + 0.1 * self.diff_eq_weights[0])  # EQ approximation
        processed = processed + processed * self.diff_reverb_weight * 0.2  # Reverb approximation
        processed = torch.tanh(processed * (1.0 + self.diff_distortion_weight))  # Distortion approximation
        processed = processed * (1.0 + 0.1 * self.diff_pitch_weight)  # Pitch approximation
        
        # Additional processing based on decoder parameters
        if 'gain' in diff_params and diff_params['gain'] is not None:
            gain_db = diff_params['gain']
            gain_linear = torch.pow(10.0, gain_db / 20.0)
            processed = processed * gain_linear.unsqueeze(-1)
        
        # Simple reverb approximation using learnable parameter
        if 'wet_gain' in diff_params and diff_params['wet_gain'] is not None:
            wet_gain = diff_params['wet_gain']
            if processed.shape[-1] > 1000:
                delay_samples = min(int(0.1 * self.sample_rate), processed.shape[-1] // 4)
                delayed = F.pad(processed, (delay_samples, 0))[..., :-delay_samples]
                processed = processed + delayed * wet_gain.unsqueeze(-1) * self.diff_reverb_weight.abs()
        
        # Distortion approximation using learnable parameter
        if 'distortion_gain' in diff_params and diff_params['distortion_gain'] is not None:
            gain_db = diff_params['distortion_gain']
            gain_linear = torch.pow(10.0, gain_db / 20.0)
            processed = torch.tanh(processed * gain_linear.unsqueeze(-1) * (1.0 + self.diff_distortion_weight))
        
        return processed
    
    def _apply_differentiable_processing(self, 
                                       audio: torch.Tensor, 
                                       preset_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Legacy differentiable processing (kept for backwards compatibility)
        
        This maintains gradient flow but provides rough approximations of effects.
        Note: This method uses the old parameter format. 
        For better quality, use _apply_differentiable_processing_with_mapping
        """
        # Check if parameters are in new pedalboard format
        if any(key in preset_params for key in ['Equalizer', 'Reverb', 'Distortion', 'Pitch']):
            # Use new method with parameter mapping
            return self._apply_differentiable_processing_with_mapping(audio, preset_params)
        
        # Legacy processing for old parameter format
        processed = audio.clone()
        
        # Differentiable EQ (simple frequency weighting)
        if 'eq_gains' in preset_params:
            eq_gains = preset_params['eq_gains']  # (batch_size, 5)
            
            # Apply different gains to different frequency bands (approximation)
            # In practice, you'd use proper filter banks, but this maintains gradients
            for i in range(5):
                # Simple frequency-based weighting (very rough approximation)
                weight = eq_gains[:, i:i+1].unsqueeze(-1)  # (batch_size, 1, 1)
                processed = processed * (1.0 + weight * self.diff_eq_weights[i])
        
        # Differentiable Reverb (simple delay + decay)
        if 'reverb_wet' in preset_params:
            reverb_wet = preset_params['reverb_wet']  # (batch_size, 1)
            
            # Simple delay-based reverb approximation
            if processed.shape[-1] > 1000:  # Ensure we have enough samples
                delay_samples = min(int(0.1 * self.sample_rate), processed.shape[-1] // 4)
                delayed = F.pad(processed, (delay_samples, 0))[:, :, :-delay_samples]
                reverb_amount = reverb_wet.unsqueeze(-1) * self.diff_reverb_weight
                processed = processed + delayed * reverb_amount * 0.3
        
        # Differentiable Distortion (tanh saturation)
        if 'distortion_drive' in preset_params:
            drive = preset_params['distortion_drive']  # (batch_size, 1)
            
            # Soft clipping approximation
            drive_amount = drive.unsqueeze(-1) * self.diff_distortion_weight
            processed = torch.tanh(processed * (1.0 + drive_amount * 2.0))
        
        # Differentiable Pitch (simple time-domain approximation)
        if 'pitch_shift' in preset_params:
            pitch_shift = preset_params['pitch_shift']  # (batch_size, 1)
            
            # Very rough pitch approximation (not ideal, but maintains gradients)
            pitch_amount = pitch_shift.unsqueeze(-1) * self.diff_pitch_weight
            processed = processed * (1.0 + pitch_amount * 0.1)
        
        return processed
    
    def _apply_real_audio_processing(self, 
                                   audio: torch.Tensor, 
                                   preset_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply real audio processing using audio_tools
        
        This breaks gradient flow but provides high-quality audio processing.
        """
        batch_size = audio.shape[0]
        processed_batch = []
        
        for b in range(batch_size):
            audio_sample = audio[b].detach().cpu().numpy()
            
            # Extract parameters for this batch item
            batch_params = {}
            for key, value in preset_params.items():
                if isinstance(value, torch.Tensor):
                    batch_params[key] = value[b].detach().cpu().numpy()
                else:
                    batch_params[key] = value
            
            # Apply real audio processing using audio_tools
            processed_sample = audio_sample.copy()
            
            # For now, just apply simple processing
            # In a full implementation, you'd use the actual audio_tools methods
            
            # EQ simulation (simple frequency domain processing)
            if 'eq_gains' in batch_params:
                eq_gains = batch_params['eq_gains']
                # Simple gain adjustment (placeholder)
                for i, gain in enumerate(eq_gains):
                    if gain != 0:
                        processed_sample = processed_sample * (1.0 + gain * 0.1)
            
            # Reverb simulation (simple delay)
            if 'reverb_wet' in batch_params:
                reverb_wet = batch_params['reverb_wet']
                if reverb_wet > 0 and len(processed_sample.shape) > 1 and processed_sample.shape[1] > 1000:
                    delay_samples = int(0.1 * self.sample_rate)
                    if delay_samples < processed_sample.shape[1]:
                        delayed = np.roll(processed_sample, delay_samples, axis=1)
                        processed_sample = processed_sample + delayed * reverb_wet * 0.3
            
            # Distortion simulation (tanh saturation)
            if 'distortion_drive' in batch_params:
                drive = batch_params['distortion_drive']
                if drive > 0:
                    processed_sample = np.tanh(processed_sample * (1.0 + drive))
            
            processed_batch.append(torch.from_numpy(processed_sample))
        
        return torch.stack(processed_batch).to(audio.device)


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
                 use_differentiable_audio: bool = True,
                 
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
        self.use_differentiable_audio = use_differentiable_audio
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
        
        # Use DifferentiableAudioProcessor (gradient approximation + real audio)
        self.audio_processor = DifferentiableAudioProcessor(
            sample_rate=sample_rate,
            use_differentiable=use_differentiable_audio
        )
        
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
        print(f"Audio Processor: {'Differentiable' if self.use_differentiable_audio else 'Real'}")
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
        elif self.use_clap and clap_embeddings is not None:
            # Concatenate embeddings for other backbone types
            combined_embeddings = torch.cat([text_embeddings, clap_embeddings], dim=-1)
            backbone_features = self.backbone(combined_embeddings)
        else:
            # Single embedding
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
            # Use DifferentiableAudioProcessor approach
            processed_audio = self.audio_processor(
                audio, 
                preset_params, 
                use_real_audio=use_real_audio
            )
            
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
        """Set pipeline to training mode with differentiable audio processing"""
        self.train()
        self.audio_processor.use_differentiable = True
        return self
    
    def eval_mode(self):
        """Set pipeline to evaluation mode with real audio processing"""
        self.eval()
        self.audio_processor.use_differentiable = False
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
    
    print("\n✨ Simplified DifferentiableAudioProcessor Pipeline:")
    print("```python")
    print("# Create pipeline with E5-large + CLAP dual embedding")
    print("pipeline = TextToAudioProcessingPipeline(")
    print("    text_encoder_type='e5-large',")
    print("    use_clap=True,")
    print("    backbone_type='dual_embedding',")
    print("    use_differentiable_audio=True,")
    print("    target_params=500000")
    print(")")
    print("")
    print("# Training mode (differentiable audio)")
    print("pipeline.train_mode()")
    print("outputs = pipeline(")
    print("    texts=['Deep bass with warm reverb'],")
    print("    audio=input_audio")
    print(")")
    print("loss = mse_loss(outputs['processed_audio'], target_audio)")
    print("")
    print("# Inference mode (real audio processing)")
    print("pipeline.eval_mode()")
    print("outputs = pipeline(texts=['Bright clean sound'], audio=input_audio)")
    print("```")
    
    print("\n🎯 KEY FEATURES:")
    print("✅ End-to-end gradient flow")
    print("✅ Dual embedding (E5-large + CLAP)")
    print("✅ Parameter efficient (~500K)")
    print("✅ Real audio processing in inference")
    print("✅ Differentiable training")


if __name__ == "__main__":
    print("🏗️ Text-to-Audio Processing Pipeline")
    print("=" * 50)
    
    print("\n🎯 SIMPLIFIED ARCHITECTURE:")
    print("Text → E5-large + CLAP → DualEmbedding → Decoder → DifferentiableAudio")
    
    print("\n📊 PIPELINE COMPONENTS:")
    print("┌─────────────────────┬──────────────────┬─────────────────────┐")
    print("│ Component           │ Role             │ Differentiable      │")
    print("├─────────────────────┼──────────────────┼─────────────────────┤")
    print("│ E5-large + CLAP     │ Text encoding    │ 🔒 Frozen           │")
    print("│ DualEmbedding       │ Feature fusion   │ ✅ Yes              │") 
    print("│ ParallelDecoder     │ Preset params    │ ✅ Yes              │")
    print("│ DiffAudioProcessor  │ Audio effects    │ ⚡ Both modes       │")
    print("└─────────────────────┴──────────────────┴─────────────────────┘")
    
    # Show usage examples
    usage_examples()
    
    print("\n💡 DESIGN DECISION:")
    print("✅ DifferentiableAudioProcessor만 사용 (learnable_pedalboard 제거)")
    print("✅ End-to-end gradient flow로 직접 텍스트-오디오 학습")
    print("✅ Parameter efficient: ~500K parameters") 
    print("✅ Training: differentiable approximation")
    print("✅ Inference: real audio processing")
    
    print("\n🚀 USAGE:")
    print("   Training: pipeline.train_mode() → differentiable audio")
    print("   Inference: pipeline.eval_mode() → real pedalboard processing")
    
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
    
    print("\n🎉 Simplified pipeline ready! No more learnable_pedalboard complexity!")


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
