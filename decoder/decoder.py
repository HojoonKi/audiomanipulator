#!/usr/bin/env python3
"""
Text-to-Preset Model: Convert text embeddings to audio processing parameters

This module implements various architectures for converting text embeddings
to pedalboard preset parameters, including:
1. Parallel Decoder Architecture (Recommended)
2. Diffusion-based Parameter Generation
3. Transformer-based Parameter Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union

# ===============================================
# Parallel Decoder Architecture (Recommended)
# ===============================================

class EffectDecoderBlock(nn.Module):
    """Individual decoder block for each audio effect"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 effect_name: str = "generic"):
        super().__init__()
        
        self.effect_name = effect_name
        self.input_dim = input_dim
        
        # Multi-layer decoder with residual connections
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Effect-specific parameter heads
        self.parameter_heads = self._build_parameter_heads(hidden_dim)
        
    def _build_parameter_heads(self, hidden_dim: int) -> nn.ModuleDict:
        """Build parameter heads specific to each effect"""
        
        heads = nn.ModuleDict()
        
        if self.effect_name == "equalizer":
            # EQ: 5 bands with frequency, gain, Q, filter_type for each band
            for band in range(1, 6):  # 5 bands: band_1 to band_5
                heads[f'band_{band}_freq'] = nn.Linear(hidden_dim, 1)
                heads[f'band_{band}_gain'] = nn.Linear(hidden_dim, 1)
                heads[f'band_{band}_q'] = nn.Linear(hidden_dim, 1)
                heads[f'band_{band}_filter_type'] = nn.Linear(hidden_dim, 5)  # 5 filter types: low-shelf, bell, high-shelf, highpass, lowpass
            
        elif self.effect_name == "reverb":
            # Reverb: Complete set of parameters for realistic reverb
            heads['room_size'] = nn.Linear(hidden_dim, 1)      # Room size (0-1)
            heads['pre_delay'] = nn.Linear(hidden_dim, 1)      # Pre-delay in ms
            heads['diffusion'] = nn.Linear(hidden_dim, 1)      # Diffusion (0-1)
            heads['damping'] = nn.Linear(hidden_dim, 1)        # High-freq damping (0-1)
            heads['wet_gain'] = nn.Linear(hidden_dim, 1)       # Wet signal level
            heads['dry_gain'] = nn.Linear(hidden_dim, 1)       # Dry signal level
            
        elif self.effect_name == "distortion":
            # Distortion: Complete set for realistic distortion
            heads['gain'] = nn.Linear(hidden_dim, 1)           # Drive/gain
            heads['bias'] = nn.Linear(hidden_dim, 1)           # DC bias
            heads['tone'] = nn.Linear(hidden_dim, 1)           # Tone control
            heads['mix'] = nn.Linear(hidden_dim, 1)            # Wet/dry mix
            
        elif self.effect_name == "pitch":
            # Pitch shift: Complete set for pitch shifting
            heads['pitch_shift'] = nn.Linear(hidden_dim, 1)    # Pitch ratio
            heads['formant_shift'] = nn.Linear(hidden_dim, 1)  # Formant correction
            heads['mix'] = nn.Linear(hidden_dim, 1)            # Wet/dry mix
            
        else:
            # Generic parameters
            heads['param_1'] = nn.Linear(hidden_dim, 1)
            heads['param_2'] = nn.Linear(hidden_dim, 1)
            heads['param_3'] = nn.Linear(hidden_dim, 1)
        
        return heads
    
    def forward(self, text_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert text embedding to effect parameters
        
        Args:
            text_embedding: (batch_size, embedding_dim)
            
        Returns:
            parameters: Dict of parameter tensors
        """
        # Pass through decoder layers
        hidden = self.decoder_layers(text_embedding)
        
        # Generate parameters through individual heads
        parameters = {}
        for param_name, head in self.parameter_heads.items():
            raw_param = head(hidden)
            
            # Apply parameter-specific activations and scaling
            parameters[param_name] = self._apply_parameter_constraints(
                param_name, raw_param
            )
        
        return parameters
    
    def _apply_parameter_constraints(self, param_name: str, raw_value: torch.Tensor) -> torch.Tensor:
        """Apply realistic constraints to parameters"""
        
        if self.effect_name == "equalizer":
            if "freq" in param_name:
                # Frequency: 20Hz - 20kHz, distributed logarithmically
                return torch.sigmoid(raw_value) * 19980 + 20
            elif "gain" in param_name:
                # Gain: -20dB to +20dB
                return torch.tanh(raw_value) * 20
            elif "q" in param_name:
                # Q factor: 0.1 to 10.0
                return torch.sigmoid(raw_value) * 9.9 + 0.1
            elif "filter_type" in param_name:
                # Filter type: softmax over 3 types (low-shelf, bell, high-shelf)
                return torch.softmax(raw_value, dim=-1)
                
        elif self.effect_name == "reverb":
            if param_name == "room_size":
                # Room size: 0 to 1
                return torch.sigmoid(raw_value)
            elif param_name == "pre_delay":
                # Pre-delay: 0 to 100ms
                return torch.sigmoid(raw_value) * 100
            elif param_name == "diffusion":
                # Diffusion: 0 to 1
                return torch.sigmoid(raw_value)
            elif param_name == "damping":
                # Damping: 0 to 1
                return torch.sigmoid(raw_value)
            elif param_name in ["wet_gain", "dry_gain"]:
                # Gain levels: 0 to 1
                return torch.sigmoid(raw_value)
                
        elif self.effect_name == "distortion":
            if param_name == "gain":
                # Distortion gain: 1 to 20 (higher range for more distortion)
                return torch.sigmoid(raw_value) * 19 + 1
            elif param_name == "bias":
                # Bias: -1 to 1
                return torch.tanh(raw_value)
            elif param_name == "tone":
                # Tone: 0 to 1 (0=dark, 1=bright)
                return torch.sigmoid(raw_value)
            elif param_name == "mix":
                # Mix: 0 to 1 (0=dry, 1=wet)
                return torch.sigmoid(raw_value)
                
        elif self.effect_name == "pitch":
            if param_name == "pitch_shift":
                # Pitch shift: 0.5 to 2.0 (pitch ratio)
                return torch.sigmoid(raw_value) * 1.5 + 0.5
            elif param_name == "formant_shift":
                # Formant shift: 0.8 to 1.2
                return torch.sigmoid(raw_value) * 0.4 + 0.8
            elif param_name == "mix":
                # Mix: 0 to 1
                return torch.sigmoid(raw_value)
        
        # Default: sigmoid activation
        return torch.sigmoid(raw_value)


class ParallelPresetDecoder(nn.Module):
    """
    Main model with parallel decoders for each effect
    
    Architecture:
    Text Embedding → [Shared Encoder] → Split → [EQ Decoder]     → EQ Params
                                              → [Reverb Decoder]  → Reverb Params  
                                              → [Dist Decoder]    → Dist Params
                                              → [Pitch Decoder]   → Pitch Params
    
    Now outputs differentiable format directly (no need for parameter mapping)
    """
    
    def __init__(self, 
                 text_embedding_dim: int = 1024,  # E5-large embedding size
                 shared_hidden_dim: int = 512,
                 decoder_hidden_dim: int = 256,
                 num_decoder_layers: int = 3,
                 dropout: float = 0.1,
                 output_format: str = "differentiable"):  # "differentiable" or "pedalboard"
        super().__init__()
        
        self.text_embedding_dim = text_embedding_dim
        self.output_format = output_format
        
        # Shared encoder to process text embeddings
        self.shared_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden_dim, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Parallel effect decoders
        self.eq_decoder = EffectDecoderBlock(
            shared_hidden_dim, decoder_hidden_dim, num_decoder_layers, dropout, "equalizer"
        )
        self.reverb_decoder = EffectDecoderBlock(
            shared_hidden_dim, decoder_hidden_dim, num_decoder_layers, dropout, "reverb"
        )
        self.distortion_decoder = EffectDecoderBlock(
            shared_hidden_dim, decoder_hidden_dim, num_decoder_layers, dropout, "distortion"
        )
        self.pitch_decoder = EffectDecoderBlock(
            shared_hidden_dim, decoder_hidden_dim, num_decoder_layers, dropout, "pitch"
        )
        
        # Optional cross-effect attention (for parameter interdependence)
        self.use_cross_attention = True
        if self.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=shared_hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
    
    def forward(self, text_embedding: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Convert text embedding to preset parameters
        
        Args:
            text_embedding: (batch_size, embedding_dim)
            
        Returns:
            preset: Nested dict with effect parameters
        """
        # Shared encoding
        shared_features = self.shared_encoder(text_embedding)  # (batch_size, shared_hidden_dim)
        
        # Optional cross-attention between effects
        if self.use_cross_attention:
            # Create effect tokens
            batch_size = shared_features.shape[0]
            effect_tokens = shared_features.unsqueeze(1).expand(-1, 4, -1)  # (batch_size, 4_effects, dim)
            
            attended_features, _ = self.cross_attention(
                effect_tokens, effect_tokens, effect_tokens
            )
            
            # Split attended features for each effect
            eq_features = attended_features[:, 0, :]      # EQ
            reverb_features = attended_features[:, 1, :]  # Reverb
            dist_features = attended_features[:, 2, :]    # Distortion
            pitch_features = attended_features[:, 3, :]   # Pitch
        else:
            # Use same features for all effects
            eq_features = reverb_features = dist_features = pitch_features = shared_features
        
        # Parallel decoding
        eq_params = self.eq_decoder(eq_features)
        reverb_params = self.reverb_decoder(reverb_features)
        distortion_params = self.distortion_decoder(dist_features)
        pitch_params = self.pitch_decoder(pitch_features)
        
        # Format as preset dictionary
        if self.output_format == "differentiable":
            preset = {
                "equalizer": self._format_eq_params_diff(eq_params),
                "reverb": self._format_reverb_params_diff(reverb_params),
                "distortion": self._format_distortion_params_diff(distortion_params),
                "pitch": self._format_pitch_params_diff(pitch_params)
            }
        else:  # pedalboard format (backward compatibility)
            preset = {
                "Equalizer": self._format_eq_params_pedalboard(eq_params),
                "Reverb": self._format_reverb_params_pedalboard(reverb_params),
                "Distortion": self._format_distortion_params_pedalboard(distortion_params),
                "Pitch": self._format_pitch_params_pedalboard(pitch_params)
            }
        
        # ADDITION: Raw tensor concatenation for guide loss computation (맞춘 구조)
        raw_tensors = []
        
        # EQ parameters (5 bands * 4 params = 20)
        for band in range(1, 6):
            raw_tensors.append(eq_params[f"band_{band}_freq"])
            raw_tensors.append(eq_params[f"band_{band}_gain"])
            raw_tensors.append(eq_params[f"band_{band}_q"])
            # Convert filter type probabilities to single value (argmax)
            filter_type_probs = eq_params[f"band_{band}_filter_type"]
            filter_type_val = torch.argmax(filter_type_probs, dim=-1, keepdim=True).float()
            raw_tensors.append(filter_type_val)
        
        # Reverb parameters (5개만 - guide preset과 일치)
        raw_tensors.extend([
            reverb_params["room_size"],
            reverb_params["pre_delay"], 
            reverb_params["diffusion"],
            reverb_params["damping"],
            reverb_params["wet_gain"]
            # dry_gain 제외
        ])
        
        # Distortion parameters (2개만 - guide preset과 일치)
        raw_tensors.extend([
            distortion_params["gain"],
            distortion_params["bias"]  # bias를 color로 매핑
            # tone, mix 제외
        ])
        
        # Pitch parameters (1개만 - guide preset과 일치)
        raw_tensors.extend([
            pitch_params["pitch_shift"]  # pitch_shift를 scale로 매핑
            # formant_shift, mix 제외
        ])
        
        # Concatenate all raw tensors (총 28개: 20 + 5 + 2 + 1)
        raw_params_tensor = torch.cat(raw_tensors, dim=-1)  # (batch_size, 28)
        
        # Add raw tensor to output
        preset["_raw_params"] = raw_params_tensor
        
        return preset
    
    def _format_eq_params_diff(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format EQ parameters into differentiable format"""
        eq_params = {}
        
        # Extract parameters for each band
        for band in range(1, 6):  # 5 bands
            band_key = f"band_{band}"
            
            # Get filter type from softmax probabilities
            filter_type_probs = params[f"band_{band}_filter_type"]
            filter_type_idx = torch.argmax(filter_type_probs, dim=-1)
            
            eq_params[band_key] = {
                "center_freq": params[f"band_{band}_freq"],
                "gain_db": params[f"band_{band}_gain"],
                "q": params[f"band_{band}_q"],
                "filter_type": filter_type_probs,  # Keep probabilities for differentiability
                "filter_type_idx": filter_type_idx  # Index for actual filter selection
            }
        
        return eq_params
    
    def _format_reverb_params_diff(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format reverb parameters into differentiable format"""
        return {
            "room_size": params["room_size"],
            "pre_delay": params["pre_delay"],
            "diffusion": params["diffusion"],
            "damping": params["damping"],
            "wet_gain": params["wet_gain"],
            "dry_gain": params["dry_gain"]
        }
    
    def _format_distortion_params_diff(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format distortion parameters into differentiable format"""
        return {
            "gain": params["gain"],
            "bias": params["bias"],
            "tone": params["tone"],
            "mix": params["mix"]
        }
    
    def _format_pitch_params_diff(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format pitch parameters into differentiable format"""
        return {
            "pitch_shift": params["pitch_shift"],
            "formant_shift": params["formant_shift"],
            "mix": params["mix"]
        }
    
    # Backward compatibility: pedalboard format methods
    def _format_eq_params_pedalboard(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format EQ parameters into pedalboard format (backward compatibility)"""
        filter_types = ["low-shelf", "bell", "high-shelf", "highpass", "lowpass"]  # 5개 타입 지원
        
        pedalboard_params = {}
        for band in range(1, 6):  # All 5 bands
            # Get filter type from softmax probabilities
            filter_type_probs = params[f"band_{band}_filter_type"]
            filter_type_idx = torch.argmax(filter_type_probs, dim=-1)
            
            # Index 범위 체크 (안전성)
            if filter_type_idx.item() >= len(filter_types):
                filter_type_idx = torch.tensor(1)  # Default to 'bell'
            filter_type = filter_types[filter_type_idx.item()]
            
            pedalboard_params[band] = {
                "frequency": params[f"band_{band}_freq"],
                "Gain": params[f"band_{band}_gain"],
                "Q": params[f"band_{band}_q"],
                "Filter-type": filter_type
            }
        
        return pedalboard_params
    
    def _format_reverb_params_pedalboard(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format reverb parameters into pedalboard format (backward compatibility)"""
        return {
            "room_size": params["room_size"],
            "pre_delay": params["pre_delay"],
            "diffusion": params["diffusion"],
            "damping": params["damping"],
            "wet_gain": params["wet_gain"]
            # dry_gain 제외 - guide preset에 없음
        }
    
    def _format_distortion_params_pedalboard(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format distortion parameters into pedalboard format (backward compatibility)"""
        return {
            "gain": params["gain"],
            "color": params["bias"]  # Map bias to color for compatibility
        }
    
    def _format_pitch_params_pedalboard(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format pitch parameters into pedalboard format (backward compatibility)"""
        return {
            "scale": params["pitch_shift"]  # Map pitch_shift to scale for compatibility
        }


# ===============================================
# Diffusion-based Parameter Generation (Advanced)
# ===============================================

class DiffusionPresetGenerator(nn.Module):
    """
    Diffusion model for generating audio effect parameters
    
    This approach treats parameter generation as a denoising process,
    potentially producing more diverse and realistic parameter combinations.
    """
    
    def __init__(self,
                 text_embedding_dim: int = 1024,
                 param_dim: int = 16,  # Total number of parameters
                 hidden_dim: int = 512,
                 num_timesteps: int = 1000):
        super().__init__()
        
        self.param_dim = param_dim
        self.num_timesteps = num_timesteps
        
        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Linear(param_dim + text_embedding_dim + 1, hidden_dim),  # +1 for timestep
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, param_dim)
        )
        
        # Timestep embedding
        self.timestep_embedding = nn.Embedding(num_timesteps, hidden_dim // 4)
        
        # Parameter mapping heads (same as parallel decoder)
        self.param_mapper = self._build_param_mapper()
    
    def _build_param_mapper(self):
        """Map raw parameters to effect-specific parameters"""
        return nn.ModuleDict({
            'eq_freq_1': nn.Linear(1, 1),
            'eq_gain_1': nn.Linear(1, 1),
            'eq_q_1': nn.Linear(1, 1),
            'eq_freq_2': nn.Linear(1, 1),
            'eq_gain_2': nn.Linear(1, 1),
            'eq_q_2': nn.Linear(1, 1),
            'reverb_room': nn.Linear(1, 1),
            'reverb_delay': nn.Linear(1, 1),
            'reverb_diffusion': nn.Linear(1, 1),
            'reverb_damping': nn.Linear(1, 1),
            'reverb_wet': nn.Linear(1, 1),
            'dist_gain': nn.Linear(1, 1),
            'dist_color': nn.Linear(1, 1),
            'pitch_scale': nn.Linear(1, 1),
        })
    
    def forward(self, text_embedding: torch.Tensor, num_inference_steps: int = 50) -> Dict:
        """
        Generate parameters using diffusion process
        
        Args:
            text_embedding: (batch_size, embedding_dim)
            num_inference_steps: Number of denoising steps
            
        Returns:
            preset: Generated preset parameters
        """
        batch_size = text_embedding.shape[0]
        device = text_embedding.device
        
        # Start with random noise
        x = torch.randn(batch_size, self.param_dim, device=device)
        
        # Denoising process
        for t in reversed(range(0, self.num_timesteps, self.num_timesteps // num_inference_steps)):
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            timestep_emb = self.timestep_embedding(timestep).unsqueeze(1)
            
            # Predict noise
            model_input = torch.cat([
                x, 
                text_embedding, 
                timestep_emb.squeeze(1)
            ], dim=1)
            
            predicted_noise = self.noise_predictor(model_input)
            
            # Denoising step (simplified DDPM)
            if t > 0:
                noise = torch.randn_like(x)
                alpha = 0.999  # Simplified noise schedule
                x = (x - predicted_noise) * alpha + noise * (1 - alpha)
            else:
                x = x - predicted_noise
        
        # Map to preset format
        return self._map_to_preset(x)
    
    def _map_to_preset(self, raw_params: torch.Tensor) -> Dict:
        """Map raw parameters to preset format"""
        # Implementation similar to ParallelPresetDecoder
        # This is a simplified version
        preset = {
            "Equalizer": {
                1: {"frequency": 1000, "Gain": 0, "Q": 1, "Filter-type": "bell"},
                2: {"frequency": 5000, "Gain": 0, "Q": 1, "Filter-type": "high-shelf"}
            },
            "Reverb": {"Room Size": 5, "Pre Delay": 0.1, "Diffusion": 0.5, "Damping": 0.5, "Wet Gain": 0.3},
            "Distortion": {"Gain": 10, "Color": 0.5},
            "Pitch": {"Scale": 0}
        }
        return preset


# ===============================================
# Transformer-based Parameter Prediction
# ===============================================

class TransformerPresetGenerator(nn.Module):
    """
    Transformer-based approach treating parameters as a sequence
    
    This approach models parameter generation as sequence-to-sequence translation:
    Text tokens → Parameter tokens
    """
    
    def __init__(self,
                 text_embedding_dim: int = 1024,
                 param_vocab_size: int = 1000,  # Discretized parameter values
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6):
        super().__init__()
        
        self.d_model = d_model
        self.param_vocab_size = param_vocab_size
        
        # Project text embedding to transformer dimension
        self.text_proj = nn.Linear(text_embedding_dim, d_model)
        
        # Parameter embeddings
        self.param_embedding = nn.Embedding(param_vocab_size, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, param_vocab_size)
        
        # Parameter sequence length (number of parameters to generate)
        self.max_param_length = 16
    
    def forward(self, text_embedding: torch.Tensor) -> Dict:
        """
        Generate parameters using transformer
        
        Args:
            text_embedding: (batch_size, embedding_dim)
            
        Returns:
            preset: Generated preset parameters
        """
        batch_size = text_embedding.shape[0]
        device = text_embedding.device
        
        # Project text embedding
        text_features = self.text_proj(text_embedding).unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Generate parameters autoregressively
        generated_params = []
        current_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for i in range(self.max_param_length):
            # Embed current input
            embedded_input = self.param_embedding(current_input)
            
            # Pass through transformer
            output = self.transformer(embedded_input, text_features)
            
            # Predict next parameter
            logits = self.output_proj(output[:, -1:, :])
            next_param = torch.argmax(logits, dim=-1)
            
            generated_params.append(next_param)
            current_input = torch.cat([current_input, next_param], dim=1)
        
        # Convert discrete parameters to continuous values and format as preset
        return self._discrete_to_preset(torch.cat(generated_params, dim=1))
    
    def _discrete_to_preset(self, discrete_params: torch.Tensor) -> Dict:
        """Convert discrete parameter tokens to preset format"""
        # Convert discrete values to continuous parameters
        continuous_params = discrete_params.float() / self.param_vocab_size
        
        # Map to preset (simplified)
        preset = {
            "Equalizer": {
                1: {"frequency": 1000, "Gain": 0, "Q": 1, "Filter-type": "bell"},
                2: {"frequency": 5000, "Gain": 0, "Q": 1, "Filter-type": "high-shelf"}
            },
            "Reverb": {"Room Size": 5, "Pre Delay": 0.1, "Diffusion": 0.5, "Damping": 0.5, "Wet Gain": 0.3},
            "Distortion": {"Gain": 10, "Color": 0.5},
            "Pitch": {"Scale": 0}
        }
        return preset


# ===============================================
# Model Factory and Recommendations
# ===============================================

class PresetGeneratorFactory:
    """Factory for creating different preset generation models"""
    
    @staticmethod
    def create_model(model_type: str = "parallel", **kwargs):
        """
        Create preset generation model
        
        Args:
            model_type: "parallel", "diffusion", or "transformer"
            **kwargs: Model-specific parameters
            
        Returns:
            Model instance
        """
        if model_type == "parallel":
            return ParallelPresetDecoder(**kwargs)
        elif model_type == "diffusion":
            return DiffusionPresetGenerator(**kwargs)
        elif model_type == "transformer":
            return TransformerPresetGenerator(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def recommend_architecture():
    """Provide architecture recommendations"""
    
    recommendations = {
        "🎯 Parallel Decoder (Recommended)": {
            "pros": [
                "명확한 parameter 제어",
                "빠른 inference",
                "interpretable",
                "각 effect별 specialized learning"
            ],
            "cons": [
                "effect 간 상호작용 제한적",
                "단순한 아키텍처"
            ],
            "best_for": "안정적이고 예측 가능한 결과가 필요한 경우"
        },
        
        "🌟 Diffusion Model": {
            "pros": [
                "높은 품질의 parameter 생성",
                "다양한 결과 생성 가능",
                "SOTA generative 기술"
            ],
            "cons": [
                "느린 inference (multiple steps)",
                "복잡한 구현",
                "학습 어려움"
            ],
            "best_for": "최고 품질과 다양성이 필요한 경우"
        },
        
        "🔧 Transformer": {
            "pros": [
                "sequence modeling 장점",
                "attention mechanism",
                "scalable"
            ],
            "cons": [
                "parameter discretization 필요",
                "복잡한 구현"
            ],
            "best_for": "sequential parameter dependencies가 중요한 경우"
        }
    }
    
    print("🏗️ PRESET GENERATION ARCHITECTURE RECOMMENDATIONS")
    print("=" * 60)
    
    for name, info in recommendations.items():
        print(f"\n{name}")
        print(f"  Pros: {', '.join(info['pros'])}")
        print(f"  Cons: {', '.join(info['cons'])}")
        print(f"  Best for: {info['best_for']}")
    
    print(f"\n💡 RECOMMENDATION FOR YOUR PROJECT:")
    print(f"  Start with: Parallel Decoder (간단하고 효과적)")
    print(f"  Upgrade to: Diffusion Model (고품질 결과 필요시)")
    print(f"  Consider: Cross-attention between effects for parameter interdependence")


if __name__ == "__main__":
    print("🎛️ Text-to-Preset Model Architectures")
    print("=" * 50)
    
    recommend_architecture()
    
    print(f"\n🧪 Testing Parallel Decoder...")
    
    # Test parallel decoder
    model = ParallelPresetDecoder(text_embedding_dim=1024)
    dummy_text_embedding = torch.randn(2, 1024)  # Batch of 2
    
    with torch.no_grad():
        preset = model(dummy_text_embedding)
    
    print(f"✅ Model created successfully!")
    print(f"   Input shape: {dummy_text_embedding.shape}")
    print(f"   Output effects: {list(preset.keys())}")
    print(f"   EQ bands: {list(preset['Equalizer'].keys())}")
    print(f"   Reverb params: {list(preset['Reverb'].keys())}")
