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

class EQBandDecoder(nn.Module):
    """Individual decoder for a single EQ band"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,  # 더 작은 hidden_dim (밴드별로 특화)
                 num_layers: int = 2,    # 더 간단한 구조
                 dropout: float = 0.1,
                 band_id: int = 1):
        super().__init__()
        
        self.band_id = band_id
        self.input_dim = input_dim
        
        # 밴드별 특화된 소형 decoder
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),  # SiLU activation for smoother gradients
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # 밴드별 파라미터 헤드 (4개: freq, gain, q, filter_type)
        self.freq_head = nn.Linear(hidden_dim, 1)
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.q_head = nn.Linear(hidden_dim, 1)
        self.filter_type_head = nn.Linear(hidden_dim, 5)  # 5 filter types
        
    def forward(self, text_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert text embedding to single EQ band parameters
        
        Args:
            text_embedding: (batch_size, embedding_dim)
            
        Returns:
            band_parameters: Dict of this band's parameters
        """
        # Pass through decoder layers
        hidden = self.decoder_layers(text_embedding)
        
        # Generate band-specific parameters
        freq_raw = self.freq_head(hidden)
        gain_raw = self.gain_head(hidden)
        q_raw = self.q_head(hidden)
        filter_type_raw = self.filter_type_head(hidden)
        
        # Apply parameter-specific constraints
        freq = self._apply_freq_constraints(freq_raw)
        gain = self._apply_gain_constraints(gain_raw)
        q = self._apply_q_constraints(q_raw)
        filter_type = torch.softmax(filter_type_raw, dim=-1)
        
        return {
            f'band_{self.band_id}_freq': freq,
            f'band_{self.band_id}_gain': gain,
            f'band_{self.band_id}_q': q,
            f'band_{self.band_id}_filter_type': filter_type
        }
    
    def _apply_freq_constraints(self, raw_value: torch.Tensor) -> torch.Tensor:
        """주파수 제약 적용 - 밴드별 특화 없이 전체 범위에서 자유롭게"""
        sigmoid_val = torch.sigmoid(raw_value)
        log_min = torch.log(torch.tensor(20.0))
        log_max = torch.log(torch.tensor(20000.0))
        log_freq = log_min + sigmoid_val * (log_max - log_min)
        return torch.exp(log_freq)
    
    def _apply_gain_constraints(self, raw_value: torch.Tensor) -> torch.Tensor:
        """게인 제약: -30dB to +30dB"""
        return torch.tanh(raw_value) * 30
    
    def _apply_q_constraints(self, raw_value: torch.Tensor) -> torch.Tensor:
        """Q factor 제약: 0.1 to 30.0 (로그 스케일)"""
        sigmoid_val = torch.sigmoid(raw_value)
        log_min = torch.log(torch.tensor(0.1))
        log_max = torch.log(torch.tensor(30.0))
        log_q = log_min + sigmoid_val * (log_max - log_min)
        return torch.exp(log_q)


class EffectDecoderBlock(nn.Module):
    """Individual decoder block for each audio effect (non-EQ effects)"""
    
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
        
        # Effect-specific parameter heads (EQ 제외)
        self.parameter_heads = self._build_parameter_heads(hidden_dim)
        
    def _build_parameter_heads(self, hidden_dim: int) -> nn.ModuleDict:
        """Build parameter heads specific to each effect (EQ 제외 - 별도 EQBandDecoder 사용)"""
        
        heads = nn.ModuleDict()
        
        if self.effect_name == "reverb":
            # Reverb: Complete set of parameters for realistic reverb
            heads['room_size'] = nn.Linear(hidden_dim, 1)      # Room size (0-1)
            heads['pre_delay'] = nn.Linear(hidden_dim, 1)      # Pre-delay in ms
            heads['diffusion'] = nn.Linear(hidden_dim, 1)      # Diffusion (0-1)
            heads['damping'] = nn.Linear(hidden_dim, 1)        # High-freq damping (0-1)
            heads['wet_gain'] = nn.Linear(hidden_dim, 1)       # Wet signal level
            heads['dry_gain'] = nn.Linear(hidden_dim, 1)       # Dry signal level
            
        elif self.effect_name == "distortion":
            # Distortion: Simplified set (processor에서 실제 사용하는 것만)
            heads['gain'] = nn.Linear(hidden_dim, 1)           # Drive/gain
            heads['color'] = nn.Linear(hidden_dim, 1)          # Color/bias (processor의 bias와 매핑)
            
        elif self.effect_name == "pitch":
            # Pitch shift: Simplified set (processor에서 실제 사용하는 것만)
            heads['scale'] = nn.Linear(hidden_dim, 1)          # Pitch scale (processor의 pitch_shift와 매핑)
            
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
                # 완전히 자유로운 frequency 생성 - 밴드별 bias 완전 제거
                # 전체 주파수 범위 (20Hz - 20kHz)에서 균등하게 생성
                sigmoid_val = torch.sigmoid(raw_value)
                log_min = torch.log(torch.tensor(20.0))
                log_max = torch.log(torch.tensor(20000.0))
                
                # 로그 스케일에서 균등 분포
                log_freq = log_min + sigmoid_val * (log_max - log_min)
                
                return torch.exp(log_freq)
            elif "gain" in param_name:
                # Gain: -30dB to +30dB (범위 확장)
                return torch.tanh(raw_value) * 30
            elif "q" in param_name:
                # Q factor: 0.1 to 30.0 (로그 스케일)
                sigmoid_val = torch.sigmoid(raw_value)
                log_min = torch.log(torch.tensor(0.1))
                log_max = torch.log(torch.tensor(30.0))
                log_q = log_min + sigmoid_val * (log_max - log_min)
                return torch.exp(log_q)
            elif "filter_type" in param_name:
                # Filter type: softmax over 5 types
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
                # Distortion gain: 1 to 10 (torchaudio_processor 범위에 맞춤)
                return torch.sigmoid(raw_value) * 9 + 1
            elif param_name == "color":
                # Color/bias: -1 to 1 (torchaudio_processor의 bias와 매핑)
                return torch.tanh(raw_value)
                
        elif self.effect_name == "pitch":
            if param_name == "scale":
                # Pitch scale: 0.5 to 2.0 (torchaudio_processor의 pitch_shift와 매핑)
                return torch.sigmoid(raw_value) * 1.5 + 0.5
        
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
        # EQ: 5개의 독립적인 밴드 decoder
        self.eq_band_decoders = nn.ModuleList([
            EQBandDecoder(
                shared_hidden_dim, 
                decoder_hidden_dim // 2,  # 더 작은 hidden_dim (128)
                num_decoder_layers - 1,   # 더 간단한 구조 (2 layers)
                dropout, 
                band_id=band_id
            ) for band_id in range(1, 6)  # 5개 밴드
        ])
        
        # 다른 effect들
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
            # 9개 effect tokens에 대한 positional encoding 추가
            self.effect_position_embedding = nn.Parameter(torch.randn(9, shared_hidden_dim) * 0.02)
    
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
            # Create effect tokens with positional encoding
            batch_size = shared_features.shape[0]
            # 9개 토큰: 5개 EQ 밴드 + 3개 다른 effect
            effect_tokens = shared_features.unsqueeze(1).expand(-1, 9, -1)  # (batch_size, 9_effects, dim)
            
            # Add positional encoding to distinguish different effects/bands
            effect_tokens = effect_tokens + self.effect_position_embedding.unsqueeze(0)
            
            # Add small random noise to encourage diversity
            if self.training:
                noise = torch.randn_like(effect_tokens) * 0.01
                effect_tokens = effect_tokens + noise
            
            attended_features, _ = self.cross_attention(
                effect_tokens, effect_tokens, effect_tokens
            )
            
            # Split attended features for each effect
            eq_band_features = [attended_features[:, i, :] for i in range(5)]  # 5개 EQ 밴드
            reverb_features = attended_features[:, 5, :]  # Reverb
            dist_features = attended_features[:, 6, :]    # Distortion
            pitch_features = attended_features[:, 7, :]   # Pitch
        else:
            # Use same features for all effects with small perturbations
            if self.training:
                base_noise = torch.randn_like(shared_features) * 0.005
                eq_band_features = [shared_features + torch.randn_like(shared_features) * 0.005 for _ in range(5)]
                reverb_features = shared_features + torch.randn_like(shared_features) * 0.005
                dist_features = shared_features + torch.randn_like(shared_features) * 0.005  
                pitch_features = shared_features + torch.randn_like(shared_features) * 0.005
            else:
                eq_band_features = [shared_features for _ in range(5)]
                reverb_features = dist_features = pitch_features = shared_features
        
        # Parallel decoding
        # EQ: 5개 밴드 독립적으로 디코딩
        eq_params = {}
        for i, eq_band_decoder in enumerate(self.eq_band_decoders):
            band_params = eq_band_decoder(eq_band_features[i])
            eq_params.update(band_params)
        
        # 다른 effect들
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
            distortion_params["color"]  # color 파라미터 사용
        ])
        
        # Pitch parameters (1개만 - guide preset과 일치)
        raw_tensors.extend([
            pitch_params["scale"]  # scale 파라미터 사용
        ])
        
        # Concatenate all raw tensors (총 28개: 20 + 5 + 2 + 1)
        raw_params_tensor = torch.cat(raw_tensors, dim=-1)  # (batch_size, 28)
        
        # Add raw tensor to output
        preset["_raw_params"] = raw_params_tensor
        
        # 파라미터 범위 검증 및 로깅 (디버깅용)
        if hasattr(self, '_debug_params') and self._debug_params:
            self._validate_parameter_ranges(preset)
        
        # EQ frequency diversity loss 계산 (선택적)
        if hasattr(self, '_enable_diversity_loss') and self._enable_diversity_loss:
            diversity_loss = self._compute_frequency_diversity_loss(preset)
            preset["_diversity_loss"] = diversity_loss
        
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
            "color": params["color"]
        }
    
    def _format_pitch_params_diff(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format pitch parameters into differentiable format"""
        return {
            "scale": params["scale"]
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
            "color": params["color"]
        }
    
    def _format_pitch_params_pedalboard(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format pitch parameters into pedalboard format (backward compatibility)"""
        return {
            "scale": params["scale"]
        }
    
    def _validate_parameter_ranges(self, preset: Dict) -> None:
        """파라미터 범위 검증 및 경고 출력"""
        warnings = []
        
        # EQ 파라미터 검증
        if "equalizer" in preset:
            eq_params = preset["equalizer"]
            for band_key, band_params in eq_params.items():
                if isinstance(band_params, dict):
                    freq = band_params.get("center_freq")
                    gain = band_params.get("gain_db")
                    q = band_params.get("q")
                    
                    if freq is not None:
                        freq_val = freq.item() if hasattr(freq, 'item') else freq
                        if not (20 <= freq_val <= 20000):
                            warnings.append(f"EQ {band_key} frequency {freq_val:.1f}Hz out of range [20-20000]")
                    
                    if gain is not None:
                        gain_val = gain.item() if hasattr(gain, 'item') else gain
                        if not (-30 <= gain_val <= 30):
                            warnings.append(f"EQ {band_key} gain {gain_val:.1f}dB out of range [-30-30]")
                    
                    if q is not None:
                        q_val = q.item() if hasattr(q, 'item') else q
                        if not (0.1 <= q_val <= 30):
                            warnings.append(f"EQ {band_key} Q {q_val:.2f} out of range [0.1-30]")
        
        # Reverb 파라미터 검증
        if "reverb" in preset:
            reverb_params = preset["reverb"]
            for param_name, expected_range in [
                ("room_size", (0, 1)),
                ("pre_delay", (0, 100)),
                ("diffusion", (0, 1)),
                ("damping", (0, 1)),
                ("wet_gain", (0, 1)),
                ("dry_gain", (0, 1))
            ]:
                if param_name in reverb_params:
                    val = reverb_params[param_name]
                    val_item = val.item() if hasattr(val, 'item') else val
                    min_val, max_val = expected_range
                    if not (min_val <= val_item <= max_val):
                        warnings.append(f"Reverb {param_name} {val_item:.3f} out of range [{min_val}-{max_val}]")
        
        # Distortion 파라미터 검증
        if "distortion" in preset:
            dist_params = preset["distortion"]
            
            gain = dist_params.get("gain")
            if gain is not None:
                gain_val = gain.item() if hasattr(gain, 'item') else gain
                if not (1 <= gain_val <= 10):
                    warnings.append(f"Distortion gain {gain_val:.2f} out of range [1-10]")
            
            color = dist_params.get("color")
            if color is not None:
                color_val = color.item() if hasattr(color, 'item') else color
                if not (-1 <= color_val <= 1):
                    warnings.append(f"Distortion color {color_val:.3f} out of range [-1-1]")
        
        # Pitch 파라미터 검증
        if "pitch" in preset:
            pitch_params = preset["pitch"]
            
            scale = pitch_params.get("scale")
            if scale is not None:
                scale_val = scale.item() if hasattr(scale, 'item') else scale
                if not (0.5 <= scale_val <= 2.0):
                    warnings.append(f"Pitch scale {scale_val:.3f} out of range [0.5-2.0]")
        
        # 경고 출력
        if warnings:
            print(f"⚠️ Parameter range warnings:")
            for warning in warnings[:5]:  # 최대 5개만 출력
                print(f"   - {warning}")
            if len(warnings) > 5:
                print(f"   - ... and {len(warnings) - 5} more warnings")
    
    def enable_debug_mode(self):
        """디버그 모드 활성화 (파라미터 범위 검증)"""
        self._debug_params = True
        print("🔍 Decoder debug mode enabled - parameter ranges will be validated")
    
    def disable_debug_mode(self):
        """디버그 모드 비활성화"""
        self._debug_params = False
    
    def enable_diversity_loss(self):
        """Frequency diversity loss 활성화"""
        self._enable_diversity_loss = True
        print("🎯 EQ frequency diversity loss enabled")
    
    def disable_diversity_loss(self):
        """Frequency diversity loss 비활성화"""
        self._enable_diversity_loss = False
    
    def _compute_frequency_diversity_loss(self, preset: Dict) -> torch.Tensor:
        """EQ 밴드들 간의 frequency 다양성을 장려하는 loss"""
        try:
            if "equalizer" not in preset:
                return torch.tensor(0.0)
            
            eq_params = preset["equalizer"]
            frequencies = []
            
            # 모든 밴드의 frequency 수집
            for band in range(1, 6):
                band_key = f"band_{band}"
                if band_key in eq_params and "center_freq" in eq_params[band_key]:
                    freq = eq_params[band_key]["center_freq"]
                    frequencies.append(freq)
            
            if len(frequencies) < 2:
                return torch.tensor(0.0)
            
            # Frequency들을 로그 스케일로 변환
            log_frequencies = torch.stack([torch.log(f + 1e-6) for f in frequencies])
            
            # 모든 frequency 쌍 간의 거리 계산
            num_freqs = len(frequencies)
            diversity_loss = torch.tensor(0.0)
            
            for i in range(num_freqs):
                for j in range(i + 1, num_freqs):
                    # 로그 스케일에서의 거리
                    log_distance = torch.abs(log_frequencies[i] - log_frequencies[j])
                    
                    # 너무 가까운 frequency들에 페널티 (로그 스케일에서 0.5 이하)
                    penalty = torch.exp(-log_distance * 2.0)  # 가까울수록 큰 페널티
                    diversity_loss += penalty
            
            # 평균 페널티 반환
            num_pairs = num_freqs * (num_freqs - 1) / 2
            return diversity_loss / num_pairs
            
        except Exception as e:
            return torch.tensor(0.0)


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
