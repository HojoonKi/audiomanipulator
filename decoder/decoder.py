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
import os


def _should_debug_decoder() -> bool:
    try:
        return os.getenv("DEBUG_DECODER", "0") == "1"
    except Exception:
        return False


def _decoder_debug_print(message: str) -> None:
    if _should_debug_decoder():
        try:
            print(message)
        except Exception:
            pass


def _tensor_stats_brief_dec(name: str, tensor: torch.Tensor) -> None:
    if not _should_debug_decoder():
        return
    try:
        shape = tuple(tensor.shape) if hasattr(tensor, 'shape') else 'NA'
        dtype = str(tensor.dtype) if hasattr(tensor, 'dtype') else 'NA'
        device = str(tensor.device) if hasattr(tensor, 'device') else 'NA'
        dim = int(tensor.dim()) if hasattr(tensor, 'dim') else -1
        tmin = float(torch.min(tensor).item()) if hasattr(tensor, 'numel') and tensor.numel() > 0 else float('nan')
        tmax = float(torch.max(tensor).item()) if hasattr(tensor, 'numel') and tensor.numel() > 0 else float('nan')
        _decoder_debug_print(f"[DECODER-DEBUG] {name}: dim={dim}, shape={shape}, dtype={dtype}, device={device}, min={tmin:.4f}, max={tmax:.4f}")
    except Exception as e:
        _decoder_debug_print(f"[DECODER-DEBUG] {name}: stats failed: {e}")

# ===============================================
# Parallel Decoder Architecture (Recommended)
# ===============================================

class UnifiedEQDecoder(nn.Module):
    """
    Unified decoder that predicts parameters for all EQ bands.
    It ensures monotonic frequency ordering AND an upper bound of 20kHz
    by predicting the ratio of the remaining frequency range for subsequent bands.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 total_bands: int = 6,
                 num_freq_bins: int = 256,
                 num_gain_bins: int = 128,
                 num_q_bins: int = 64):
        super().__init__()
        
        self.total_bands = total_bands
        self.num_freq_bins = num_freq_bins
        self.num_gain_bins = num_gain_bins
        self.num_q_bins = num_q_bins
        
        # 1. Shared Backbone Network
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # 2. --- 수정된 주파수 출력 헤드 ---
        # Band 1: Predicts the absolute starting frequency (Classification + Regression)
        self.freq1_class_head = nn.Linear(hidden_dim, num_freq_bins)
        self.freq1_offset_head = nn.Linear(hidden_dim, 1)

        # Bands 2-6: Predict (total_bands - 1) coefficients (0~1) for the remaining range
        self.freq_coeffs_head = nn.Linear(hidden_dim, total_bands - 1)

        # --- Gain and Q Heads (No changes needed) ---
        self.gain_heads = nn.ModuleDict()
        self.q_heads = nn.ModuleDict()
        for i in range(total_bands):
            band_idx = i + 1
            self.gain_heads[f'band_{band_idx}_class'] = nn.Linear(hidden_dim, num_gain_bins)
            self.gain_heads[f'band_{band_idx}_offset'] = nn.Linear(hidden_dim, 1)
            self.q_heads[f'band_{band_idx}_class'] = nn.Linear(hidden_dim, num_q_bins)
            self.q_heads[f'band_{band_idx}_offset'] = nn.Linear(hidden_dim, 1)

        # --- Filter Type Heads (No changes needed) ---
        self.filter_type_head_first = nn.Linear(hidden_dim, 2)
        self.filter_type_head_last = nn.Linear(hidden_dim, 2)

    def forward(self, text_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = text_embedding.shape[0]
        device = text_embedding.device

        hidden = self.backbone(text_embedding)

        # --- Reconstruct Frequencies (NEW LOGIC) ---
        log_freqs_reconstructed = []
        log_max_full = torch.log(torch.tensor(20000.0, device=device))

        # Step 1: Calculate Band 1 frequency
        log_min_full = torch.log(torch.tensor(20.0, device=device))
        freq1_logits = self.freq1_class_head(hidden)
        freq1_offset_raw = self.freq1_offset_head(hidden)
        freq1_offset = torch.tanh(freq1_offset_raw) * 0.5
        
        current_log_freq = self._reconstruct_single_freq(
            freq1_logits, freq1_offset, log_min_full, log_max_full
        )
        log_freqs_reconstructed.append(current_log_freq)

        # Step 2: Predict coefficients for bands 2-6 and calculate frequencies
        # Use sigmoid to ensure coefficients are between 0 and 1
        coeffs = torch.sigmoid(self.freq_coeffs_head(hidden)) # Shape: (batch_size, 5)

        for i in range(self.total_bands - 1):
            c = coeffs[:, i]
            remaining_range = log_max_full - current_log_freq
            # Add a proportion of the remaining range to the current frequency
            current_log_freq = current_log_freq + c * remaining_range
            log_freqs_reconstructed.append(current_log_freq)

        final_freqs = torch.exp(torch.stack(log_freqs_reconstructed, dim=1))

        # --- Reconstruct Gain, Q, and Filter Type (OLD LOGIC) ---
        final_gains = []
        final_qs = []
        raw_gain_logits, raw_gain_offsets = {}, {}
        raw_q_logits, raw_q_offsets = {}, {}

        for i in range(self.total_bands):
            band_idx = i + 1
            raw_gain_logits[band_idx] = self.gain_heads[f'band_{band_idx}_class'](hidden)
            raw_gain_offsets[band_idx] = torch.tanh(self.gain_heads[f'band_{band_idx}_offset'](hidden)) * 0.5
            raw_q_logits[band_idx] = self.q_heads[f'band_{band_idx}_class'](hidden)
            raw_q_offsets[band_idx] = torch.tanh(self.q_heads[f'band_{band_idx}_offset'](hidden)) * 0.5

            gain = self._reconstruct_gain(raw_gain_logits[band_idx], raw_gain_offsets[band_idx])
            q = self._reconstruct_q(raw_q_logits[band_idx], raw_q_offsets[band_idx])
            final_gains.append(gain)
            final_qs.append(q)

        filter_type_first = torch.softmax(self.filter_type_head_first(hidden), dim=-1)
        filter_type_last = torch.softmax(self.filter_type_head_last(hidden), dim=-1)

        # --- Populate final output dictionary ---
        output_dict = {}
        for i in range(self.total_bands):
            band_idx = i + 1
            # For training: raw logits and offsets
            # For band 1 freq
            if band_idx == 1:
                output_dict[f'band_{band_idx}_freq_logits'] = freq1_logits
                output_dict[f'band_{band_idx}_freq_offset'] = freq1_offset
            # For band 2-6 freq, we output the raw coefficient value (pre-sigmoid)
            else:
                output_dict[f'band_{band_idx}_freq_coeff'] = self.freq_coeffs_head(hidden)[:, i-1].unsqueeze(-1)
            
            output_dict[f'band_{band_idx}_gain_logits'] = raw_gain_logits[band_idx]
            output_dict[f'band_{band_idx}_gain_offset'] = raw_gain_offsets[band_idx]
            output_dict[f'band_{band_idx}_q_logits'] = raw_q_logits[band_idx]
            output_dict[f'band_{band_idx}_q_offset'] = raw_q_offsets[band_idx]
            
            # Filter type
            if band_idx == 1:
                output_dict[f'band_{band_idx}_filter_type'] = filter_type_first
            elif band_idx == self.total_bands:
                output_dict[f'band_{band_idx}_filter_type'] = filter_type_last
            else:
                output_dict[f'band_{band_idx}_filter_type'] = torch.ones(batch_size, 1, device=device) # Peaking
            
            # For inference: final continuous values
            output_dict[f'band_{band_idx}_freq'] = final_freqs[:, i].unsqueeze(-1)
            output_dict[f'band_{band_idx}_gain'] = final_gains[i].unsqueeze(-1)
            output_dict[f'band_{band_idx}_q'] = final_qs[i].unsqueeze(-1)

        return output_dict

    # --- Helper methods (argmax 제거 및 리팩토링) ---
    def _differentiable_reconstruct(self, logits, offset, min_val, max_val, num_bins, log_scale=False):
        """
        Differentiable parameter reconstruction using a softmax weighted average.
        This is a generic helper function to replace argmax-based reconstruction.
        """
        device = logits.device
        
        # If log_scale is True, perform calculations in log space
        if log_scale:
            # torch.tensor() wrapper removed. min_val/max_val are already numbers or tensors.
            log_min_val = torch.log(torch.as_tensor(min_val, device=device))
            log_max_val = torch.log(torch.as_tensor(max_val, device=device))
        else:
            log_min_val = torch.as_tensor(min_val, device=device)
            log_max_val = torch.as_tensor(max_val, device=device)

        # Create a tensor of all bin centers
        bin_width = (log_max_val - log_min_val) / num_bins
        bin_centers = torch.linspace(
            log_min_val + bin_width / 2, 
            log_max_val - bin_width / 2, 
            steps=num_bins, 
            device=device
        )
        
        # Calculate probabilities for each bin
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate the weighted average of bin centers ("soft argmax")
        soft_center = torch.sum(probs * bin_centers, dim=-1)

        # Apply the fine-tuning offset
        final_val = soft_center + offset.squeeze(-1) * bin_width
        
        # If log_scale, convert back from log space
        if log_scale:
            final_val = torch.exp(final_val)
                
        return final_val.clamp(min_val, max_val)

    def _reconstruct_single_freq(self, logits, offset, log_min, log_max):
        """
        Reconstructs the log-frequency for Band 1. 
        Note: This function should return the LOG frequency to be used in the ratio calculation.
        The differentiable helper returns linear scale, so we take the log again.
        """
        # We need to convert log_min/max to linear scale for the helper
        min_freq = torch.exp(log_min)
        max_freq = torch.exp(log_max)
        
        reconstructed_freq = self._differentiable_reconstruct(
            logits, offset, min_freq, max_freq, self.num_freq_bins, log_scale=True
        )
        # Return the log of the frequency as required by the rest of the forward pass
        return torch.log(reconstructed_freq)

    def _reconstruct_gain(self, gain_logits, gain_offset):
        return self._differentiable_reconstruct(
            gain_logits, 
            gain_offset, 
            min_val=-20.0, 
            max_val=20.0, 
            num_bins=self.num_gain_bins, 
            log_scale=False
        )

    def _reconstruct_q(self, q_logits, q_offset):
        return self._differentiable_reconstruct(
            q_logits, 
            q_offset, 
            min_val=0.1, 
            max_val=30.0, 
            num_bins=self.num_q_bins, 
            log_scale=True
        )


class EffectDecoderBlock(nn.Module):
    """Individual decoder block for each audio effect (non-EQ effects)"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 effect_name: str = "generic",
                 # 하이브리드 헤드 설정
                 num_bins_config: dict = None):
        super().__init__()
        
        self.effect_name = effect_name
        self.input_dim = input_dim
        
        # 기본 빈 개수 설정
        if num_bins_config is None:
            num_bins_config = {
                'reverb': {'room_size': 64, 'pre_delay': 64, 'diffusion': 64, 'damping': 64, 'wet_gain': 64, 'decay_time': 64},
                'distortion': {'gain': 64, 'color': 64},
                # pitch는 -12..+12 총 25개 클래스
                'pitch': {'scale': 25}
            }
        self.num_bins_config = num_bins_config.get(effect_name, {})
        
        # Multi-layer decoder with residual connections
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Effect-specific hybrid parameter heads
        self.parameter_heads = self._build_hybrid_parameter_heads(hidden_dim)
        
    def _build_hybrid_parameter_heads(self, hidden_dim: int) -> nn.ModuleDict:
        """Build hybrid parameter heads (classification + regression) for each effect"""
        
        heads = nn.ModuleDict()
        
        if self.effect_name == "reverb":
            # Reverb parameters with hybrid heads
            for param_name, num_bins in self.num_bins_config.items():
                heads[f'{param_name}_class'] = nn.Linear(hidden_dim, num_bins)  # 분류 헤드
                heads[f'{param_name}_offset'] = nn.Linear(hidden_dim, 1)        # 회귀 헤드
                
        elif self.effect_name == "distortion":
            # Distortion parameters with hybrid heads
            for param_name, num_bins in self.num_bins_config.items():
                heads[f'{param_name}_class'] = nn.Linear(hidden_dim, num_bins)
                heads[f'{param_name}_offset'] = nn.Linear(hidden_dim, 1)
                
        elif self.effect_name == "pitch":
            # Pitch는 분류 전용(-12..+12 semitones)
            for param_name, num_bins in self.num_bins_config.items():
                heads[f'{param_name}_class'] = nn.Linear(hidden_dim, num_bins)
                
        else:
            # Generic parameters (fallback)
            heads['param_1_class'] = nn.Linear(hidden_dim, 64)
            heads['param_1_offset'] = nn.Linear(hidden_dim, 1)
            heads['param_2_class'] = nn.Linear(hidden_dim, 64)
            heads['param_2_offset'] = nn.Linear(hidden_dim, 1)
            heads['param_3_class'] = nn.Linear(hidden_dim, 64)
            heads['param_3_offset'] = nn.Linear(hidden_dim, 1)
        
        return heads
    
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
            heads['decay_time'] = nn.Linear(hidden_dim, 1)    # Decay time in ms
            
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
        Convert text embedding to effect parameters using hybrid heads
        
        Args:
            text_embedding: (batch_size, embedding_dim)
            
        Returns:
            parameters: Dict of parameter tensors (both classification and regression)
        """
        # Pass through decoder layers
        hidden = self.decoder_layers(text_embedding)
        
        # Generate hybrid parameters
        parameters = {}
        
        # 하이브리드 헤드 처리
        class_outputs = {}
        offset_outputs = {}
        
        for param_name, head in self.parameter_heads.items():
            raw_output = head(hidden)
            
            if param_name.endswith('_class'):
                # 분류 헤드: softmax 적용
                base_name = param_name[:-6]  # '_class' 제거
                class_outputs[base_name] = raw_output
                parameters[param_name] = torch.softmax(raw_output, dim=-1)
                
            elif param_name.endswith('_offset'):
                # 회귀 헤드: tanh로 [-0.5, 0.5] 제약
                base_name = param_name[:-7]  # '_offset' 제거
                offset_outputs[base_name] = torch.tanh(raw_output) * 0.5
                parameters[param_name] = offset_outputs[base_name]
        
        # 최종 연속값 계산 (추론용)
        for param_name in class_outputs.keys():
            if self.effect_name == "pitch":
                # pitch는 분류만: 클래스 → 정수 반음값(-12..+12)
                logits = class_outputs[param_name]
                class_idx = torch.argmax(logits, dim=-1, keepdim=True)  # [B,1]
                # 0..24 → -12..+12 매핑, [B,1] 유지
                parameters[param_name] = class_idx.float() - 12.0
            elif param_name in offset_outputs:
                final_value = self._reconstruct_parameter(
                    param_name, 
                    class_outputs[param_name], 
                    offset_outputs[param_name]
                )
                # 모든 최종 값은 [B,1] 컬럼 텐서로 표준화
                if final_value.dim() == 1:
                    final_value = final_value.unsqueeze(-1)
                parameters[param_name] = final_value
        
        return parameters
    
    def _reconstruct_parameter(self, param_name: str, class_logits: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """분류 + 회귀로부터 최종 파라미터 값 복원 (미분 가능 버전)"""
        
        # 1. 파라미터별 범위와 빈 개수 설정
        num_bins = self.num_bins_config.get(param_name, 64)
        min_val, max_val = 0.0, 1.0 # 기본값

        if self.effect_name == "reverb":
            if param_name == "room_size": min_val, max_val = 0.0, 10.0
            elif param_name == "pre_delay": min_val, max_val = 0.0, 0.2
            elif param_name in ["diffusion", "damping", "wet_gain"]: min_val, max_val = 0.0, 1.0
            elif param_name == "decay_time": min_val, max_val = 0.1, 10.0
                
        elif self.effect_name == "distortion":
            if param_name == "gain": min_val, max_val = 1.0, 10.0
            elif param_name == "color": min_val, max_val = -1.0, 1.0

        # 2. 모든 빈의 중심값 텐서 생성
        # shape: (num_bins,)
        bin_width = (max_val - min_val) / num_bins
        bin_centers = torch.linspace(
            min_val + bin_width / 2, 
            max_val - bin_width / 2, 
            steps=num_bins, 
            device=class_logits.device
        )

        # 3. Softmax를 이용한 가중 평균으로 "Soft Argmax" 구현
        # probs shape: (batch_size, num_bins)
        probs = torch.softmax(class_logits, dim=-1)
        
        # 가중 평균 계산. 결과 shape: (batch_size,)
        soft_bin_center = torch.sum(probs * bin_centers, dim=-1)

        # 4. 오프셋 적용
        # offset.squeeze(-1) shape: (batch_size,)
        final_val = soft_bin_center + offset.squeeze(-1) * bin_width
        
        return final_val.clamp(min_val, max_val)
    
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
            elif param_name == "wet_gain":
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden_dim, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Parallel effect decoders
        # EQ: 통합된 EQ decoder (단조 증가 주파수 보장)
        self.eq_decoder = UnifiedEQDecoder(
            shared_hidden_dim, 
            decoder_hidden_dim,  # 통합 모델이므로 더 큰 hidden_dim
            num_decoder_layers,  # 더 깊은 구조
            dropout, 
            total_bands=6
        )
        
        # 다른 effect들
        self.reverb_decoder = EffectDecoderBlock(
            shared_hidden_dim, decoder_hidden_dim, num_decoder_layers, dropout, "reverb"
        )
        self.distortion_decoder = EffectDecoderBlock(
            shared_hidden_dim, decoder_hidden_dim, num_decoder_layers, dropout, "distortion"
        )
        # self.pitch_decoder = EffectDecoderBlock(
        #     shared_hidden_dim, decoder_hidden_dim, num_decoder_layers, dropout, "pitch"
        # )
        
        # Optional cross-effect attention (for parameter interdependence)
        self.use_cross_attention = True
        if self.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=shared_hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            # 4개 effect tokens에 대한 positional encoding 추가 (1개 EQ + 3개 다른 effect)
            self.effect_position_embedding = nn.Parameter(torch.randn(4, shared_hidden_dim) * 0.02)
    
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
            # 4개 토큰: 1개 EQ + 3개 다른 effect
            effect_tokens = shared_features.unsqueeze(1).expand(-1, 4, -1)  # (batch_size, 4_effects, dim)
            
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
            eq_features = attended_features[:, 0, :]      # EQ (통합)
            reverb_features = attended_features[:, 1, :]  # Reverb
            dist_features = attended_features[:, 2, :]    # Distortion
            # pitch_features = attended_features[:, 3, :]   # Pitch
        else:
            # Use same features for all effects with small perturbations
            if self.training:
                eq_features = shared_features + torch.randn_like(shared_features) * 0.005
                reverb_features = shared_features + torch.randn_like(shared_features) * 0.005
                dist_features = shared_features + torch.randn_like(shared_features) * 0.005  
                # pitch_features = shared_features + torch.randn_like(shared_features) * 0.005
            else:
                eq_features = reverb_features = dist_features = pitch_features = shared_features
        
        # Parallel decoding
        # EQ: 통합 디코더로 모든 밴드 한 번에 디코딩 (단조 증가 보장)
        eq_params = self.eq_decoder(eq_features)
        
        # 다른 effect들
        reverb_params = self.reverb_decoder(reverb_features)
        distortion_params = self.distortion_decoder(dist_features)
        # pitch_params = self.pitch_decoder(pitch_features)
        
        # Format as preset dictionary (exclude pitch from final return)
        if self.output_format == "differentiable":
            preset = {
                "equalizer": self._format_eq_params_diff(eq_params),
                "reverb": self._format_reverb_params_diff(reverb_params),
                "distortion": self._format_distortion_params_diff(distortion_params),
                # "pitch": self._format_pitch_params_diff(pitch_params)
            }
        else:  # pedalboard format (backward compatibility)
            preset = {
                "Equalizer": self._format_eq_params_pedalboard(eq_params),
                "Reverb": self._format_reverb_params_pedalboard(reverb_params),
                "Distortion": self._format_distortion_params_pedalboard(distortion_params),
                # "Pitch": self._format_pitch_params_pedalboard(pitch_params)
            }
        
        # ADDITION: Raw tensor concatenation for guide loss computation (맞춘 구조)
        raw_tensors = []
        
        def _dbg(name: str, t: torch.Tensor):
            try:
                if os.getenv("DEBUG_DECODER", "0") == "1":
                    shape = tuple(t.shape) if hasattr(t, 'shape') else 'NA'
                    dim = t.dim() if hasattr(t, 'dim') else -1
                    print(f"[DECODER-DEBUG] {name}: dim={dim}, shape={shape}")
            except Exception:
                pass
        
        # EQ parameters (6 bands * 4 params = 24)
        for band in range(1, 7):
            f = eq_params[f"band_{band}_freq"]
            g = eq_params[f"band_{band}_gain"]
            qv = eq_params[f"band_{band}_q"]
            _dbg(f"band_{band}_freq", f)
            _dbg(f"band_{band}_gain", g)
            _dbg(f"band_{band}_q", qv)
            raw_tensors.append(f)
            raw_tensors.append(g)
            raw_tensors.append(qv)
            # Convert filter type probabilities to single value (argmax)
            filter_type_probs = eq_params[f"band_{band}_filter_type"]
            filter_type_val = torch.argmax(filter_type_probs, dim=-1, keepdim=True).float()
            _dbg(f"band_{band}_filter_idx", filter_type_val)
            raw_tensors.append(filter_type_val)
        
        # Reverb parameters (6개 - decay_time 추가)
        rs = reverb_params["room_size"]
        pd = reverb_params["pre_delay"]
        df = reverb_params["diffusion"]
        dm = reverb_params["damping"]
        wg = reverb_params["wet_gain"]
        dt = reverb_params["decay_time"]
        _dbg("room_size", rs)
        _dbg("pre_delay", pd)
        _dbg("diffusion", df)
        _dbg("damping", dm)
        _dbg("wet_gain", wg)
        _dbg("decay_time", dt)
        raw_tensors.extend([rs, pd, df, dm, wg, dt])
        
        # Distortion parameters (2개만 - guide preset과 일치)
        dg = distortion_params["gain"]
        dc = distortion_params["color"]
        _dbg("dist_gain", dg)
        _dbg("dist_color", dc)
        raw_tensors.extend([dg, dc])
        
        # Pitch parameters (1개만 - guide preset과 일치)
        # ps = pitch_params["scale"]
        # _dbg("pitch_semitone", ps)
        # raw_tensors.extend([ps])
        
        # Concatenate all raw tensors (총 33개: 24 + 6 + 2 + 1) --> 32개(pitch 제외)
        try:
            raw_params_tensor = torch.cat(raw_tensors, dim=-1)  # (batch_size, 32)
        except Exception as e:
            print(f"[DECODER-DEBUG] torch.cat failed: {e}")
            for idx, t in enumerate(raw_tensors):
                _dbg(f"raw_tensors[{idx}]", t)
            raise
        
        # Add raw tensor to output
        preset["_raw_params"] = raw_params_tensor

        # 보조 출력: 하이브리드 헤드 로짓/오프셋 제공 (loss에서 사용)
        hybrid_aux = {"eq": {}, "reverb": {}, "distortion": {}} #, "pitch": {}}
        # EQ: 각 밴드의 freq/gain/q 로짓/오프셋
        for band in range(1, 7):
            band_key = f"band_{band}"
            band_dict = {}
            # 존재하는 키만 채움 (안전)
            for name in ["freq", "gain", "q"]:
                logit_key = f"{band_key}_{name}_logits"
                offset_key = f"{band_key}_{name}_offset"
                if logit_key in eq_params:
                    band_dict[f"{name}_logits"] = eq_params[logit_key]
                if offset_key in eq_params:
                    band_dict[f"{name}_offset"] = eq_params[offset_key]
            if band_dict:
                hybrid_aux["eq"][band_key] = band_dict

        # Reverb/Distortion/Pitch: *_class / *_offset 보존
        for effect_name, params in [("reverb", reverb_params), ("distortion", distortion_params)]: #, ("pitch", pitch_params)]:
            effect_aux = {}
            for k, v in params.items():
                if k.endswith("_class") or k.endswith("_offset"):
                    effect_aux[k] = v
            hybrid_aux[effect_name] = effect_aux

        preset["_hybrid"] = hybrid_aux
        
        # 파라미터 범위 검증 및 로깅 (디버깅용)
        if hasattr(self, '_debug_params') and self._debug_params:
            self._validate_parameter_ranges(preset)
        
        return preset
    
    def _format_eq_params_diff(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format EQ parameters into differentiable format"""
        eq_params = {}
        
        # Extract parameters for each band
        for band in range(1, 7):  # 6 bands
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
            "decay_time": params["decay_time"],
        }
    
    def _format_distortion_params_diff(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format distortion parameters into differentiable format"""
        return {
            "gain": params["gain"],
            "color": params["color"]
        }
    
    # def _format_pitch_params_diff(self, params: Dict[str, torch.Tensor]) -> Dict:
    #     """Format pitch parameters into differentiable format"""
    #     return {
    #         # pitch는 semitone 정수
    #         "scale": params["scale"]
    #     }
    
    # Backward compatibility: pedalboard format methods
    def _format_eq_params_pedalboard(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format EQ parameters into pedalboard format (backward compatibility)"""
        
        pedalboard_params = {}
        for band in range(1, 7):  # All 6 bands
            # Get filter type from softmax probabilities
            filter_type_probs = params[f"band_{band}_filter_type"]
            filter_type_idx = torch.argmax(filter_type_probs, dim=-1)
            
            # 밴드별 필터 타입 설정
            if band == 1:  # 첫 번째 밴드
                filter_types = ["low-shelf", "highpass"]
            elif band == 6:  # 마지막 밴드
                filter_types = ["high-shelf", "lowpass"]
            else:  # 중간 밴드들
                filter_types = ["bell"]  # peaking
            
            # Index 범위 체크 (안전성)
            if filter_type_idx.item() >= len(filter_types):
                filter_type_idx = torch.tensor(0)  # Default to first option
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
            "wet_gain": params["wet_gain"],
            "decay_time": params["decay_time"]
        }
    
    def _format_distortion_params_pedalboard(self, params: Dict[str, torch.Tensor]) -> Dict:
        """Format distortion parameters into pedalboard format (backward compatibility)"""
        return {
            "gain": params["gain"],
            "color": params["color"]
        }
    
    # def _format_pitch_params_pedalboard(self, params: Dict[str, torch.Tensor]) -> Dict:
    #     """Format pitch parameters into pedalboard format (backward compatibility)"""
    #     return {
    #         # pitch는 semitone 정수
    #         "scale": params["scale"]
    #     }
    
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
                ("decay_time", (0.1, 10.0)),
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
        
        # Pitch 파라미터 검증 (semitones)
        if "pitch" in preset:
            pitch_params = preset["pitch"]
            
            scale = pitch_params.get("scale")
            if scale is not None:
                scale_val = scale.item() if hasattr(scale, 'item') else scale
                if not (-12 <= scale_val <= 12):
                    warnings.append(f"Pitch semitones {scale_val:.3f} out of range [-12, 12]")
        
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
    import traceback
    try:
        print('Testing UnifiedEQDecoder...')
        decoder = UnifiedEQDecoder(input_dim=1024)
        text_embedding = torch.randn(2, 1024)  # batch_size=2

        output_dict = decoder(text_embedding)
        print('✓ Forward pass successful')
        print('\\n--- Output Dictionary Structure ---')
        for key, value in output_dict.items():
            print(f"- {key:<28} shape={value.shape}")
        
        final_freqs = torch.cat([output_dict[f'band_{i}_freq'] for i in range(1, 7)], dim=1)
        
        for i in range(text_embedding.shape[0]):
            freqs_list = final_freqs[i].tolist()
            print(f'\\n--- Testing item {i+1} in batch ---')
            print(f'Frequencies: {[f"{f:.2f}" for f in freqs_list]}')
            
            is_monotonic = all(freqs_list[j] <= freqs_list[j+1] for j in range(len(freqs_list) - 1))
            is_in_range = freqs_list[0] >= 20 and freqs_list[-1] <= 20000
            print(f'✓ Monotonic increasing: {is_monotonic}')
            print(f'✓ Within 20-20k Hz range: {is_in_range}')

        loss = final_freqs.sum()
        loss.backward()
        print('\n✓ Gradient flow successful')

    except Exception as e:
        print(f'✗ Error: {e}')
        traceback.print_exc()
