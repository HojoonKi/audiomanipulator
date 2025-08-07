#!/usr/bin/env python3
"""
Backbone Model: Advanced Neural Network Backbones for Audio Processing

This module provides sophisticated backbone networks including Transformer-based
architectures for processing input embeddings before feeding to decoder heads.

Architecture Options:
1. Simple MLP Backbone (lightweight)
2. Multi-Scale Backbone (multi-resolution processing)
3. Residual Backbone (ResNet-style)
4. Transformer Backbone (SOTA, using attention mechanisms)
5. Hybrid Backbone (Transformer + CNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, List, Tuple

# Import attention components
try:
    from attention import (
        CrossAttention, 
        BasicTransformerBlock, 
        FeedForward,
        default,
        exists
    )
    ATTENTION_AVAILABLE = True
except ImportError:
    try:
        from .attention import (
            CrossAttention, 
            BasicTransformerBlock, 
            FeedForward,
            default,
            exists
        )
        ATTENTION_AVAILABLE = True
    except ImportError:
        try:
            from model.attention import (
                CrossAttention, 
                BasicTransformerBlock, 
                FeedForward,
                default,
                exists
            )
            ATTENTION_AVAILABLE = True
        except ImportError:
            print("Warning: attention.py not found. Transformer backbones disabled.")
            ATTENTION_AVAILABLE = False


class CrossModalFusionBlock(nn.Module):
    """
    Cross-modal fusion block using cross-attention
    
    Allows CLAP embeddings to attend to text embeddings for richer context understanding.
    """
    
    def __init__(self, 
                 text_dim: int,
                 clap_dim: int, 
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        if not ATTENTION_AVAILABLE:
            raise ImportError("attention.py required for CrossModalFusionBlock")
        
        self.text_dim = text_dim
        self.clap_dim = clap_dim
        self.hidden_dim = hidden_dim
        
        # Project both modalities to same dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.clap_proj = nn.Linear(clap_dim, hidden_dim)
        
        # Cross-attention: CLAP queries attend to text keys/values
        self.clap_to_text_attention = CrossAttention(
            query_dim=hidden_dim,
            context_dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout
        )
        
        # Cross-attention: Text queries attend to CLAP keys/values  
        self.text_to_clap_attention = CrossAttention(
            query_dim=hidden_dim,
            context_dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout
        )
        
        # Self-attention for refined representations
        self.text_self_attention = CrossAttention(
            query_dim=hidden_dim,
            context_dim=None,  # Self-attention
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout
        )
        
        self.clap_self_attention = CrossAttention(
            query_dim=hidden_dim,
            context_dim=None,  # Self-attention
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout
        )
        
        # Layer norms
        self.text_norm1 = nn.LayerNorm(hidden_dim)
        self.text_norm2 = nn.LayerNorm(hidden_dim)
        self.clap_norm1 = nn.LayerNorm(hidden_dim)
        self.clap_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks
        self.text_ff = FeedForward(hidden_dim, dropout=dropout, glu=True)
        self.clap_ff = FeedForward(hidden_dim, dropout=dropout, glu=True)
        
        # Final fusion
        self.fusion_attention = CrossAttention(
            query_dim=hidden_dim,
            context_dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout
        )
        
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_ff = FeedForward(hidden_dim, dropout=dropout, glu=True)
        
        print(f"✅ CrossModalFusionBlock initialized:")
        print(f"   Text dim: {text_dim} -> {hidden_dim}")
        print(f"   CLAP dim: {clap_dim} -> {hidden_dim}")
        print(f"   Heads: {num_heads}")
    
    def forward(self, text_emb: torch.Tensor, clap_emb: torch.Tensor):
        """
        Forward pass through cross-modal fusion
        
        Args:
            text_emb: Text embeddings (batch_size, text_dim)
            clap_emb: CLAP embeddings (batch_size, clap_dim)
            
        Returns:
            fused_features: Cross-attended and fused features (batch_size, hidden_dim)
        """
        batch_size = text_emb.shape[0]
        
        # Project to common dimension and add sequence dimension
        text_feat = self.text_proj(text_emb).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        clap_feat = self.clap_proj(clap_emb).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Cross-attention: CLAP attends to text for richer context
        clap_attended = clap_feat + self.clap_to_text_attention(
            self.clap_norm1(clap_feat), context=self.text_norm1(text_feat)
        )
        
        # Cross-attention: Text attends to CLAP for audio-aware context
        text_attended = text_feat + self.text_to_clap_attention(
            self.text_norm1(text_feat), context=self.clap_norm1(clap_feat)
        )
        
        # Self-attention for refined representations
        text_refined = text_attended + self.text_self_attention(self.text_norm2(text_attended))
        clap_refined = clap_attended + self.clap_self_attention(self.clap_norm2(clap_attended))
        
        # Feed-forward processing
        text_final = text_refined + self.text_ff(self.text_norm2(text_refined))
        clap_final = clap_refined + self.clap_ff(self.clap_norm2(clap_refined))
        
        # Final fusion: Use enhanced CLAP as query, enhanced text as context
        # This allows CLAP to query the rich textual context
        fused = clap_final + self.fusion_attention(
            self.fusion_norm(clap_final), context=self.fusion_norm(text_final)
        )
        
        # Final processing
        fused = fused + self.fusion_ff(self.fusion_norm(fused))
        
        # Remove sequence dimension
        return fused.squeeze(1)  # (batch_size, hidden_dim)


class DynamicBackbone(nn.Module):
    """
    Dynamic backbone network with cross-attention fusion
    
    Supports various text encoders and CLAP embeddings with sophisticated 
    cross-attention mechanisms for richer multimodal understanding.
    """
    
    def __init__(self, 
                 text_dim: Optional[int] = None,  # Text embedding dimension (dynamic)
                 clap_dim: int = 512,  # CLAP embedding dimension (fixed)
                 hidden_dims: list = None,
                 dropout_rate: float = 0.2,
                 activation: str = 'relu',
                 fusion_type: str = 'cross_attention'):  # 'concat', 'add', 'cross_attention'
        super().__init__()
        
        self.text_dim = text_dim
        self.clap_dim = clap_dim
        self.fusion_type = fusion_type
        
        # Enhanced fusion options
        if fusion_type == 'cross_attention':
            # Use sophisticated cross-attention fusion
            if text_dim is not None:
                self.cross_modal_fusion = CrossModalFusionBlock(
                    text_dim=text_dim,
                    clap_dim=clap_dim,
                    hidden_dim=512,  # Common dimension after fusion
                    num_heads=8,
                    dropout=dropout_rate
                )
                input_dim = 512  # Output of cross-modal fusion
            else:
                # Will be built dynamically
                self.cross_modal_fusion = None
                input_dim = None
        elif fusion_type == 'concat':
            # Simple concatenation
            if text_dim is not None:
                input_dim = text_dim + clap_dim
            else:
                input_dim = None  # Runtime determination
        elif fusion_type == 'add':
            # Addition requires same dimensions
            if text_dim is not None and text_dim != clap_dim:
                # Need projection layers
                self.text_proj = nn.Linear(text_dim, clap_dim)
                self.clap_proj = nn.Linear(clap_dim, clap_dim)
            input_dim = clap_dim
        else:
            input_dim = clap_dim  # Default fallback
        
        # Hidden dimensions for processing after fusion
        if hidden_dims is None:
            if fusion_type == 'cross_attention':
                hidden_dims = [512, 256, 128]  # Start from fusion output
            else:
                hidden_dims = [512, 256, 128]
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Activation function
        if activation.lower() == 'relu':
            self.activation_fn = nn.ReLU(inplace=True)
        elif activation.lower() == 'gelu':
            self.activation_fn = nn.GELU()
        elif activation.lower() == 'swish':
            self.activation_fn = nn.SiLU()
        else:
            self.activation_fn = nn.ReLU(inplace=True)
        
        # Build layers if input_dim is known
        if input_dim is not None:
            self._build_layers(input_dim)
        else:
            self.backbone = None  # Will be built dynamically
        
        print(f"✅ DynamicBackbone initialized:")
        print(f"   Text dim: {text_dim}")
        print(f"   CLAP dim: {clap_dim}")
        print(f"   Fusion type: {fusion_type}")
        print(f"   Hidden dims: {hidden_dims}")
        if fusion_type == 'cross_attention':
            print(f"   🎯 Using cross-attention for rich multimodal fusion!")
    
    def _build_cross_modal_fusion(self, text_dim: int):
        """Dynamically build cross-modal fusion block"""
        self.cross_modal_fusion = CrossModalFusionBlock(
            text_dim=text_dim,
            clap_dim=self.clap_dim,
            hidden_dim=512,
            num_heads=8,
            dropout=self.dropout_rate
        )
        
        if torch.cuda.is_available():
            self.cross_modal_fusion = self.cross_modal_fusion.cuda()
            
        print(f"🔧 Dynamic cross-modal fusion built: {text_dim} + {self.clap_dim} -> 512")
    
    def _build_layers(self, input_dim: int):
        """Dynamically build processing layers"""
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation_fn,
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Remove last dropout
        if layers:
            layers = layers[:-1]
        
        self.backbone = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
        print(f"🔧 Dynamic layers built with input_dim: {input_dim} -> output_dim: {self.output_dim}")
    
    def _auto_detect_and_build(self, combined_input: torch.Tensor):
        """Auto-detect dimensions and build layers on first forward pass"""
        input_dim = combined_input.shape[-1]
        self._build_layers(input_dim)
        # Move to same device
        if combined_input.is_cuda:
            self.backbone = self.backbone.cuda()
    
    def forward(self, text_emb: Optional[torch.Tensor] = None, clap_emb: Optional[torch.Tensor] = None):
        """
        Forward pass through dynamic backbone with cross-attention fusion
        
        Args:
            text_emb: Text embeddings (batch_size, text_dim)
            clap_emb: CLAP embeddings (batch_size, clap_dim)
            
        Returns:
            features: Processed features (batch_size, output_dim)
        """
        # Handle different input scenarios
        if text_emb is not None and clap_emb is not None:
            # Both embeddings available - use sophisticated fusion
            if self.fusion_type == 'cross_attention':
                # Build cross-modal fusion dynamically if needed
                if self.cross_modal_fusion is None:
                    text_dim = text_emb.shape[-1]
                    self._build_cross_modal_fusion(text_dim)
                
                # Cross-attention fusion for rich multimodal understanding
                combined = self.cross_modal_fusion(text_emb, clap_emb)
                
            elif self.fusion_type == 'concat':
                combined = torch.cat([text_emb, clap_emb], dim=-1)
            elif self.fusion_type == 'add':
                if hasattr(self, 'text_proj'):
                    text_proj = self.text_proj(text_emb)
                    clap_proj = self.clap_proj(clap_emb)
                    combined = text_proj + clap_proj
                else:
                    combined = text_emb + clap_emb
            else:
                combined = torch.cat([text_emb, clap_emb], dim=-1)
                
        elif text_emb is not None:
            # Only text embedding - handle dynamic dimensions
            if self.fusion_type == 'cross_attention':
                # For cross-attention, we need to project to common dimension
                if self.cross_modal_fusion is None:
                    text_dim = text_emb.shape[-1]
                    self._build_cross_modal_fusion(text_dim)
                # Use text projection only
                combined = self.cross_modal_fusion.text_proj(text_emb)
            else:
                combined = text_emb
        elif clap_emb is not None:
            # Only CLAP embedding
            combined = clap_emb
        else:
            raise ValueError("At least one of text_emb or clap_emb must be provided")
        
        # Build processing layers dynamically if needed
        if self.backbone is None:
            self._auto_detect_and_build(combined)
        
        return self.backbone(combined)


class SharedBackbone(DynamicBackbone):
    """
    SharedBackbone은 이제 DynamicBackbone의 alias
    하위 호환성을 위해 유지
    """
    
    def __init__(self, 
                 input_dim: int = 1024,
                 hidden_dims: list = None,
                 hidden_dim: int = None,
                 output_dim: int = None,
                 dropout_rate: float = 0.2,
                 activation: str = 'relu'):
        
        # 기존 인터페이스를 DynamicBackbone으로 변환
        if hidden_dims is None and hidden_dim is not None:
            if output_dim is None:
                output_dim = hidden_dim // 2
            hidden_dims = [hidden_dim, output_dim]
        elif hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # DynamicBackbone 초기화 (concat 방식으로 고정)
        super().__init__(
            text_dim=input_dim,
            clap_dim=0,  # CLAP 사용 안함
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            fusion_type='concat'
        )
        
        # 하위 호환성을 위한 속성
        self.input_dim = input_dim
    
    def forward(self, x):
        """하위 호환성을 위한 forward"""
        # x의 차원이 예상과 다를 수 있으므로 동적 처리
        if x.shape[-1] != self.input_dim and self.text_dim != x.shape[-1]:
            # 차원이 다르면 text_dim을 업데이트하고 백본을 다시 빌드
            self.text_dim = x.shape[-1]
            if self.backbone is None or (hasattr(self.backbone, '__len__') and len(self.backbone) > 0 and self.backbone[0].in_features != x.shape[-1]):
                self._auto_detect_and_build(x)
        
        return super().forward(text_emb=x, clap_emb=None)
    
    def get_layer_outputs(self, x):
        """
        Get outputs from each layer for analysis
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            layer_outputs: List of outputs from each layer
        """
        layer_outputs = []
        current_x = x
        
        # Process through DynamicBackbone
        combined = x
        if self.backbone is None:
            self._auto_detect_and_build(combined)
        
        for layer in self.backbone:
            current_x = layer(current_x)
            if isinstance(layer, nn.Linear):
                layer_outputs.append(current_x)
        
        return layer_outputs


class DynamicTransformerBackbone(nn.Module):
    """
    Dynamic Transformer backbone with cross-attention fusion
    
    Advanced transformer-based backbone supporting various text encoders 
    and CLAP embeddings with sophisticated cross-attention mechanisms.
    """
    
    def __init__(self,
                 text_dim: Optional[int] = None,
                 clap_dim: int = 512,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.1,
                 fusion_type: str = 'cross_attention'):
        super().__init__()
        
        if not ATTENTION_AVAILABLE:
            raise ImportError("attention.py required for DynamicTransformerBackbone")
        
        self.text_dim = text_dim
        self.clap_dim = clap_dim
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        
        # Enhanced fusion with cross-attention
        if fusion_type == 'cross_attention':
            if text_dim is not None:
                self.cross_modal_fusion = CrossModalFusionBlock(
                    text_dim=text_dim,
                    clap_dim=clap_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
                input_dim = hidden_dim  # Output of cross-modal fusion
            else:
                self.cross_modal_fusion = None
                input_dim = None
        elif fusion_type == 'concat':
            if text_dim is not None:
                input_dim = text_dim + clap_dim
            else:
                input_dim = None  # Runtime determination
        elif fusion_type == 'add':
            if text_dim is not None and text_dim != clap_dim:
                self.text_proj = nn.Linear(text_dim, clap_dim)
                self.clap_proj = nn.Linear(clap_dim, clap_dim)
            input_dim = clap_dim
        else:
            input_dim = clap_dim
        
        # Input projection (dynamic if needed)
        if input_dim is not None:
            if fusion_type == 'cross_attention':
                # No additional projection needed, fusion outputs hidden_dim
                self.input_proj = nn.Identity()
            else:
                self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = None
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Multi-modal transformer layers
        self.transformer_layers = nn.ModuleList([
            BasicTransformerBlock(
                dim=hidden_dim,
                n_heads=num_heads,
                d_head=dim_head,
                dropout=dropout,
                context_dim=None,
                gated_ff=True,
                checkpoint=False
            ) for _ in range(num_layers)
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim
        
        print(f"✅ DynamicTransformerBackbone initialized:")
        print(f"   Text dim: {text_dim}")
        print(f"   CLAP dim: {clap_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Fusion type: {fusion_type}")
        if fusion_type == 'cross_attention':
            print(f"   🎯 Using cross-attention transformer fusion!")
    
    def _build_cross_modal_fusion(self, text_dim: int):
        """Dynamically build cross-modal fusion block"""
        self.cross_modal_fusion = CrossModalFusionBlock(
            text_dim=text_dim,
            clap_dim=self.clap_dim,
            hidden_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        if torch.cuda.is_available():
            self.cross_modal_fusion = self.cross_modal_fusion.cuda()
            
        print(f"🔧 Dynamic cross-modal transformer fusion built: {text_dim} + {self.clap_dim} -> {self.hidden_dim}")
    
    def _build_input_projection(self, input_dim: int):
        """Dynamically build input projection"""
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        print(f"🔧 Dynamic input projection built: {input_dim} -> {self.hidden_dim}")
    
    def forward(self, text_emb: Optional[torch.Tensor] = None, clap_emb: Optional[torch.Tensor] = None):
        """
        Forward pass through dynamic transformer backbone with cross-attention
        
        Args:
            text_emb: Text embeddings (batch_size, text_dim)
            clap_emb: CLAP embeddings (batch_size, clap_dim)
            
        Returns:
            features: Processed features (batch_size, hidden_dim)
        """
        # Handle different input scenarios with enhanced fusion
        if text_emb is not None and clap_emb is not None:
            if self.fusion_type == 'cross_attention':
                # Build cross-modal fusion dynamically if needed
                if self.cross_modal_fusion is None:
                    text_dim = text_emb.shape[-1]
                    self._build_cross_modal_fusion(text_dim)
                
                # Cross-attention fusion for rich multimodal understanding
                combined = self.cross_modal_fusion(text_emb, clap_emb)
                
            elif self.fusion_type == 'concat':
                combined = torch.cat([text_emb, clap_emb], dim=-1)
            elif self.fusion_type == 'add':
                if hasattr(self, 'text_proj'):
                    text_proj = self.text_proj(text_emb)
                    clap_proj = self.clap_proj(clap_emb)
                    combined = text_proj + clap_proj
                else:
                    combined = text_emb + clap_emb
            else:
                combined = torch.cat([text_emb, clap_emb], dim=-1)
        elif text_emb is not None:
            combined = text_emb
        elif clap_emb is not None:
            combined = clap_emb
        else:
            raise ValueError("At least one of text_emb or clap_emb must be provided")
        
        # Build input projection dynamically if needed
        if self.input_proj is None:
            input_dim = combined.shape[-1]
            self._build_input_projection(input_dim)
            if combined.is_cuda:
                self.input_proj = self.input_proj.cuda()
        
        batch_size = combined.shape[0]
        
        # Project to hidden dimension (unless already done by cross-attention)
        if self.fusion_type == 'cross_attention':
            x = combined  # Already in hidden_dim from cross-modal fusion
        else:
            x = self.input_proj(combined)  # (batch_size, hidden_dim)
        
        # Add positional embedding and expand to sequence
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        x = x + self.pos_embedding
        
        # Pass through transformer layers with enhanced representations
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
        
        # Remove sequence dimension and normalize
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        x = self.output_norm(x)
        
        return x


class TransformerBackbone(DynamicTransformerBackbone):
    """
    TransformerBackbone은 이제 DynamicTransformerBackbone의 alias
    하위 호환성을 위해 유지
    """
    
    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.1,
                 use_layer_scale: bool = True,
                 layer_scale_init: float = 0.1):
        
        # DynamicTransformerBackbone 초기화
        super().__init__(
            text_dim=input_dim,
            clap_dim=0,  # CLAP 사용 안함
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            fusion_type='concat'
        )
        
        # 하위 호환성을 위한 속성
        self.input_dim = input_dim
    
    def forward(self, x):
        """하위 호환성을 위한 forward"""
        return super().forward(text_emb=x, clap_emb=None)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            BasicTransformerBlock(
                dim=hidden_dim,
                n_heads=num_heads,
                d_head=dim_head,
                dropout=dropout,
                context_dim=None,  # Self-attention only
                gated_ff=True,
                checkpoint=False
            ) for _ in range(num_layers)
        ])
        
        # Layer scale for training stability (from ConvNext/ResMLP)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scales = nn.ParameterList([
                nn.Parameter(torch.ones(hidden_dim) * layer_scale_init)
                for _ in range(num_layers)
            ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        self.output_dim = hidden_dim
        
        print(f"✅ TransformerBackbone initialized:")
        print(f"   Input dim: {input_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Layers: {num_layers}")
        print(f"   Heads: {num_heads}")
        print(f"   Head dim: {dim_head}")
        print(f"   Layer Scale: {use_layer_scale}")
    
    def forward(self, x):
        """
        Forward pass through transformer backbone
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            features: Processed features (batch_size, hidden_dim)
        """
        batch_size = x.shape[0]
        
        # Project to hidden dimension
        x = self.input_proj(x)  # (batch_size, hidden_dim)
        
        # Add positional embedding and expand to sequence
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        x = x + self.pos_embedding
        
        # Pass through transformer layers
        for i, layer in enumerate(self.transformer_layers):
            residual = x
            x = layer(x)  # Self-attention + FFN
            
            # Apply layer scale if enabled
            if self.use_layer_scale:
                x = residual + self.layer_scales[i] * (x - residual)
        
        # Remove sequence dimension and normalize
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        x = self.output_norm(x)
        
        return x


class HierarchicalTransformerBackbone(nn.Module):
    """
    Hierarchical Transformer with multiple scales of processing
    
    Processes embeddings at different resolutions and combines them,
    inspired by Swin Transformer and ConvNext architectures.
    """
    
    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dims: List[int] = [256, 512, 768],
                 num_layers_per_stage: List[int] = [2, 2, 2],
                 num_heads: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        if not ATTENTION_AVAILABLE:
            raise ImportError("attention.py required for HierarchicalTransformerBackbone")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_stages = len(hidden_dims)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Multi-stage processing
        self.stages = nn.ModuleList()
        
        for stage_idx in range(self.num_stages):
            stage_dim = hidden_dims[stage_idx]
            num_layers = num_layers_per_stage[stage_idx]
            
            # Stage layers
            stage_layers = nn.ModuleList([
                BasicTransformerBlock(
                    dim=stage_dim,
                    n_heads=num_heads,
                    d_head=dim_head,
                    dropout=dropout,
                    context_dim=None,
                    gated_ff=True,
                    checkpoint=False
                ) for _ in range(num_layers)
            ])
            
            # Downsampling between stages
            if stage_idx < self.num_stages - 1:
                downsample = nn.Linear(stage_dim, hidden_dims[stage_idx + 1])
            else:
                downsample = None
            
            self.stages.append(nn.ModuleDict({
                'layers': stage_layers,
                'downsample': downsample,
                'norm': nn.LayerNorm(stage_dim)
            }))
        
        # Global fusion
        total_dim = sum(hidden_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = hidden_dims[-1]
        
        print(f"✅ HierarchicalTransformerBackbone initialized:")
        print(f"   Input dim: {input_dim}")
        print(f"   Hidden dims: {hidden_dims}")
        print(f"   Layers per stage: {num_layers_per_stage}")
        print(f"   Output dim: {self.output_dim}")
    
    def forward(self, x):
        """
        Forward pass through hierarchical transformer
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            features: Multi-scale fused features (batch_size, output_dim)
        """
        # Project input
        x = self.input_proj(x).unsqueeze(1)  # (batch_size, 1, hidden_dims[0])
        
        stage_outputs = []
        
        # Process through stages
        for stage_idx, stage in enumerate(self.stages):
            # Apply transformer layers
            for layer in stage['layers']:
                x = layer(x)
            
            # Normalize and store output
            x_normed = stage['norm'](x)
            stage_outputs.append(x_normed.squeeze(1))  # Remove sequence dim
            
            # Downsample for next stage
            if stage['downsample'] is not None:
                x = stage['downsample'](x)
        
        # Fuse multi-scale features
        fused = torch.cat(stage_outputs, dim=-1)
        output = self.fusion(fused)
        
        return output


class CrossModalTransformerBackbone(nn.Module):
    """
    Cross-modal Transformer for conditioning on multiple modalities
    
    Can handle both text embeddings and audio features with cross-attention.
    Useful for more sophisticated audio processing guided by text.
    """
    
    def __init__(self,
                 text_dim: int = 1024,
                 audio_dim: Optional[int] = None,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        if not ATTENTION_AVAILABLE:
            raise ImportError("attention.py required for CrossModalTransformerBackbone")
        
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.use_audio = audio_dim is not None
        
        # Input projections
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        if self.use_audio:
            self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Cross-modal transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': BasicTransformerBlock(
                    dim=hidden_dim,
                    n_heads=num_heads,
                    d_head=dim_head,
                    dropout=dropout,
                    context_dim=None,  # Self-attention
                    gated_ff=True,
                    checkpoint=False
                ),
                'cross_attn': BasicTransformerBlock(
                    dim=hidden_dim,
                    n_heads=num_heads,
                    d_head=dim_head,
                    dropout=dropout,
                    context_dim=hidden_dim if self.use_audio else None,
                    gated_ff=True,
                    checkpoint=False
                ) if self.use_audio else None
            }) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = hidden_dim
        
        print(f"✅ CrossModalTransformerBackbone initialized:")
        print(f"   Text dim: {text_dim}")
        print(f"   Audio dim: {audio_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Layers: {num_layers}")
        print(f"   Use audio: {self.use_audio}")
    
    def forward(self, text_emb, audio_emb=None):
        """
        Forward pass through cross-modal transformer
        
        Args:
            text_emb: Text embeddings (batch_size, text_dim)
            audio_emb: Audio embeddings (batch_size, audio_dim), optional
            
        Returns:
            features: Cross-modal features (batch_size, hidden_dim)
        """
        # Project inputs
        text_feat = self.text_proj(text_emb).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        if self.use_audio and audio_emb is not None:
            audio_feat = self.audio_proj(audio_emb).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        else:
            audio_feat = None
        
        # Process through transformer layers
        x = text_feat
        
        for layer_dict in self.transformer_layers:
            # Self-attention on text features
            x = layer_dict['self_attn'](x)
            
            # Cross-attention with audio features (if available)
            if audio_feat is not None and layer_dict['cross_attn'] is not None:
                x = layer_dict['cross_attn'](x, context=audio_feat)
        
        # Output projection
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.output_proj(x)
        
        return x


class PerceiverBackbone(nn.Module):
    """
    Perceiver-inspired backbone for handling variable-length inputs
    
    Uses cross-attention to map from input embeddings to a fixed set of latents,
    then processes latents with self-attention. Good for handling different
    embedding sizes and capturing long-range dependencies.
    """
    
    def __init__(self,
                 input_dim: int = 1024,
                 latent_dim: int = 512,
                 num_latents: int = 64,
                 num_cross_layers: int = 2,
                 num_self_layers: int = 6,
                 num_heads: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        if not ATTENTION_AVAILABLE:
            raise ImportError("attention.py required for PerceiverBackbone")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        
        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)
        
        # Cross-attention layers (input -> latents)
        self.cross_attention_layers = nn.ModuleList([
            BasicTransformerBlock(
                dim=latent_dim,
                n_heads=num_heads,
                d_head=dim_head,
                dropout=dropout,
                context_dim=latent_dim,  # Cross-attention
                gated_ff=True,
                checkpoint=False
            ) for _ in range(num_cross_layers)
        ])
        
        # Self-attention layers (latents -> latents)
        self.self_attention_layers = nn.ModuleList([
            BasicTransformerBlock(
                dim=latent_dim,
                n_heads=num_heads,
                d_head=dim_head,
                dropout=dropout,
                context_dim=None,  # Self-attention
                gated_ff=True,
                checkpoint=False
            ) for _ in range(num_self_layers)
        ])
        
        # Output pooling and projection
        self.output_pooling = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = latent_dim
        
        print(f"✅ PerceiverBackbone initialized:")
        print(f"   Input dim: {input_dim}")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Num latents: {num_latents}")
        print(f"   Cross layers: {num_cross_layers}")
        print(f"   Self layers: {num_self_layers}")
    
    def forward(self, x):
        """
        Forward pass through Perceiver backbone
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            features: Processed features (batch_size, latent_dim)
        """
        batch_size = x.shape[0]
        
        # Project input and expand to sequence
        x_proj = self.input_proj(x).unsqueeze(1)  # (batch_size, 1, latent_dim)
        
        # Initialize latents for this batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_latents, latent_dim)
        
        # Cross-attention: input -> latents
        for cross_layer in self.cross_attention_layers:
            latents = cross_layer(latents, context=x_proj)
        
        # Self-attention: latents -> latents
        for self_layer in self.self_attention_layers:
            latents = self_layer(latents)
        
        # Pool and project output
        # Global average pooling over latent dimension
        latents_pooled = latents.mean(dim=1)  # (batch_size, latent_dim)
        output = self.output_proj(latents_pooled)
        
        return output


class ResidualBackbone(nn.Module):
    """
    Backbone with residual connections for better gradient flow
    
    Uses residual connections similar to ResNet to enable deeper networks
    and better gradient flow during training.
    """
    
    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dim: int = 512,
                 num_blocks: int = 4,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout_rate)
            )
            self.residual_blocks.append(block)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        self.output_dim = hidden_dim
        
        print(f"✅ ResidualBackbone initialized:")
        print(f"   Input dim: {input_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Num blocks: {num_blocks}")
        print(f"   Output dim: {hidden_dim}")
    
    def forward(self, x):
        """
        Forward pass through residual backbone
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            features: Processed features with residual connections
        """
        # Input projection
        x = self.input_proj(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
        
        # Output normalization
        x = self.output_norm(x)
        
        return x


def create_backbone(backbone_type: str = 'simple', 
                   text_dim: Optional[int] = None,
                   clap_dim: int = 512,
                   fusion_type: str = 'cross_attention',  # Default to cross-attention
                   **kwargs):
    """
    Factory function to create different backbone types with cross-attention fusion
    
    Args:
        backbone_type: 'simple', 'dynamic', 'transformer', 'dynamic_transformer', 
                      'residual', 'hierarchical', 'crossmodal', 'perceiver'
        text_dim: Text embedding dimension (can be None for auto-detection)
        clap_dim: CLAP embedding dimension (default 512)
        fusion_type: How to combine embeddings ('concat', 'add', 'cross_attention')
                    Defaults to 'cross_attention' for richer multimodal understanding
        **kwargs: Backbone-specific parameters
        
    Returns:
        Backbone model instance with enhanced cross-attention capabilities
    """
    backbone_type = backbone_type.lower()
    
    print(f"🎯 Creating {backbone_type} backbone with {fusion_type} fusion")
    
    if backbone_type == 'simple':
        # Use DynamicBackbone with cross-attention for simple case
        return DynamicBackbone(
            text_dim=text_dim,
            clap_dim=clap_dim,
            fusion_type=fusion_type,
            **kwargs
        )
    elif backbone_type == 'dynamic':
        return DynamicBackbone(
            text_dim=text_dim,
            clap_dim=clap_dim,
            fusion_type=fusion_type,
            **kwargs
        )
    elif backbone_type == 'transformer':
        # Use DynamicTransformerBackbone with cross-attention
        return DynamicTransformerBackbone(
            text_dim=text_dim,
            clap_dim=clap_dim,
            fusion_type=fusion_type,
            **kwargs
        )
    elif backbone_type == 'dynamic_transformer':
        return DynamicTransformerBackbone(
            text_dim=text_dim,
            clap_dim=clap_dim,
            fusion_type=fusion_type,
            **kwargs
        )
    elif backbone_type == 'residual':
        # Use DynamicBackbone with residual-style hidden layers
        if 'hidden_dims' not in kwargs:
            # Set default residual-style dimensions
            kwargs['hidden_dims'] = [512, 512, 256, 128]
        return DynamicBackbone(
            text_dim=text_dim,
            clap_dim=clap_dim,
            fusion_type=fusion_type,
            **kwargs
        )
            
    elif backbone_type == 'crossmodal':
        return CrossModalTransformerBackbone(
            text_dim=text_dim or 1024,
            audio_dim=clap_dim,
            **kwargs
        )
    elif backbone_type == 'perceiver':
        if text_dim is not None:
            if fusion_type == 'cross_attention':
                input_dim = 512  # Cross-attention output
            elif fusion_type == 'concat':
                input_dim = text_dim + clap_dim
            else:
                input_dim = text_dim
        else:
            input_dim = kwargs.get('input_dim', 1024)
            
        if fusion_type == 'cross_attention' and text_dim is not None:
            # Create combined cross-attention + perceiver backbone
            class CrossAttentionPerceiverBackbone(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.cross_modal_fusion = CrossModalFusionBlock(
                        text_dim=text_dim,
                        clap_dim=clap_dim,
                        hidden_dim=512,
                        num_heads=8,
                        dropout=kwargs.get('dropout', 0.1)
                    )
                    self.perceiver_backbone = PerceiverBackbone(input_dim=512, **kwargs)
                    self.output_dim = self.perceiver_backbone.output_dim
                
                def forward(self, text_emb=None, clap_emb=None):
                    if text_emb is not None and clap_emb is not None:
                        fused = self.cross_modal_fusion(text_emb, clap_emb)
                    elif text_emb is not None:
                        fused = text_emb
                    else:
                        fused = clap_emb
                    return self.perceiver_backbone(fused)
            
            return CrossAttentionPerceiverBackbone()
        else:
            return PerceiverBackbone(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}. "
                        f"Available: simple, dynamic, transformer, dynamic_transformer, "
                        f"residual, crossmodal, perceiver")


def test_backbones():
    """Test different backbone architectures"""
    
    print("🧪 TESTING BACKBONE ARCHITECTURES")
    print("=" * 50)
    
    input_dim = 1024
    batch_size = 4
    
    # Test data
    dummy_input = torch.randn(batch_size, input_dim)
    
    print(f"\n📊 Test input shape: {dummy_input.shape}")
    
    # Test different backbone types
    backbone_configs = {
        'Simple Backbone': {
            'type': 'simple',
            'kwargs': {
                'input_dim': input_dim,
                'hidden_dims': [512, 256, 128],
                'dropout_rate': 0.2,
                'activation': 'relu'
            }
        },
        'Multi-Scale Backbone': {
            'type': 'multiscale',
            'kwargs': {
                'input_dim': input_dim,
                'scales': [256, 512, 768],
                'final_dim': 128
            }
        },
        'Residual Backbone': {
            'type': 'residual',
            'kwargs': {
                'input_dim': input_dim,
                'hidden_dim': 512,
                'num_blocks': 4,
                'dropout_rate': 0.2
            }
        }
    }
    
    for name, config in backbone_configs.items():
        print(f"\n🔧 Testing {name}:")
        
        # Create backbone
        backbone = create_backbone(config['type'], **config['kwargs'])
        
        # Forward pass
        with torch.no_grad():
            output = backbone(dummy_input)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Output dim: {backbone.output_dim}")
        
        # Check output statistics
        mean_val = output.mean().item()
        std_val = output.std().item()
        print(f"   Output stats: mean={mean_val:.3f}, std={std_val:.3f}")


def analyze_backbone_capacity():
    """Analyze the representational capacity of different backbones"""
    
    print("\n📈 BACKBONE CAPACITY ANALYSIS")
    print("=" * 40)
    
    input_dim = 1024
    
    backbones = {
        'Simple (3-layer)': create_backbone('simple', input_dim=input_dim, hidden_dims=[512, 256, 128]),
        'Simple (5-layer)': create_backbone('simple', input_dim=input_dim, hidden_dims=[512, 384, 256, 192, 128]),
        'Multi-Scale': create_backbone('multiscale', input_dim=input_dim, scales=[256, 512, 768], final_dim=128),
        'Residual (4-block)': create_backbone('residual', input_dim=input_dim, hidden_dim=512, num_blocks=4)
    }
    
    for name, backbone in backbones.items():
        # Count parameters
        num_params = sum(p.numel() for p in backbone.parameters())
        
        print(f"{name:20}: {num_params:,} parameters, output_dim={backbone.output_dim}")


if __name__ == "__main__":
    print("🏗️ BACKBONE MODEL ARCHITECTURES")
    print("=" * 50)
    
    print("\n🎯 Purpose: Shared backbone for processing embeddings")
    print("   Input: Text embeddings (E5-large: 1024D)")
    print("   Output: Processed features for decoder heads")
    
    print("\n🏗️ Available Architectures:")
    print("   1. Simple Backbone: Standard multi-layer network")
    print("   2. Multi-Scale Backbone: Processes at multiple scales")
    print("   3. Residual Backbone: ResNet-style with skip connections")
    
    # Test all backbone types
    test_backbones()
    
    # Analyze capacity
    analyze_backbone_capacity()
    
    print(f"\n💡 Recommendations:")
    print(f"   • Simple Backbone: Good default choice, efficient")
    print(f"   • Multi-Scale: Better for complex text understanding")
    print(f"   • Residual: Best for very deep networks")
    print(f"   • Use with pedalboard_decoder.py for complete system")

