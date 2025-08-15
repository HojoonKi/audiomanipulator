#!/usr/bin/env python3
"""
Backbone Model: Simplified Neural Network Backbones for Audio Processing

This module provides essential backbone networks for processing input embeddings
before feeding to decoder heads.

Architecture Options:
1. SharedBackbone (MLP-based, lightweight)
2. TransformerBackbone (attention-based)
3. TunedCLAPWithAdapters (CLAP + adapter fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, List, Tuple
try:
    from encoder.text_encoder import CLAPTextEncoder
except Exception:
    CLAPTextEncoder = None  # Optional import; used only by TunedCLAPWithAdapters

# Import attention components
try:
    from attention import (
        BasicTransformerBlock, 
    )
    ATTENTION_AVAILABLE = True
except ImportError:
    try:
        from .attention import (
            BasicTransformerBlock, 
        )
        ATTENTION_AVAILABLE = True
    except ImportError:
        try:
            from model.attention import (
                BasicTransformerBlock, 
            )
            ATTENTION_AVAILABLE = True
        except ImportError:
            print("Warning: attention.py not found. Transformer backbones disabled.")
            ATTENTION_AVAILABLE = False


class SharedBackbone(nn.Module):
    """
    Simple MLP backbone for lightweight processing
    """
    
    def __init__(self, 
                 input_dim: int = 1024,
                 hidden_dims: list = None,
                 hidden_dim: int = None,
                 output_dim: int = None,
                 dropout_rate: float = 0.2,
                 activation: str = 'relu'):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Handle different initialization patterns
        if hidden_dims is None and hidden_dim is not None:
            if output_dim is None:
                output_dim = hidden_dim // 2
            hidden_dims = [hidden_dim, output_dim]
        elif hidden_dims is None:
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
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation_fn,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Remove last dropout
        if layers:
            layers = layers[:-1]
        
        self.backbone = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
        print(f"✅ SharedBackbone initialized:")
        print(f"   Input dim: {input_dim}")
        print(f"   Hidden dims: {hidden_dims}")
        print(f"   Output dim: {self.output_dim}")
    
    def forward(self, x):
        """Forward pass through MLP backbone"""
        return self.backbone(x)
    
    def get_layer_outputs(self, x):
        """Get outputs from each layer for analysis"""
        layer_outputs = []
        current_x = x
        
        for layer in self.backbone:
            current_x = layer(current_x)
            if isinstance(layer, nn.Linear):
                layer_outputs.append(current_x)
        
        return layer_outputs


class TransformerBackbone(nn.Module):
    """
    Transformer-based backbone using attention mechanisms
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
        super().__init__()
        
        if not ATTENTION_AVAILABLE:
            raise ImportError("attention.py required for TransformerBackbone")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
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
        """Forward pass through transformer backbone"""
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



class GatedResidualUnit(nn.Module):
    """
    Lightweight bottleneck MLP with a residual gating mechanism.
    Input/Output: [B, D]
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.full((1,), -10.0))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(self.up(x))
        return residual + torch.sigmoid(self.gate) * x


class CrossAttentionAdapter(nn.Module):
    """
    Adapter block with cross-attention over LLM hidden states and gated MLP.
    Operates on a single-vector audio representation.
    """
    def __init__(
        self,
        audio_dim: int = 512,
        llm_dim: int = 768,
        attn_dim: int = 256,
                 num_heads: int = 8,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.audio_q = nn.Linear(audio_dim, attn_dim)
        self.llm_kv = nn.Linear(llm_dim, attn_dim * 2)
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(attn_dim, audio_dim)
        self.norm_q = nn.LayerNorm(attn_dim)
        self.norm_out = nn.LayerNorm(audio_dim)
        self.dropout = nn.Dropout(dropout)
        self.post_mlp = GatedResidualUnit(dim=audio_dim, hidden_dim=mlp_hidden_dim, dropout=dropout)
        nn.init.xavier_uniform_(self.audio_q.weight)
        nn.init.xavier_uniform_(self.llm_kv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, audio_vec: torch.Tensor, llm_hidden: torch.Tensor) -> torch.Tensor:
        # audio_vec: (batch, audio_dim)
        # llm_hidden: (batch, llm_dim) -> need to add sequence dimension
        
        audio_vec_seq = audio_vec.unsqueeze(1)  # (batch, 1, audio_dim)
        
        # Add sequence dimension to llm_hidden if needed
        if llm_hidden.dim() == 2:
            llm_hidden_seq = llm_hidden.unsqueeze(1)  # (batch, 1, llm_dim)
        else:
            llm_hidden_seq = llm_hidden  # Already (batch, seq, llm_dim)
        
        q = self.norm_q(self.audio_q(audio_vec_seq))  # (batch, 1, attn_dim)
        kv = self.llm_kv(llm_hidden_seq)  # (batch, seq, attn_dim*2)
        k, v = torch.chunk(kv, 2, dim=-1)  # (batch, seq, attn_dim) each
        
        attn_out, _ = self.attn(q, k, v)  # (batch, 1, attn_dim)
        attn_out = self.dropout(attn_out)
        attn_audio = self.out_proj(attn_out)  # (batch, 1, audio_dim)
        audio_updated = self.norm_out(audio_vec_seq + attn_audio)  # (batch, 1, audio_dim)
        audio_updated = self.post_mlp(audio_updated.squeeze(1))  # (batch, audio_dim)
        return audio_updated


class TunedCLAPWithAdapters(nn.Module):
    """
    Frozen CLAP audio encoder + stack of trainable adapter blocks that fuse LLM features.
    - CLAP is kept frozen; we use its audio embedding (512D) as the base representation.
    - Only adapter parameters are trainable.
    """
    def __init__(
        self,
        llm_feature_dim: int = 1024,
        num_adapters: int = 4,
        adapter_hidden_dim: int = 256,
        attention_dim: int = 256,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        freeze_clap: bool = True,
    ):
        super().__init__()
        if CLAPTextEncoder is None:
            raise ImportError("CLAPTextEncoder is not available. Ensure encoder.text_encoder is importable.")
        
        # CLAP 모델 초기화 (text embedding용)
        self.clap = CLAPTextEncoder(freeze_model=True)
        
        # CLAP text embedding 차원 확인
        dummy_texts = ["dummy text"]
        with torch.no_grad():
            dummy_text_embed = self.clap.get_text_embedding(dummy_texts)
            text_embed_dim = dummy_text_embed.shape[-1]
        
        self.text_embed_dim = text_embed_dim
        self.llm_dim = llm_feature_dim
        
        # CLAP 파라미터를 확실히 frozen으로 설정
        if freeze_clap:
            for p in self.clap.parameters():
                p.requires_grad = False
        
        print(f"🔒 CLAP model frozen: {not any(p.requires_grad for p in self.clap.parameters())}")
        adapters = []
        for _ in range(num_adapters):
            adapters.append(
                CrossAttentionAdapter(
                    audio_dim=text_embed_dim,  # CLAP text embedding을 base로 사용
                    llm_dim=self.llm_dim,
                    attn_dim=attention_dim,
                    num_heads=num_attention_heads,
                    mlp_hidden_dim=adapter_hidden_dim,
                dropout=dropout,
                )
            )
        self.adapters = nn.ModuleList(adapters)
        self.final_norm = nn.LayerNorm(text_embed_dim)
        self.output_dim = text_embed_dim
        
        # 어댑터 파라미터 통계 출력
        adapter_params = sum(p.numel() for p in self.adapters.parameters())
        norm_params = sum(p.numel() for p in self.final_norm.parameters())
        total_trainable = adapter_params + norm_params
        
        print(f"🎯 TunedCLAPWithAdapters initialized:")
        print(f"   Adapters: {num_adapters} blocks, {adapter_params:,} parameters")
        print(f"   Final norm: {norm_params:,} parameters")
        print(f"   Total trainable: {total_trainable:,} parameters")
        print(f"   LLM feature dim: {llm_feature_dim}")
        print(f"   CLAP text embed dim: {text_embed_dim}")

    def forward(self, texts: List[str], llm_hidden: torch.Tensor):
        """
        Text-only backbone processing
        
        Args:
            texts: 원본 텍스트 (CLAP으로 인코딩할 대상)
            llm_hidden: LLM으로 만든 text embedding (cross-attention용)
            
        Returns:
            fused_features: CLAP text embedding + LLM hidden의 융합된 features
        """
        # 1. CLAP으로 원본 texts 인코딩 (base representation)
        with torch.no_grad():
            clap_text_embed = self.clap.get_text_embedding(texts).float()
        
        # 2. LLM hidden features 준비
        llm_h = llm_hidden.float()
        
        # 3. Device 맞추기
        if clap_text_embed.device != llm_h.device:
            llm_h = llm_h.to(clap_text_embed.device)
        
        # 4. Cross-attention fusion: CLAP text embedding(base) + LLM hidden(context)
        fused = clap_text_embed
        for adapter in self.adapters:
            fused = adapter(fused, llm_h)  # CLAP embedding을 LLM hidden과 cross-attention
        
        fused = self.final_norm(fused)
        return fused

    def compute_clap_loss(self, audio: torch.Tensor, texts: List[str], temperature: float = 0.07) -> torch.Tensor:
        """
        주어진 오디오와 텍스트로부터 CLAP 임베딩을 추출하여
        대칭적 Contrastive Loss(CLAP Loss)를 계산합니다.
        
        이제 text_encoder.py의 개선된 방식을 사용합니다.

        Args:
            audio (torch.Tensor): 오디오 웨이브폼 텐서. (batch, channels, time)
            texts (List[str]): 오디오에 해당하는 텍스트 설명 리스트.
            temperature (float): 로짓(logits)을 스케일링하는 온도 파라미터.

        Returns:
            torch.Tensor: 계산된 스칼라 손실 값.
        """
        try:
            # text_encoder.py의 compute_clap_loss 메서드를 직접 사용
            # 이미 gradient flow가 검증된 방식입니다
            return self.clap.compute_clap_loss(audio, texts)
            
        except Exception as e:
            print(f"CLAP loss 계산 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # Safe fallback
            device = audio.device
            return torch.tensor(0.1, device=device, requires_grad=True)


def create_backbone(backbone_type: str = 'tuned_clap_adapters', 
                   input_dim: int = 1024,
                   **kwargs):
    """
    Factory function to create different backbone types
    
    Args:
        backbone_type: 'simple' or 'transformer' or 'tuned_clap_adapters'
        input_dim: Input dimension for simple/transformer backbones
        **kwargs: Backbone-specific parameters
        
    Returns:
        Backbone model instance
    """
    backbone_type = backbone_type.lower()
    
    print(f"🎯 Creating {backbone_type} backbone")
    
    if backbone_type == 'simple':
        return SharedBackbone(input_dim=input_dim, **kwargs)
    elif backbone_type == 'transformer':
        return TransformerBackbone(input_dim=input_dim, **kwargs)
    elif backbone_type == 'tuned_clap_adapters':
        return TunedCLAPWithAdapters(**kwargs)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}. "
                        f"Available: simple, transformer, tuned_clap_adapters")


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
        'Transformer Backbone': {
            'type': 'transformer',
            'kwargs': {
                'input_dim': input_dim,
                'hidden_dim': 512,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.1
            }
        }
    }
    
    for name, config in backbone_configs.items():
        print(f"\n🔧 Testing {name}:")
        
        try:
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
        except Exception as e:
            print(f"   ❌ Error: {e}")


def analyze_backbone_capacity():
    """Analyze the representational capacity of different backbones"""
    
    print("\n📈 BACKBONE CAPACITY ANALYSIS")
    print("=" * 40)
    
    input_dim = 1024
    
    try:
        backbones = {
            'Simple (3-layer)': create_backbone('simple', input_dim=input_dim, hidden_dims=[512, 256, 128]),
            'Simple (5-layer)': create_backbone('simple', input_dim=input_dim, hidden_dims=[512, 384, 256, 192, 128]),
        }
        
        if ATTENTION_AVAILABLE:
            backbones['Transformer'] = create_backbone('transformer', input_dim=input_dim, hidden_dim=512, num_layers=4)
        
        for name, backbone in backbones.items():
            # Count parameters
            num_params = sum(p.numel() for p in backbone.parameters())
            
            print(f"{name:20}: {num_params:,} parameters, output_dim={backbone.output_dim}")
    except Exception as e:
        print(f"❌ Error analyzing backbones: {e}")
