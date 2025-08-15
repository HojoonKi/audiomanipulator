#!/usr/bin/env python3
"""
Discriminator for Adversarial Training

이 모듈은 생성된 preset 파라미터와 실제 GT 파라미터를 구별하는 
Discriminator 네트워크를 구현합니다.

목적: 모드 붕괴(Mode Collapse) 방지 및 다양성 증진
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PresetDiscriminator(nn.Module):
    """
    Preset 파라미터(27개)를 입력으로 받아 Real/Fake를 판별하는 Discriminator
    
    Architecture:
    - Input: 27개 preset 파라미터 (normalized)
    - Output: Real/Fake probability (0~1)
    - 작고 빠른 네트워크로 설계 (Generator보다 학습 속도 조절 가능)
    """
    
    def __init__(self, input_dim=27, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(PresetDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 입력 정규화 레이어
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # 히든 레이어들 구성
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (첫 번째 레이어 제외)
            if i > 0:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 최종 출력 레이어
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 0~1 확률 출력
        
        self.discriminator = nn.Sequential(*layers)
        
        # 가중치 초기화
        self._initialize_weights()
        
        print(f"🎯 Discriminator 생성됨:")
        print(f"   - 입력 차원: {input_dim}")
        print(f"   - 히든 레이어: {hidden_dims}")
        print(f"   - Dropout: {dropout_rate}")
        print(f"   - 총 파라미터: {sum(p.numel() for p in self.parameters()):,}")
    
    def _initialize_weights(self):
        """가중치 초기화 - Discriminator에 적합한 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier 초기화 (더 안정적인 학습)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, preset_params):
        """
        Args:
            preset_params: [batch_size, 27] - 정규화된 preset 파라미터
            
        Returns:
            torch.Tensor: [batch_size, 1] - Real 확률 (0~1)
        """
        # 입력 차원 확인
        if preset_params.dim() == 1:
            preset_params = preset_params.unsqueeze(0)  # [27] -> [1, 27]
        
        batch_size = preset_params.size(0)
        
        # 입력 정규화
        if batch_size > 1:  # BatchNorm은 배치 크기가 1보다 커야 함
            x = self.input_norm(preset_params)
        else:
            x = preset_params
        
        # Discriminator 통과
        output = self.discriminator(x)
        
        return output
    
    def compute_adversarial_loss(self, real_params, fake_params, label_smoothing=0.1):
        """
        Discriminator를 위한 적대적 손실 계산
        
        Args:
            real_params: [batch_size, 27] - 실제 GT 파라미터
            fake_params: [batch_size, 27] - 생성된 파라미터 (detached)
            label_smoothing: float - 레이블 스무딩 강도
            
        Returns:
            torch.Tensor: Discriminator loss
        """
        device = real_params.device
        batch_size = real_params.size(0)
        
        # Real 데이터에 대한 예측
        real_pred = self.forward(real_params)
        
        # Fake 데이터에 대한 예측 (gradient 차단)
        fake_pred = self.forward(fake_params.detach())
        
        # 레이블 생성 (레이블 스무딩 적용)
        real_labels = torch.ones(batch_size, 1, device=device) - label_smoothing
        fake_labels = torch.zeros(batch_size, 1, device=device) + label_smoothing
        
        # Binary Cross Entropy Loss
        real_loss = F.binary_cross_entropy(real_pred, real_labels)
        fake_loss = F.binary_cross_entropy(fake_pred, fake_labels)
        
        # 총 Discriminator 손실
        discriminator_loss = (real_loss + fake_loss) / 2
        
        return discriminator_loss, real_pred, fake_pred
    
    def compute_generator_adversarial_loss(self, fake_params):
        """
        Generator를 위한 적대적 손실 계산
        
        Args:
            fake_params: [batch_size, 27] - 생성된 파라미터 (gradient 유지)
            
        Returns:
            torch.Tensor: Generator adversarial loss
        """
        device = fake_params.device
        batch_size = fake_params.size(0)
        
        # Generator가 생성한 파라미터에 대한 Discriminator 예측
        fake_pred = self.forward(fake_params)
        
        # Generator는 Discriminator를 속이려고 함 (Real로 분류되길 원함)
        real_labels = torch.ones(batch_size, 1, device=device)
        
        # Generator adversarial loss
        generator_adversarial_loss = F.binary_cross_entropy(fake_pred, real_labels)
        
        return generator_adversarial_loss
    
    def get_discrimination_accuracy(self, real_params, fake_params):
        """
        Discriminator의 판별 정확도 계산 (모니터링용)
        
        Returns:
            dict: {'real_acc': float, 'fake_acc': float, 'total_acc': float}
        """
        with torch.no_grad():
            real_pred = self.forward(real_params)
            fake_pred = self.forward(fake_params.detach())
            
            # 0.5를 기준으로 분류
            real_correct = (real_pred > 0.5).float().mean().item()
            fake_correct = (fake_pred <= 0.5).float().mean().item()
            total_correct = (real_correct + fake_correct) / 2
            
            return {
                'real_acc': real_correct,
                'fake_acc': fake_correct, 
                'total_acc': total_correct
            }


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss - GAN 학습 안정화를 위한 추가 손실
    
    Discriminator의 중간 feature를 매칭하여 더 안정적인 학습 유도
    """
    
    def __init__(self, discriminator):
        super(FeatureMatchingLoss, self).__init__()
        self.discriminator = discriminator
        
        # Discriminator의 중간 레이어에서 feature 추출을 위한 hook 등록
        self.features_real = []
        self.features_fake = []
        self._register_hooks()
    
    def _register_hooks(self):
        """중간 레이어에서 feature를 추출하기 위한 hook 등록"""
        def hook_fn_real(module, input, output):
            self.features_real.append(output)
        
        def hook_fn_fake(module, input, output):
            self.features_fake.append(output)
        
        # Discriminator의 중간 레이어들에 hook 등록
        layers = list(self.discriminator.discriminator.children())
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear) and i < len(layers) - 2:  # 마지막 출력 레이어 제외
                layer.register_forward_hook(hook_fn_real)
                layer.register_forward_hook(hook_fn_fake)
    
    def forward(self, real_params, fake_params):
        """
        Feature Matching Loss 계산
        
        Args:
            real_params: 실제 파라미터
            fake_params: 생성된 파라미터
            
        Returns:
            torch.Tensor: Feature matching loss
        """
        # Feature 리스트 초기화
        self.features_real.clear()
        self.features_fake.clear()
        
        # Discriminator 통과하여 feature 추출
        with torch.no_grad():
            _ = self.discriminator(real_params)
        _ = self.discriminator(fake_params)
        
        # Feature 매칭 손실 계산
        fm_loss = 0
        for feat_real, feat_fake in zip(self.features_real, self.features_fake):
            fm_loss += F.mse_loss(feat_fake, feat_real.detach())
        
        return fm_loss / len(self.features_real) if self.features_real else torch.tensor(0.0)


def create_discriminator(config=None):
    """
    Discriminator 생성 팩토리 함수
    
    Args:
        config: dict - Discriminator 설정
        
    Returns:
        PresetDiscriminator: 생성된 discriminator
    """
    if config is None:
        config = {
            'input_dim': 27,
            'hidden_dims': [128, 64, 32],
            'dropout_rate': 0.3
        }
    
    discriminator = PresetDiscriminator(**config)
    
    return discriminator


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Discriminator 테스트...")
    
    # Discriminator 생성
    discriminator = create_discriminator()
    
    # 더미 데이터 생성
    batch_size = 8
    real_params = torch.randn(batch_size, 28)
    fake_params = torch.randn(batch_size, 28)
    
    # Forward pass 테스트
    real_pred = discriminator(real_params)
    fake_pred = discriminator(fake_params)
    
    print(f"✅ Real prediction shape: {real_pred.shape}")
    print(f"✅ Fake prediction shape: {fake_pred.shape}")
    print(f"✅ Real prediction range: [{real_pred.min().item():.3f}, {real_pred.max().item():.3f}]")
    print(f"✅ Fake prediction range: [{fake_pred.min().item():.3f}, {fake_pred.max().item():.3f}]")
    
    # Loss 계산 테스트
    disc_loss, _, _ = discriminator.compute_adversarial_loss(real_params, fake_params)
    gen_loss = discriminator.compute_generator_adversarial_loss(fake_params)
    
    print(f"✅ Discriminator loss: {disc_loss.item():.4f}")
    print(f"✅ Generator adversarial loss: {gen_loss.item():.4f}")
    
    # 정확도 계산 테스트
    accuracy = discriminator.get_discrimination_accuracy(real_params, fake_params)
    print(f"✅ Discrimination accuracy: {accuracy}")
    
    print("🎯 Discriminator 테스트 완료!")
