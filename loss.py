#!/usr/bin/env python3
"""
Loss Functions for Audio Manipulation Training

This module contains all loss functions used in the audio manipulation training pipeline:
- CLAP-based self-supervised loss
- Guide preset losses (MSE, L1, normalized)
- Parameter normalization and weighting
- Frequency regularization losses
"""

import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
import os


def _should_debug_loss() -> bool:
    try:
        return os.getenv("DEBUG_LOSS", "0") == "1"
    except Exception:
        return False


def _debug_loss_print(message: str) -> None:
    if _should_debug_loss():
        try:
            print(message)
        except Exception:
            pass


def _tensor_stats_brief(name: str, tensor: torch.Tensor) -> None:
    if not _should_debug_loss():
        return
    try:
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype)
        device = str(tensor.device)
        tmin = float(torch.min(tensor).item()) if tensor.numel() > 0 else float('nan')
        tmax = float(torch.max(tensor).item()) if tensor.numel() > 0 else float('nan')
        tmean = float(torch.mean(tensor).item()) if tensor.numel() > 0 else float('nan')
        _debug_loss_print(f"[LOSS-DEBUG] {name}: shape={shape}, dtype={dtype}, device={device}, min={tmin:.4f}, max={tmax:.4f}, mean={tmean:.4f}")
    except Exception as e:
        _debug_loss_print(f"[LOSS-DEBUG] {name}: stats failed: {e}")


def compute_self_supervised_clap_loss(
    fx_model: torch.nn.Module,         # 훈련시킬 모델 (학생)
    clap_model: torch.nn.Module,        # 고정된 평가자 모델 (교사)
    original_audios: torch.Tensor,    # 원본 오디오 배치
    fx_texts: List[str],              # 적용된 FX에 대한 텍스트 설명 배치
    temperature: float = 0.07
) -> torch.Tensor:
    """
    자기 지도 방식으로 FX 모델을 훈련하기 위한 CLAP Loss를 계산합니다.
    업데이트된 CLAPTextEncoder의 새로운 메서드들을 사용합니다.
    
    Args:
        fx_model: 훈련 대상인 FX 적용 모델 (TunedCLAPPipeline).
        clap_model: 가중치가 고정된(frozen) 사전 훈련된 CLAP 모델.
        original_audios: 원본 오디오 텐서 배치.
        fx_texts: 각 오디오에 적용된 랜덤 FX를 설명하는 텍스트 리스트.
        temperature: Contrastive Loss의 온도 파라미터.
        
    Returns:
        torch.Tensor: FX 모델 업데이트를 위한 스칼라 손실 값.
    """
    device = original_audios.device

    try:
        # 1. 학생(fx_model)이 과제 수행: 텍스트 설명에 맞춰 오디오 처리
        with torch.cuda.amp.autocast(enabled=False):  # 안정성을 위해 FP32 유지
            outputs = fx_model(texts=fx_texts, audio=original_audios, use_real_audio=True)
            predicted_audios = outputs['processed_audio']

        # 2. 오디오 차원 정규화: 최종적으로 (batch, samples)
        x = predicted_audios
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() >= 3:
            # 마지막 축은 samples로 유지하고, 중간 축(채널/불필요 차원)은 평균으로 모노화
            reduce_dims = list(range(1, x.dim() - 1))
            if len(reduce_dims) > 0:
                x = x.mean(dim=reduce_dims)
        predicted_audios_mono = x  # (batch, samples)
        if _should_debug_loss():
            _tensor_stats_brief("predicted_audios_mono.after_reduce", predicted_audios_mono)

        # 텍스트 길이를 배치 크기에 맞춤 (필요 시 반복/자름)
        try:
            B = predicted_audios_mono.size(0)
            if isinstance(fx_texts, list) and len(fx_texts) != B:
                if len(fx_texts) == 0:
                    fx_texts = ["Apply audio effect"] * B
                else:
                    reps = (B + len(fx_texts) - 1) // len(fx_texts)
                    fx_texts = (fx_texts * reps)[:B]
        except Exception:
            pass
        
        # 디버그: 텐서 디바이스/형상 출력
        if _should_debug_loss():
            _tensor_stats_brief("predicted_audios_mono", predicted_audios_mono)
            try:
                ref_param = next(clap_model.parameters())
                _debug_loss_print(f"[LOSS-DEBUG] clap_model.param_device={ref_param.device}")
            except Exception:
                pass

        # 3. 업데이트된 CLAPTextEncoder의 compute_clap_loss 메서드 직접 사용
        # 이 메서드는 내부적으로 gradient flow가 보장된 방식으로 구현되어 있음
        loss = clap_model.compute_clap_loss(predicted_audios_mono, fx_texts)
        
        return loss
        
    except Exception as e:
        print(f"자기지도 CLAP loss 실패: {e}")
        import traceback
        traceback.print_exc()
        return torch.tensor(0.1, device=device, requires_grad=True)


def extract_guide_values(guide_preset):
    """Guide preset에서 28개 값 추출 - fined_presets_filtered.py 구조에 맞춤"""
    values = []
    
    # Filter type 매핑 (문자열 -> 숫자)
    filter_type_mapping = {
        "low-shelf": 0,
        "bell": 1, 
        "high-shelf": 2,
        "highpass": 3,
        "lowpass": 4,
        # 추가 호환성
        "notch": 1,  # notch는 bell과 유사
        "low_shelf": 0,  # 언더스코어 버전
        "high_shelf": 2,
        "low_pass": 4,
        "high_pass": 3
    }
    
    # EQ (20개): frequency, gain, q, filter_type × 5
    eq_section = guide_preset['Equalizer']  # 리스트 형태
    for eq_band in eq_section:  # 5개 밴드
        # filter_type을 숫자로 변환
        filter_type_str = eq_band['filter_type']
        filter_type_num = filter_type_mapping.get(filter_type_str, 1)  # 기본값: bell (1)
        
        values.extend([
            eq_band['frequency'],
            eq_band['gain'], 
            eq_band['q'],
            filter_type_num
        ])
    
    # Reverb (5개)
    reverb = guide_preset['Reverb']
    values.extend([
        reverb['room_size'],
        reverb['pre_delay'], 
        reverb['diffusion'],
        reverb['damping'],
        reverb['wet_gain']
    ])
    
    # Distortion (2개)
    dist = guide_preset['Distortion']
    values.extend([
        dist['gain'],
        dist['color']
    ])
    
    # Pitch (1개)
    pitch = guide_preset['Pitch']
    values.append(pitch['scale'])
    
    return values  # 28개 값 보장됨


def compute_guide_loss(model, generated_preset, guide_preset, device):
    """Guide preset과의 차이를 이용한 정규화된 MSE loss"""
    try:
        # 디코더에서 _raw_params 직접 추출
        if isinstance(generated_preset, dict) and "_raw_params" in generated_preset:
            generated_tensor = generated_preset["_raw_params"].to(device)
            
            # Guide preset을 간단한 tensor로 변환
            guide_values = extract_guide_values(guide_preset)
            if guide_values is None:
                return torch.tensor(0.1, device=device, requires_grad=True)
            
            guide_tensor = torch.FloatTensor(guide_values).to(device)
            
            # 배치 차원 처리
            if generated_tensor.dim() == 2:
                generated_tensor = generated_tensor.squeeze(0)
            
            # 단일 샘플을 배치 형태로 변환하여 정규화 함수 재사용
            generated_batch = generated_tensor.unsqueeze(0)
            guide_batch = guide_tensor.unsqueeze(0)

            # 길이 불일치 대비: 공통 최소 차원으로 정렬 (pitch 제거 호환)
            min_dim = min(generated_batch.shape[-1], guide_batch.shape[-1])
            generated_batch = generated_batch[..., :min_dim]
            guide_batch = guide_batch[..., :min_dim]
            
            # 파라미터별 정규화 적용
            normalized_generated, normalized_guide = _normalize_parameters_for_loss(
                generated_batch.to(device), guide_batch.to(device)
            )
            
            # 가중치 기반 MSE loss 계산
            mse_loss = _compute_weighted_mse_loss(normalized_generated, normalized_guide).squeeze()
            
            return mse_loss
        else:
            return torch.tensor(0.1, device=device, requires_grad=True)
            
    except Exception as e:
        print(f"❌ Guide loss 실패: {e}")
        return torch.tensor(0.1, device=device, requires_grad=True)


def compute_batch_guide_loss(model, batch_generated_preset, batch_guide_presets, device):
    """배치 단위 Guide Loss 계산 - 파라미터별 정규화 적용"""
    try:
        # 배치 단위로 _raw_params 추출
        if isinstance(batch_generated_preset, dict) and "_raw_params" in batch_generated_preset:
            generated_batch_tensor = batch_generated_preset["_raw_params"].to(device)
            
            if generated_batch_tensor.dim() == 3:
                generated_batch_tensor = generated_batch_tensor.squeeze(1)  # [batch_size, 28]
            
            # 모든 preset이 이미 검증됨, 직접 변환
            batch_guide_values = []
            
            for guide_preset in batch_guide_presets:
                guide_values = extract_guide_values(guide_preset)
                batch_guide_values.append(guide_values)
            
            # Guide values를 배치 텐서로 스택
            guide_batch_tensor = torch.FloatTensor(batch_guide_values).to(device)

            # 길이 불일치 대비: 공통 최소 차원으로 정렬 (pitch 제거 호환)
            min_dim = min(generated_batch_tensor.shape[-1], guide_batch_tensor.shape[-1])
            generated_batch_tensor = generated_batch_tensor[..., :min_dim]
            guide_batch_tensor = guide_batch_tensor[..., :min_dim]
            
            # 파라미터별 정규화 적용
            normalized_generated, normalized_guide = _normalize_parameters_for_loss(
                generated_batch_tensor.to(device), guide_batch_tensor.to(device)
            )
            
            # 가중치 기반 MSE loss 계산
            batch_mse_loss = _compute_weighted_mse_loss(normalized_generated, normalized_guide)
            
            return batch_mse_loss
            
        else:
            return torch.tensor(0.1, device=device, requires_grad=True)
            
    except Exception as e:
        print(f"❌ Batch guide loss 실패: {e}")
        return torch.tensor(0.1, device=device, requires_grad=True)




def compute_batch_guide_loss_normalized_l1(model, batch_generated_preset, batch_guide_presets, device):
    """배치 단위 정규화된 L1 Guide Loss 계산 - 사전훈련용 (frequency 스케일 문제 해결)"""
    try:
        # 배치 단위로 _raw_params 추출
        if isinstance(batch_generated_preset, dict) and "_raw_params" in batch_generated_preset:
            generated_batch_tensor = batch_generated_preset["_raw_params"].to(device)
            
            if generated_batch_tensor.dim() == 3:
                generated_batch_tensor = generated_batch_tensor.squeeze(1)  # [batch_size, 28]
            
            # 모든 preset이 이미 검증됨, 직접 변환
            batch_guide_values = []
            
            for guide_preset in batch_guide_presets:
                guide_values = extract_guide_values(guide_preset)
                batch_guide_values.append(guide_values)
            
            # Guide values를 배치 텐서로 스택
            guide_batch_tensor = torch.FloatTensor(batch_guide_values).to(device)

            # 길이 불일치 대비: 공통 최소 차원으로 정렬 (pitch 제거 호환)
            min_dim = min(generated_batch_tensor.shape[-1], guide_batch_tensor.shape[-1])
            generated_batch_tensor = generated_batch_tensor[..., :min_dim]
            guide_batch_tensor = guide_batch_tensor[..., :min_dim]
            
            # 파라미터별 정규화 적용
            normalized_generated, normalized_guide = _normalize_parameters_for_loss(
                generated_batch_tensor.to(device), guide_batch_tensor.to(device)
            )
            
            # 정규화된 L1 loss 계산 - 평균값 함정 방지
            batch_l1_loss = nn.L1Loss()(normalized_generated, normalized_guide)
            
            return batch_l1_loss
            
        else:
            return torch.tensor(0.1, device=device, requires_grad=True)
            
    except Exception as e:
        print(f"❌ Normalized batch L1 loss 실패: {e}")
        return torch.tensor(0.1, device=device, requires_grad=True)


def compute_batch_hybrid_guide_loss(
    batch_generated_preset: Dict,
    batch_guide_presets: List[Dict],
    device,
    lambda_regression: float = 0.3,
    use_gated_offset: bool = True
):
    """
    Refactored hybrid guide loss with configuration-driven design
    and correct log-scale handling.
    """
    # --- 헬퍼 함수 정의 ---
    def to_linear_bin_offset(x, min_v, max_v, n_bins):
        x = torch.clamp(x, min_v, max_v)
        bin_width = (max_v - min_v) / n_bins
        t = (x - min_v) / bin_width
        bin_idx = torch.clamp(t.floor().long(), 0, n_bins - 1)
        center = min_v + (bin_idx.float() + 0.5) * bin_width
        offset = (x - center) / bin_width
        return bin_idx, offset

    def to_log_bin_offset(x, min_v, max_v, n_bins):
        x = torch.clamp(x, min_v, max_v)
        log_x = torch.log(x)
        log_min = torch.log(torch.tensor(min_v, device=x.device, dtype=x.dtype))
        log_max = torch.log(torch.tensor(max_v, device=x.device, dtype=x.dtype))
        # 로그 스케일에서 선형으로 binning
        bin_idx, offset = to_linear_bin_offset(log_x, log_min, log_max, n_bins)
        return bin_idx, offset

    # --- 파라미터 설정 (Single Source of Truth) ---
    PARAM_CONFIG = [
        # EQ (5 bands * 3 params = 15)
        *[{"group": "eq", "band": b, "name": p, "idx": 4*(b-1)+i, "range": r, "bins": n, "scale": s}
          for b in range(1, 6)
          for i, (p, r, n, s) in enumerate([
              ("freq", (20.0, 20000.0), 256, "log"),
              ("gain", (-20.0, 20.0), 128, "linear"),
              ("q",    (0.1, 30.0),    64, "log")
          ])],
        # Reverb (5 params)
        *[{"group": "reverb", "name": p, "idx": 20+i, "range": r, "bins": n, "scale": "linear"}
          for i, (p, r, n) in enumerate([
              ("room_size", (0.0, 1.0), 64), ("pre_delay", (0.0, 100.0), 64),
              ("diffusion", (0.0, 1.0), 64), ("damping", (0.0, 1.0), 64),
              ("wet_gain",  (0.0, 1.0), 64)
          ])],
        # Distortion (2 params)
        *[{"group": "distortion", "name": p, "idx": 25+i, "range": r, "bins": n, "scale": "linear"}
          for i, (p, r, n) in enumerate([("gain", (1.0, 10.0), 64), ("color", (-1.0, 1.0), 64)])],
        # Pitch 제거: decoder 최종 출력에서 제외되므로 학습에서도 사용하지 않음
    ]

    hybrid_outputs = batch_generated_preset["_hybrid"]
    batch_guide_values = [extract_guide_values(g) for g in batch_guide_presets]
    guide_tensor = torch.FloatTensor(batch_guide_values).to(device)

    if _should_debug_loss():
        try:
            raw = batch_generated_preset.get("_raw_params", None)
            if raw is not None:
                _tensor_stats_brief("gen._raw_params", raw)
        except Exception:
            pass
        _tensor_stats_brief("guide_tensor", guide_tensor)

    total_class_loss = 0.0
    total_offset_loss = 0.0
    num_class_terms = 0
    num_offset_terms = 0
    try:
        # --- 통합된 손실 계산 루프 ---
        for config in PARAM_CONFIG:
            group, name = config["group"], config["name"]
            
            # 모델 출력에서 logits/offset 가져오기
            if group == "eq":
                band_key = f"band_{config['band']}"
                if band_key not in hybrid_outputs.get(group, {}): continue
                aux = hybrid_outputs[group][band_key]
                logits_key, offset_key = f"{name}_logits", f"{name}_offset"
            else:
                aux = hybrid_outputs.get(group, {})
                logits_key, offset_key = f"{name}_class", f"{name}_offset"
            
            if logits_key not in aux: continue
            logits = aux[logits_key]
            if _should_debug_loss():
                _tensor_stats_brief(f"hyb.{group}.{name}.logits", logits)

            # GT 값 가져오기
            gt_val = guide_tensor[:, config["idx"]].unsqueeze(-1)
            if _should_debug_loss():
                _tensor_stats_brief(f"guide.{group}.{name}.gt_val", gt_val)
            min_val, max_val, num_bins = config["range"][0], config["range"][1], config["bins"]
            
            # 분류 손실 계산
            if config["scale"] == "classification_only":
                gt_bins = torch.round(torch.clamp(gt_val.squeeze(-1), min_val, max_val)).long()
                gt_bins = (gt_bins - int(min_val)) # 0-24 범위로 변환
                class_loss = F.cross_entropy(logits, gt_bins)
            else:
                binning_fn = to_log_bin_offset if config["scale"] == "log" else to_linear_bin_offset
                gt_bin, gt_off = binning_fn(gt_val, min_val, max_val, num_bins)
                class_loss = F.cross_entropy(logits, gt_bin.squeeze(-1))
            if _should_debug_loss():
                _debug_loss_print(f"[LOSS-DEBUG] class_loss({group}.{name})={float(class_loss.item()):.6f}")

            total_class_loss += class_loss
            num_class_terms += 1
            
            # 회귀 (오프셋) 손실 계산
            if offset_key in aux:
                offset = aux[offset_key]
                probs = F.softmax(logits, dim=-1)
                prob_correct = probs.gather(1, gt_bin)
                
                offset_weight = prob_correct.detach() if use_gated_offset else 1.0
                off_loss = F.l1_loss(offset, gt_off, reduction='none')
                off_loss = (off_loss * offset_weight).mean()
                if _should_debug_loss():
                    _debug_loss_print(f"[LOSS-DEBUG] off_loss({group}.{name})={float(off_loss.item()):.6f}")

                total_offset_loss += off_loss
                num_offset_terms += 1

        # 최종 손실 계산
        if num_class_terms > 0: total_class_loss /= num_class_terms
        if num_offset_terms > 0: total_offset_loss /= num_offset_terms
        total_loss = total_class_loss + lambda_regression * total_offset_loss
        return {
            'total_loss': total_loss,
            'classification_loss': total_class_loss,
            'offset_loss': total_offset_loss,
            'num_class_terms': torch.tensor(float(num_class_terms), device=device),
            'num_offset_terms': torch.tensor(float(num_offset_terms), device=device),
        }

    except Exception as e:
        print(f"❌ Hybrid guide loss 실패: {e}")
        return {
            'total_loss': torch.tensor(0.1, device=device, requires_grad=True),
            'classification_loss': torch.tensor(0.1, device=device),
            'offset_loss': torch.tensor(0.0, device=device)
        }


def _normalize_parameters_for_loss(generated_tensor, guide_tensor):
    """파라미터별 정규화 - 각 파라미터 타입에 맞는 스케일 적용"""
    normalized_generated = generated_tensor.clone()
    normalized_guide = guide_tensor.clone()
    
    # EQ parameters (20개): frequency, gain, q, filter_type × 5
    for band in range(5):
        base_idx = band * 4
        
        # Frequency (로그 스케일 정규화)
        freq_idx = base_idx + 0
        if freq_idx < generated_tensor.size(-1):
            # 로그 스케일 정규화: log(freq/20) / log(20000/20)
            gen_freq = torch.clamp(generated_tensor[..., freq_idx], min=20, max=20000)
            guide_freq = torch.clamp(guide_tensor[..., freq_idx], min=20, max=20000)
            
            normalized_generated[..., freq_idx] = torch.log(gen_freq / 20) / torch.log(torch.tensor(1000.0))
            normalized_guide[..., freq_idx] = torch.log(guide_freq / 20) / torch.log(torch.tensor(1000.0))
        
        # Gain (dB, 선형 정규화)
        gain_idx = base_idx + 1
        if gain_idx < generated_tensor.size(-1):
            normalized_generated[..., gain_idx] = (generated_tensor[..., gain_idx] + 30) / 60  # -30~30 → 0~1
            normalized_guide[..., gain_idx] = (guide_tensor[..., gain_idx] + 30) / 60
        
        # Q (로그 스케일 정규화)
        q_idx = base_idx + 2
        if q_idx < generated_tensor.size(-1):
            gen_q = torch.clamp(generated_tensor[..., q_idx], min=0.1, max=30)
            guide_q = torch.clamp(guide_tensor[..., q_idx], min=0.1, max=30)
            
            normalized_generated[..., q_idx] = torch.log(gen_q / 0.1) / torch.log(torch.tensor(300.0))
            normalized_guide[..., q_idx] = torch.log(guide_q / 0.1) / torch.log(torch.tensor(300.0))
        
        # Filter type (카테고리, 선형)
        type_idx = base_idx + 3
        if type_idx < generated_tensor.size(-1):
            normalized_generated[..., type_idx] = generated_tensor[..., type_idx] / 4  # 0~4 → 0~1
            normalized_guide[..., type_idx] = guide_tensor[..., type_idx] / 4
    
    # Reverb parameters (20~24)
    reverb_ranges = [(0, 1), (0, 100), (0, 1), (0, 1), (0, 1)]
    for i, (min_val, max_val) in enumerate(reverb_ranges):
        idx = 20 + i
        if idx < generated_tensor.size(-1):
            range_size = max_val - min_val
            if range_size > 0:
                normalized_generated[..., idx] = (generated_tensor[..., idx] - min_val) / range_size
                normalized_guide[..., idx] = (guide_tensor[..., idx] - min_val) / range_size
    
    # Distortion parameters (25~26)
    # Gain (로그 스케일)
    if 25 < generated_tensor.size(-1):
        gen_dist_gain = torch.clamp(generated_tensor[..., 25], min=1, max=10)
        guide_dist_gain = torch.clamp(guide_tensor[..., 25], min=1, max=10)
        normalized_generated[..., 25] = torch.log(gen_dist_gain) / torch.log(torch.tensor(10.0))
        normalized_guide[..., 25] = torch.log(guide_dist_gain) / torch.log(torch.tensor(10.0))
    
    # Color (선형)
    if 26 < generated_tensor.size(-1):
        normalized_generated[..., 26] = (generated_tensor[..., 26] + 1) / 2  # -1~1 → 0~1
        normalized_guide[..., 26] = (guide_tensor[..., 26] + 1) / 2
    
    # Pitch parameters 제거: decoder 최종 출력에서 제외. 존재하더라도 무시.
    
    return normalized_generated, normalized_guide


def _compute_weighted_mse_loss(normalized_generated, normalized_guide):
    """가중치 기반 MSE loss - 파라미터 타입별로 다른 가중치 적용"""
    # 파라미터별 가중치 정의
    weights = []
    
    # EQ parameters (20개) - frequency에 더 높은 가중치
    for band in range(5):
        weights.extend([
            3.0,  # frequency - 높은 가중치 (로그 스케일 보상)
            1.0,  # gain
            1.5,  # q - 중간 가중치 (로그 스케일 보상)
            2.0   # filter_type - 카테고리, 높은 가중치
        ])
    
    # Reverb parameters (5개)
    weights.extend([1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Distortion parameters (2개)
    weights.extend([1.5, 1.0])  # gain에 약간 높은 가중치
    
    # Pitch parameters (1개)
    weights.extend([2.0])  # pitch는 중요하므로 높은 가중치
    
    # 동적 길이 대응: pitch 제거 후에도 길이가 맞도록 가중치 길이 조정
    current_dim = normalized_generated.shape[-1]
    if len(weights) < current_dim:
        # 부족하면 0으로 패딩 (무시)
        weights = weights + [0.0] * (current_dim - len(weights))
    elif len(weights) > current_dim:
        # 길면 잘라냄
        weights = weights[:current_dim]

    weight_tensor = torch.tensor(weights, device=normalized_generated.device, dtype=normalized_generated.dtype)
    
    # 배치 차원에 맞춰 확장
    if normalized_generated.dim() > 1:
        weight_tensor = weight_tensor.unsqueeze(0).expand_as(normalized_generated)
    
    # 가중치 적용된 MSE 계산
    squared_errors = (normalized_generated - normalized_guide) ** 2
    weighted_errors = squared_errors * weight_tensor
    
    # 평균 계산 (가중치 합으로 나누어 정규화)
    total_weight = weight_tensor.sum(dim=-1, keepdim=True)
    weighted_mse = weighted_errors.sum(dim=-1, keepdim=True) / total_weight
    
    return weighted_mse.mean()



def compute_adversarial_training_loss(
    model,
    discriminator,
    batch_generated_preset,
    batch_guide_presets,
    device,
    adversarial_weight: float = 1.0,
    guide_weight: float = 1.0,
    use_feature_matching: bool = False,
    # Guide 손실 모드: 'hybrid' 또는 'normalized_l1'
    guide_mode: str = 'hybrid',
    # Hybrid 전용 옵션
    lambda_regression: float = 0.3,
    use_gated_offset: bool = True,
    feature_matching_weight: float = 0.1,
    # Discriminator 업데이트를 내부에서 수행할지 여부 및 옵티마이저/accelerator 전달
    discriminator_optimizer=None,
    accelerator=None,
):
    """
    적대적 학습을 위한 복합 손실 계산
    
    Args:
        model: Generator 모델
        discriminator: Discriminator 모델
        batch_generated_preset: 생성된 preset 파라미터
        batch_guide_presets: GT preset 데이터
        device: 디바이스
        adversarial_weight: 적대적 손실 가중치
        guide_weight: 가이드 손실 가중치
        use_feature_matching: Feature matching 사용 여부
        
    Returns:
        dict: 각종 손실값들
    """
    try:
        # 1. Guide Loss 계산 (모드에 따라 선택)
        if guide_mode == 'hybrid':
            hybrid = compute_batch_hybrid_guide_loss(
                batch_generated_preset=batch_generated_preset,
                batch_guide_presets=batch_guide_presets,
                device=device,
                lambda_regression=lambda_regression,
                use_gated_offset=use_gated_offset,
            )
            guide_loss = hybrid['total_loss']
            if _should_debug_loss():
                _debug_loss_print(f"[LOSS-DEBUG] guide_mode=hybrid, guide_total={float(guide_loss.item()):.6f}")
        else:
            guide_loss = compute_batch_guide_loss_normalized_l1(
                model, batch_generated_preset, batch_guide_presets, device
            )
            if _should_debug_loss():
                _debug_loss_print(f"[LOSS-DEBUG] guide_mode=normalized_l1, guide_total={float(guide_loss.item()):.6f}")
        
        # 2. 생성된 파라미터 정규화 (Discriminator 입력용)
        if isinstance(batch_generated_preset, dict) and "_raw_params" in batch_generated_preset:
            generated_raw = batch_generated_preset["_raw_params"].to(device)
            if generated_raw.dim() == 3:
                generated_raw = generated_raw.squeeze(1)  # [batch_size, 28]
            
            # GT 파라미터 준비
            batch_guide_values = []
            for guide_preset in batch_guide_presets:
                guide_values = extract_guide_values(guide_preset)
                batch_guide_values.append(guide_values)
            guide_raw = torch.FloatTensor(batch_guide_values).to(device)
            
            # 파라미터 정규화 (Discriminator는 정규화된 입력을 받음)
            # 길이 불일치 대비: 공통 최소 차원으로 정렬 (pitch 제거 호환)
            min_dim = min(generated_raw.shape[-1], guide_raw.shape[-1])
            generated_raw = generated_raw[..., :min_dim]
            guide_raw = guide_raw[..., :min_dim]

            normalized_generated, normalized_guide = _normalize_parameters_for_loss(
                generated_raw.to(device), guide_raw.to(device)
            )
            # 2.1 Discriminator update step (선택적)
            if discriminator_optimizer is not None:
                try:
                    disc = discriminator.module if hasattr(discriminator, 'module') else discriminator
                    discriminator_optimizer.zero_grad()
                    disc_loss, _, _ = disc.compute_adversarial_loss(
                        real_params=normalized_guide.detach(),
                        fake_params=normalized_generated.detach()
                    )
                    if accelerator is not None:
                        accelerator.backward(disc_loss)
                    else:
                        disc_loss.backward()
                    discriminator_optimizer.step()
                    if _should_debug_loss():
                        _debug_loss_print(f"[LOSS-DEBUG] discriminator_loss={float(disc_loss.item()):.6f}")
                except Exception as e:
                    _debug_loss_print(f"[LOSS-DEBUG] D step failed: {e}")
            if _should_debug_loss():
                _tensor_stats_brief("generated_raw", generated_raw)
                _tensor_stats_brief("guide_raw", guide_raw)
                _tensor_stats_brief("normalized_generated", normalized_generated)
                _tensor_stats_brief("normalized_guide", normalized_guide)
            
            # 3. Generator Adversarial Loss 계산 (DDP wrapping 대응)
            disc = discriminator.module if hasattr(discriminator, 'module') else discriminator
            if not hasattr(disc, 'compute_generator_adversarial_loss'):
                raise AttributeError("discriminator has no method compute_generator_adversarial_loss (check DDP wrapping)")
            generator_adv_loss = disc.compute_generator_adversarial_loss(normalized_generated)
            if _should_debug_loss():
                _debug_loss_print(f"[LOSS-DEBUG] generator_adv_loss={float(generator_adv_loss.item()):.6f}")
            
            # 4. Feature Matching Loss (선택적)
            feature_matching_loss = torch.tensor(0.0, device=device)
            if use_feature_matching:
                # 간단한 feature matching - L2 distance between generated and real
                feature_matching_loss = F.mse_loss(normalized_generated, normalized_guide.detach())
                if _should_debug_loss():
                    _debug_loss_print(f"[LOSS-DEBUG] feature_matching_loss={float(feature_matching_loss.item()):.6f}")
            
            # 5. 총 Generator Loss 계산
            total_generator_loss = (
                guide_weight * guide_loss +
                adversarial_weight * generator_adv_loss +
                feature_matching_weight * feature_matching_loss
            )
            if _should_debug_loss():
                _debug_loss_print(f"[LOSS-DEBUG] weights: guide={guide_weight}, adv={adversarial_weight}, fm={feature_matching_weight}")
                _debug_loss_print(f"[LOSS-DEBUG] total_generator_loss={float(total_generator_loss.item()):.6f}")
            
            return {
                'total_loss': total_generator_loss,
                'guide_loss': guide_loss,
                'adversarial_loss': generator_adv_loss,
                'feature_matching_loss': feature_matching_loss,
                'normalized_generated': normalized_generated,
                'normalized_guide': normalized_guide
            }
        else:
            return {
                'total_loss': torch.tensor(0.1, device=device, requires_grad=True),
                'guide_loss': torch.tensor(0.1, device=device),
                'adversarial_loss': torch.tensor(0.0, device=device),
                'feature_matching_loss': torch.tensor(0.0, device=device)
            }
            
    except Exception as e:
        print(f"❌ Adversarial training loss 실패: {e}")
        traceback.print_exc()
        return {
            'total_loss': torch.tensor(0.1, device=device, requires_grad=True),
            'guide_loss': torch.tensor(0.1, device=device),
            'adversarial_loss': torch.tensor(0.0, device=device),
            'feature_matching_loss': torch.tensor(0.0, device=device)
        }


def compute_diversity_metrics(generated_params_batch, guide_params_batch=None):
    """
    생성된 파라미터의 다양성 메트릭 계산
    
    Args:
        generated_params_batch: [batch_size, 28] - 생성된 파라미터
        guide_params_batch: [batch_size, 28] - GT 파라미터 (선택적)
        
    Returns:
        dict: 다양성 메트릭들
    """
    try:
        if generated_params_batch.dim() == 3:
            generated_params_batch = generated_params_batch.squeeze(1)
        
        batch_size, param_dim = generated_params_batch.shape
        
        metrics = {}
        
        # 1. 전체 파라미터 다양성
        param_std = torch.std(generated_params_batch, dim=0).mean().item()
        param_range = (generated_params_batch.max(dim=0)[0] - generated_params_batch.min(dim=0)[0]).mean().item()
        
        metrics.update({
            'overall_std': param_std,
            'overall_range': param_range,
        })
        
        # 2. EQ Frequency 다양성 (가장 중요)
        freq_indices = [0, 4, 8, 12, 16]  # EQ frequency 파라미터들
        if all(i < param_dim for i in freq_indices):
            freq_params = generated_params_batch[:, freq_indices]
            freq_std = torch.std(freq_params, dim=0).mean().item()
            freq_range = (freq_params.max(dim=0)[0] - freq_params.min(dim=0)[0]).mean().item()
            
            # Frequency 간 거리 다양성
            freq_distances = []
            for i in range(len(freq_indices)):
                for j in range(i + 1, len(freq_indices)):
                    distances = torch.abs(freq_params[:, i] - freq_params[:, j])
                    freq_distances.append(distances.mean().item())
            
            metrics.update({
                'freq_std': freq_std,
                'freq_range': freq_range,
                'freq_avg_distance': np.mean(freq_distances) if freq_distances else 0.0,
            })
        
        # 3. 배치 내 유사도 (Mode Collapse 지표)
        # 각 샘플 간의 L2 거리 계산
        pairwise_distances = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                distance = torch.norm(generated_params_batch[i] - generated_params_batch[j], p=2).item()
                pairwise_distances.append(distance)
        
        avg_pairwise_distance = np.mean(pairwise_distances) if pairwise_distances else 0.0
        min_pairwise_distance = np.min(pairwise_distances) if pairwise_distances else 0.0
        
        metrics.update({
            'avg_pairwise_distance': avg_pairwise_distance,
            'min_pairwise_distance': min_pairwise_distance,
            'mode_collapse_risk': 1.0 / (1.0 + avg_pairwise_distance)  # 거리가 작을수록 위험
        })
        
        # 4. GT와의 비교 (제공된 경우)
        if guide_params_batch is not None:
            if guide_params_batch.dim() == 3:
                guide_params_batch = guide_params_batch.squeeze(1)
            
            # GT 다양성과 비교
            gt_std = torch.std(guide_params_batch, dim=0).mean().item()
            gt_range = (guide_params_batch.max(dim=0)[0] - guide_params_batch.min(dim=0)[0]).mean().item()
            
            metrics.update({
                'diversity_ratio_std': param_std / (gt_std + 1e-6),
                'diversity_ratio_range': param_range / (gt_range + 1e-6),
            })
        
        return metrics
        
    except Exception as e:
        print(f"❌ Diversity metrics 계산 실패: {e}")
        return {
            'overall_std': 0.0,
            'overall_range': 0.0,
            'freq_std': 0.0,
            'freq_range': 0.0,
            'avg_pairwise_distance': 0.0,
            'mode_collapse_risk': 1.0
        }
