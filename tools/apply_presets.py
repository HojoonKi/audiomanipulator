#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
import torch.nn.functional as F


def _infer_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_audio_mono(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav  # [1, T]


def _filter_type_to_logits(filter_type: str, device: torch.device) -> torch.Tensor:
    # Order used in processor: [low_shelf, peaking, high_shelf, highpass, lowpass]
    mapping = {
        "low-shelf": 0,
        "lowshelf": 0,
        "bell": 1,
        "peaking": 1,
        "high-shelf": 2,
        "highshelf": 2,
        "highpass": 3,
        "lowpass": 4,
        "low_pass": 4,
        "high_pass": 3,
        "low-pass": 4,
        "high-pass": 3,
        "low_shelf": 0,
        "high_shelf": 2,
    }
    idx = mapping.get(str(filter_type).lower(), 1)
    
    # 선택된 인덱스에 높은 로짓 값, 나머지는 낮은 값
    logits = torch.full((5,), -10.0, device=device, dtype=torch.float32)
    logits[idx] = 10.0
    return logits


def _build_batch_preset_from_fined(first_presets: List[Dict], batch_size: int, device: torch.device) -> Dict:
    # Initialize containers for batch tensors
    def col() -> torch.Tensor:
        return torch.zeros(batch_size, 1, device=device, dtype=torch.float32)

    # EQ 파라미터: differentiable_flexible_eq가 기대하는 형태로 구성
    eq: Dict[str, Dict[str, torch.Tensor]] = {}
    for b in range(1, 6):
        eq[f"band_{b}"] = {
            "center_freq": col(),  # center_freq -> cutoff_freq로 변경
            "gain_db": col(),
            "q": col(),     # q -> q_factor로 변경
            "filter_type": torch.zeros(batch_size, 5, device=device, dtype=torch.float32),  # 로짓 텐서
        }

    reverb = {
        "wet_gain": col(),
        "room_size": col(),
        "pre_delay": col(),  # 초 단위 (ms 변환 제거)
        "damping": col(),
        "diffusion": col(),
    }

    distortion = {
        "gain": col(),
        "color": col(),
    }

    for i, p in enumerate(first_presets):
        # Equalizer: 5 bands
        bands = p.get("Equalizer", [])
        for b in range(1, 6):
            band = bands[b - 1] if b - 1 < len(bands) else None
            if band is None:
                # 기본값 설정 (bypass)
                eq[f"band_{b}"]["center_freq"][i, 0] = 1000.0
                eq[f"band_{b}"]["gain_db"][i, 0] = 0.0
                eq[f"band_{b}"]["q"][i, 0] = 1.0
                eq[f"band_{b}"]["filter_type"][i] = _filter_type_to_logits("peaking", device)
                continue
                
            eq[f"band_{b}"]["center_freq"][i, 0] = float(band.get("frequency", 1000.0))
            eq[f"band_{b}"]["gain_db"][i, 0] = float(band.get("gain", 0.0))
            eq[f"band_{b}"]["q"][i, 0] = float(band.get("q", 1.0))
            eq[f"band_{b}"]["filter_type"][i] = _filter_type_to_logits(
                band.get("filter_type", "bell"), device
            )

        # Reverb: pre_delay는 초 단위로 그대로 사용 (ms 변환 제거)
        rv = p.get("Reverb", {})
        reverb["wet_gain"][i, 0] = float(rv.get("wet_gain", 0.5))
        reverb["room_size"][i, 0] = float(rv.get("room_size", 0.5))
        reverb["pre_delay"][i, 0] = float(rv.get("pre_delay", 0.02))  # 초 단위 그대로
        reverb["damping"][i, 0] = float(rv.get("damping", 0.5))
        reverb["diffusion"][i, 0] = float(rv.get("diffusion", 0.5))

        # Distortion
        ds = p.get("Distortion", {})
        distortion["gain"][i, 0] = float(ds.get("gain", 1.0))
        distortion["color"][i, 0] = float(ds.get("color", 0.0))

    return {"equalizer": eq, "reverb": reverb, "distortion": distortion}


def _slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text)[:60].strip("_")


def main():
    parser = argparse.ArgumentParser(description="Apply first N fined presets to an audio file and save outputs")
    parser.add_argument("--input", type=str, default="audio_dataset/speech/male/1320-122612-0000.flac", help="Path to input audio (wav)")
    parser.add_argument("--outdir", type=str, default="output/test", help="Directory to save processed wavs")
    parser.add_argument("--num", type=int, default=5, help="Number of presets from the top to apply")
    parser.add_argument("--sr", type=int, default=48000, help="Sample rate for processing")
    args = parser.parse_args()

    device = _infer_device()

    # Prepare imports for project modules
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]  # /audiomanipulator
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Lazy imports after sys.path setup
    from audio_tools.torchaudio_processor import TorchAudioProcessor
    from descriptions.fined_presets_filtered import fined_presets

    os.makedirs(args.outdir, exist_ok=True)

    # Load and prepare audio batch
    wav = _load_audio_mono(args.input, args.sr).to(device)  # [1, T]
    batch_wav = wav.repeat(args.num, 1)  # [B, T]

    # Build batch preset from first N presets
    first_presets = fined_presets[: args.num]
    batch_preset = _build_batch_preset_from_fined(first_presets, args.num, device)
    # Process
    processor = TorchAudioProcessor(sample_rate=args.sr).to(device)
    processor.eval()
    with torch.no_grad():
        out = processor(batch_wav, batch_preset)  # [B, 1, T]

    # Save outputs
    for i, preset in enumerate(first_presets):
        prompt = preset.get("prompt", f"preset_{i+1}")
        stem = f"{i+1:02d}_" + _slugify(prompt)
        out_i = out[i].detach().cpu()  # [1, T]
        torchaudio.save(os.path.join(args.outdir, f"{stem}.wav"), out_i, args.sr)

    # Also save dry reference once
    dry_path = os.path.join(args.outdir, "00_input_dry.wav")
    if not os.path.exists(dry_path):
        torchaudio.save(dry_path, wav.cpu(), args.sr)

    print(f"Saved {args.num} processed files to {args.outdir}")


if __name__ == "__main__":
    main()


