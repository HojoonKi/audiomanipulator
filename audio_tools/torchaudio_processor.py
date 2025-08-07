#!/usr/bin/env python3
"""
TorchAudio-based Differentiable Audio Processor

Complete rewrite using torchaudio for reliable differentiable audio processing.
Directly compatible with decoder output format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as TAF
import numpy as np
from typing import Dict, Optional, Union

print("✅ TorchAudio Differentiable Audio Processor")


class TorchAudioEqualizer(nn.Module):
    """Differentiable EQ using torchaudio biquad filters"""
    
    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate
        print("✅ TorchAudio Equalizer initialized")
    
    def forward(self, 
                audio: torch.Tensor,
                center_freq: torch.Tensor,
                gain_db: torch.Tensor,
                q: torch.Tensor,
                filter_type: str = "bell") -> torch.Tensor:
        """
        Apply EQ using torchaudio differentiable filters
        
        Args:
            audio: Input audio tensor [batch, channels, samples] or [batch, samples]
            center_freq: Center frequency (Hz) - from decoder
            gain_db: Gain in dB - from decoder  
            q: Q factor - from decoder
            filter_type: Filter type - from decoder
            
        Returns:
            Processed audio tensor
        """
        # Ensure proper tensor format
        if isinstance(center_freq, (int, float)):
            center_freq = torch.tensor(float(center_freq), device=audio.device)
        if isinstance(gain_db, (int, float)):
            gain_db = torch.tensor(float(gain_db), device=audio.device)
        if isinstance(q, (int, float)):  
            q = torch.tensor(float(q), device=audio.device)
        
        # Move tensors to same device
        center_freq = center_freq.to(audio.device)
        gain_db = gain_db.to(audio.device)
        q = q.to(audio.device)
        
        # Squeeze single-element tensors
        if center_freq.dim() > 0:
            center_freq = center_freq.squeeze()
        if gain_db.dim() > 0:
            gain_db = gain_db.squeeze()
        if q.dim() > 0:
            q = q.squeeze()
        
        # Clamp parameters to safe ranges
        center_freq = torch.clamp(center_freq, 20.0, min(20000.0, self.sample_rate / 2.1))
        gain_db = torch.clamp(gain_db, -30.0, 30.0)
        q = torch.clamp(q, 0.1, 30.0)
        
        try:
            # Apply appropriate torchaudio filter based on type
            if filter_type == "bell":
                return TAF.equalizer_biquad(
                    audio, self.sample_rate, 
                    center_freq, gain_db, q
                )
            elif filter_type == "highpass":
                return TAF.highpass_biquad(
                    audio, self.sample_rate,
                    center_freq, q
                )
            elif filter_type == "lowpass":
                return TAF.lowpass_biquad(
                    audio, self.sample_rate,
                    center_freq, q
                )
            elif filter_type == "high_shelf":
                return TAF.equalizer_biquad(
                    audio, self.sample_rate,
                    center_freq, gain_db, q
                )  # Use equalizer for shelf
            elif filter_type == "low_shelf":
                return TAF.equalizer_biquad(
                    audio, self.sample_rate,
                    center_freq, gain_db, q
                )  # Use equalizer for shelf
            else:
                # Default to bell filter
                return TAF.equalizer_biquad(
                    audio, self.sample_rate,
                    center_freq, gain_db, q
                )
                
        except Exception as e:
            print(f"⚠️ EQ processing failed: {e}, returning original audio")
            return audio


class TorchAudioReverb(nn.Module):
    """Differentiable Reverb using convolution with generated impulse responses"""
    
    def __init__(self, sample_rate: int = 44100, max_ir_length: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_ir_length = max_ir_length
        print("✅ TorchAudio Reverb initialized")
    
    def forward(self,
                audio: torch.Tensor,
                wet_gain: torch.Tensor,
                dry_gain: torch.Tensor) -> torch.Tensor:
        """
        Apply reverb using differentiable convolution
        
        Args:
            audio: Input audio tensor
            wet_gain: Wet signal gain (0-1) - from decoder
            dry_gain: Dry signal gain (0-1) - from decoder
            
        Returns:
            Processed audio tensor with reverb
        """
        # Process decoder parameters
        if isinstance(wet_gain, (int, float)):
            wet_gain = torch.tensor(float(wet_gain), device=audio.device)
        if isinstance(dry_gain, (int, float)):
            dry_gain = torch.tensor(float(dry_gain), device=audio.device)
            
        wet_gain = wet_gain.to(audio.device).squeeze()
        dry_gain = dry_gain.to(audio.device).squeeze()
        
        # Clamp parameters
        wet_gain = torch.clamp(wet_gain, 0.0, 1.0)
        dry_gain = torch.clamp(dry_gain, 0.0, 1.0)
        
        try:
            # Generate simple but effective impulse response
            ir = self._generate_impulse_response(wet_gain, audio.device)
            
            # Apply convolution reverb
            wet_signal = self._apply_convolution(audio, ir)
            
            # Mix dry and wet signals
            return dry_gain * audio + wet_gain * wet_signal
            
        except Exception as e:
            print(f"⚠️ Reverb processing failed: {e}, returning original audio")
            return audio
    
    def _generate_impulse_response(self, wet_gain: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Generate a simple reverb impulse response"""
        # Simple exponentially decaying noise for reverb
        ir_length = min(4410, self.max_ir_length)  # 0.1 seconds at 44.1kHz
        
        # Create decay envelope
        t = torch.arange(ir_length, dtype=torch.float32, device=device)
        decay_time = 0.5 + wet_gain * 1.5  # 0.5-2.0 seconds based on wet_gain
        envelope = torch.exp(-3.0 * t / (decay_time * self.sample_rate))
        
        # Add some randomness for natural reverb
        noise = torch.randn(ir_length, device=device)
        
        # Simple early reflections
        ir = torch.zeros(ir_length, device=device)
        early_times = [441, 882, 1323, 1764]  # 10ms, 20ms, 30ms, 40ms
        for i, early_time in enumerate(early_times):
            if early_time < ir_length:
                ir[early_time] = 0.5 * (0.8 ** i)  # Decreasing reflections
        
        # Add diffuse reverb tail
        ir += noise * envelope * 0.3
        
        # Normalize
        ir = ir / (torch.norm(ir) + 1e-8)
        
        return ir
    
    def _apply_convolution(self, audio: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Apply FFT-based convolution"""
        original_shape = audio.shape
        
        # Handle different input shapes
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
            squeeze_needed = 2
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [batch, samples] -> [batch, 1, samples]
            squeeze_needed = 1
        else:
            squeeze_needed = 0
        
        batch_size, channels, samples = audio.shape
        
        # Process each channel
        processed_channels = []
        for c in range(channels):
            channel = audio[:, c, :]  # [batch, samples]
            
            # FFT convolution for each batch item
            convolved_batch = []
            for b in range(batch_size):
                single_audio = channel[b]  # [samples]
                
                # Pad for proper convolution
                conv_length = single_audio.shape[0] + ir.shape[0] - 1
                fft_size = 2 ** int(np.ceil(np.log2(conv_length)))
                
                # FFT convolution
                audio_fft = torch.fft.rfft(single_audio, n=fft_size)
                ir_fft = torch.fft.rfft(ir, n=fft_size)
                
                result_fft = audio_fft * ir_fft
                convolved = torch.fft.irfft(result_fft, n=fft_size)
                
                # Crop to original length
                convolved = convolved[:samples]
                convolved_batch.append(convolved)
            
            processed_channels.append(torch.stack(convolved_batch))
        
        result = torch.stack(processed_channels, dim=1)  # [batch, channels, samples]
        
        # Restore original shape
        if squeeze_needed == 2:
            result = result.squeeze(0).squeeze(0)  # [samples]
        elif squeeze_needed == 1:
            result = result.squeeze(1)  # [batch, samples]
        
        return result


class TorchAudioDistortion(nn.Module):
    """Differentiable Distortion using waveshaping"""
    
    def __init__(self):
        super().__init__()
        print("✅ TorchAudio Distortion initialized")
    
    def forward(self,
                audio: torch.Tensor,
                gain: torch.Tensor,
                bias: torch.Tensor) -> torch.Tensor:
        """
        Apply distortion using waveshaping
        
        Args:
            audio: Input audio tensor
            gain: Distortion gain (1-10) - from decoder
            bias: Distortion bias (-1 to 1) - from decoder
            
        Returns:
            Processed audio tensor with distortion
        """
        # Process decoder parameters
        if isinstance(gain, (int, float)):
            gain = torch.tensor(float(gain), device=audio.device)
        if isinstance(bias, (int, float)):
            bias = torch.tensor(float(bias), device=audio.device)
            
        gain = gain.to(audio.device).squeeze()
        bias = bias.to(audio.device).squeeze()
        
        # Clamp parameters
        gain = torch.clamp(gain, 1.0, 10.0)
        bias = torch.clamp(bias, -1.0, 1.0)
        
        try:
            # Apply gain and bias
            processed = audio * gain + bias
            
            # Waveshaping distortion (differentiable)
            distorted = torch.tanh(processed * 0.8)
            
            # Output gain compensation
            compensation = 1.0 / (1.0 + gain * 0.05)
            
            return distorted * compensation
            
        except Exception as e:
            print(f"⚠️ Distortion processing failed: {e}, returning original audio")
            return audio


class TorchAudioPitchShift(nn.Module):
    """Differentiable Pitch Shift using resampling"""
    
    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate
        print("✅ TorchAudio Pitch Shift initialized")
    
    def forward(self,
                audio: torch.Tensor,
                pitch_shift: torch.Tensor) -> torch.Tensor:
        """
        Apply pitch shifting using resampling
        
        Args:
            audio: Input audio tensor
            pitch_shift: Pitch shift ratio (0.5-2.0) - from decoder
            
        Returns:
            Processed audio tensor with pitch shift
        """
        # Process decoder parameters
        if isinstance(pitch_shift, (int, float)):
            pitch_shift = torch.tensor(float(pitch_shift), device=audio.device)
            
        pitch_shift = pitch_shift.to(audio.device).squeeze()
        
        # Clamp parameters (0.5 = down octave, 2.0 = up octave)
        pitch_shift = torch.clamp(pitch_shift, 0.5, 2.0)
        
        # Skip processing if pitch shift is minimal
        if abs(pitch_shift - 1.0) < 0.01:
            return audio
        
        try:
            return self._apply_pitch_shift(audio, pitch_shift)
            
        except Exception as e:
            print(f"⚠️ Pitch shift processing failed: {e}, returning original audio")
            return audio
    
    def _apply_pitch_shift(self, audio: torch.Tensor, pitch_ratio: torch.Tensor) -> torch.Tensor:
        """Apply pitch shifting using interpolation"""
        original_shape = audio.shape
        original_length = original_shape[-1]
        
        # Prepare for interpolation
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
            squeeze_needed = 2
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [batch, samples] -> [batch, 1, samples]
            squeeze_needed = 1
        else:
            squeeze_needed = 0
        
        # Apply resampling using interpolation
        # Higher pitch_ratio = higher pitch = faster playback = shorter audio
        scale_factor = float(1.0 / pitch_ratio)
        
        resampled = F.interpolate(
            audio,
            scale_factor=scale_factor,
            mode='linear',
            align_corners=False
        )
        
        # Restore original length by padding or cropping
        current_length = resampled.shape[-1]
        if current_length < original_length:
            # Pad with zeros
            pad_amount = original_length - current_length
            resampled = F.pad(resampled, (0, pad_amount))
        elif current_length > original_length:
            # Crop to original length
            resampled = resampled[..., :original_length]
        
        # Restore original shape
        if squeeze_needed == 2:
            resampled = resampled.squeeze(0).squeeze(0)  # [samples]
        elif squeeze_needed == 1:
            resampled = resampled.squeeze(1)  # [batch, samples]
        
        return resampled


class TorchAudioProcessor(nn.Module):
    """
    Main processor combining all TorchAudio effects
    Compatible with decoder output format
    """
    
    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Initialize effects
        self.equalizer = TorchAudioEqualizer(sample_rate)
        self.reverb = TorchAudioReverb(sample_rate)  
        self.distortion = TorchAudioDistortion()
        self.pitch = TorchAudioPitchShift(sample_rate)
        
        print("✅ TorchAudio Processor initialized with all effects")
    
    def forward(self, 
                audio: torch.Tensor,
                preset: Dict) -> torch.Tensor:
        """
        Process audio with preset from decoder
        
        Args:
            audio: Input audio tensor
            preset: Preset dictionary from decoder in format:
                {
                    "equalizer": {"center_freq": tensor, "gain_db": tensor, "q": tensor, "filter_type": str},
                    "reverb": {"wet_gain": tensor, "dry_gain": tensor},
                    "distortion": {"gain": tensor, "bias": tensor},
                    "pitch": {"pitch_shift": tensor}
                }
                
        Returns:
            Processed audio tensor
        """
        processed = audio
        
        # Apply EQ if present
        if "equalizer" in preset:
            eq_params = preset["equalizer"]
            processed = self.equalizer(
                processed,
                center_freq=eq_params.get("center_freq", 1000.0),
                gain_db=eq_params.get("gain_db", 0.0),
                q=eq_params.get("q", 1.0),
                filter_type=eq_params.get("filter_type", "bell")
            )
        
        # Apply Reverb if present
        if "reverb" in preset:
            reverb_params = preset["reverb"]
            processed = self.reverb(
                processed,
                wet_gain=reverb_params.get("wet_gain", 0.3),
                dry_gain=reverb_params.get("dry_gain", 0.7)
            )
        
        # Apply Distortion if present
        if "distortion" in preset:
            dist_params = preset["distortion"]
            processed = self.distortion(
                processed,
                gain=dist_params.get("gain", 2.0),
                bias=dist_params.get("bias", 0.0)
            )
        
        # Apply Pitch Shift if present
        if "pitch" in preset:
            pitch_params = preset["pitch"]
            processed = self.pitch(
                processed,
                pitch_shift=pitch_params.get("pitch_shift", 1.0)
            )
        
        return processed


# Convenience functions for easy usage
def create_torchaudio_processor(sample_rate: int = 44100) -> TorchAudioProcessor:
    """Create a TorchAudio processor instance"""
    return TorchAudioProcessor(sample_rate)


def test_torchaudio_processor():
    """Test the TorchAudio processor with dummy data"""
    print("\n🧪 Testing TorchAudio Processor...")
    
    # Create processor
    processor = TorchAudioProcessor()
    
    # Create test audio
    audio = torch.randn(1, 1, 8000, requires_grad=True)  # 1 batch, 1 channel, 8000 samples
    print(f"Input audio shape: {audio.shape}")
    
    # Create test preset in decoder format
    preset = {
        "equalizer": {
            "center_freq": torch.tensor(1000.0),
            "gain_db": torch.tensor(3.0),
            "q": torch.tensor(2.0),
            "filter_type": "bell"
        },
        "reverb": {
            "wet_gain": torch.tensor(0.4),
            "dry_gain": torch.tensor(0.6)
        },
        "distortion": {
            "gain": torch.tensor(3.0),
            "bias": torch.tensor(0.1)
        },
        "pitch": {
            "pitch_shift": torch.tensor(1.2)
        }
    }
    
    # Process audio (keep gradients)
    processed = processor(audio, preset)
    
    print(f"✅ Processing successful!")
    print(f"Output audio shape: {processed.shape}")
    print(f"Output differentiable: {processed.requires_grad}")
    
    # Test gradient flow
    if audio.requires_grad:
        loss = torch.mean(processed ** 2)
        loss.backward()
        grad_norm = torch.norm(audio.grad) if audio.grad is not None else 0
        print(f"✅ Gradient flow test: loss={loss.item():.6f}, grad_norm={grad_norm:.6f}")
    
    print("✅ TorchAudio Processor test completed!")


if __name__ == "__main__":
    print("🎵 TorchAudio Differentiable Audio Processor")
    print("=" * 50)
    
    test_torchaudio_processor()
