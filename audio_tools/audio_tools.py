#!/usr/bin/env python3
"""
Audio Tools for Audio Processing

This module provides various audio processing tools including:
- Equalizer (EQ)
- Reverb
- Distortion  
- Pitch Shifting

Dependencies:
    pip install librosa soundfile scipy numpy pedalboard
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import butter, sosfilt, iirfilter
import warnings
from typing import Union, Tuple, Optional
from pathlib import Path

try:
    from pedalboard import Pedalboard, Reverb, Distortion, PitchShift, HighpassFilter, LowpassFilter, PeakFilter
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    warnings.warn("Pedalboard not available. Some features will use fallback implementations.")


class AudioProcessor:
    """Base class for audio processing tools"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=None)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file {file_path}: {e}")
    
    def save_audio(self, audio: np.ndarray, file_path: str, sample_rate: int = None):
        """Save audio to file"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            sf.write(file_path, audio, sample_rate)
            print(f"Audio saved to: {file_path}")
        except Exception as e:
            raise ValueError(f"Error saving audio file {file_path}: {e}")


class Equalizer(AudioProcessor):
    """
    Equalizer with configurable bands
    
    Parameters:
    - frequency: Center frequency in Hz
    - gain: Gain in dB (positive = boost, negative = cut)
    - q: Q factor (bandwidth control, higher = narrower)
    - filter_type: 'peak', 'highpass', 'lowpass', 'highshelf', 'lowshelf'
    """
    
    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self.bands = []
    
    def add_band(self, frequency: float, gain: float, q: float = 1.0, filter_type: str = 'peak'):
        """
        Add EQ band
        
        Parameters:
        -----------
        frequency : float
            Center frequency in Hz (20 - 20000)
            The frequency around which the filter will operate
            
        gain : float
            Gain in dB (-20 to +20 typical range)
            Positive values boost the frequency, negative values cut it
            
        q : float, default=1.0
            Q factor (0.1 - 10.0 typical range)
            Controls the bandwidth of the filter
            Higher Q = narrower band, more surgical EQ
            Lower Q = wider band, more musical EQ
            
        filter_type : str, default='peak'
            Type of filter to apply
            Options: 'peak', 'highpass', 'lowpass', 'highshelf', 'lowshelf'
            - 'peak': Bell-shaped curve, boosts/cuts around center frequency
            - 'highpass': Removes frequencies below the cutoff
            - 'lowpass': Removes frequencies above the cutoff
            - 'highshelf': Boosts/cuts all frequencies above the cutoff
            - 'lowshelf': Boosts/cuts all frequencies below the cutoff
        """
        band = {
            'frequency': frequency,
            'gain': gain,
            'q': q,
            'filter_type': filter_type
        }
        self.bands.append(band)
        return self
    
    def apply(self, audio: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """Apply EQ to audio"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        processed_audio = audio.copy()
        
        if PEDALBOARD_AVAILABLE:
            # Use Pedalboard for better quality
            board = Pedalboard()
            
            for band in self.bands:
                freq = band['frequency']
                gain = band['gain']
                q = band['q']
                filter_type = band['filter_type']
                
                if filter_type == 'peak':
                    board.append(PeakFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))
                elif filter_type == 'highpass':
                    board.append(HighpassFilter(cutoff_frequency_hz=freq))
                elif filter_type == 'lowpass':
                    board.append(LowpassFilter(cutoff_frequency_hz=freq))
            
            processed_audio = board(processed_audio, sample_rate)
        
        else:
            # Fallback implementation using scipy
            for band in self.bands:
                processed_audio = self._apply_band_scipy(processed_audio, band, sample_rate)
        
        return processed_audio
    
    def _apply_band_scipy(self, audio: np.ndarray, band: dict, sample_rate: int) -> np.ndarray:
        """Apply single EQ band using scipy"""
        freq = band['frequency']
        gain = band['gain']
        q = band['q']
        filter_type = band['filter_type']
        
        # Normalize frequency
        nyquist = sample_rate / 2
        norm_freq = freq / nyquist
        
        if norm_freq >= 1.0:
            return audio
        
        try:
            if filter_type == 'peak':
                # Peaking filter implementation
                A = 10**(gain/40)
                w = 2 * np.pi * norm_freq
                alpha = np.sin(w) / (2 * q)
                
                b0 = 1 + alpha * A
                b1 = -2 * np.cos(w)
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * np.cos(w)
                a2 = 1 - alpha / A
                
                # Normalize coefficients
                b = [b0/a0, b1/a0, b2/a0]
                a = [1, a1/a0, a2/a0]
                
                filtered = signal.lfilter(b, a, audio)
                
            elif filter_type == 'highpass':
                sos = butter(2, norm_freq, btype='high', output='sos')
                filtered = sosfilt(sos, audio)
                
            elif filter_type == 'lowpass':
                sos = butter(2, norm_freq, btype='low', output='sos')
                filtered = sosfilt(sos, audio)
                
            else:
                filtered = audio
            
            return filtered
            
        except Exception as e:
            warnings.warn(f"EQ band application failed: {e}")
            return audio
    
    def reset(self):
        """Reset all EQ bands"""
        self.bands = []
        return self


class ReverbProcessor(AudioProcessor):
    """
    Reverb processor
    
    Parameters:
    - room_size: Room size (0.0 to 1.0)
    - pre_delay: Pre-delay in seconds
    - diffusion: Diffusion amount (0.0 to 1.0)  
    - damping: High-frequency damping (0.0 to 1.0)
    - wet_gain: Wet signal gain in dB
    """
    
    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self.room_size = 0.5
        self.pre_delay = 0.03
        self.diffusion = 0.5
        self.damping = 0.5
        self.wet_gain = -6.0
    
    def set_parameters(self, room_size: float = 0.5, pre_delay: float = 0.03, 
                      diffusion: float = 0.5, damping: float = 0.5, wet_gain: float = -6.0):
        """
        Set reverb parameters
        
        Parameters:
        -----------
        room_size : float, default=0.5
            Size of the simulated room (0.0 - 1.0)
            0.0 = very small room (tight, short reverb)
            1.0 = very large hall (spacious, long reverb)
            
        pre_delay : float, default=0.03
            Pre-delay time in seconds (0.0 - 0.2)
            Time between direct sound and first reflection
            Larger values create sense of bigger space
            
        diffusion : float, default=0.5
            Diffusion amount (0.0 - 1.0)
            Controls how scattered the reflections are
            0.0 = clear, distinct echoes
            1.0 = smooth, dense reverb tail
            
        damping : float, default=0.5
            High-frequency damping (0.0 - 1.0)
            Simulates absorption of high frequencies by surfaces
            0.0 = bright, reflective surfaces (glass, concrete)
            1.0 = dark, absorptive surfaces (carpet, curtains)
            
        wet_gain : float, default=-6.0
            Wet signal gain in dB (-20 to 0)
            Controls the level of reverb effect
            -20 dB = very subtle reverb
            0 dB = equal mix of dry and wet signal
        """
        self.room_size = np.clip(room_size, 0.0, 1.0)
        self.pre_delay = max(0.0, pre_delay)
        self.diffusion = np.clip(diffusion, 0.0, 1.0)
        self.damping = np.clip(damping, 0.0, 1.0)
        self.wet_gain = wet_gain
        return self
    
    def apply(self, audio: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """Apply reverb to audio"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        if PEDALBOARD_AVAILABLE:
            # Use Pedalboard reverb
            reverb = Reverb(
                room_size=self.room_size,
                damping=self.damping,
                wet_level=10**(self.wet_gain/20),  # Convert dB to linear
                dry_level=0.8,
                width=1.0,
                freeze_mode=0.0
            )
            
            processed_audio = reverb(audio, sample_rate)
        
        else:
            # Simple reverb implementation using convolution
            processed_audio = self._apply_simple_reverb(audio, sample_rate)
        
        return processed_audio
    
    def _apply_simple_reverb(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Simple reverb implementation"""
        try:
            # Create impulse response
            duration = 0.5 + self.room_size * 2.0  # 0.5 to 2.5 seconds
            length = int(duration * sample_rate)
            
            # Generate exponential decay
            t = np.linspace(0, duration, length)
            decay = np.exp(-t * (2.0 - self.room_size * 1.5))
            
            # Add some randomness for diffusion
            np.random.seed(42)  # For reproducibility
            noise = np.random.randn(length)
            impulse = decay * noise * self.diffusion + decay * (1 - self.diffusion)
            
            # Apply damping (low-pass filter)
            if self.damping > 0:
                cutoff = 8000 * (1 - self.damping)  # 8kHz to 0Hz
                nyquist = sample_rate / 2
                if cutoff < nyquist:
                    sos = butter(2, cutoff/nyquist, btype='low', output='sos')
                    impulse = sosfilt(sos, impulse)
            
            # Normalize impulse response
            impulse = impulse / np.max(np.abs(impulse))
            
            # Apply pre-delay
            pre_delay_samples = int(self.pre_delay * sample_rate)
            if pre_delay_samples > 0:
                impulse = np.concatenate([np.zeros(pre_delay_samples), impulse])
            
            # Convolve with audio
            wet_signal = np.convolve(audio, impulse, mode='same')
            
            # Mix dry and wet signals
            wet_gain_linear = 10**(self.wet_gain/20)
            dry_gain_linear = 0.7  # -3dB dry signal
            
            processed_audio = dry_gain_linear * audio + wet_gain_linear * wet_signal
            
            return processed_audio
            
        except Exception as e:
            warnings.warn(f"Reverb application failed: {e}")
            return audio


class DistortionProcessor(AudioProcessor):
    """
    Distortion processor
    
    Parameters:
    - gain: Input gain in dB
    - color: Distortion character/color (0.0 to 1.0)
    """
    
    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self.gain = 10.0
        self.color = 0.5
    
    def set_parameters(self, gain: float = 10.0, color: float = 0.5):
        """
        Set distortion parameters
        
        Parameters:
        -----------
        gain : float, default=10.0
            Input gain in dB (0 - 30 typical range)
            Amount of overdrive applied to the signal
            0 dB = clean signal
            10 dB = light distortion
            20+ dB = heavy distortion/fuzz
            
        color : float, default=0.5
            Distortion character/color (0.0 - 1.0)
            Controls the type and harmonic content of distortion
            0.0 - 0.33 = Soft clipping (tube/valve-like, warm)
            0.34 - 0.66 = Hard clipping (transistor-like, aggressive)
            0.67 - 1.0 = Asymmetric clipping (diode-like, buzzy)
        """
        self.gain = gain
        self.color = np.clip(color, 0.0, 1.0)
        return self
    
    def apply(self, audio: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """Apply distortion to audio"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        if PEDALBOARD_AVAILABLE:
            # Use Pedalboard distortion
            distortion = Distortion(drive_db=self.gain)
            processed_audio = distortion(audio, sample_rate)
        
        else:
            # Custom distortion implementation
            processed_audio = self._apply_custom_distortion(audio)
        
        return processed_audio
    
    def _apply_custom_distortion(self, audio: np.ndarray) -> np.ndarray:
        """Custom distortion implementation"""
        try:
            # Apply input gain
            gained_audio = audio * 10**(self.gain/20)
            
            # Different distortion curves based on color
            if self.color < 0.33:
                # Soft clipping (tube-like)
                processed_audio = np.tanh(gained_audio * 2.0) * 0.5
            elif self.color < 0.66:
                # Hard clipping (transistor-like)  
                processed_audio = np.clip(gained_audio, -0.7, 0.7)
            else:
                # Asymmetric clipping (diode-like)
                processed_audio = np.where(gained_audio > 0, 
                                         np.tanh(gained_audio * 1.5) * 0.6,
                                         np.tanh(gained_audio * 2.5) * 0.4)
            
            # Apply some harmonic enhancement
            harmonics = 0.1 * self.color * np.sin(gained_audio * 4 * np.pi)
            processed_audio += harmonics
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(processed_audio))
            if max_val > 0.95:
                processed_audio = processed_audio * 0.95 / max_val
            
            return processed_audio
            
        except Exception as e:
            warnings.warn(f"Distortion application failed: {e}")
            return audio


class PitchProcessor(AudioProcessor):
    """
    Pitch shifter
    
    Parameters:
    - scale: Pitch scaling factor (0.5 = one octave down, 2.0 = one octave up)
    """
    
    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self.scale = 1.0
    
    def set_parameters(self, scale: float = 1.0):
        """
        Set pitch parameters
        
        Parameters:
        -----------
        scale : float, default=1.0
            Pitch scaling factor (0.25 - 4.0)
            Multiplier for the fundamental frequency
            0.5 = one octave down (half frequency)
            1.0 = no change (original pitch)
            2.0 = one octave up (double frequency)
            1.05946 = one semitone up
            0.94387 = one semitone down
            
            Common musical intervals:
            - 0.5 = -12 semitones (octave down)
            - 0.707 = -7 semitones (perfect fifth down)
            - 0.841 = -3 semitones (minor third down)  
            - 1.189 = +3 semitones (minor third up)
            - 1.414 = +7 semitones (perfect fifth up)
            - 2.0 = +12 semitones (octave up)
        """
        self.scale = max(0.25, min(4.0, scale))  # Limit range
        return self
    
    def apply(self, audio: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """Apply pitch shifting to audio"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        if abs(self.scale - 1.0) < 0.001:  # No change needed
            return audio
        
        if PEDALBOARD_AVAILABLE:
            # Use Pedalboard pitch shift
            semitones = 12 * np.log2(self.scale)
            pitch_shift = PitchShift(semitones=semitones)
            processed_audio = pitch_shift(audio, sample_rate)
        
        else:
            # Use librosa for pitch shifting
            processed_audio = self._apply_librosa_pitch_shift(audio, sample_rate)
        
        return processed_audio
    
    def _apply_librosa_pitch_shift(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Pitch shift using librosa"""
        try:
            # Convert scale to semitones
            semitones = 12 * np.log2(self.scale)
            
            # Apply pitch shift
            processed_audio = librosa.effects.pitch_shift(
                audio, sr=sample_rate, n_steps=semitones
            )
            
            return processed_audio
            
        except Exception as e:
            warnings.warn(f"Pitch shifting failed: {e}")
            return audio


class AudioToolbox:
    """
    Combined audio processing toolbox
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.equalizer = Equalizer(sample_rate)
        self.reverb = ReverbProcessor(sample_rate)
        self.distortion = DistortionProcessor(sample_rate)
        self.pitch = PitchProcessor(sample_rate)
    
    def process_with_config(self, input_path: str, output_path: str, config: dict):
        """
        Process audio with configuration dictionary
        
        Parameters:
        -----------
        input_path : str
            Path to input audio file
            
        output_path : str
            Path for output audio file
            
        config : dict
            Configuration dictionary with the following structure:
            {
                "Equalizer": {
                    1: {"frequency": 100, "Gain": 5, "Q": 1.0, "Filter-type": "bell"},
                    2: {"frequency": 3500, "Gain": -3, "Q": 0.7, "Filter-type": "high-shelf"}
                },
                "Reverb": {
                    "Room Size": 0.9,
                    "Pre Delay": 0.1,
                    "Diffusion": 0.7,
                    "Damping": 0.4,
                    "Wet Gain": -3.0
                },
                "Distortion": {
                    "Gain": 15,
                    "Color": 0.8
                },
                "Pitch": {
                    "Scale": 0.5  # or semitones: "Semitones": -12
                }
            }
            
        Processing Order: EQ → Pitch → Distortion → Reverb
        """
        
        # Load audio
        try:
            audio, sr = librosa.load(input_path, sr=None)
            print(f"Loaded audio: {input_path} (SR: {sr}Hz, Duration: {len(audio)/sr:.2f}s)")
        except Exception as e:
            raise ValueError(f"Error loading audio: {e}")
        
        processed_audio = audio.copy()
        
        # 1. Apply EQ first
        if "Equalizer" in config:
            eq_config = config["Equalizer"]
            self.equalizer.reset()
            
            for band_id, band_params in eq_config.items():
                # Convert parameter names to match our function
                eq_params = self._normalize_eq_params(band_params)
                self.equalizer.add_band(**eq_params)
            
            processed_audio = self.equalizer.apply(processed_audio, sr)
            print("Applied EQ")
        
        # 2. Apply Pitch Shift
        if "Pitch" in config:
            pitch_config = config["Pitch"]
            pitch_params = self._normalize_pitch_params(pitch_config)
            self.pitch.set_parameters(**pitch_params)
            processed_audio = self.pitch.apply(processed_audio, sr)
            print("Applied Pitch Shift")
        
        # 3. Apply Distortion
        if "Distortion" in config:
            distortion_config = config["Distortion"]
            distortion_params = self._normalize_distortion_params(distortion_config)
            self.distortion.set_parameters(**distortion_params)
            processed_audio = self.distortion.apply(processed_audio, sr)
            print("Applied Distortion")
        
        # 4. Apply Reverb last
        if "Reverb" in config:
            reverb_config = config["Reverb"]
            reverb_params = self._normalize_reverb_params(reverb_config)
            self.reverb.set_parameters(**reverb_params)
            processed_audio = self.reverb.apply(processed_audio, sr)
            print("Applied Reverb")
        
        # Save processed audio
        try:
            sf.write(output_path, processed_audio, sr)
            print(f"Processed audio saved to: {output_path}")
        except Exception as e:
            raise ValueError(f"Error saving audio: {e}")
        
        return processed_audio, sr
    
    def _normalize_eq_params(self, params: dict) -> dict:
        """Normalize EQ parameter names"""
        normalized = {}
        
        # Handle different parameter name formats
        if "frequency" in params:
            normalized["frequency"] = params["frequency"]
        elif "Frequency" in params:
            normalized["frequency"] = params["Frequency"]
        
        if "Gain" in params:
            normalized["gain"] = params["Gain"]
        elif "gain" in params:
            normalized["gain"] = params["gain"]
        
        if "Q" in params:
            normalized["q"] = params["Q"]
        elif "q" in params:
            normalized["q"] = params["q"]
        else:
            normalized["q"] = 1.0  # Default
        
        # Handle filter type mapping
        filter_type = params.get("Filter-type", params.get("filter_type", "peak"))
        
        # Map common filter type names
        filter_mapping = {
            "bell": "peak",
            "Bell": "peak",
            "peak": "peak",
            "Peak": "peak",
            "high-shelf": "highshelf",
            "High-shelf": "highshelf",
            "highshelf": "highshelf",
            "low-shelf": "lowshelf",
            "Low-shelf": "lowshelf",
            "lowshelf": "lowshelf",
            "highpass": "highpass",
            "lowpass": "lowpass"
        }
        
        normalized["filter_type"] = filter_mapping.get(filter_type, "peak")
        
        return normalized
    
    def _normalize_reverb_params(self, params: dict) -> dict:
        """Normalize Reverb parameter names"""
        normalized = {}
        
        # Room Size (0.0-1.0)
        if "Room Size" in params:
            # If value is 0-10 scale, convert to 0.0-1.0
            room_size = params["Room Size"]
            if room_size > 1.0:
                room_size = room_size / 10.0
            normalized["room_size"] = room_size
        elif "room_size" in params:
            normalized["room_size"] = params["room_size"]
        
        # Pre Delay (seconds)
        if "Pre Delay" in params:
            normalized["pre_delay"] = params["Pre Delay"]
        elif "pre_delay" in params:
            normalized["pre_delay"] = params["pre_delay"]
        
        # Diffusion (0.0-1.0)
        if "Diffusion" in params:
            normalized["diffusion"] = params["Diffusion"]
        elif "diffusion" in params:
            normalized["diffusion"] = params["diffusion"]
        
        # Damping (0.0-1.0)
        if "Damping" in params:
            normalized["damping"] = params["Damping"]
        elif "damping" in params:
            normalized["damping"] = params["damping"]
        
        # Wet Gain (dB)
        if "Wet Gain" in params:
            wet_gain = params["Wet Gain"]
            # If value is 0.0-1.0, convert to dB
            if 0.0 <= wet_gain <= 1.0:
                # Convert linear to dB: 0.8 -> -1.94 dB
                wet_gain = 20 * np.log10(wet_gain) if wet_gain > 0 else -20
            normalized["wet_gain"] = wet_gain
        elif "wet_gain" in params:
            normalized["wet_gain"] = params["wet_gain"]
        
        return normalized
    
    def _normalize_distortion_params(self, params: dict) -> dict:
        """Normalize Distortion parameter names"""
        normalized = {}
        
        if "Gain" in params:
            normalized["gain"] = params["Gain"]
        elif "gain" in params:
            normalized["gain"] = params["gain"]
        
        if "Color" in params:
            normalized["color"] = params["Color"]
        elif "color" in params:
            normalized["color"] = params["color"]
        
        return normalized
    
    def _normalize_pitch_params(self, params: dict) -> dict:
        """Normalize Pitch parameter names"""
        normalized = {}
        
        if "Scale" in params:
            scale = params["Scale"]
            # If negative number, treat as semitones
            if scale < 0:
                # Convert semitones to scale: -7 semitones -> 2^(-7/12) = 0.659
                normalized["scale"] = 2**(scale/12)
            else:
                normalized["scale"] = scale
        elif "scale" in params:
            normalized["scale"] = params["scale"]
        elif "Semitones" in params:
            # Direct semitone input: convert to scale
            semitones = params["Semitones"]
            normalized["scale"] = 2**(semitones/12)
        elif "semitones" in params:
            semitones = params["semitones"]
            normalized["scale"] = 2**(semitones/12)
        
        return normalized
    
    def process_audio_file(self, input_path: str, output_path: str, 
                          eq_bands: list = None,
                          reverb_params: dict = None,
                          distortion_params: dict = None,
                          pitch_params: dict = None):
        """
        Process audio file with multiple effects
        
        Parameters:
        -----------
        input_path : str
            Path to input audio file
            Supported formats: .wav, .mp3, .flac, .m4a, .aac
            
        output_path : str  
            Path for output audio file
            Will be saved as .wav format
            
        eq_bands : list of dict, optional
            List of EQ band dictionaries, each containing:
            - 'frequency': float (20-20000 Hz)
            - 'gain': float (-20 to +20 dB)
            - 'q': float (0.1-10.0, default=1.0)
            - 'filter_type': str ('peak', 'highpass', 'lowpass', etc.)
            
        reverb_params : dict, optional
            Reverb parameters dictionary:
            - 'room_size': float (0.0-1.0, default=0.5)
            - 'pre_delay': float (0.0-0.2 seconds, default=0.03)
            - 'diffusion': float (0.0-1.0, default=0.5)
            - 'damping': float (0.0-1.0, default=0.5)
            - 'wet_gain': float (-20 to 0 dB, default=-6.0)
            
        distortion_params : dict, optional
            Distortion parameters dictionary:
            - 'gain': float (0-30 dB, default=10.0)
            - 'color': float (0.0-1.0, default=0.5)
            
        pitch_params : dict, optional
            Pitch parameters dictionary:
            - 'scale': float (0.25-4.0, default=1.0)
            
        Returns:
        --------
        tuple: (processed_audio, sample_rate)
            processed_audio : np.ndarray
                The processed audio signal
            sample_rate : int
                Sample rate of the audio
                
        Processing Order:
        ----------------
        1. EQ (frequency shaping first)
        2. Distortion (before reverb to avoid muddy sound)
        3. Pitch Shift (before reverb for natural sound)
        4. Reverb (last for realistic space simulation)
        """
        
        # Load audio
        try:
            audio, sr = librosa.load(input_path, sr=None)
            print(f"Loaded audio: {input_path} (SR: {sr}Hz, Duration: {len(audio)/sr:.2f}s)")
        except Exception as e:
            raise ValueError(f"Error loading audio: {e}")
        
        processed_audio = audio.copy()
        
        # Apply EQ
        if eq_bands:
            self.equalizer.reset()
            for band in eq_bands:
                self.equalizer.add_band(**band)
            processed_audio = self.equalizer.apply(processed_audio, sr)
            print("Applied EQ")
        
        # Apply Distortion  
        if distortion_params:
            self.distortion.set_parameters(**distortion_params)
            processed_audio = self.distortion.apply(processed_audio, sr)
            print("Applied Distortion")
        
        # Apply Pitch Shift
        if pitch_params:
            self.pitch.set_parameters(**pitch_params)
            processed_audio = self.pitch.apply(processed_audio, sr)
            print("Applied Pitch Shift")
        
        # Apply Reverb (usually last)
        if reverb_params:
            self.reverb.set_parameters(**reverb_params)
            processed_audio = self.reverb.apply(processed_audio, sr)
            print("Applied Reverb")
        
        # Save processed audio
        try:
            sf.write(output_path, processed_audio, sr)
            print(f"Processed audio saved to: {output_path}")
        except Exception as e:
            raise ValueError(f"Error saving audio: {e}")
        
        return processed_audio, sr


# Monster Voice and Example Configurations
def create_monster_voice_presets():
    """Create monster voice presets"""
    
    presets = {
        'deep_monster': {
            "Equalizer": {
                1: {"frequency": 100, "Gain": 5, "Q": 1.0, "Filter-type": "bell"},       # Deep bass boost
                2: {"frequency": 3500, "Gain": -3, "Q": 0.7, "Filter-type": "high-shelf"} # Cut sibilance
            },
            "Pitch": {
                "Scale": -7  # -7 semitones = deeper voice
            },
            "Distortion": {
                "Gain": 15,     # Heavy distortion for growl
                "Color": 0.8    # Aggressive character
            },
            "Reverb": {
                "Room Size": 9,     # Large cave-like space
                "Pre Delay": 0.1,   # Big space delay
                "Diffusion": 0.7,   # Thick reflections
                "Damping": 0.4,     # Slightly dark
                "Wet Gain": 0.8     # Heavy reverb mix
            }
        },
        
        'demon_voice': {
            "Equalizer": {
                1: {"frequency": 80, "Gain": 6, "Q": 1.2, "Filter-type": "bell"},        # Even deeper bass
                2: {"frequency": 200, "Gain": 3, "Q": 0.8, "Filter-type": "bell"},       # Growl emphasis
                3: {"frequency": 5000, "Gain": -5, "Q": 0.5, "Filter-type": "high-shelf"} # Cut highs
            },
            "Pitch": {
                "Scale": -12  # Full octave down
            },
            "Distortion": {
                "Gain": 20,     # Very heavy distortion
                "Color": 1.0    # Maximum aggressive
            },
            "Reverb": {
                "Room Size": 10,    # Massive space
                "Pre Delay": 0.15,  # Long delay
                "Diffusion": 0.9,   # Very diffuse
                "Damping": 0.2,     # Dark, ominous
                "Wet Gain": 0.9     # Almost full wet
            }
        },
        
        'robot_voice': {
            "Equalizer": {
                1: {"frequency": 400, "Gain": -2, "Q": 2.0, "Filter-type": "bell"},      # Remove human warmth
                2: {"frequency": 2000, "Gain": 4, "Q": 1.5, "Filter-type": "bell"},     # Metallic emphasis
                3: {"frequency": 8000, "Gain": -3, "Q": 0.7, "Filter-type": "high-shelf"} # Cut natural highs
            },
            "Pitch": {
                "Scale": -3  # Slightly lower
            },
            "Distortion": {
                "Gain": 8,      # Moderate distortion
                "Color": 0.6    # Digital character
            },
            "Reverb": {
                "Room Size": 2,     # Small metallic space
                "Pre Delay": 0.02,  # Short delay
                "Diffusion": 0.3,   # Less diffuse, more mechanical
                "Damping": 0.8,     # Very damped, metallic
                "Wet Gain": 0.3     # Subtle reverb
            }
        }
    }
    
    return presets


def test_monster_voice(input_file: str, output_dir: str = "monster_outputs"):
    """
    Test function to apply monster voice effects
    
    Parameters:
    -----------
    input_file : str
        Path to input audio file
    output_dir : str
        Directory to save processed files
    """
    
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get presets
    presets = create_monster_voice_presets()
    
    # Initialize toolbox
    toolbox = AudioToolbox(sample_rate=44100)
    
    print("🎭 Monster Voice Processor")
    print("=" * 50)
    
    for preset_name, config in presets.items():
        output_file = os.path.join(output_dir, f"{preset_name}_{os.path.basename(input_file)}")
        
        print(f"\n🔄 Processing: {preset_name}")
        print(f"Config: {config}")
        
        try:
            toolbox.process_with_config(input_file, output_file, config)
            print(f"✅ Saved: {output_file}")
        except Exception as e:
            print(f"❌ Error processing {preset_name}: {e}")
    
    print(f"\n🎉 All monster voices saved to: {output_dir}")


# Example usage and utility functions
def create_example_presets():
    """Create example presets for different effects"""
    
    presets = {
        'bright_eq': [
            {'frequency': 2000, 'gain': 3, 'q': 1.0, 'filter_type': 'peak'},
            {'frequency': 8000, 'gain': 2, 'q': 0.7, 'filter_type': 'peak'}
        ],
        
        'warm_eq': [
            {'frequency': 200, 'gain': 2, 'q': 1.0, 'filter_type': 'peak'},
            {'frequency': 1000, 'gain': -1, 'q': 0.5, 'filter_type': 'peak'},
            {'frequency': 4000, 'gain': -2, 'q': 1.0, 'filter_type': 'peak'}
        ],
        
        'hall_reverb': {
            'room_size': 0.8, 'pre_delay': 0.05, 'diffusion': 0.7, 
            'damping': 0.3, 'wet_gain': -9
        },
        
        'room_reverb': {
            'room_size': 0.4, 'pre_delay': 0.02, 'diffusion': 0.5,
            'damping': 0.5, 'wet_gain': -12
        },
        
        'soft_distortion': {'gain': 8, 'color': 0.2},
        'hard_distortion': {'gain': 15, 'color': 0.8},
        
        'octave_up': {'scale': 2.0},
        'octave_down': {'scale': 0.5},
        'slight_detune': {'scale': 1.05}
    }
    
    return presets


if __name__ == "__main__":
    # Example usage
    toolbox = AudioToolbox(sample_rate=44100)
    presets = create_example_presets()
    monster_presets = create_monster_voice_presets()
    
    print("🎵 Audio Tools - Monster Voice Edition")
    print("=" * 60)
    
    # Example with your monster voice config
    your_config = {
        "Equalizer": {
            1: {"frequency": 100, "Gain": 5, "Q": 1.0, "Filter-type": "bell"},       # Deep bass for monster tone
            2: {"frequency": 3500, "Gain": -3, "Q": 0.7, "Filter-type": "high-shelf"} # Cut sibilance for more 'inhuman' quality
        },
        "Reverb": {
            "Room Size": 9,         # Large room effect
            "Pre Delay": 0.1,       # To simulate a big space
            "Diffusion": 0.7,       # Thicker reflections
            "Damping": 0.4,         # Slightly darker tail
            "Wet Gain": 0.8         # Emphasize echo/space
        },
        "Distortion": {
            "Gain": 15,             # More aggressive, monstrous formant
            "Color": 0.8
        },
        "Pitch": {
            "Scale": -7             # Lower for monster effect
        }
    }
    
    print("\n📋 Your Monster Voice Config:")
    for effect, params in your_config.items():
        print(f"  {effect}: {params}")
    
    # Example processing (uncomment to use with actual audio file)
    """
    # Process with your config
    toolbox.process_with_config(
        input_path="input_voice.wav",
        output_path="monster_voice.wav",
        config=your_config
    )
    
    # Or test all monster presets
    test_monster_voice("input_voice.wav", "monster_outputs")
    """
    
    print("\n🎭 Available Monster Presets:")
    for preset_name in monster_presets.keys():
        print(f"  - {preset_name}")
    
    print("\n🔧 Available Standard Presets:")
    for preset_name in presets.keys():
        print(f"  - {preset_name}")
    
    print("\n💡 Usage Examples:")
    print("1. Process with your config:")
    print('   toolbox.process_with_config("input.wav", "output.wav", your_config)')
    print("\n2. Test all monster presets:")
    print('   test_monster_voice("input.wav", "monster_outputs")')
    print("\n3. Use a preset:")
    print('   toolbox.process_with_config("input.wav", "demon.wav", monster_presets["demon_voice"])')