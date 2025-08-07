# Audio Manipulator: Text-to-Audio Effect Generation

## 🎯 Experiment Purpose

This project aims to develop an **AI model that automatically generates audio effects based on text descriptions**.

### Core Research Goals
- **Natural Language → Audio Effects**: Convert text like "warm vintage sound" into actual audio processing parameters
- **Multimodal Learning**: Combination of text encoders (SentenceTransformer + CLAP) and audio processors
- **Real-time Application**: System capable of immediately applying generated parameters to actual audio

### Technical Innovations
1. **Differentiable Audio Processing**: End-to-end learning with differentiable audio effects
2. **Environment-Aware Reverb**: Intelligent parameter generation considering environment-specific reverb characteristics
3. **Cross-Attention Multimodal Fusion**: Sophisticated bidirectional attention between text and audio embeddings for richer understanding
4. **Parallel Decoder Architecture**: Parallel prediction of each effect (EQ, Reverb, Distortion, Pitch) with enhanced backbone features

## 🚀 Model Testing Usage

### Basic Usage

```bash
# Basic test (automatically loads latest checkpoint)
python test.py \
    --input_audio /path/to/audio.wav \
    --text_prompt "add warm reverb and make it sound spacious"

# Specific effect requests
python test.py \
    --input_audio ./audio_dataset/instrumentals/flute/852.wav \
    --text_prompt "vintage analog warmth with tape saturation"
```

### Advanced Usage

```bash
# Use specific checkpoint
python test.py \
    --input_audio ./audio_dataset/instrumentals/flute/3591.wav \
    --text_prompt "make it brighter with crisp highs" \
    --checkpoint_path ./checkpoints/checkpoint_epoch_50.pt

# Specify output path
python test.py \
    --input_audio ./audio_dataset/speech/sample.wav \
    --text_prompt "add studio quality compression" \
    --output_path ./my_results/compressed_audio.wav

# Limit audio length (default: maintain original length)
python test.py \
    --input_audio ./long_audio.wav \
    --text_prompt "deep space reverb" \
    --audio_length 10.0

# Run on CPU
python test.py \
    --input_audio ./audio.wav \
    --text_prompt "lo-fi vintage sound" \
    --device cpu
```

### 📋 Parameter Description

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--input_audio` | ✅ | - | Input audio file path (.wav, .mp3, .flac supported) |
| `--text_prompt` | ✅ | - | Effect description (natural language) |
| `--output_path` | ❌ | Auto-generated | Output file path |
| `--audio_length` | ❌ | Original length | Audio length to process (seconds) |
| `--checkpoint_dir` | ❌ | `./checkpoints` | Checkpoint folder |
| `--checkpoint_path` | ❌ | Latest auto-selected | Specific checkpoint file |
| `--device` | ❌ | `auto` | Device (`cuda`, `cpu`, `auto`) |

### 🎵 Text Prompt Examples

#### Environment/Spatial Effects
```bash
# Add spatial feel
python test.py --input_audio audio.wav --text_prompt "make it sound like it's in a cathedral"
python test.py --input_audio audio.wav --text_prompt "intimate bedroom recording"
python test.py --input_audio audio.wav --text_prompt "wide stereo field with ambient space"

# Environment-specific reverb
python test.py --input_audio audio.wav --text_prompt "concert hall acoustics"
python test.py --input_audio audio.wav --text_prompt "small studio room"
python test.py --input_audio audio.wav --text_prompt "outdoor natural reverb"
```

#### Tone/Timbre Changes
```bash
# Vintage/analog sound
python test.py --input_audio audio.wav --text_prompt "warm analog tape saturation"
python test.py --input_audio audio.wav --text_prompt "vintage 1970s sound"
python test.py --input_audio audio.wav --text_prompt "tube amplifier warmth"

# Modern sound
python test.py --input_audio audio.wav --text_prompt "crystal clear digital quality"
python test.py --input_audio audio.wav --text_prompt "modern pop production"
python test.py --input_audio audio.wav --text_prompt "professional studio sound"
```

#### Frequency/EQ Adjustment
```bash
# Brightness control
python test.py --input_audio audio.wav --text_prompt "brighter with crisp highs"
python test.py --input_audio audio.wav --text_prompt "warm and mellow"
python test.py --input_audio audio.wav --text_prompt "bass-heavy and punchy"

# Specific frequency enhancement
python test.py --input_audio audio.wav --text_prompt "enhance vocal presence"
python test.py --input_audio audio.wav --text_prompt "boost midrange clarity"
```

#### Dynamics/Compression
```bash
# Compression effects
python test.py --input_audio audio.wav --text_prompt "gentle compression for smoothness"
python test.py --input_audio audio.wav --text_prompt "heavy compression for punch"
python test.py --input_audio audio.wav --text_prompt "vintage opto compressor sound"
```

## 📁 Output Results

After execution, results are saved in the following structure:

```
output/
└── 20250807_074237_warm_analog_tape_saturation/
    ├── warm_analog_tape_saturation.wav           # Processed audio
    ├── warm_analog_tape_saturation_preset.json   # Generated effect parameters
    └── metadata.txt                              # Processing information summary
```

### JSON Parameter Structure
```json
{
  "timestamp": "2025-08-07T07:42:38.348077",
  "text_prompt": "warm analog tape saturation",
  "model_info": {
    "text_encoder": "sentence-transformer-large",
    "sample_rate": 44100,
    "backbone_type": "residual",
    "decoder_type": "parallel"
  },
  "preset_parameters": {
    "equalizer": {
      "band_1": {"center_freq": 100, "gain_db": 2.5, "q": 0.7, "filter_type": "high_pass"},
      "band_2": {"center_freq": 500, "gain_db": -1.2, "q": 1.4, "filter_type": "bell"},
      "band_3": {"center_freq": 2000, "gain_db": 0.8, "q": 2.1, "filter_type": "bell"},
      "band_4": {"center_freq": 8000, "gain_db": -0.5, "q": 1.8, "filter_type": "bell"},
      "band_5": {"center_freq": 15000, "gain_db": -2.0, "q": 0.9, "filter_type": "low_pass"}
    },
    "reverb": {
      "room_size": 0.3,
      "pre_delay": 15.2,
      "diffusion": 0.7,
      "damping": 0.4,
      "wet_gain": 0.2,
      "dry_gain": 0.8
    },
    "distortion": {
      "gain": 3.2,
      "bias": 0.1,
      "tone": 0.6,
      "mix": 0.3
    },
    "pitch": {
      "pitch_shift": 1.0,
      "formant_shift": 1.0,
      "mix": 0.0
    }
  }
}
```

## 🛠️ System Requirements

- **Python**: 3.9+
- **GPU**: CUDA support recommended (CPU also possible but slower)
- **Memory**: Minimum 8GB RAM
- **Storage**: ~1GB for model checkpoints

## 📊 Model Architecture

1. **Text Encoder**: SentenceTransformer-large (768D) + CLAP (512D)
2. **Enhanced Backbone**: Cross-Attention Fusion + Processing Network
   - **Cross-Modal Fusion Block**: Sophisticated bidirectional attention between text and CLAP embeddings
   - **CLAP-to-Text Attention**: CLAP embeddings attend to rich textual context for nuanced understanding
   - **Text-to-CLAP Attention**: Text embeddings become audio-aware through CLAP context
   - **Self-Attention Refinement**: Further processing for enhanced representations
   - **Learned Fusion**: Context-dependent combination (not fixed like concatenation)
3. **Parallel Effect Decoders**: 
   - EQ Decoder: 5 bands × 3 parameters = 15 outputs
   - Reverb Decoder: 6 parameters (room_size, pre_delay, diffusion, damping, wet_gain, dry_gain)
   - Distortion Decoder: 4 parameters (gain, bias, tone, mix)
   - Pitch Decoder: 3 parameters (pitch_shift, formant_shift, mix)

### 🔄 Cross-Attention Fusion Benefits

- **Bidirectional Information Flow**: CLAP embeddings leverage rich textual context while text embeddings become audio-aware
- **Context-Dependent Fusion**: Learned attention weights adapt to different audio concepts (vs. fixed concatenation/addition)
- **Richer Multimodal Understanding**: Sophisticated interaction between text semantics and audio characteristics
- **Better Parameter Generation**: More nuanced audio effect parameters through enhanced cross-modal representations

## 🎯 Performance Metrics

- **Total Parameters**: ~271M (Base) + ~15M (Cross-Attention Fusion) = ~286M
- **Trainable Parameters**: ~3.3M (Base) + ~15M (Cross-Attention) = ~18.3M  
- **Inference Time**: ~2-3 seconds (GPU), ~10-15 seconds (CPU)
- **Supported Formats**: WAV, MP3, FLAC
- **Sample Rate**: 44.1kHz (automatic conversion)
- **Cross-Attention Heads**: 8 (for optimal multimodal fusion)
- **Fusion Architecture**: Bidirectional attention with self-refinement

---

*This project explores new possibilities in text-based audio effect generation through sophisticated cross-attention fusion, aiming to revolutionize music production and audio post-production workflows with enhanced multimodal understanding.*