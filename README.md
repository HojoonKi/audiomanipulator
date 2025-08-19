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

## 🐳 Quick Start with Docker (Recommended)

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) (20.10+)
- For GPU support (RTX 20xx/30xx/40xx): [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### 🔧 GPU Setup (RTX 3090/4090 Users)

Before building the container, install NVIDIA Container Toolkit:

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify installation
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### 🚀 Complete Setup Guide (run.sh 기반)

```bash
# 1) Clone
git clone https://github.com/HojoonKi/audiomanipulator.git
cd audiomanipulator

# 2) (권장) 허깅페이스 토큰 설정: 모델/토크나이저 다운로드를 위해 필요할 수 있음
export HUGGINGFACE_HUB_TOKEN=hf_xxx

# 3) dasp-pytorch clone
git clone https://github.com/HojoonKi/dasp-pytorch.git

# 4) 컨테이너 빌드 및 진입
./run.sh

# 5) (컨테이너 내부) conda 환경 자동 활성화
python train.py --help
```

- 모든 의존성은 environment.yml로 관리됩니다.
- requirements.txt, venv 등은 더 이상 사용하지 않습니다.

### 🎯 Using the Container

컨테이너는 `run.sh` 실행 시 자동으로 진입합니다. 이미 실행 중인 컨테이너에 다시 접속하려면:

```bash
docker exec -it audiomanipulator bash

# 컨테이너 내부 예시
python test.py --help
python train.py --help
nvidia-smi
```

### 🎛️ Container Management

```bash
# Build & run and attach
./run.sh

# Attach to running container later
docker exec -it audiomanipulator bash

# Stop / remove
docker stop audiomanipulator
docker rm audiomanipulator

# Logs (follow)
docker logs -f audiomanipulator
```

### 🔧 GPU Support (CUDA 12.1 Ready)

The container is pre-configured with CUDA 12.1 for RTX 3090/4090 support. Once inside the container, verify it's working:

```bash
# Check CUDA availability (run inside container)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Check GPU memory (run inside container)
nvidia-smi
```

**Note**: 
- CUDA 12.1 기반으로 구성됨
- NVIDIA GPU가 없으면 자동으로 CPU 모드로 동작
- 구형 GPU는 Dockerfile에서 CUDA 11.8 등으로 변경 필요할 수 있음

## 🛠️ Alternative Installation Methods


<details>
<summary>📋 Native Installation (Click to expand)</summary>

### Option 1: Using Conda (Recommended)

```bash
# 1. Create conda environment with Python 3.10
conda create -n audio_tools python=3.10 -y
conda activate audio_tools

# 2. Clone the repository
git clone https://github.com/HojoonKi/audiomanipulator.git
cd audiomanipulator

# 3. Install dependencies from environment.yml
conda env update -f environment.yml

# 4. Install additional pip packages if needed
pip install -r requirements.txt
```

### Option 2: Using pip only

```bash
# 1. Create virtual environment with Python 3.10+ (recommended)
python3.10 -m venv audio_env
source audio_env/bin/activate  # On Windows: audio_env\Scripts\activate

# 2. Upgrade pip and install basic tools
pip install --upgrade pip setuptools wheel

# 3. Clone the repository
git clone https://github.com/HojoonKi/audiomanipulator.git
cd audiomanipulator

# 4. Install dependencies
pip install -r requirements.txt
```

### System Requirements (Native)
- **Python**: 3.10+ (recommended, minimum 3.8)
- **GPU**: CUDA-compatible GPU recommended (for training)
- **RAM**: Minimum 8GB, 16GB+ recommended for training
- **Storage**: At least 5GB free space for models and datasets
- **OS**: Linux, macOS, or Windows with WSL2

</details>


### Model Parameters & Effects

| Effect Type | Parameters | Description |
|-------------|------------|-------------|
| **Equalizer** | filter_type (5 types)<br/>freq (20-20kHz)<br/>gain (-20 to +20 dB)<br/>Q factor (0.1-10) | 5-band EQ: low-shelf, bell, high-shelf, low-pass, high-pass |
| **Reverb** | room_size (0-1)<br/>damping (0-1)<br/>wet_level (0-1)<br/>dry_level (0-1) | Environmental reverb simulation |
| **Distortion** | drive (0-1)<br/>gain (0-1) | Analog-style saturation |
| **Pitch** | pitch (-12 to +12 semitones) | Pitch shifting without tempo change |

### Environment Variables

```bash
# Optional Docker environment variables
export CUDA_VISIBLE_DEVICES=0          # Specify GPU
export TORCH_HOME=/app/cache/torch      # PyTorch model cache
export HF_HOME=/app/cache/huggingface   # HuggingFace cache
export WANDB_CACHE_DIR=/app/cache/wandb # Weights & Biases cache
```

## 🔬 Model Architecture

### Core Components

1. **Text Encoder**: 
   - SentenceTransformer (`all-mpnet-base-v2`) for semantic understanding
   - CLAP encoder for audio-text alignment

2. **Audio Encoder**: 
   - Mel-spectrogram features with differentiable processing
   - Cross-attention fusion with text embeddings

3. **Parallel Decoder**:
   - Separate heads for each effect type
   - Constraint-aware parameter generation
   - 5-class filter type classification for EQ

### Training Process

```bash
# Inside Docker container, run training with desired parameters
python train.py \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --save_checkpoint_every 10

# With custom dataset
python train.py \
    --dataset_path /app/custom_dataset \
    --description_path /app/custom_descriptions.txt
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Pedalboard**: Real-time audio effects processing
- **Text2FX**: Former research on text to preset parameters
- **HuggingFace Transformers**: Text encoding and model hosting

---


## 🚀 Model Testing Usage

### Basic Usage

Once inside the Docker container, you can test the model directly:

```bash
# Basic test (automatically loads latest checkpoint)
python test.py \
    --input_audio ./audio_dataset/instrumentals/flute/852.wav \
    --text_prompt "add warm reverb and make it sound spacious"

# Specific effect requests
python test.py \
    --input_audio ./audio_dataset/instrumentals/Piano/1234.wav \
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
    --input_audio ./audio_dataset/instrumentals/Violin/sample.wav \
    --text_prompt "add studio quality compression" \
    --output_path ./output/my_results/compressed_audio.wav

# Limit audio length (default: maintain original length)
python test.py \
    --input_audio ./audio_dataset/instrumentals/Guitar/long_audio.wav \
    --text_prompt "deep space reverb" \
    --audio_length 10.0

# Run on CPU (if GPU not available)
python test.py \
    --input_audio ./audio_dataset/instrumentals/flute/852.wav \
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
# Add spatial feel (inside container)
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
# Vintage/analog sound (inside container)
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
# Brightness control (inside container)
python test.py --input_audio audio.wav --text_prompt "brighter with crisp highs"
python test.py --input_audio audio.wav --text_prompt "warm and mellow"
python test.py --input_audio audio.wav --text_prompt "bass-heavy and punchy"

# Specific frequency enhancement
python test.py --input_audio audio.wav --text_prompt "enhance vocal presence"
python test.py --input_audio audio.wav --text_prompt "boost midrange clarity"
```

#### Dynamics/Compression
```bash
# Compression effects (inside container)
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
    "equalizer": [
      {"center_freq": 100, "gain_db": 2.5, "q": 0.7, "filter_type": "high_pass"},
      {"center_freq": 500, "gain_db": -1.2, "q": 1.4, "filter_type": "bell"},
      {"center_freq": 2000, "gain_db": 0.8, "q": 2.1, "filter_type": "bell"},
      {"center_freq": 8000, "gain_db": -0.5, "q": 1.8, "filter_type": "bell"},
      {"center_freq": 15000, "gain_db": -2.0, "q": 0.9, "filter_type": "low_pass"}
    ],
    "reverb": {
      "room_size": 0.3,
      "pre_delay": 15.2,
      "diffusion": 0.7,
      "damping": 0.4,
      "wet_gain": 0.2,
    },
    "distortion": {
      "gain": 3.2,
      "color": 0.1,
    },
    "pitch": {
      "pitch": 1.0,
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
2. **Backbone**: Cross-Attention Fusion + Processing Network
   - **Cross-Modal Fusion Block**: Sophisticated bidirectional attention between text and CLAP embeddings
   - **CLAP-to-Text Attention**: CLAP embeddings attend to rich textual context for nuanced understanding
   - **Text-to-CLAP Attention**: Text embeddings become audio-aware through CLAP context
   - **Self-Attention Refinement**: Further processing for enhanced representations
   - **Learned Fusion**: Context-dependent combination (not fixed like concatenation)
3. **Parallel Effect Decoders**: 
   - EQ Decoder: 5 bands × 4 parameters = 20 outputs
   - Reverb Decoder: 5 parameters (room_size, pre_delay, diffusion, damping, wet_gain)
   - Distortion Decoder: 2 parameters (gain, color)
   - Pitch Decoder: 1 parameters (pitch)

### 🔄 Cross-Attention Fusion Benefits

- **Bidirectional Information Flow**: CLAP embeddings leverage rich textual context while text embeddings become audio-aware
- **Context-Dependent Fusion**: Learned attention weights adapt to different audio concepts (vs. fixed concatenation/addition)
- **Richer Multimodal Understanding**: Sophisticated interaction between text semantics and audio characteristics
- **Better Parameter Generation**: More nuanced audio effect parameters through enhanced cross-modal representations

## 📁 Project Structure

```
AudioManipulator/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 environment.yml              # Conda environment file
├── 📄 train.py                     # Main training script
├── 📄 test.py                      # Model testing script
├── 📄 dynamic_pipeline_factory.py  # Model architecture factory
├── 📄 dataset.py                   # Data loading utilities
├── 📁 encoder/                     # Text encoders
│   └── text_encoder.py
├── 📁 decoder/                     # Audio effect decoders
│   └── decoder.py
├── 📁 model/                       # Core model components
│   ├── attention.py               # Cross-attention mechanisms
│   └── backbone_model.py          # Enhanced backbone with fusion
├── 📁 audio_tools/                # Audio processing utilities
│   ├── audio_tools.py
│   └── torchaudio_processor.py
├── 📁 utils/                      # Utility functions
│   └── parameter_mapper.py
├── 📁 descriptions/               # Training descriptions
│   ├── descriptions.txt
│   └── fined_presets_filtered.py
├── 📁 audio_dataset/             # Training audio data
│   ├── instrumentals/
│   └── speech/
├── 📁 checkpoints/               # Model checkpoints
└── 📁 output/                    # Generated results
```

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