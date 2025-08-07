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
- [Docker Compose](https://docs.docker.com/compose/install/) (1.29+)
- For GPU support (RTX 3090/4090): [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

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

### 🚀 Complete Setup Guide

```bash
# 1. Clone the repository
git clone https://github.com/HojoonKi/audiomanipulator.git
cd audiomanipulator

# 2. Build and start container in background
docker-compose up -d --build

# 3. Verify container is running
docker-compose ps

# 4. Enter the container for interactive use
docker-compose exec audiomanipulator bash
```

### 🎯 Using the Container

Once the container is built and running, enter it and use the tools directly:

```bash
# Enter the container
docker-compose exec audiomanipulator bash

# Now you're inside the container - run any commands you need:
python test.py --help     # See test options
python train.py --help    # See training options
ls audio_dataset/         # Browse available audio files
nvidia-smi               # Check GPU status
```

That's it! From here, you can run training, testing, or any other commands directly inside the container environment.

### 🎛️ Container Management

```bash
# Essential commands
docker-compose up -d --build     # Build and start container
docker-compose exec audiomanipulator bash  # Enter container  
docker-compose ps                # Check running status
docker-compose logs -f           # View container logs
docker-compose restart           # Restart container
docker-compose down             # Stop and remove container
```

### 🔧 GPU Support (RTX 3090/4090 Ready)

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
- CUDA 12.1 support is enabled for RTX 3090/4090 compatibility
- If you don't have an NVIDIA GPU, the container will automatically fall back to CPU mode
- For older GPUs, you may need to modify the Dockerfile to use CUDA 11.8

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

## ⚡ Quick Reference

### Docker Commands Cheat Sheet

```bash
# 🚀 Essential Commands
docker-compose up -d --build           # Build and start container in background
docker-compose exec audiomanipulator bash  # Enter container
docker-compose ps                      # Check running status
docker-compose logs -f                 # View logs
docker-compose down                    # Stop and remove container

# 🔍 Debug & Monitoring Commands
docker-compose restart                 # Restart container
docker system prune -f                # Clean up (careful!)
```

### Model Parameters & Effects

| Effect Type | Parameters | Description |
|-------------|------------|-------------|
| **Equalizer** | filter_type (3 types)<br/>freq (20-20kHz)<br/>gain (-20 to +20 dB)<br/>Q factor (0.1-10) | 3-band EQ: low-shelf, bell, high-shelf |
| **Reverb** | room_size (0-1)<br/>damping (0-1)<br/>wet_level (0-1)<br/>dry_level (0-1) | Environmental reverb simulation |
| **Distortion** | drive (0-1)<br/>gain (0-1) | Analog-style saturation |
| **Pitch** | pitch_shift (-12 to +12 semitones) | Pitch shifting without tempo change |

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
   - 3-class filter type classification for EQ

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

## 📁 Project Structure

```
AudioManipulator/
├── 🐳 Docker Configuration
│   ├── Dockerfile                 # Main container definition
│   ├── docker-compose.yml         # Multi-service orchestration
│   └── .dockerignore              # Build context exclusions
├── 🎵 Audio Processing
│   ├── audio_tools/               # Audio I/O and preprocessing
│   ├── audio_dataset/             # Training data (instrumentals/speech)
│   └── output/                    # Generated audio outputs
├── 🧠 Model Components
│   ├── model/                     # Neural network architectures
│   ├── encoder/                   # Text encoding modules
│   ├── decoder/                   # Parameter generation
│   └── utils/                     # Parameter mapping utilities
├── 📊 Training & Evaluation
│   ├── train.py                   # Main training script
│   ├── test.py                    # Inference and testing
│   ├── pipeline.py               # Complete processing pipeline
│   └── checkpoints/              # Saved model weights
├── 📝 Data & Descriptions
│   ├── descriptions/              # Text-effect pair datasets
│   └── prompt/                    # Template prompts
└── 📚 Documentation
    ├── README.md                  # This file
    └── requirements.txt           # Python dependencies
```

## 🚀 Advanced Usage

### Custom Dataset Training

1. **Prepare Your Data**:
   ```bash
   # In Docker container
   mkdir -p /app/custom_dataset/audio
   mkdir -p /app/custom_dataset/descriptions
   
   # Copy your audio files
   cp /host/my_audio/* /app/custom_dataset/audio/
   
   # Create description file
   echo "warm vintage sound with analog saturation" > /app/custom_dataset/descriptions/custom.txt
   ```

2. **Train with Custom Data**:
   ```bash
   python train.py \
       --dataset_path /app/custom_dataset \
       --description_path /app/custom_dataset/descriptions/custom.txt \
       --epochs 50
   ```

### API Integration

```python
# Example Python integration (in container)
from pipeline import AudioManipulatorPipeline

# Initialize pipeline
pipeline = AudioManipulatorPipeline(device="cuda")

# Process audio
result = pipeline.process(
    audio_path="/app/input.wav",
    text_prompt="bright and crisp studio sound",
    output_path="/app/output.wav"
)

print(f"Generated parameters: {result['parameters']}")
```

### Batch Processing

```bash
# Process multiple files with Docker
cat audio_list.txt | while read audio_file description; do
    docker-compose exec audiomanipulator python test.py \
        --input_audio "$audio_file" \
        --text_prompt "$description" \
        --output_audio "output_$(basename $audio_file)"
done
```

## 🎮 Interactive Development

### Jupyter Notebook Environment

```bash
# Start Jupyter service
docker-compose --profile notebook up -d audiomanipulator-notebook

# Access at http://localhost:8888
# Default password: audiomanipulator
```

The notebook environment includes:
- Pre-configured audio processing tools
- Model experimentation notebooks
- Real-time parameter visualization
- Interactive effect demonstration

### Development Workflow

```bash
# 1. Code with live reload
docker-compose exec audiomanipulator bash
cd /app && python -m pytest tests/  # Run tests

# 2. Train with monitoring
docker-compose --profile training up audiomanipulator-train

# 3. Experiment in Jupyter
# Visit http://localhost:8888 and open groundit_demo.ipynb
```

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/HojoonKi/audiomanipulator.git
cd audiomanipulator

# Start development container
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Install pre-commit hooks (optional)
docker-compose exec audiomanipulator pre-commit install
```

### Code Style

```bash
# Format code (in container)
black .
isort .
flake8 .
```

## 📋 Troubleshooting

### Common Docker Issues

**Container won't start**:
```bash
# Check logs
docker-compose logs audiomanipulator

# Rebuild if needed
docker-compose build --no-cache audiomanipulator
```

**GPU not detected (RTX 3090/4090)**:
```bash
# Verify NVIDIA Container Toolkit installation
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Check if NVIDIA Container Toolkit is properly configured
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# For older GPUs, try CUDA 11.8 instead
# Modify Dockerfile: FROM nvidia/cuda:11.8-devel-ubuntu22.04
```

**Permission issues**:
```bash
# Fix volume permissions
sudo chown -R $USER:$USER output/
sudo chown -R $USER:$USER checkpoints/
```

**Out of memory**:
```bash
# Reduce batch size in training
python train.py --batch_size 16  # Instead of 32

# Or use CPU mode
python train.py --device cpu
```

### Native Installation Issues

<details>
<summary>Click to expand native installation troubleshooting</summary>

**ImportError: No module named 'torch'**:
```bash
# Verify environment activation
conda activate audio_tools  # or source audio_env/bin/activate
pip list | grep torch
```

**CUDA not available**:
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Audio processing errors**:
```bash
# Install system audio libraries (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg

# macOS
brew install libsndfile ffmpeg
```

</details>

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **AudioLDM2**: Advanced audio generation model integration
- **GrounDiT**: Grounding and text-to-audio synthesis
- **Pedalboard**: Real-time audio effects processing
- **HuggingFace Transformers**: Text encoding and model hosting

---

**📞 Support**: For issues or questions, please create a GitHub issue or contact the development team.

**🔄 Updates**: This project is actively maintained. Check for updates regularly.

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
2. **Enhanced Backbone**: Cross-Attention Fusion + Processing Network
   - **Cross-Modal Fusion Block**: Sophisticated bidirectional attention between text and CLAP embeddings
   - **CLAP-to-Text Attention**: CLAP embeddings attend to rich textual context for nuanced understanding
   - **Text-to-CLAP Attention**: Text embeddings become audio-aware through CLAP context
   - **Self-Attention Refinement**: Further processing for enhanced representations
   - **Learned Fusion**: Context-dependent combination (not fixed like concatenation)
3. **Parallel Effect Decoders**: 
   - EQ Decoder: 5 bands × 4 parameters = 20 outputs
   - Reverb Decoder: 6 parameters (room_size, pre_delay, diffusion, damping, wet_gain, dry_gain)
   - Distortion Decoder: 4 parameters (gain, bias, tone, mix)
   - Pitch Decoder: 3 parameters (pitch_shift, formant_shift, mix)

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