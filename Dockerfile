# Use CUDA 12.1 for RTX 3090/4090 compatibility
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    tmux \
    vim \
    htop \
    libsndfile1 \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
RUN mkdir -p /app

WORKDIR /app

# Setup Python environment (as root for universal access)
RUN python3.10 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Add virtual environment activation to .bashrc for interactive shells
RUN echo 'source /app/venv/bin/activate' >> /root/.bashrc

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt 

# Copy application
COPY . .

# Create necessary directories with full permissions
RUN mkdir -p checkpoints output audio_dataset \
    && chmod 777 checkpoints output audio_dataset

# Switch back to audiomanipulator user
USER audiomanipulator
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["bash"]
EXPOSE 8080

# Labels
LABEL maintainer="HojoonKi"
LABEL description="Audio Manipulator: Text-to-Audio Effect Generation"
LABEL version="1.0"
