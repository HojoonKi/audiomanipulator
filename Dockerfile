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

# Create audiomanipulator user (will be overridden by docker-compose if needed)
RUN groupadd -g 1000 audiomanipulator && \
    useradd -u 1000 -g 1000 -m -s /bin/bash audiomanipulator && \
    usermod -aG sudo audiomanipulator

# Create workspace directory with proper ownership
RUN mkdir -p /home/audiomanipulator/app && \
    chown -R audiomanipulator:audiomanipulator /home/audiomanipulator

WORKDIR /home/audiomanipulator/app

# Setup Python environment
USER audiomanipulator
RUN python3.10 -m venv venv
ENV PATH="/home/audiomanipulator/app/venv/bin:$PATH"

# Add virtual environment activation to .bashrc for interactive shells
RUN echo 'source /home/audiomanipulator/app/venv/bin/activate' >> ~/.bashrc

# Install Python packages
COPY --chown=audiomanipulator:audiomanipulator requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy application with proper ownership
COPY --chown=audiomanipulator:audiomanipulator . .

# Create necessary directories with proper permissions
RUN mkdir -p checkpoints output audio_dataset \
    && chmod 755 checkpoints output audio_dataset

# Switch to root to setup entrypoint and sudo
USER root

# Copy and setup entrypoint script with proper permissions
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Install sudo for entrypoint script
RUN apt-get update && apt-get install -y sudo && rm -rf /var/lib/apt/lists/* \
    && echo "audiomanipulator ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch back to audiomanipulator user
USER audiomanipulator
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["bash"]
EXPOSE 8080

# Labels
LABEL maintainer="HojoonKi"
LABEL description="Audio Manipulator: Text-to-Audio Effect Generation"
LABEL version="1.0"
