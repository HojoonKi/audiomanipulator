FROM nvidia/cuda:11.8-devel-ubuntu22.04

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
    && rm -rf /var/lib/apt/lists/*

# Create user and workspace
RUN useradd -m audiomanipulator
USER audiomanipulator
WORKDIR /home/audiomanipulator/app

# Setup Python environment
RUN python3.10 -m venv venv
ENV PATH="/home/audiomanipulator/app/venv/bin:$PATH"

# Install Python packages
COPY --chown=audiomanipulator:audiomanipulator requirements.txt .
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

# Copy application
COPY --chown=audiomanipulator:audiomanipulator . .

# Create directories
RUN mkdir -p checkpoints output

CMD ["bash"]
EXPOSE 8080

# Labels
LABEL maintainer="HojoonKi"
LABEL description="Audio Manipulator: Text-to-Audio Effect Generation"
LABEL version="1.0"
