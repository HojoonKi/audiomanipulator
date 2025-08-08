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
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
RUN mkdir -p /app

WORKDIR /app

# Setup Python environment (as root for universal access)
RUN python3.10 -m venv venv \
    && chmod -R 755 /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt 

# Copy application
COPY . .

# Create necessary directories with full permissions for any user
RUN mkdir -p checkpoints output audio_dataset \
    && chmod -R 755 checkpoints output audio_dataset \
    && chmod -R 755 /app

CMD ["bash"]
EXPOSE 8080

# Labels
LABEL maintainer="HojoonKi"
LABEL description="Audio Manipulator: Text-to-Audio Effect Generation"
LABEL version="1.0"

# Create entrypoint script for dynamic user setup
RUN printf '#!/bin/bash\n\
# Create user dynamically if not root\n\
if [ "$(id -u)" != "0" ]; then\n\
    USER_ID=$(id -u)\n\
    GROUP_ID=$(id -g)\n\
    # Create group if it does not exist\n\
    if ! getent group $GROUP_ID > /dev/null 2>&1; then\n\
        groupadd -g $GROUP_ID usergroup\n\
    fi\n\
    # Create user if it does not exist\n\
    if ! getent passwd $USER_ID > /dev/null 2>&1; then\n\
        useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash user\n\
        echo "source /app/venv/bin/activate" >> /home/user/.bashrc\n\
    fi\n\
else\n\
    # Root user - just add venv activation to bashrc\n\
    echo "source /app/venv/bin/activate" >> /root/.bashrc\n\
fi\n\
# Activate virtual environment\n\
source /app/venv/bin/activate\n\
exec "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
