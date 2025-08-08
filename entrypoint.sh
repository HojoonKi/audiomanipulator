#!/bin/bash
# Docker entrypoint script to handle permissions and activate virtual environment

# Fix permissions for mounted volumes
if [ -d "/home/audiomanipulator/app" ]; then
    # Ensure the audiomanipulator user owns the app directory
    sudo chown -R audiomanipulator:audiomanipulator /home/audiomanipulator/app 2>/dev/null || true
    
    # Ensure mounted volumes have correct permissions
    sudo chmod -R 755 /home/audiomanipulator/app/checkpoints 2>/dev/null || true
    sudo chmod -R 755 /home/audiomanipulator/app/output 2>/dev/null || true
    sudo chmod -R 755 /home/audiomanipulator/app/audio_dataset 2>/dev/null || true
fi

# Activate virtual environment if it exists
if [ -f "/home/audiomanipulator/app/venv/bin/activate" ]; then
    source /home/audiomanipulator/app/venv/bin/activate
    echo "✅ Virtual environment activated"
    echo "🐍 Python: $(which python)"
    echo "📦 PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
else
    echo "⚠️  Virtual environment not found, using system Python"
fi

# Execute the command as audiomanipulator user
exec "$@"
