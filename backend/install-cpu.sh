#!/bin/bash
# Install script for CPU-only dependencies

set -e

cd /home/rag/voicebird/backend

# Remove existing venv if it exists
rm -rf venv

# Create new virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch CPU version first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install fastapi>=0.104.0
pip install "uvicorn[standard]>=0.24.0"
pip install python-multipart>=0.0.6
pip install python-dotenv>=1.0.0
pip install transformers>=4.36.0
pip install accelerate>=0.25.0
pip install librosa>=0.10.0
pip install soundfile>=0.12.0
pip install yt-dlp>=2024.1.0
pip install pymysql>=1.1.0
pip install cryptography>=41.0.0

# Try to install Intel extension (optional, may fail)
pip install intel-extension-for-pytorch>=2.0.0 || echo "Intel extension not available, skipping..."

echo "Installation complete!"
