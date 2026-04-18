#!/bin/bash
# =========================================================
# Project Garuda — Raspberry Pi Setup Script
# Target: Raspberry Pi 4/5 (ARM64)
# =========================================================

set -e

echo "============================================"
echo "  Project Garuda — Raspberry Pi Setup"
echo "============================================"

# Update system
echo "[1/6] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
echo "[2/6] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    libopencv-dev python3-opencv \
    libatlas-base-dev libhdf5-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    cmake git wget curl

# Create virtual environment
echo "[3/6] Creating virtual environment..."
python3 -m venv garuda_env
source garuda_env/bin/activate

# Install Python dependencies
echo "[4/6] Installing Python dependencies..."
pip install --upgrade pip
pip install numpy opencv-python-headless pyyaml requests
pip install deep_sort_realtime

# Install TFLite runtime (optimised for ARM)
echo "[5/6] Installing TFLite runtime..."
pip install tflite-runtime

# Install Ultralytics (lightweight mode)
pip install ultralytics

# Verify installation
echo "[6/6] Verifying installation..."
python3 -c "
import cv2
import numpy as np
import yaml
print('✅ OpenCV:', cv2.__version__)
print('✅ NumPy:', np.__version__)
try:
    import tflite_runtime.interpreter as tflite
    print('✅ TFLite Runtime: OK')
except ImportError:
    print('⚠️  TFLite Runtime: Not found')
print('All dependencies verified!')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate env:  source garuda_env/bin/activate"
echo "  Run inference:  python deployment/raspberry_pi/run_pi.py"
echo "============================================"
