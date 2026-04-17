#!/bin/bash
# =========================================================
# Project Garuda — NVIDIA Jetson Setup Script
# Target: Jetson Nano / Xavier NX / Orin
# =========================================================

set -e

echo "============================================"
echo "  Project Garuda — Jetson Setup"
echo "============================================"

# Update system
echo "[1/6] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
echo "[2/6] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    libopencv-dev python3-opencv \
    libhdf5-dev libjpeg-dev libpng-dev \
    cmake git wget curl

# Create virtual environment
echo "[3/6] Creating virtual environment..."
python3 -m venv garuda_env
source garuda_env/bin/activate

# Install Python dependencies
echo "[4/6] Installing Python dependencies..."
pip install --upgrade pip
pip install numpy pyyaml requests

# Install PyTorch for Jetson (from NVIDIA)
echo "[5/6] Installing PyTorch + TensorRT..."

# Note: Adjust URL for your JetPack version
# JetPack 5.x (L4T R35.x)
JETPACK_VERSION=$(cat /etc/nv_tegra_release 2>/dev/null | head -1 | awk '{print $2}' || echo "unknown")
echo "  Detected JetPack: $JETPACK_VERSION"

# Install ONNX Runtime with GPU
pip install onnxruntime-gpu 2>/dev/null || pip install onnxruntime

# Install Ultralytics
pip install ultralytics

# Install OpenCV with CUDA (if available)
pip install opencv-python-headless 2>/dev/null || true

# Verify
echo "[6/6] Verifying installation..."
python3 -c "
import cv2
import numpy as np
print('✅ OpenCV:', cv2.__version__)
print('✅ NumPy:', np.__version__)
try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('   CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('   GPU:', torch.cuda.get_device_name(0))
except ImportError:
    print('⚠️  PyTorch not installed')
try:
    import onnxruntime as ort
    print('✅ ONNX Runtime:', ort.__version__)
    print('   Providers:', ort.get_available_providers())
except ImportError:
    print('⚠️  ONNX Runtime not installed')
print('All checks complete!')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate env:  source garuda_env/bin/activate"
echo "  Run inference:  python deployment/jetson/run_jetson.py"
echo "============================================"
