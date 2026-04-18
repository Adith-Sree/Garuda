#!/bin/bash
# =========================================================
# Project Garuda — Raspberry Pi 4 Setup Script (Minimal)
# Target : Raspberry Pi OS 64-bit (Bullseye / Bookworm)
# Footprint: ~120 MB total Python packages
# =========================================================

set -euo pipefail

PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$PI_DIR/../.." && pwd)"

echo "============================================"
echo "  Project Garuda — Raspberry Pi 4 Setup"
echo "  Project root: $PROJECT_ROOT"
echo "============================================"

# ----------------------------------------------------------
# 1. System dependencies (minimal — no libhdf5, no cmake)
# ----------------------------------------------------------
echo ""
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    libopenblas0 \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    wget

# ----------------------------------------------------------
# 2. Create virtual environment
# ----------------------------------------------------------
echo ""
echo "[2/5] Creating virtual environment..."
cd "$PROJECT_ROOT"
python3 -m venv garuda_env
source garuda_env/bin/activate
pip install --upgrade pip --quiet

# ----------------------------------------------------------
# 3. Install Pi-only Python dependencies
# ----------------------------------------------------------
echo ""
echo "[3/5] Installing Python dependencies (~120 MB)..."
# opencv-python-headless avoids pulling in Qt/GTK/display libs
pip install --quiet \
    "numpy>=1.24.0,<2.0" \
    "opencv-python-headless>=4.8.0" \
    "PyYAML>=6.0"

# ----------------------------------------------------------
# 4. Install ONNX Runtime (ARM64 wheel shipped on PyPI)
# ----------------------------------------------------------
echo ""
echo "[4/5] Installing ONNX Runtime for ARM64..."
pip install --quiet "onnxruntime>=1.16.0"
# NOTE: We do NOT install onnxruntime-gpu — Pi 4 has no CUDA/NPU.
# NOTE: We do NOT install torch / ultralytics — model is already ONNX.
# NOTE: We do NOT install deep_sort_realtime — built-in IOU tracker is used.
# NOTE: We do NOT install requests — webhook alerts are disabled by default.

# ----------------------------------------------------------
# 5. Verify installation
# ----------------------------------------------------------
echo ""
echo "[5/5] Verifying installation..."
python3 - <<'PYEOF'
import sys

errors = []

try:
    import cv2
    print(f"  ✅ OpenCV       : {cv2.__version__}")
except ImportError as e:
    errors.append(f"OpenCV: {e}")

try:
    import numpy as np
    print(f"  ✅ NumPy        : {np.__version__}")
except ImportError as e:
    errors.append(f"NumPy: {e}")

try:
    import yaml
    print(f"  ✅ PyYAML       : {yaml.__version__}")
except ImportError as e:
    errors.append(f"PyYAML: {e}")

try:
    import onnxruntime as ort
    print(f"  ✅ ONNX Runtime : {ort.__version__}")
    print(f"     Providers    : {ort.get_available_providers()}")
except ImportError as e:
    errors.append(f"onnxruntime: {e}")

if errors:
    print("\n  ❌ The following packages failed to import:")
    for err in errors:
        print(f"     {err}")
    sys.exit(1)
else:
    print("\n  All dependencies verified successfully!")
PYEOF

# Create required directories
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/models/pi4"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate env   :  source garuda_env/bin/activate"
echo ""
echo "  ⚠  Before running, export your model:"
echo "     yolo export model=yolov8n.pt format=onnx imgsz=320 simplify=True"
echo "     mv yolov8n.onnx models/pi4/yolov8n_pi4_320.onnx"
echo ""
echo "  Run (with display)  :"
echo "     python deployment/raspberry_pi/run_pi.py \\"
echo "       --model models/pi4/yolov8n_pi4_320.onnx \\"
echo "       --config deployment/raspberry_pi/pi_config.yaml \\"
echo "       --source 0"
echo ""
echo "  Run (headless / SSH):"
echo "     python deployment/raspberry_pi/run_pi.py \\"
echo "       --model models/pi4/yolov8n_pi4_320.onnx \\"
echo "       --config deployment/raspberry_pi/pi_config.yaml \\"
echo "       --source 0 --no-display"
echo "============================================"
