# 🦅 Project Garuda

**UAV Real-Time Object Detection & Tracking System**

A modular, production-ready system for real-time aerial object detection using YOLOv8, DeepSORT tracking, and configurable alert generation — optimised for edge deployment on Raspberry Pi and NVIDIA Jetson.

---

## 📁 Project Structure

```
Garuda/
├── configs/
│   ├── data.yaml              # Dataset configuration (VisDrone / custom)
│   └── model.yaml             # Model, training, inference, tracking, alert settings
├── models/
│   └── weights/               # Trained & exported model weights
├── src/
│   ├── detection/
│   │   └── yolo_detector.py   # YOLOv8 detector wrapper
│   ├── tracking/
│   │   └── tracker.py         # DeepSORT tracker (with IOU fallback)
│   ├── pipeline/
│   │   └── pipeline.py        # Full capture → detect → track → alert pipeline
│   ├── alerts/
│   │   └── alert_manager.py   # Flagged-class alert system
│   └── utils/
│       ├── preprocessing.py   # Frame resize, letterbox, normalization
│       ├── visualization.py   # Bounding box / track ID / FPS drawing
│       └── logger.py          # Centralized logging
├── scripts/
│   ├── train.py               # Training pipeline
│   ├── export.py              # Model export (ONNX, TFLite, TensorRT)
│   ├── run.py                 # Real-time inference
│   └── optimize.py            # Benchmark, quantize, profile
├── deployment/
│   ├── raspberry_pi/
│   │   ├── setup.sh           # Pi environment setup
│   │   ├── run_pi.py          # ONNX inference + Web Stream (3–10 FPS)
│   │   ├── requirements_pi.txt # Minimal dependencies for Pi
│   │   └── pi_config.yaml     # Pi-specific mission config
│   └── jetson/
│       ├── setup.sh           # Jetson environment setup
│       └── run_jetson.py      # TensorRT/ONNX inference (10–25 FPS)
├── tests/
│   └── test_pipeline.py       # Unit tests
├── main.py                    # CLI entry point
├── requirements.txt           # Main dependencies
├── README.md
└── walkthrough.md             # Detailed deployment guide
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/Garuda.git
cd Garuda

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference (Webcam)

```bash
python main.py run --source 0
```

### 3. Run on a Video File

```bash
python main.py run --source path/to/video.mp4 --save
```

---

## 🏋️ Training

### Download VisDrone Dataset

```bash
# Download VisDrone2019-DET
mkdir -p datasets/VisDrone
cd datasets/VisDrone

# Train set
wget https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1/VisDrone2019-DET-train.zip
unzip VisDrone2019-DET-train.zip -d train

# Validation set
wget https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1/VisDrone2019-DET-val.zip
unzip VisDrone2019-DET-val.zip -d val
```

### Train a Model

```bash
# Train with default config
python main.py train --data configs/data.yaml --epochs 100

# Train with custom settings
python main.py train --data configs/data.yaml --epochs 50 --batch 32 --imgsz 640

# Resume training
python main.py train --resume last
python main.py train --resume runs/train/garuda_exp/weights/last.pt
```

### Custom Dataset

1. Organise your dataset in YOLO format:
   ```
   datasets/custom/
   ├── train/
   │   ├── images/
   │   └── labels/
   └── val/
       ├── images/
       └── labels/
   ```
2. Update `configs/data.yaml` with your paths and class names.
3. Run training as above.

---

## 📦 Model Export

```bash
# Export to ONNX
python main.py export --weights runs/train/garuda_exp/weights/best.pt --format onnx

# Export to TFLite (for Raspberry Pi)
python main.py export --weights best.pt --format tflite --imgsz 320

# Export to TFLite with INT8 quantization
python main.py export --weights best.pt --format tflite --int8

# Export to TensorRT (for Jetson)
python main.py export --weights best.pt --format engine --half

# Export all formats
python main.py export --weights best.pt --batch-all
```

---

## 🖥️ Deployment

### Raspberry Pi 4 (3–10 FPS)

Optimised for minimal footprint (~120 MB deps). No PyTorch or Ultralytics required.

1. **On the Pi:**
   ```bash
   chmod +x deployment/raspberry_pi/setup.sh
   ./deployment/raspberry_pi/setup.sh
   source garuda_env/bin/activate
   ```

2. **Run inference with ONNX model:**
   ```bash
   # Run with web stream (View at http://<PI_IP>:5000)
   python deployment/raspberry_pi/run_pi.py \
       --model models/pi4/best_pi4_320.onnx \
       --stream --no-display

   # Flags: --stream (web view), --no-display (headless), --frame-skip 3
   ```

### NVIDIA Jetson (10–25 FPS)

```bash
# On the Jetson:
chmod +x deployment/jetson/setup.sh
./deployment/jetson/setup.sh

# Run inference with TensorRT
python deployment/jetson/run_jetson.py \
    --model models/weights/best.engine \
    --source 0 \
    --imgsz 640

# Or with ONNX + CUDA
python deployment/jetson/run_jetson.py \
    --model models/weights/best.onnx \
    --source 0
```

---

## 📊 Optimization & Benchmarking

```bash
# Benchmark inference speed
python main.py optimize --task benchmark --weights yolov8n.pt --device cpu --runs 100

# Profile across image sizes
python main.py optimize --task profile --weights yolov8n.pt

# Compare formats (PyTorch vs ONNX)
python main.py optimize --task compare --weights yolov8n.pt --formats pt onnx

# Quantize to FP16 + INT8
python main.py optimize --task quantize --weights yolov8n.pt
```

---

## 🚨 Alert System

Configure alerts in `configs/model.yaml`:

```yaml
alerts:
  enabled: true
  flagged_classes:
    - car
    - truck
    - person
  cooldown_seconds: 5
  webhook_url: "https://your-webhook.example.com/alert"
  log_file: logs/alerts.log
```

Alerts trigger when flagged classes are detected:
- **Console**: Red warning messages with class, confidence, and track ID
- **File Log**: JSON-line format in the configured log file
- **Webhook**: HTTP POST to the configured URL

---

## ⌨️ Runtime Controls

While running inference:

| Key | Action |
|-----|--------|
| `q` | Quit |
| `l` | Lock/Unlock Gimbal Target |
| `+` | Increase confidence threshold (+0.05) |
| `-` | Decrease confidence threshold (-0.05) |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run from project root
cd Garuda
python -m pytest tests/test_pipeline.py -v
```

---

## ⚡ Performance Tuning

| Parameter | Raspberry Pi | Jetson | Desktop GPU |
|-----------|-------------|--------|-------------|
| Model | yolov8n | yolov8s | yolov8m/l |
| Image Size | 320 | 640 | 640–1280 |
| Frame Skip | 3 | 1 | 1 |
| Format | TFLite | TensorRT | PyTorch |
| Quantization | INT8 | FP16 | FP32/FP16 |
| Target FPS | 3–8 | 10–25 | 30–60+ |

### Tips

- **Lower image size** → faster inference, lower accuracy
- **Higher frame skip** → more responsive UI, fewer detections
- **Use `yolov8n`** for edge, `yolov8s/m` for Jetson, `yolov8l/x` for desktop
- **INT8 quantization** gives ~2x speedup on TFLite
- **TensorRT FP16** gives ~3x speedup on Jetson

---

## 🏗️ Architecture

```
Camera/Video → FramePreprocessor → YOLODetector → ObjectTracker → AlertManager → Visualizer → Display/Save
                                         ↑                ↑              ↑
                                    model.yaml      tracking config   flagged classes
```

All components are config-driven and plug-and-play. Replace any module by implementing the same interface.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request
