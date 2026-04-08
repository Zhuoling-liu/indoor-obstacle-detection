# 🤖 Indoor Obstacle Detection

YOLOv8-based indoor obstacle detection model designed for robot navigation.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Detectable Classes

| ID | Class |
|----|-------|
| 0 | closed_door |
| 1 | door |
| 2 | elevator |
| 3 | escalator |
| 4 | footpath |
| 5 | obstacle |
| 6 | person |
| 7 | wall |

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP50 | 56.3% |
| mAP50-95 | 34.9% |
| Model Size | 6 MB (PyTorch) |
| Inference Speed | ~90ms/image (CPU) |

## 🚀 Quick Start

### Installation

```bash
pip install ultralytics
```

### Download Model

Download `best.pt` or `best.onnx` from [Releases](../../releases)

### Usage Example

```python
from ultralytics import YOLO

# Load model
model = YOLO('weights/best.pt')

# Detect single image
results = model.predict(source='test.jpg', save=True, conf=0.5)

# Real-time camera detection
results = model.predict(source=0, show=True, conf=0.5)

# Detect video file
results = model.predict(source='video.mp4', save=True, conf=0.5)
```

### Get Detection Results

```python
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])           # Class ID
        conf = float(box.conf[0])          # Confidence
        x1, y1, x2, y2 = box.xyxy[0]       # Bounding box coordinates
        class_name = model.names[cls_id]   # Class name
        
        print(f"Detected: {class_name}, Confidence: {conf:.2%}")
```

## 📁 Project Structure

```
indoor-obstacle-detection/
├── README.md
├── requirements.txt
├── weights/
│   ├── best.pt          # PyTorch model
│   └── best.onnx        # ONNX model (cross-platform deployment)
├── notebooks/
│   └── train.ipynb      # Training code
└── examples/
    └── inference.py     # Inference example
```

## 🔧 Train Your Own Model

1. Prepare dataset (YOLOv8 format)
2. Modify `data.yaml` configuration
3. Run training:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=50, imgsz=640)
```

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

MIT License

## 👥 Author

Joseph Lab @ McGill University
