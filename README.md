# 🤖 Indoor Obstacle Detection (室内障碍物检测)

基于 YOLOv8 的室内障碍物检测模型，专为机器人导航设计。

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 可检测类别

| ID | 类别 | 英文 |
|----|------|------|
| 0 | 关闭的门 | closed_door |
| 1 | 门 | door |
| 2 | 电梯 | elevator |
| 3 | 扶梯 | escalator |
| 4 | 人行道 | footpath |
| 5 | 障碍物 | obstacle |
| 6 | 行人 | person |
| 7 | 墙 | wall |

## 📊 模型性能

| 指标 | 值 |
|------|-----|
| mAP50 | 56.3% |
| mAP50-95 | 34.9% |
| 模型大小 | 6 MB (PyTorch) |
| 推理速度 | ~90ms/image (CPU) |

## 🚀 快速开始

### 安装

```bash
pip install ultralytics
```

### 下载模型

从 [Releases](../../releases) 下载 `best.pt` 或 `best.onnx`

### 使用示例

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('weights/best.pt')

# 检测单张图片
results = model.predict(source='test.jpg', save=True, conf=0.5)

# 实时摄像头检测
results = model.predict(source=0, show=True, conf=0.5)

# 检测视频文件
results = model.predict(source='video.mp4', save=True, conf=0.5)
```

### 获取检测结果

```python
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])           # 类别ID
        conf = float(box.conf[0])          # 置信度
        x1, y1, x2, y2 = box.xyxy[0]       # 边界框坐标
        class_name = model.names[cls_id]   # 类别名称
        
        print(f"检测到: {class_name}, 置信度: {conf:.2%}")
```

## 📁 项目结构

```
indoor-obstacle-detection/
├── README.md
├── requirements.txt
├── weights/
│   ├── best.pt          # PyTorch 模型
│   └── best.onnx        # ONNX 模型 (跨平台部署)
├── notebooks/
│   └── train.ipynb      # 训练代码
└── examples/
    └── inference.py     # 推理示例
```

## 🔧 训练自己的模型

1. 准备数据集 (YOLOv8 格式)
2. 修改 `data.yaml` 配置
3. 运行训练:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=50, imgsz=640)
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

## 👥 作者

Joseph Lab @ McGill University
