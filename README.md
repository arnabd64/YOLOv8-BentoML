# YOLOv8 Object Detection using BentoML

A simple RESTful application to deploy an __YOLOv8 Object Detection__ model using __BentoML__ framework. I have tried the best of my knowedge to build this application that is highly configurable and help the community to build more Machine Learning Applications using BentoML.

## Ultralytics - YOLOv8

__YOLOv8__ was the state-of-the-art Object Detection model released back in early 2023. It is primarily known for balancing between prediction accuracy and latency, these models can perform efficiently even on a dual core CPU with 2GB of system RAM. There are 5 models in the YOLOv8 family from Nano Model(`yolov8n`) to eXtra Large Model(`yolov8x`).

| Model  | Model Name   | Params (millions) | mAP (50-75) |
| ------ | ------------ | ----------------- | ----------- |
| Nano   | `yolov8n.pt` | $3.2$  | $37.3$ |
| Small  | `yolov8s.pt` | $11.2$ | $44.9$ |
| Medium | `yolov8m.pt` | $25.9$ | $50.2$ |
| Large  | `yolov8l.pt` | $43.7$ | $52.9$ |
| XL     | `yolov8x.pt` | $68.2$ | $53.9$ |

Run the following code to download and use YOLOv8 models in your code:
```python
from ultralytics import YOLO

model = YOLO(`yolov8s.pt`)
```

## BentoML

__BentoML__ is a python framework to speed the deployment process for ML models. The framework speeds up the creation of RESTful APIs, Inference optimizations and Containerization of the application.



## Links

* [BentoML](https://docs.bentoml.com/en/latest)
* [Ultralytics](https://docs.ultralytics.com/)
* [Ultralytics - YOLOv8](https://docs.ultralytics.com/models/yolov8/#key-features)
