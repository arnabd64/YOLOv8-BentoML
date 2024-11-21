# YOLOv8 Object Detection using BentoML

A simple RESTful application to deploy an __YOLOv8 Object Detection__ model using __BentoML__ framework. I have tried the best of my knowedge to build this application that is highly configurable and help the community to build more Machine Learning Applications using BentoML.

Here are the reasons why I think BentoML is an excellent framework.

1. __Easy to Code__ which leads to a significant speed up in developing inference APIs for out models.
2. __Integration with Popular Frameworks__ like Pytorch, Tensorflow, Huggingface, ONNX and more.
3. __Inbuilt Adaptive batching__ helps to speed up the API requests in a high traffic scenario.
4. __Inbuilt Logging, Tracing and Metrics__ for better observability.
5. __Integration with Gradio__ that helps me built easy to use UI for faster prototyping and testing.

## Ultralytics - YOLOv8

__YOLOv8__ was the state-of-the-art Object Detection model released back in early 2023. It is primarily known for balancing between prediction accuracy and latency, these models can perform efficiently even on a dual core CPU with 2GB of system RAM. There are 5 models in the YOLOv8 family from Nano Model(`yolov8n`) to eXtra Large Model(`yolov8x`).

| Model  | Model Name   | Params (millions) | mAP (50-75) |
| ------ | ------------ | ----------------- | ----------- |
| Nano   | `yolov8n.pt` | 3.2  | 37.3 |
| Small  | `yolov8s.pt` | 11.2 | 44.9 |
| Medium | `yolov8m.pt` | 25.9 | 50.2 |
| Large  | `yolov8l.pt` | 43.7 | 52.9 |
| XL     | `yolov8x.pt` | 68.2 | 53.9 |

Run the following code to download and use YOLOv8 models in your code:
```python
from ultralytics import YOLO
model = YOLO(`yolov8s.pt`)
```

## BentoML

__BentoML__ is a python framework to speed the deployment process for ML models. The framework speeds up the creation of RESTful APIs, Inference optimizations and Containerization of the application.

In this project we will be exploring the `Service` API of BentoML in depth along with integrating a Gradio Application. A `Service` can be thought of as a collection of endpoints serving a single model.

I have written the inference code for YOLOv8 on the python script `service.py`, a `Service` is defined by a python class decorated with the `bentoml.serivce` decorator. All the settings are passed as arguments to this decorator which is stored in an YAML file named `bento-settings.yaml` The documentation for all the settings can be found on the [BentoML docs](https://docs.bentoml.org/en/latest/guides/configurations.html) page as well as a sample YAML file on [GitHub](https://github.com/bentoml/BentoML/blob/1.3/src/bentoml/_internal/configuration/v2/default_configuration.yaml).

## Bentofile

The `bentofile.yaml` is a contains a the [build options](https://docs.bentoml.org/en/latest/guides/build-options.html) needed to containerize the project. You can think of a `bentofile` similar to a `Dockerfile`. The purpose of both these scripts to build a __Docker Image__ for our project but unlike a Dockerfile, a bentofile contains only configurations and settings instead of commands, in a Dockerfile, in order to build the image. This in-turn simplifies the building process.

To build a docker image using `bentofile`, run the following commands on the terminal:

```bash
# build a 'Bento'
$ bentoml build

# build a Docker image
# you will find the command to containerize the project after a sucessful run of the above command
$ bentoml containerize yolo_service:<unique_tag>

# Run the container
$ docker run -itd --rm --name=yolo -p 5000:5000 yolo_service:<unique_tag>

# Check the logs
$ docker compose logs yolo
```

## Run a Local Server

On your terminal, run:
```bash
# {py file containing bento service}:{Service Name}
$ bentoml serve service:YoloService
```


## Links

* [BentoML](https://docs.bentoml.com/en/latest)
* [Ultralytics](https://docs.ultralytics.com/)
* [Ultralytics - YOLOv8](https://docs.ultralytics.com/models/yolov8/#key-features)
