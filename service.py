import json
import os
from pathlib import Path
from typing import Annotated, List

import bentoml
import yaml
from bentoml.validators import ContentType
from pydantic import Field
from ultralytics import YOLO

# load settings
with open("bento-settings.yaml", "r") as cfg:
    settings = yaml.load(cfg, yaml.SafeLoader)

# download model
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8s.pt")
_ = YOLO(MODEL_NAME)


Image = Annotated[Path, ContentType("image/*")]
CutOffScore = Annotated[float, Field(0.5, gt=0, lt=1)]


@bentoml.service(**settings["service"])
class YoloService:
    def __init__(self):
        self.model = YOLO(MODEL_NAME)

    @bentoml.api(batchable=True)
    def inference(self, images: List[Image]):
        """
        performs object detection on images
        """
        output = self.model.predict(images, verbose=False)
        responses = [
            {
                "image": image.stem + image.suffix,
                "objects": json.loads(result.to_json()) if len(result) > 0 else None,
            }
            for image, result in zip(images, output)
        ]
        return responses

    @bentoml.api(batchable=False)
    def annotate(self, image: Image) -> Image:
        result = self.model.predict(image).pop()
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output

    @bentoml.api(batchable=False)
    def object_id_map(self):
        return self.model.names
