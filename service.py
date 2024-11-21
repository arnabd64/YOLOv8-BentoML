import json
from pathlib import Path
from typing import Annotated, Any, Dict, List

import bentoml
import yaml
from bentoml.validators import ContentType
from pydantic import Field
from ultralytics import YOLO

# load settings
with open("bento-settings.yaml", "r") as cfg:
    settings = yaml.load(cfg, yaml.SafeLoader)

# download model
_ = YOLO("yolov8n.pt")


Image = Annotated[Path, ContentType("image/*")]
CutOffScore = Annotated[float, Field(0.5, gt=0, lt=1)]


@bentoml.service(**settings["service"])
class YoloService:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    @bentoml.api(batchable=True, max_batch_size=8)
    def inference(self, images: List[Image]) -> List[List[Dict[str, Any]]]:
        """
        performs object detection on images
        """
        output = self.model.predict(images, verbose=False)
        responses = [json.loads(response.to_json(decimals=2)) for response in output]
        return responses

    @bentoml.api(batchable=False)
    def annotate(self, image: Image) -> Image:
        result = self.model.predict(image).pop()
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output
