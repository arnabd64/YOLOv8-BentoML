import json
from pathlib import Path
from typing import Annotated, Any, Dict, List

import bentoml
from bentoml.validators import ContentType
from pydantic import Field
from ultralytics import YOLO
import yaml

with open("bento-settings.yml", "r") as cfg:
    settings = yaml.load(cfg, yaml.SafeLoader)


Image = Annotated[Path, ContentType("image/*")]
CutOffScore = Annotated[float, Field(0.5, gt=0, lt=1)]


@bentoml.service(**settings['service'])
class YoloService:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    @bentoml.api(**settings['api'])
    def inference(
        self, image: Image, cutoff_score: CutOffScore
    ) -> List[Dict[str, Any]]:
        response = (
            self.model.predict(image, conf=cutoff_score, verbose=False)
            .pop()
            .to_json(decimals=2)
        )
        return json.loads(response)
