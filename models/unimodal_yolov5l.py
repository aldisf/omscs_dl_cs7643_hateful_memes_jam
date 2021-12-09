import torch

from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.utils.build import build_classifier_layer
from .encoders import Yolov5lEncoder


@registry.register_model("unimodal_yolov5l")
class UnimodalYolov5l(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/yolov5l/unimodal.yaml"

    def build(self):
        self.image_encoder = Yolov5lEncoder(self.config.image_encoder)
        self.classifier = build_classifier_layer(self.config.classifier)

    def forward(self, sample_list):
        # Unpack
        image = sample_list["image"]

        # Encoders
        image_features = self.image_encoder(image)

        # Flatten
        image_features = torch.flatten(image_features, start_dim=1)

        # Classifier
        logits = self.classifier(image_features)

        # MMF will automatically calculate loss
        return {"scores": logits}
