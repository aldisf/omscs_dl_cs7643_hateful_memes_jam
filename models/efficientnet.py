import torch

from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.utils.build import build_classifier_layer
from mmf.utils.build import build_image_encoder
from mmf.utils.build import build_text_encoder

from .encoders import EfficientNetEncoder


@registry.register_model("unimodal_efficientnet")
class UnimodalEfficientNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/efficientnet/unimodal.yaml"

    def build(self):
        self.image_encoder = EfficientNetEncoder(self.config.image_encoder)
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
        return {"scores": logits}


@registry.register_model("late_fusion_efficientnet")
class LateFusionEfficientNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/efficientnet/late_fusion.yaml"

    def build(self):
        self.image_encoder = EfficientNetEncoder(self.config.image_encoder)
        self.text_encoder = build_text_encoder(self.config.text_encoder)
        self.image_classifier = build_classifier_layer(self.config.image_classifier)
        self.text_classifier = build_classifier_layer(self.config.text_classifier)

    def forward(self, sample_list):
        # Unpack
        image = sample_list["image"]
        text = sample_list["input_ids"]

        # Encoders
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)[1]

        # Flatten
        image_features = torch.flatten(image_features, start_dim=1)
        text_features = torch.flatten(text_features, start_dim=1)

        # Classifiers
        image_logits = self.image_classifier(image_features)
        text_logits = self.text_classifier(text_features)
        
        # Fusion
        logits = (image_logits + text_logits) / 2
        return {"scores": logits}



@registry.register_model("concat_bert_efficientnet")
class ConcatBERTEfficientNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/efficientnet/concat_bert.yaml"

    def build(self):
        self.image_encoder = EfficientNetEncoder(self.config.image_encoder)
        self.text_encoder = build_text_encoder(self.config.text_encoder)
        self.classifier = build_classifier_layer(self.config.classifier)

    def forward(self, sample_list):
        # Unpack
        image = sample_list["image"]
        text = sample_list["input_ids"]

        # Encoders
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)[1]

        # Flatten
        image_features = torch.flatten(image_features, start_dim=1)
        text_features = torch.flatten(text_features, start_dim=1)

        # Concatenate
        concat_features = torch.cat([text_features, image_features], dim=1)

        # Classifier
        logits = self.classifier(concat_features)
        return {"scores": logits}
