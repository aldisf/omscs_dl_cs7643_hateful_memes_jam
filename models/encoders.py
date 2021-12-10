import torch
import torchvision
from torch import nn
from dataclasses import dataclass

from mmf.common.registry import registry
from mmf.modules.encoders import Encoder

from efficientnet_pytorch import EfficientNet


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)


@registry.register_encoder("yolov5s")
class Yolov5sEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "yolov5s"
        pool_type: str = "avg"
        num_output_features: int = 1

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config
        model = torch.hub.load(
            "ultralytics/yolov5", "yolov5s", autoshape=False, force_reload=True
        )
        modules = list(model.model.children())[0][:11]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out


@registry.register_encoder("yolov5m")
class Yolov5mEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "yolov5m"
        pool_type: str = "avg"
        num_output_features: int = 1

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config
        model = torch.hub.load(
            "ultralytics/yolov5", "yolov5m", autoshape=False, force_reload=True
        )
        modules = list(model.model.children())[0][:11]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out


@registry.register_encoder("yolov5l")
class Yolov5lEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "yolov5l"
        pool_type: str = "avg"
        num_output_features: int = 1

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config
        model = torch.hub.load(
            "ultralytics/yolov5", "yolov5l", autoshape=False, force_reload=True
        )
        modules = list(model.model.children())[0][:11]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out


@registry.register_encoder("yolov5x")
class Yolov5xEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "yolov5x"
        pool_type: str = "avg"
        num_output_features: int = 1

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config
        model = torch.hub.load(
            "ultralytics/yolov5", "yolov5x", autoshape=False, force_reload=True
        )
        modules = list(model.model.children())[0][:11]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out


@registry.register_encoder("mobilenetv3_large")
class MobileNetv3LargeEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "mobilenetv3_large"
        pool_type: str = "avg"
        num_output_features: int = 1

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config
        model = torchvision.models.mobilenet_v3_large(pretrained=True)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out


@registry.register_encoder("efficientnet")
class EfficientNetEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "efficientnet-b0"
        pool_type: str = "avg"
        num_output_features: int = 1

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config

        self.model = EfficientNet.from_pretrained(
            model_name=config.name, num_classes=config.num_output_features
        )
        self.model._global_params = self.model._global_params._replace(
            include_top=False
        )

    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out
