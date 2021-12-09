import collections
from copy import deepcopy

import torch
import transformers
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.encoders import MultiModalEncoderBase
from mmf.utils.build import build_classifier_layer


class UnimodalBaseModified(MultiModalEncoderBase):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def build(self):
        encoders = self._build_encoders(self.config)
        # Text Encoder mode
        if "modal_encoder" not in self.config:
            self.encoder = encoders[0]
        # Modal encoder mode
        elif "text_encoder" not in self.config:
            self.encoder = encoders[1]
        else:
            raise RuntimeError(
                "Unimodal Encoder can't have both text and modal encoder"
            )

    def forward(self, x, *args, **kwargs):
        x = self.encoder(x, *args, **kwargs)
        # Case of bert encoder, we only need pooled output
        is_instance = isinstance(x, collections.abc.Sequence) or isinstance(
            x,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        )
        if is_instance and len(x) >= 2:
            x = x[1]

        x = torch.flatten(x, start_dim=1)
        return x


@registry.register_model("unimodal_text_modified_roberta")
class UnimodalText(BaseModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/unimodal/text.yaml"

    def build(self):
        self.base = UnimodalBaseModified(self.config)
        # As the in_dim is dynamically calculated we need to copy classifier_config
        classifier_config = deepcopy(self.config.classifier)
        classifier_config.params.in_dim = self.config.text_hidden_size
        self.classifier = build_classifier_layer(classifier_config)

    def forward(self, sample_list):
        # BERT Based Encoders
        args = []
        if "input_ids" in sample_list:
            text = sample_list.input_ids
            args.append(sample_list.input_mask)
            args.append(sample_list.segment_ids)
        else:
            text = sample_list.text

        embedding = self.base(text, *args)
        output = {}
        output["scores"] = self.classifier(embedding)

        return output
