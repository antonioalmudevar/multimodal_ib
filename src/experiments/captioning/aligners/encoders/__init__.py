from torch import nn

from .text import BertConfig, BertLMHeadModel
from .vision import init_vision_encoder

ENCODERS_MODALITY = {
    'vision': init_vision_encoder,
}


def init_modality_encoders(encoders_kwargs):
    modality_encoders, encoder_widths = dict(), dict()
    for modality_name, kwargs_encoder in encoders_kwargs.items():
        assert modality_name in list(ENCODERS_MODALITY.keys()), \
            "{} is not a valid modality".format(modality_name)
        encoder = ENCODERS_MODALITY[modality_name](**kwargs_encoder)
        modality_encoders[modality_name] = encoder
        encoder_widths[modality_name] = encoder.encoder_width()
    return nn.ModuleDict(modality_encoders), encoder_widths