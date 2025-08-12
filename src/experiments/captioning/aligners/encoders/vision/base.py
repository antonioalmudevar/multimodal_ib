import logging

import torch
from torch.cuda.amp import autocast as autocast

from .clip_vit import create_clip_vit_L
from .eva_vit import create_eva_vit_g
from src.experiments.captioning.utils import LayerNorm, disabled_train

class VisionEcoder(torch.nn.Module):
    
    def __init__(self, vision_encoder, ln_vision):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.ln_vision = ln_vision

    def forward(self, img):
        return self.ln_vision(self.vision_encoder(img))
    
    def encoder_width(self):
        return self.vision_encoder.num_features


def init_vision_encoder(
        model_name, 
        freeze, 
        img_size, 
        drop_path_rate, 
        use_grad_checkpoint, 
        precision
    ):
    if model_name == "eva_clip_g":
        model = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )
    elif model_name == "clip_L":
        model = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
    else:
        raise ValueError(f"unsupported model: {model_name}")
    if freeze:
        for _, param in model.named_parameters():
            param.requires_grad = False
        model = model.eval()
        model.train = disabled_train
        logging.info("freeze vision encoder")
    ln_vision = LayerNorm(model.num_features)
    return VisionEcoder(model, ln_vision)