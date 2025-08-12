from torch import nn
import torchvision.models.vision_transformer as vit

MODELS = [
    "tiny_vit"
]

class TinyViT(nn.Module):
    def __init__(self, image_size=64, ch_in=3):

        super().__init__()
        self.vit = vit.VisionTransformer(
            image_size=image_size,   
            patch_size=4,    # Smaller patch size (4x4) to capture more local details
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=768,
            num_classes=10
        )
        
        # Modify the first convolutional layer to accept 1-channel input
        if ch_in!=3:
            old_conv = self.vit.conv_proj
            self.vit.conv_proj = nn.Conv2d(
                in_channels=ch_in,
                out_channels=old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride, 
                padding=old_conv.padding
            )
        self.vit.heads = nn.Identity()
        
        self.size_code = 384


    def forward(self, x):
        return self.vit(x)  # Returns embeddings



def select_vit(arch, **kwargs):
    if arch=="tiny_vit":
        return TinyViT(**kwargs)