from . import cifar_resnet
from . import resnet
from . import vit
from .mlp import MLPEncoder, MLPFlattenEncoder

def get_encoder(arch, ch_in=1, image_size=64, **kwargs):
    
    if arch in cifar_resnet.MODELS:
        return cifar_resnet.select_cifar_resnet(arch=arch, ch_in=ch_in, **kwargs)
    
    elif arch in resnet.MODELS:
        return resnet.ResNet(model_size=arch, **kwargs)
    
    elif arch.upper() == 'MLP':
        return MLPEncoder(**kwargs)
    
    elif arch.upper() == 'MLP-FLATTEN':
        input_dim = int(ch_in * image_size**2)
        return MLPFlattenEncoder(input_dim=input_dim, **kwargs)

    elif arch in vit.MODELS:
        return vit.select_vit(
            arch=arch,
            ch_in=ch_in,
            image_size=image_size,
            **kwargs
        )
    
    else:
        raise ValueError