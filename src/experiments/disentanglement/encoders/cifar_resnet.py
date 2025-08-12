"""
https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""
from typing import List, Type

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


MODELS = [
    'resnet20', 
    'resnet32', 
    'resnet44', 
    'resnet56', 
    'resnet110', 
    'resnet218', 
    'resnet326', 
    'resnet434', 
    'resnet1202',
]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self, 
        in_planes: int, 
        planes: int, 
        stride: int=1, 
        option: str='A',
    ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0)
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, x: Tensor) -> Tensor:

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out



class CifarResNet(nn.Module):
    
    def __init__(
        self, 
        block: Type[BasicBlock], 
        num_blocks: List[int], 
        ch_in: int,
        channels: List[int]=[16, 32, 64],
    ):
        super().__init__()
        assert len(channels)==3, "This resnet must have 3 layers"

        self.in_planes = channels[0]

        self.conv1 = nn.Conv2d(
            ch_in, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, channels[0], num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, channels[1], num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, channels[2], num_blocks[2], stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.apply(_weights_init)
        self.size_code = self.in_planes


    def _make_layer(
        self, 
        block: Type[BasicBlock], 
        planes: int, 
        num_blocks: int, 
        stride: int=1,
    ) -> nn.Sequential:
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 'A'))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
        

    def forward(self, x: Tensor) -> Tensor:

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        return out


#==========Original ResNets===============
def resnet20(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [3, 3, 3], ch_in, channels, **kwargs)


def resnet32(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [5, 5, 5], ch_in, channels, **kwargs)


def resnet44(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [7, 7, 7], ch_in, channels, **kwargs)


def resnet56(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [9, 9, 9], ch_in, channels, **kwargs)


def resnet110(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [18, 18, 18], ch_in, channels, **kwargs)


def resnet218(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [36, 36, 36], ch_in, channels, **kwargs)


def resnet326(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [54, 54, 54], ch_in, channels, **kwargs)


def resnet434(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [72, 72, 72], ch_in, channels, **kwargs)


def resnet1202(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [200, 200, 200], ch_in, channels, **kwargs)


def select_cifar_resnet(arch, **kwargs):
    return eval(arch)(**kwargs)