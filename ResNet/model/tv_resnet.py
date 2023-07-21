from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_places: int, out_places: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_places,
        out_places,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_places: int, out_places: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_places, out_places, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(
            self, 
            in_places: int,
            out_places: int,
            stride: int = 1,
            ) -> None:
        super().__init__()

        self.conv1 = conv3x3(in_places, out_places, stride)
        self.bn1 = nn.BatchNorm2d(out_places)
        self.relu = nn.ReLU(True)

        self.conv2 = conv3x3(out_places, out_places)
        self.bn2 = nn.BatchNorm2d(out_places)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity

        out = self.relu(out)

        return out

class BottleNeck(nn.Module):

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
            ) -> None:
        super().__init__()
        if norm_layer == None:
            