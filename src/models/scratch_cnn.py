"""CNN trained from scratch (Milestone 1 baseline).

Architecture (ResNet-style, ~11 M parameters with the default widths):

    stem   : conv3x3(s2) -> conv3x3          224 -> 112   (32 -> widths[0] channels)
    stage 1: blocks_per_stage blocks, s2      112 ->  56   widths[0]
    stage 2: blocks_per_stage blocks, s2       56 ->  28   widths[1]
    stage 3: blocks_per_stage blocks, s2       28 ->  14   widths[2]
    stage 4: blocks_per_stage blocks, s2       14 ->   7   widths[3]

Each block is two 3x3 conv-BN-ReLU layers. With ``residual=True`` the block adds a
skip connection (He et al., 2016), which lets gradients flow through deep networks;
``residual=False`` gives a plain VGG-style network for the architecture comparison.

The four stage outputs are returned by ``forward_features`` so the same encoder
can later serve as the U-Net encoder (colorization) and the image feature grid
(captioning).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder
from .registry import register_encoder


def conv_bn_relu(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class ConvBlock(nn.Module):
    """Two 3x3 convolutions; optional residual (skip) connection."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv1 = conv_bn_relu(c_in, c_out, stride)
        self.conv2 = nn.Sequential(nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if residual and (stride != 1 or c_in != c_out):
            # 1x1 projection so the skip connection matches shape
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        if self.residual:
            out = out + self.shortcut(x)
        return self.relu(out)


@register_encoder("scratch_cnn")
class ScratchCNN(Encoder):
    def __init__(self, widths: list[int] = (64, 128, 256, 512), blocks_per_stage: int = 2,
                 residual: bool = True, in_channels: int = 3, pretrained: bool = False, **_):
        super().__init__()
        if pretrained:
            raise ValueError("scratch_cnn has no pretrained weights (set model.pretrained=false)")
        widths = list(widths)
        self.out_channels = widths
        self.reductions = [4, 8, 16, 32]
        self.stem = nn.Sequential(conv_bn_relu(in_channels, 32, stride=2), conv_bn_relu(32, widths[0]))
        stages, c_in = [], widths[0]
        for c_out in widths:
            blocks = [ConvBlock(c_in, c_out, stride=2, residual=residual)]
            blocks += [ConvBlock(c_out, c_out, residual=residual) for _ in range(blocks_per_stage - 1)]
            stages.append(nn.Sequential(*blocks))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    def last_conv_layer(self) -> nn.Module:
        return self.stages[-1]
