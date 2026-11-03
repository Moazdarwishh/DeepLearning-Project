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
