"""ImageNet-pretrained ResNet encoders for transfer learning.

ResNet-50 (He et al., 2016) is the main transfer model: it is the standard
backbone reported on CUB-200-2011, and its 4 stages (256/512/1024/2048 channels
at 1/4, 1/8, 1/16, 1/32 resolution) give natural U-Net skip connections for
colorization in Milestone 2. ResNet-18/34 are lighter options (faster on a laptop).

Two transfer strategies are compared in Milestone 1 (set in the config):

* linear probe   (model.freeze_backbone=true)  - backbone frozen, only the new
  200-class head is trained. Measures how good ImageNet features are as-is.
* fine-tuning    (model.freeze_backbone=false) - all layers trained, backbone with a
  10x smaller learning rate (train.backbone_lr) so pretrained features are adapted
  gently instead of being destroyed by large updates.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder
from .registry import register_encoder

_RESNETS = {
    "resnet18": ("resnet18", "ResNet18_Weights", [64, 128, 256, 512]),
    "resnet34": ("resnet34", "ResNet34_Weights", [64, 128, 256, 512]),
    "resnet50": ("resnet50", "ResNet50_Weights", [256, 512, 1024, 2048]),
}


class ResNetEncoder(Encoder):
    def __init__(self, arch: str = "resnet50", pretrained: bool = True, **_):
        super().__init__()
        import torchvision

        fn_name, weights_name, channels = _RESNETS[arch]
        weights = getattr(torchvision.models, weights_name).DEFAULT if pretrained else None
        net = getattr(torchvision.models, fn_name)(weights=weights)
        self.arch = arch
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.out_channels = channels
        self.reductions = [4, 8, 16, 32]
        # net.fc (ImageNet 1000-class head) is dropped; Classifier adds a 200-class head.

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

    def last_conv_layer(self) -> nn.Module:
        return self.layer4


for _name in _RESNETS:
    register_encoder(_name)(lambda pretrained=True, _arch=_name, **kw: ResNetEncoder(_arch, pretrained, **kw))
