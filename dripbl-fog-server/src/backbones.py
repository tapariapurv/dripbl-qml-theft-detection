"""Frame encoders shared by training (xd_experiments / train_xd) and the live node.

Every backbone maps a (B, C, 64, 64) float tensor in [0, 1] to a 128-d latent, which
the quantum head compresses to 4 qubit angles. C is 3 (RGB) or 6 (RGB + motion: the
frame minus the frame 0.33 s earlier, encoded around 0.5).

  tiny      LightweightCNN trained from scratch (the original deployed encoder)
  resnet18  ImageNet ResNet-18, fine-tuned, frames upsampled to `input_size`
  mobilenet ImageNet MobileNetV3-Small, fine-tuned, frames upsampled to `input_size`

For pretrained kinds the ImageNet weights are only downloaded when training
(`pretrained=True`); a deployed artifact carries its own fine-tuned weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge_pipeline import LightweightCNN  # noqa: E402

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Backbone(nn.Module):
    """(B, C, 64, 64) in [0, 1] -> (B, 128) latent."""

    def __init__(self, kind: str, channels: int, input_size: int = 64, pretrained: bool = False) -> None:
        super().__init__()
        self.kind, self.channels, self.input_size = kind, channels, input_size
        if kind == "tiny":
            self.net = LightweightCNN(128, channels)
            return
        from torchvision import models
        if kind == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            stem, feat_dim = m.conv1, 512
            m.fc = nn.Identity()
        elif kind == "mobilenet":
            m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            stem, feat_dim = m.features[0][0], 576
            m.classifier = nn.Identity()
        else:
            raise ValueError(f"unknown backbone {kind!r}")
        if channels != 3:  # motion channels start from the RGB filters, scaled down
            new = nn.Conv2d(channels, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, bias=False)
            with torch.no_grad():
                new.weight[:, :3] = stem.weight
                new.weight[:, 3:] = stem.weight * 0.5
            if kind == "resnet18":
                m.conv1 = new
            else:
                m.features[0][0] = new
        self.net = nn.Sequential(m, nn.Linear(feat_dim, 128))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "tiny":
            return self.net(x)
        rgb = (x[:, :3] - _MEAN.to(x.device)) / _STD.to(x.device)
        x = torch.cat([rgb, (x[:, 3:] - 0.5) / 0.25], 1) if self.channels > 3 else rgb
        if self.input_size != x.shape[-1]:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.net(x)
