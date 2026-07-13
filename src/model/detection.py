import math

import torch
import torch.nn as nn

from src.model.yolov8_backbone import ConvBNSiLU


class CenterNetHead(nn.Module):
    """
    Lightweight CenterNet-style heatmap head.

    The repository predicts class center heatmaps plus the CenterNet local
    sub-cell offset. Size/box branches are omitted because this project does not
    have box-size labels.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int | None = None,
        num_convs: int = 2,
        prior_prob: float = 0.01,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = min(in_channels, 128)

        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_convs):
            layers.append(ConvBNSiLU(current_channels, hidden_channels, 3))
            current_channels = hidden_channels

        self.feature = nn.Sequential(*layers)
        self.heatmap = nn.Conv2d(hidden_channels, num_classes, 1)
        self.offset = nn.Conv2d(hidden_channels, 2, 1)
        bias = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.heatmap.bias, bias)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.feature(x)
        return {
            "center_feature": feature,
            "heatmap_logits": self.heatmap(feature),
            "offset": self.offset(feature),
        }
