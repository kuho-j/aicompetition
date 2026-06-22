import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.yolov8_backbone import ConvBNSiLU

class FPNNeck(nn.Module):
    """
    Feature Pyramid Network (top-down pathway).
    integrate feature that fusioned in each scales
    """

    def __init__(self, in_channels: list[int], out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList(
            [ConvBNSiLU(c, out_channels, 1, 1, 0) for c in in_channels]
        )
        self.smooth = nn.ModuleList(
            [ConvBNSiLU(out_channels, out_channels, 3) for _ in in_channels]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # Lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral, features)]

        # Top-down: 큰 스케일(고해상도)로 정보 흘려보내기
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return [smooth(lat) for smooth, lat in zip(self.smooth, laterals)]
