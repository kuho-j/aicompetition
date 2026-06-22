import torch
import torch.nn as nn
from src.model.yolov8_backbone import ConvBNSiLU
import math

class CenterNetHead(nn.Module):
    '''
    CenterNet style
    
    detecting center point on 2D plane

    output:
        heatmap : [B, num_classes, H, W]

    '''

    def __init__(self, in_channels : int, num_classes : int = 60):
        super().__init__()

        # Heatmap
        self.heatmap = nn.Sequential(
                ConvBNSiLU(in_channels, in_channels, 3),
                ConvBNSiLU(in_channels, in_channels, 3),
                nn.Conv2d(in_channels, num_classes, 1),
                )

        # initiate Focal Loss
        nn.init.constant_(self.heatmap[-1].bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.heatmap(x).sigmoid()
