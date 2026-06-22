from src.model.yolov8_backbone import *
from src.model.cross_attention import *
from src.model.fpn_neck import *
from src.model.detection import *

import torch
import torch.nn as nn


class MultiViewDetector(nn.Module):

    '''
    '''

    def __init__(
            self,
            num_views : int = 5,
            num_classes : int = 60,
            img_channels : int = 3,
            fpn_out_channels : int = 256,
            backbone_width : float = 0.25,
            backbone_depth : float = 0.33,
            attn_heads : int = 4,
            spatial_ds : int = 2, # spatial compression before the attention
            ):
        super().__init__()
        self.num_views = num_views
        self.num_classes = num_classes

        # backbone
        self.backbone = YOLOv8Backbone(img_channels, backbone_width, backbone_depth)
        bb_ch = self.backbone.out_channels # [C_p3, C_p4, C_p5]

        # cross-view attention
        self.view_fusion = nn.ModuleList([
            CrossViewAttentionFusion(
                embed_dim=ch,
                num_views=num_views,
                num_heads=attn_heads,
                spatial_downsample=spatial_ds,
                )
            for ch in bb_ch
            ])

        # fpn neck
        self.fpn = FPNNeck(bb_ch, fpn_out_channels)

        # detection head
        self.head = CenterNetHead(fpn_out_channels, num_classes)

    def forward(self, images : torch.Tensor) -> torch.Tensor:
        '''
        images : [B, N_views, C, H, W]
        '''
        B, N, C, H, W = images.shape
        
        assert N == self.num_views, f'Expected {self.num_views} views, got {N}'

        # extrack feature in each views
        per_view_feats : list[list[torch.Tensor]] = []
        for v in range(N):
            feats = self.backbone(images[:, v]) # [P3, P4, P5]
            per_view_feats.append(feats)
            
        # cross-view fusion
        num_scales = len(per_view_feats[0])
        fused_feats : list[torch.Tensor] = []
        for s in range(num_scales):
            scale_feats  = [per_view_feats[v][s] for v in range(N)]
            fused = self.view_fusion[s](scale_feats)
            fused_feats.append(fused)
        
        # FPN
        fpn_out = self.fpn(fused_feats)
        
        # detect ... P3
        return self.head(fpn_out[0])




