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
        
        # 병렬 처리를 위해 batch에 view를 넣기
        x = images.reshape(B * N, C, H, W)

        feats = self.backbone(x)

        per_scale_feats = []
        for feat in feats:
            _, C_s, H_s, W_s = feat.shape
            feat = feat.view(B, N, C_s, H_s, W_s)
            per_scale_feats.append(feat)
            
        # cross-view fusion
        fused_feats : list[torch.Tensor] = []
        for s, scale_feat in enumerate(per_scale_feats):
            fused = self.view_fusion[s](scale_feat)
            fused_feats.append(fused)
        
        # FPN
        fpn_out = self.fpn(fused_feats)
        
        # detect ... P3
        return self.head(fpn_out[0])




