import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.yolov8_backbone import ConvBNSiLU

class ViewFiLM(nn.Module):
    '''
    카메라 시점 별 feature calibration
    '''

    def __init__(self, num_views : int, embed_dim : int):
        super().__init__()
        self.num_views = num_views
        self.embed_dim = embed_dim

        self.gamma = nn.Parameter(torch.ones(num_views, embed_dim))
        self.beta = nn.Parameter(torch.zeros(num_views, embed_dim))

    def forward(self, feat : torch.Tensor) -> torch.Tensor:
        '''feat : [B, N, C, H, W]'''

        assert feat.shape[1] == self.num_views
        assert feat.shape[2] == self.embed_dim

        gamma = self.gamma.view(1, self.num_views, self.embed_dim, 1, 1)
        beta = self.beta.view(1, self.num_views, self.embed_dim, 1, 1)
        
        return feat * gamma + beta

class CrossViewAttentionFusion(nn.Module):
    '''
    n개 view의 feature를 multi-head attention으로 융합
    '''

    def __init__(
            self,
            embed_dim : int,
            num_views : int = 5,
            num_heads : int = 8,
            dropout : float = 0.0,
            spatial_downsample : int = 1, # > 1이면 attention 이전 공간 압축
            ):
        super().__init__()
        self.num_views = num_views
        self.embed_dim = embed_dim
        self.ds = spatial_downsample

        self.view_film = ViewFiLM(num_views, embed_dim)

        self.pool = (
            nn.AvgPool2d(spatial_downsample, spatial_downsample)
            if spatial_downsample > 1
            else nn.Identity()
            )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = embed_dim * 4,
            dropout = dropout,
            activation = 'gelu',
            batch_first = True,
            norm_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.view_gate = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )
        nn.init.zeros_(self.view_gate[-1].weight)
        nn.init.zeros_(self.view_gate[-1].bias)
        self.out_conv = ConvBNSiLU(embed_dim, embed_dim, 1, 1, 0)

    def forward(self, view_feats : torch.Tensor) -> torch.Tensor:
        '''
        view_feats : [B, N, C, H, W]
        returns    : [B, C, H, W] fused feature
        '''

        B, N, C, H, W = view_feats.shape

        assert N == self.num_views
        assert C == self.embed_dim
        
        x = view_feats

        # view-conditioned feature calibration
        x = self.view_film(x)

        # optional spatial downsample
        if self.ds > 1:
            x = x.reshape(B * N, C, H, W)
            x = self.pool(x)
            _, _, h_, w_ = x.shape
            x = x.view(B, N, C, h_, w_)
        else:
            h_, w_ = H, W
        
        # attention over views at each spatial location
        x = x.permute(0, 3, 4, 1, 2).reshape(B * h_ * w_, N, C)
        x = self.transformer(x)

        # fuse views with learned confidence weights
        weights = torch.softmax(self.view_gate(x), dim=1)
        x = (x * weights).sum(dim=1)
        fused = x.view(B, h_, w_, C).permute(0, 3, 1, 2).contiguous()

        
        if self.ds > 1:
            fused = F.interpolate(
                fused,
                size= (H, W),
                mode= 'bilinear',
                align_corners=False,
            )
        
        return self.out_conv(fused)
