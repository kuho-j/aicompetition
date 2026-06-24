import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.yolov8_backbone import ConvBNSiLU

class ViewPositionalEncoding(nn.Module):
    '''
    카메라 시점 별 학습 가능한 임베딩
    '''

    def __init__(self, num_views : int, embed_dim : int):
        super().__init__()

        self.embed = nn.Parameter(torch.zeros(num_views, embed_dim))
        nn.init.trunc_normal_(self.embed, std=0.02)

    def forward(self, feat : torch.Tensor, view_idx : int) -> torch.Tensor:
        '''feat : [B, C, H, W]'''

        emb = self.embed[view_idx].view(1, -1, 1, 1)
        
        return feat + emb

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

        self.pos_enc = ViewPositionalEncoding(num_views, embed_dim)

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

        # view positional embedding
        view_emb = self.pos_enc.embed.view(1, N, C, 1, 1)
        x = x + view_emb

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

        # fused views
        x = x.mean(dim=1)
        fused = x.view(B, h_, w_, C).permute(0, 3, 1, 2).contiguous()

        
        if self.ds > 1:
            fused = F.interpolate(
                fused,
                size= (H, W),
                mode= 'bilinear',
                align_corners=False,
            )
        
        return self.out_conv(fused)