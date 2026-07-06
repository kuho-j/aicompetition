import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.yolov8_backbone import ConvBNSiLU

def load_homography_matrices(path: str | Path, num_views: int) -> torch.Tensor:
    '''
    Load per-view homography matrices from a pickle file.

    The file may contain either a list of matrices or a dict indexed by view id.
    Matrices are expected to map each camera image coordinate into the shared
    training plane coordinate.
    '''

    path = Path(path)
    with path.open('rb') as f:
        homographies = pickle.load(f)

    if isinstance(homographies, dict):
        matrices = [homographies[i] for i in range(num_views)]
    else:
        matrices = list(homographies)

    if len(matrices) != num_views:
        raise ValueError(
            f'Expected {num_views} homography matrices, got {len(matrices)}'
        )

    tensor = torch.as_tensor(matrices, dtype=torch.float32)
    if tensor.shape != (num_views, 3, 3):
        raise ValueError(
            f'Expected homographies with shape ({num_views}, 3, 3), got {tuple(tensor.shape)}'
        )

    return tensor

def _clamp_homogeneous_denominator(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sign = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    return torch.where(x.abs() < eps, sign * eps, x)

class LearnableHomographyAlign(nn.Module):
    '''
    Warp per-view feature maps into a shared plane before cross-view attention.

    Homographies are initialized from calibration and kept learnable as 8 free
    parameters per view. The bottom-right element is fixed to 1 to remove the
    arbitrary projective scale.
    '''

    def __init__(
            self,
            homographies: torch.Tensor,
            source_to_target: bool = True,
            align_corners: bool = True,
            padding_mode: str = 'zeros',
            ):
        super().__init__()

        if homographies.ndim != 3 or homographies.shape[-2:] != (3, 3):
            raise ValueError(
                f'homographies must have shape [N, 3, 3], got {tuple(homographies.shape)}'
            )

        homographies = homographies.float().clone()
        homographies = homographies / _clamp_homogeneous_denominator(
            homographies[:, 2:3, 2:3]
        )

        params = torch.stack(
            [
                homographies[:, 0, 0],
                homographies[:, 0, 1],
                homographies[:, 0, 2],
                homographies[:, 1, 0],
                homographies[:, 1, 1],
                homographies[:, 1, 2],
                homographies[:, 2, 0],
                homographies[:, 2, 1],
            ],
            dim=1,
        )

        self.homography_params = nn.Parameter(params)
        self.source_to_target = source_to_target
        self.align_corners = align_corners
        self.padding_mode = padding_mode

    @property
    def num_views(self) -> int:
        return self.homography_params.shape[0]

    def homography_matrices(self) -> torch.Tensor:
        h = self.homography_params
        ones = torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)
        return torch.stack(
            [
                torch.stack([h[:, 0], h[:, 1], h[:, 2]], dim=1),
                torch.stack([h[:, 3], h[:, 4], h[:, 5]], dim=1),
                torch.stack([h[:, 6], h[:, 7], ones[:, 0]], dim=1),
            ],
            dim=1,
        )

    def _make_sampling_grid(
            self,
            feat_h: int,
            feat_w: int,
            image_h: int,
            image_w: int,
            device: torch.device,
            dtype: torch.dtype,
            ) -> torch.Tensor:
        calc_dtype = torch.float32
        matrices = self.homography_matrices().to(device=device, dtype=calc_dtype)
        if self.source_to_target:
            matrices = torch.linalg.inv(matrices)

        xs = torch.linspace(0, image_w - 1, feat_w, device=device, dtype=calc_dtype)
        ys = torch.linspace(0, image_h - 1, feat_h, device=device, dtype=calc_dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        ones = torch.ones_like(grid_x)
        target_points = torch.stack([grid_x, grid_y, ones], dim=0).reshape(3, -1)
        source_points = matrices @ target_points

        z = source_points[:, 2]
        z = _clamp_homogeneous_denominator(z)
        src_x = source_points[:, 0] / z
        src_y = source_points[:, 1] / z

        norm_x = 2.0 * src_x / max(image_w - 1, 1) - 1.0
        norm_y = 2.0 * src_y / max(image_h - 1, 1) - 1.0

        return torch.stack([norm_x, norm_y], dim=-1).view(
            self.num_views, feat_h, feat_w, 2
        ).to(dtype=dtype)

    def forward(self, view_feats: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        '''
        view_feats : [B, N, C, H, W]
        image_size : (image_h, image_w) from the input images
        '''

        B, N, C, H, W = view_feats.shape
        if N != self.num_views:
            raise ValueError(f'Expected {self.num_views} views, got {N}')

        image_h, image_w = image_size
        grid = self._make_sampling_grid(
            H,
            W,
            image_h,
            image_w,
            view_feats.device,
            view_feats.dtype,
        )
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * N, H, W, 2)

        x = view_feats.reshape(B * N, C, H, W)
        aligned = F.grid_sample(
            x,
            grid,
            mode='bilinear',
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        return aligned.view(B, N, C, H, W)

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
