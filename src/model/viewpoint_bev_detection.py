import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.grid_bev_layer import GridToBEVLayer
from src.model.yolov8_backbone import ConvBNSiLU


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNSiLU(channels, channels, 3),
            ConvBNSiLU(channels, channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ViewpointConditioner(nn.Module):
    """
    Turns viewpoint-estimator outputs into a fixed conditioning vector.

    Accepted inputs:
        None
        Tensor [B, D]
        dict with any of: "embedding", "logits", "probs", "params"
    """

    def __init__(self, viewpoint_dim: int = 16, embed_dim: int = 128):
        super().__init__()
        self.viewpoint_dim = viewpoint_dim
        self.null_viewpoint = nn.Parameter(torch.zeros(1, viewpoint_dim))
        self.mlp = nn.Sequential(
            nn.Linear(viewpoint_dim, embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        viewpoint: torch.Tensor | dict[str, torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if viewpoint is None:
            vector = self.null_viewpoint.expand(batch_size, -1)
        elif isinstance(viewpoint, dict):
            parts = []
            for key in ("embedding", "probs", "logits", "params"):
                value = viewpoint.get(key)
                if value is not None:
                    parts.append(self._as_2d(value))
            if len(parts) == 0:
                vector = self.null_viewpoint.expand(batch_size, -1)
            else:
                vector = torch.cat(parts, dim=1)
        else:
            vector = self._as_2d(viewpoint)

        vector = vector.to(device=device, dtype=dtype)
        vector = self._fit_dim(vector)
        return self.mlp(vector)

    def _fit_dim(self, vector: torch.Tensor) -> torch.Tensor:
        dim = vector.shape[1]
        if dim == self.viewpoint_dim:
            return vector
        if dim > self.viewpoint_dim:
            return vector[:, : self.viewpoint_dim]

        pad = vector.new_zeros(vector.shape[0], self.viewpoint_dim - dim)
        return torch.cat([vector, pad], dim=1)

    @staticmethod
    def _as_2d(value: torch.Tensor) -> torch.Tensor:
        if value.dim() == 1:
            return value.unsqueeze(1)
        return value.flatten(1)


class FiLM2d(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1.0 + torch.tanh(scale)) + shift


class BEVRefinementDecoder(nn.Module):
    """
    Fast BEV U-Net style decoder.

    It keeps the high-resolution BEV grid, adds two lower-resolution context paths,
    and conditions every stage with the viewpoint embedding.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 192,
        cond_dim: int = 128,
        num_classes: int = 60,
    ):
        super().__init__()
        self.input_proj = ConvBNSiLU(in_channels + 3, hidden_channels, 3)

        self.enc1 = ResidualConvBlock(hidden_channels)
        self.down1 = ConvBNSiLU(hidden_channels, hidden_channels, 3, 2)
        self.enc2 = ResidualConvBlock(hidden_channels)
        self.down2 = ConvBNSiLU(hidden_channels, hidden_channels, 3, 2)
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(hidden_channels),
            ResidualConvBlock(hidden_channels),
        )

        self.film1 = FiLM2d(hidden_channels, cond_dim)
        self.film2 = FiLM2d(hidden_channels, cond_dim)
        self.film3 = FiLM2d(hidden_channels, cond_dim)

        self.up2 = ConvBNSiLU(hidden_channels * 2, hidden_channels, 3)
        self.up1 = ConvBNSiLU(hidden_channels * 2, hidden_channels, 3)
        self.out_refine = nn.Sequential(
            ResidualConvBlock(hidden_channels),
            ConvBNSiLU(hidden_channels, hidden_channels, 3),
        )
        self.heatmap = nn.Conv2d(hidden_channels, num_classes, 1)
        nn.init.constant_(self.heatmap.bias, -math.log((1 - 0.01) / 0.01))

    def forward(
        self,
        bev_feature: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = bev_feature.shape
        coords = self._coord_channels(b, h, w, bev_feature.device, bev_feature.dtype)
        x = torch.cat([bev_feature * valid_mask, valid_mask, coords], dim=1)

        x0 = self.input_proj(x)
        x1 = self.film1(self.enc1(x0), cond)
        x2 = self.film2(self.enc2(self.down1(x1)), cond)
        x3 = self.film3(self.bottleneck(self.down2(x2)), cond)

        x = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, x2], dim=1))
        x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, x1], dim=1))
        x = self.out_refine(x)
        return self.heatmap(x)

    @staticmethod
    def _coord_channels(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=0)
        return coords.unsqueeze(0).expand(batch_size, -1, -1, -1)


class SingleViewBEVDetector(nn.Module):
    """
    Single-image detector for top-down heatmap prediction.

    Pipeline:
        image -> GridToBEVLayer -> viewpoint-conditioned BEV decoder -> heatmap

    By default forward() returns the heatmap tensor for compatibility with the
    existing training loop. Set return_aux=True to also get grid/homography
    diagnostics from GridToBEVLayer.
    """

    def __init__(
        self,
        num_classes: int = 60,
        img_channels: int = 3,
        fpn_out_channels: int = 256,
        backbone_width: float = 0.25,
        backbone_depth: float = 0.33,
        bev_size: tuple[int, int] = (128, 128),
        heatmap_size: tuple[int, int] = (60, 80),
        num_grid_points: int = 4,
        decoder_channels: int = 192,
        viewpoint_dim: int = 16,
        viewpoint_embed_dim: int = 128,
    ):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.grid_to_bev = GridToBEVLayer(
            num_grid_points=num_grid_points,
            img_channels=img_channels,
            fpn_out_channels=fpn_out_channels,
            backbone_width=backbone_width,
            backbone_depth=backbone_depth,
            bev_size=bev_size,
        )
        self.viewpoint_conditioner = ViewpointConditioner(
            viewpoint_dim=viewpoint_dim,
            embed_dim=viewpoint_embed_dim,
        )
        self.decoder = BEVRefinementDecoder(
            in_channels=fpn_out_channels,
            hidden_channels=decoder_channels,
            cond_dim=viewpoint_embed_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        image: torch.Tensor,
        viewpoint: torch.Tensor | dict[str, torch.Tensor] | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if image.dim() == 5:
            b, n, c, h, w = image.shape
            image = image.reshape(b * n, c, h, w)
            viewpoint = self._repeat_viewpoint_for_views(viewpoint, batch_size=b, num_views=n)

        bev_outputs = self.grid_to_bev(image)
        bev_feature = bev_outputs["bev_feature"]
        valid_mask = bev_outputs["bev_valid_mask"]
        cond = self.viewpoint_conditioner(
            viewpoint,
            batch_size=bev_feature.shape[0],
            device=bev_feature.device,
            dtype=bev_feature.dtype,
        )

        logits = self.decoder(bev_feature, valid_mask, cond)
        if logits.shape[-2:] != self.heatmap_size:
            logits = F.interpolate(
                logits,
                size=self.heatmap_size,
                mode="bilinear",
                align_corners=False,
            )
        heatmap = logits.sigmoid()

        if not return_aux:
            return heatmap

        return {
            "heatmap": heatmap,
            "heatmap_logits": logits,
            **bev_outputs,
        }

    @staticmethod
    def _repeat_viewpoint_for_views(
        viewpoint: torch.Tensor | dict[str, torch.Tensor] | None,
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor | dict[str, torch.Tensor] | None:
        if viewpoint is None:
            return None
        if torch.is_tensor(viewpoint):
            if viewpoint.shape[0] == batch_size:
                return viewpoint.repeat_interleave(num_views, dim=0)
            return viewpoint

        repeated = {}
        for key, value in viewpoint.items():
            if torch.is_tensor(value) and value.shape[0] == batch_size:
                repeated[key] = value.repeat_interleave(num_views, dim=0)
            else:
                repeated[key] = value
        return repeated


# Backwards-friendly alias while the repository transitions away from 5-view input.
ViewpointBEVDetector = SingleViewBEVDetector
