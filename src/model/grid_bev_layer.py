import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.fpn_neck import FPNNeck
from src.model.yolov8_backbone import ConvBNSiLU, YOLOv8Backbone


class GridPointHead(nn.Module):
    """
    Predicts one image-space heatmap per reference grid point.

    output:
        grid_logits: [B, num_grid_points, H, W]
    """

    def __init__(self, in_channels: int, num_grid_points: int):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNSiLU(in_channels, in_channels, 3),
            ConvBNSiLU(in_channels, in_channels, 3),
            nn.Conv2d(in_channels, num_grid_points, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class GridToBEVLayer(nn.Module):
    """
    Converts a single image into a BEV-aligned feature map using labeled grid points.

    Main output for the following heatmap model:
        bev_feature: [B, C, bev_h, bev_w]
        bev_valid_mask: [B, 1, bev_h, bev_w]

    Diagnostic/auxiliary outputs:
        grid_points: [B, N, 2] in input-image pixel coordinates, ordered as (x, y)
        grid_confidence: [B, N]
        homography: [B, 3, 3], mapping BEV normalized coords to feature normalized coords
        grid_logits: [B, N, feat_h, feat_w]
    """

    def __init__(
        self,
        num_grid_points: int = 4,
        img_channels: int = 3,
        fpn_out_channels: int = 256,
        backbone_width: float = 0.25,
        backbone_depth: float = 0.33,
        bev_size: tuple[int, int] = (192, 256),
        bev_grid_points: torch.Tensor | None = None,
    ):
        super().__init__()
        if num_grid_points < 4:
            raise ValueError("At least 4 grid points are required to estimate homography.")

        self.num_grid_points = num_grid_points
        self.img_channels = img_channels
        self.bev_size = bev_size

        self.backbone = YOLOv8Backbone(img_channels, backbone_width, backbone_depth)
        self.fpn = FPNNeck(self.backbone.out_channels, fpn_out_channels)
        self.grid_head = GridPointHead(fpn_out_channels, num_grid_points)

        if bev_grid_points is None:
            bev_grid_points = self._default_bev_grid_points(num_grid_points)
        if bev_grid_points.shape != (num_grid_points, 2):
            raise ValueError(
                f"bev_grid_points must have shape ({num_grid_points}, 2), "
                f"got {tuple(bev_grid_points.shape)}."
            )
        self.register_buffer("bev_grid_points", bev_grid_points.float())

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        image:
            [H, W, C], [C, H, W], [B, H, W, C], or [B, C, H, W]
        """
        image = self._to_bchw(image)
        _, _, img_h, img_w = image.shape

        features = self.backbone(image)
        fpn_features = self.fpn(features)
        image_feature = fpn_features[0]
        grid_logits = self.grid_head(image_feature)

        grid_points_feature, grid_confidence = self._softargmax_points(grid_logits)
        _, _, feat_h, feat_w = image_feature.shape

        grid_points_image = grid_points_feature.clone()
        grid_points_image[..., 0] = grid_points_image[..., 0] * img_w / feat_w
        grid_points_image[..., 1] = grid_points_image[..., 1] * img_h / feat_h
  
        dst_feature_norm = self._pixel_to_norm(
            grid_points_feature,
            height=feat_h,
            width=feat_w,
        )
        homography = self._estimate_homography(self.bev_grid_points, dst_feature_norm)

        bev_feature, bev_valid_mask = self._warp_to_bev(image_feature, homography)

        return {
            "bev_feature": bev_feature,
            "bev_valid_mask": bev_valid_mask,
            "grid_points": grid_points_image,
            "grid_confidence": grid_confidence,
            "homography": homography,
            "grid_logits": grid_logits,
        }

    def _to_bchw(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            if image.shape[-1] == self.img_channels:
                image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
        elif image.dim() == 4:
            if image.shape[-1] == self.img_channels:
                image = image.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "image must have shape [H, W, C], [C, H, W], [B, H, W, C], "
                "or [B, C, H, W]."
            )
        return image.float()

    def _softargmax_points(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, h, w = logits.shape
        probs = logits.flatten(2).softmax(dim=-1).view(b, n, h, w)

        ys = torch.linspace(0, h - 1, h, device=logits.device, dtype=logits.dtype)
        xs = torch.linspace(0, w - 1, w, device=logits.device, dtype=logits.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        x = (probs * xx).sum(dim=(2, 3))
        y = (probs * yy).sum(dim=(2, 3))
        points = torch.stack([x, y], dim=-1)

        confidence = logits.flatten(2).amax(dim=-1).sigmoid()
        return points, confidence

    def _warp_to_bev(
        self,
        feature: torch.Tensor,
        homography: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = feature.shape[0]
        bev_h, bev_w = self.bev_size
        base_grid = self._make_bev_grid(b, bev_h, bev_w, feature.device, feature.dtype)
        sample_grid = self._apply_homography(base_grid, homography)

        bev_feature = F.grid_sample(
            feature,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        ones = torch.ones(
            b,
            1,
            feature.shape[2],
            feature.shape[3],
            device=feature.device,
            dtype=feature.dtype,
        )
        bev_valid_mask = F.grid_sample(
            ones,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).clamp(0.0, 1.0)

        return bev_feature, bev_valid_mask

    def _estimate_homography(
        self,
        src_bev_norm: torch.Tensor,
        dst_feature_norm: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = dst_feature_norm.shape
        src = src_bev_norm.to(dst_feature_norm.device, dst_feature_norm.dtype)
        src = src.unsqueeze(0).expand(b, -1, -1)

        x = src[..., 0]
        y = src[..., 1]
        u = dst_feature_norm[..., 0]
        v = dst_feature_norm[..., 1]
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)

        row_u = torch.stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], dim=-1)
        row_v = torch.stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], dim=-1)
        a = torch.stack([row_u, row_v], dim=2).reshape(b, n * 2, 9)

        _, _, vh = torch.linalg.svd(a)
        h = vh[:, -1, :].reshape(b, 3, 3)
        return h / self._safe_denominator(h[:, 2:3, 2:3])

    @staticmethod
    def _pixel_to_norm(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
        denom_x = max(width - 1, 1)
        denom_y = max(height - 1, 1)
        x = points[..., 0] / denom_x * 2.0 - 1.0
        y = points[..., 1] / denom_y * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _make_bev_grid(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    @staticmethod
    def _apply_homography(grid: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = grid.shape
        ones = torch.ones(b, h, w, 1, device=grid.device, dtype=grid.dtype)
        points = torch.cat([grid, ones], dim=-1).view(b, h * w, 3)
        warped = points @ homography.transpose(1, 2)
        xy = warped[..., :2] / GridToBEVLayer._safe_denominator(warped[..., 2:])
        return xy.view(b, h, w, 2)

    @staticmethod
    def _safe_denominator(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        sign = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
        return torch.where(x.abs() < eps, sign * eps, x)

    @staticmethod
    def _default_bev_grid_points(num_grid_points: int) -> torch.Tensor:
        if num_grid_points == 4:
            return torch.tensor(
                [
                    [-1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                ]
            )

        side = int(num_grid_points**0.5)
        if side * side != num_grid_points:
            raise ValueError(
                "Provide bev_grid_points when num_grid_points is not 4 or a square number."
            )

        ys = torch.linspace(-1.0, 1.0, side)
        xs = torch.linspace(-1.0, 1.0, side)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten()], dim=-1)
