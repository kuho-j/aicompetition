import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.fpn_neck import FPNNeck
from src.model.detection import CenterNetHead
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
        grid_visible: [B, N], confidence-thresholded visibility mask
        homography_point_mask: [B, N], points used for homography estimation
        homography: [B, 3, 3], mapping BEV normalized coords to feature normalized coords
        grid_logits: [B, N, feat_h, feat_w]
        image_center_logits: [B, C_cls, feat_h, feat_w], before homography
        image_center_offset: [B, 2, feat_h, feat_w], before homography
        bev_center_logits: [B, C_cls, bev_h, bev_w], only when warp_center=True
        bev_center_offset: [B, 2, bev_h, bev_w], only when warp_center=True
    """

    def __init__(
        self,
        num_grid_points: int = 4,
        img_channels: int = 3,
        fpn_out_channels: int = 256,
        backbone_width: float = 0.25,
        backbone_depth: float = 0.33,
        bev_size: tuple[int, int] = (60, 80),
        bev_grid_points: torch.Tensor | None = None,
        visibility_threshold: float = 0.5,
        num_classes: int = 60,
        center_head_channels: int = 128,
    ):
        super().__init__()
        if num_grid_points < 4:
            raise ValueError("At least 4 grid points are required to estimate homography.")

        self.num_grid_points = num_grid_points
        self.img_channels = img_channels
        self.bev_size = bev_size
        self.visibility_threshold = visibility_threshold

        self.backbone = YOLOv8Backbone(img_channels, backbone_width, backbone_depth)
        self.fpn = FPNNeck(self.backbone.out_channels, fpn_out_channels)
        self.grid_head = GridPointHead(fpn_out_channels, num_grid_points)
        self.center_head = CenterNetHead(
            fpn_out_channels,
            num_classes=num_classes,
            hidden_channels=center_head_channels,
        )
        self.freeze_geometry = False

        if bev_grid_points is None:
            bev_grid_points = self._default_bev_grid_points(num_grid_points)
        if bev_grid_points.shape != (num_grid_points, 2):
            raise ValueError(
                f"bev_grid_points must have shape ({num_grid_points}, 2), "
                f"got {tuple(bev_grid_points.shape)}."
            )
        self.register_buffer("bev_grid_points", bev_grid_points.float())

    def set_geometry_trainable(self, trainable: bool) -> None:
        self.freeze_geometry = not trainable
        for module in (self.backbone, self.fpn, self.grid_head):
            for param in module.parameters():
                param.requires_grad = trainable
            if not trainable:
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_geometry:
            self.backbone.eval()
            self.fpn.eval()
            self.grid_head.eval()
        return self

    def forward(
        self,
        image: torch.Tensor,
        return_center: bool = True,
        use_grid_head: bool = True,
        homography: torch.Tensor | None = None,
        warp_center: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        image:
            [H, W, C], [C, H, W], [B, H, W, C], or [B, C, H, W]
        """
        image = self._to_bchw(image)

        if self.freeze_geometry:
            with torch.no_grad():
                image_feature = self._forward_image_feature(image)
                geometry_outputs = self._forward_geometry(
                    image,
                    image_feature,
                    use_grid_head=use_grid_head,
                    homography=homography,
                    warp_feature=use_grid_head or homography is not None,
                )
            image_feature = image_feature.detach()
            geometry_outputs = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in geometry_outputs.items()
            }
        else:
            image_feature = self._forward_image_feature(image)
            geometry_outputs = self._forward_geometry(
                image,
                image_feature,
                use_grid_head=use_grid_head,
                homography=homography,
                warp_feature=use_grid_head or homography is not None,
            )

        if not return_center:
            return geometry_outputs

        center_outputs = self.center_head(image_feature)
        image_center_logits = center_outputs["heatmap_logits"]
        image_center_offset = center_outputs["offset"]
        outputs = {
            **geometry_outputs,
            "center_feature": center_outputs["center_feature"],
            "image_center_logits": image_center_logits,
            "image_center_offset": image_center_offset,
        }

        if warp_center:
            if "homography" not in geometry_outputs:
                raise ValueError("warp_center=True requires grid head output or a supplied homography.")
            bev_center_logits, _ = self._warp_to_bev(
                image_center_logits,
                geometry_outputs["homography"],
            )
            bev_center_offset, _ = self._warp_to_bev(
                image_center_offset,
                geometry_outputs["homography"],
            )
            outputs.update(
                {
                    "bev_center_logits": bev_center_logits,
                    "bev_center_offset": bev_center_offset,
                }
            )

        return outputs

    def _forward_image_feature(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        fpn_features = self.fpn(features)
        return fpn_features[0]

    def _forward_geometry(
        self,
        image: torch.Tensor,
        image_feature: torch.Tensor,
        use_grid_head: bool,
        homography: torch.Tensor | None,
        warp_feature: bool,
    ) -> dict[str, torch.Tensor]:
        _, _, img_h, img_w = image.shape
        _, _, feat_h, feat_w = image_feature.shape

        outputs: dict[str, torch.Tensor] = {
            "image_feature": image_feature,
        }

        if use_grid_head:
            grid_outputs = self._forward_grid_head(image, image_feature)
            outputs.update(grid_outputs)
        elif homography is not None:
            outputs["homography"] = self._prepare_homography(
                homography,
                batch_size=image.shape[0],
                device=image_feature.device,
                dtype=image_feature.dtype,
            )

        if warp_feature:
            if "homography" not in outputs:
                raise ValueError("warp_feature=True requires grid head output or a supplied homography.")
            bev_feature, bev_valid_mask = self._warp_to_bev(
                image_feature,
                outputs["homography"],
            )
            outputs["bev_feature"] = bev_feature
            outputs["bev_valid_mask"] = bev_valid_mask

        return outputs

    def _forward_grid_head(
        self,
        image: torch.Tensor,
        image_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _, _, img_h, img_w = image.shape
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
        grid_visible = grid_confidence >= self.visibility_threshold
        homography_point_mask = self._make_homography_point_mask(
            grid_confidence,
            grid_visible,
            min_points=4,
        )
        homography = self._estimate_homography(
            self.bev_grid_points,
            dst_feature_norm,
            point_mask=homography_point_mask,
        )

        return {
            "grid_points": grid_points_image,
            "grid_confidence": grid_confidence,
            "grid_visible": grid_visible,
            "homography_point_mask": homography_point_mask,
            "homography": homography,
            "grid_logits": grid_logits,
        }

    @staticmethod
    def _prepare_homography(
        homography: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        homography = torch.as_tensor(homography, device=device, dtype=dtype)
        if homography.dim() == 2:
            if homography.shape != (3, 3):
                raise ValueError(f"homography must have shape [3, 3], got {tuple(homography.shape)}")
            homography = homography.unsqueeze(0).expand(batch_size, -1, -1)
        elif homography.dim() == 3:
            if homography.shape[1:] != (3, 3):
                raise ValueError(
                    f"homography must have shape [B, 3, 3], got {tuple(homography.shape)}"
                )
            if homography.shape[0] == 1 and batch_size != 1:
                homography = homography.expand(batch_size, -1, -1)
            elif homography.shape[0] != batch_size:
                raise ValueError(
                    f"homography batch size must be 1 or {batch_size}, got {homography.shape[0]}"
                )
        else:
            raise ValueError(f"homography must have shape [3, 3] or [B, 3, 3], got {tuple(homography.shape)}")

        return homography

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
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _ = dst_feature_norm.shape
        src = src_bev_norm.to(dst_feature_norm.device, dst_feature_norm.dtype)
        src = src.unsqueeze(0).expand(b, -1, -1)

        if point_mask is not None:
            if point_mask.shape != (b, n):
                raise ValueError(
                    f"point_mask must have shape {(b, n)}, got {tuple(point_mask.shape)}"
                )
            homographies = []
            for batch_idx in range(b):
                selected = point_mask[batch_idx].to(
                    device=dst_feature_norm.device,
                    dtype=torch.bool,
                )
                homographies.append(
                    self._estimate_homography_dense(
                        src[batch_idx : batch_idx + 1, selected],
                        dst_feature_norm[batch_idx : batch_idx + 1, selected],
                    )
                )
            return torch.cat(homographies, dim=0)

        return self._estimate_homography_dense(src, dst_feature_norm)

    @staticmethod
    def _estimate_homography_dense(
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = dst.shape

        x = src[..., 0]
        y = src[..., 1]
        u = dst[..., 0]
        v = dst[..., 1]
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)

        row_u = torch.stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], dim=-1)
        row_v = torch.stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], dim=-1)
        a = torch.stack([row_u, row_v], dim=2).reshape(b, n * 2, 9)

        _, _, vh = torch.linalg.svd(a)
        h = vh[:, -1, :].reshape(b, 3, 3)
        return h / GridToBEVLayer._safe_denominator(h[:, 2:3, 2:3])

    @staticmethod
    def _make_homography_point_mask(
        confidence: torch.Tensor,
        visible: torch.Tensor,
        min_points: int,
    ) -> torch.Tensor:
        mask = visible.clone()
        counts = mask.sum(dim=1)
        if torch.all(counts >= min_points):
            return mask

        _, topk_indices = confidence.topk(k=min_points, dim=1)
        fallback = torch.zeros_like(mask)
        fallback.scatter_(1, topk_indices, True)
        return torch.where((counts >= min_points).unsqueeze(1), mask, fallback)

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
