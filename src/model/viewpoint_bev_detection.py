import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.grid_bev_layer import GridToBEVLayer


class SingleViewBEVDetector(nn.Module):
    """
    Single-image detector for top-down heatmap prediction.

    Pipeline:
        image -> shared backbone/FPN -> grid head + CenterNet head -> BEV warp

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
        bev_size: tuple[int, int] = (60, 80),
        heatmap_size: tuple[int, int] = (60, 80),
        num_grid_points: int = 4,
        decoder_channels: int = 64,
        center_head_channels: int = 128,
        viewpoint_dim: int = 16,
        viewpoint_embed_dim: int = 128,
        grid_visibility_threshold: float = 0.5,
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
            visibility_threshold=grid_visibility_threshold,
            num_classes=num_classes,
            center_head_channels=center_head_channels,
        )
        self.freeze_grid_to_bev = False

    def set_grid_to_bev_trainable(self, trainable: bool) -> None:
        self.freeze_grid_to_bev = not trainable
        if hasattr(self.grid_to_bev, "set_geometry_trainable"):
            self.grid_to_bev.set_geometry_trainable(trainable)
            for param in self.grid_to_bev.center_head.parameters():
                param.requires_grad = True
        else:
            for param in self.grid_to_bev.parameters():
                param.requires_grad = trainable

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_grid_to_bev:
            if hasattr(self.grid_to_bev, "set_geometry_trainable"):
                self.grid_to_bev.train(mode)
                self.grid_to_bev.set_geometry_trainable(False)
            else:
                self.grid_to_bev.eval()
        return self

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
        logits = bev_outputs["bev_center_logits"]
        offset = bev_outputs["bev_center_offset"]
        if logits.shape[-2:] != self.heatmap_size:
            logits = F.interpolate(
                logits,
                size=self.heatmap_size,
                mode="bilinear",
                align_corners=False,
            )
            offset = F.interpolate(
                offset,
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
            "offset": offset,
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
