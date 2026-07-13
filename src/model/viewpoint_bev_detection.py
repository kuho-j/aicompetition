import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.grid_bev_layer import GridToBEVLayer
from src.predict import decode_predictions


class SingleViewBEVDetector(nn.Module):
    """
    Single-image detector for image-space center prediction plus optional homography.

    Pipeline:
        image -> shared backbone/FPN -> optional grid head -> homography
                                    -> CenterNet head -> image-space heatmap

    forward() returns a dict by default:
        heatmap/logits/offset from the CenterNet head before homography,
        decoded center coordinates/classes from that heatmap,
        and homography/grid outputs when the grid head or a fixed homography is used.

    Pass return_aux=False for the old heatmap-only behavior.
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
        use_grid_head: bool = True,
        fixed_homography: torch.Tensor | None = None,
        decode_topk: int = 100,
        decode_score_threshold: float = 0.3,
    ):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.use_grid_head = use_grid_head
        self.decode_topk = decode_topk
        self.decode_score_threshold = decode_score_threshold
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
        if fixed_homography is None:
            self.register_buffer("fixed_homography", torch.empty(0), persistent=False)
        else:
            self.register_buffer(
                "fixed_homography",
                torch.as_tensor(fixed_homography, dtype=torch.float32),
                persistent=False,
            )

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

    def set_grid_head_enabled(self, enabled: bool) -> None:
        self.use_grid_head = enabled

    def set_fixed_homography(self, homography: torch.Tensor | None) -> None:
        if homography is None:
            self.fixed_homography = torch.empty(
                0,
                device=self.fixed_homography.device,
                dtype=self.fixed_homography.dtype,
            )
            return
        self.fixed_homography = torch.as_tensor(
            homography,
            device=self.fixed_homography.device,
            dtype=self.fixed_homography.dtype,
        )

    def forward(
        self,
        image: torch.Tensor,
        viewpoint: torch.Tensor | dict[str, torch.Tensor] | None = None,
        return_aux: bool = True,
        decode: bool = True,
        topk: int | None = None,
        score_threshold: float | None = None,
        use_grid_head: bool | None = None,
        homography: torch.Tensor | None = None,
        warp_center: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if image.dim() == 5:
            b, n, c, h, w = image.shape
            image = image.reshape(b * n, c, h, w)
            viewpoint = self._repeat_viewpoint_for_views(viewpoint, batch_size=b, num_views=n)

        active_use_grid_head = self.use_grid_head if use_grid_head is None else use_grid_head
        active_homography = self._resolve_homography(homography)
        outputs = self.grid_to_bev(
            image,
            return_center=True,
            use_grid_head=active_use_grid_head,
            homography=active_homography,
            warp_center=warp_center,
        )
        logits = outputs["image_center_logits"]
        offset = outputs["image_center_offset"]
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

        result = {
            "heatmap": heatmap,
            "heatmap_logits": logits,
            "offset": offset,
            **outputs,
        }

        if decode:
            decoded = decode_predictions(
                heatmap,
                offset=offset,
                topk=self.decode_topk if topk is None else topk,
                score_threshold=(
                    self.decode_score_threshold
                    if score_threshold is None
                    else score_threshold
                ),
            )
            result["detections"] = decoded
            result["scores"] = [item["scores"] for item in decoded]
            result["classes"] = [item["classes"] for item in decoded]
            result["centers"] = [item["centers"] for item in decoded]

        return result

    def _resolve_homography(self, homography: torch.Tensor | None) -> torch.Tensor | None:
        if homography is not None:
            return homography
        if self.fixed_homography.numel() == 0:
            return None
        return self.fixed_homography

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
