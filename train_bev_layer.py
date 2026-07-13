import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.load_data_for_bev_train import load_data_bev
from src.aug import rotation_augmentation
from src.dataset import GridBEVDataset, collate_grid_bev_fn
from src.model.grid_bev_layer import GridToBEVLayer


ROTATION_AUG_WARMUP_EPOCHS = 10
ROTATION_AUG_MILD_END_EPOCH = 40
ROTATION_AUG_MILD_DEGREE_RANGE = (-5.0, 5.0)
ROTATION_AUG_STRONG_DEGREE_RANGE = (-15.0, 15.0)


def get_rotation_aug_degree_range(epoch: int) -> tuple[float, float] | None:
    if epoch <= ROTATION_AUG_WARMUP_EPOCHS:
        return None
    if epoch <= ROTATION_AUG_MILD_END_EPOCH:
        return ROTATION_AUG_MILD_DEGREE_RANGE
    return ROTATION_AUG_STRONG_DEGREE_RANGE


def fit_grid_points_to_count(
    grid_points: torch.Tensor,
    num_grid_points: int,
) -> torch.Tensor:
    grid_points = torch.as_tensor(grid_points, dtype=torch.float32)
    if grid_points.ndim != 2 or grid_points.shape[1] != 2:
        raise ValueError(f"grid_points must have shape [N, 2], got {tuple(grid_points.shape)}")

    current_count = grid_points.shape[0]
    if current_count == num_grid_points:
        return grid_points
    if current_count > num_grid_points:
        raise ValueError(
            f"grid_points has {current_count} points, but model is configured for "
            f"{num_grid_points}. Set --num-grid-points to match the labels."
        )
    if current_count == 0:
        pad = grid_points.new_zeros(num_grid_points, 2)
        return pad

    pad_count = num_grid_points - current_count
    pad = grid_points[-1:].expand(pad_count, -1)
    return torch.cat([grid_points, pad], dim=0)


def fit_grid_visible_to_count(
    grid_visible: torch.Tensor,
    num_grid_points: int,
) -> torch.Tensor:
    grid_visible = torch.as_tensor(grid_visible, dtype=torch.float32)
    if grid_visible.ndim != 1:
        raise ValueError(f"grid_visible must have shape [N], got {tuple(grid_visible.shape)}")

    current_count = grid_visible.shape[0]
    if current_count == num_grid_points:
        return grid_visible
    if current_count > num_grid_points:
        raise ValueError(
            f"grid_visible has {current_count} points, but model is configured for "
            f"{num_grid_points}. Set --num-grid-points to match the labels."
        )

    pad_count = num_grid_points - current_count
    pad = grid_visible.new_zeros(pad_count)
    return torch.cat([grid_visible, pad], dim=0)


def load_bev_training_samples(
    filepaths_path: str,
    num_grid_points: int,
    max_samples: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    img_list: list[torch.Tensor] = []
    grid_list: list[torch.Tensor] = []
    grid_visible_list: list[torch.Tensor] = []

    with open(filepaths_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            sample = load_data_bev(line)
            img_list.append(sample["image"])
            grid_list.append(fit_grid_points_to_count(sample["grid_points"], num_grid_points))
            grid_visible_list.append(
                fit_grid_visible_to_count(sample["grid_visible"], num_grid_points)
            )

            if max_samples is not None and len(img_list) >= max_samples:
                break

    if len(img_list) == 0:
        raise ValueError(f"no BEV training samples were loaded from {filepaths_path}")

    return img_list, grid_list, grid_visible_list


def make_train_loader(
    filepaths_path: str,
    batch_size: int,
    num_workers: int,
    num_grid_points: int,
    max_samples: int | None = None,
) -> DataLoader:
    img_list, grid_list, grid_visible_list = load_bev_training_samples(
        filepaths_path=filepaths_path,
        num_grid_points=num_grid_points,
        max_samples=max_samples,
    )
    dataset = GridBEVDataset(
        img_list=img_list,
        grid_list=grid_list,
        grid_visible_list=grid_visible_list,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_grid_bev_fn,
    )


@dataclass
class GridBEVLossWeights:
    heatmap: float = 1.0
    coord: float = 3.0
    reprojection: float = 0.0


class GridBEVLayerLoss:
    def __init__(
        self,
        weights: GridBEVLossWeights,
        heatmap_sigma: float = 1.5,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
    ):
        self.weights = weights
        self.heatmap_sigma = heatmap_sigma
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        bev_grid_points: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        images = batch["image"]
        gt_points_img = batch["grid_points"]
        visible = batch["grid_visible"].to(
            device=gt_points_img.device,
            dtype=gt_points_img.dtype,
        )

        grid_logits = outputs["grid_logits"]
        pred_points_img = outputs["grid_points"]
        homography = outputs["homography"]

        _, _, img_h, img_w = images.shape
        _, _, feat_h, feat_w = grid_logits.shape

        gt_points_feat = image_points_to_feature_points(
            gt_points_img,
            image_size=(img_h, img_w),
            feature_size=(feat_h, feat_w),
        )
        gt_heatmap = render_grid_heatmaps(
            gt_points_feat,
            visible,
            height=feat_h,
            width=feat_w,
            sigma=self.heatmap_sigma,
        )

        heatmap_loss = masked_gaussian_focal_loss(
            grid_logits.sigmoid(),
            gt_heatmap,
            visible,
            alpha=self.focal_alpha,
            beta=self.focal_beta,
        )
        coord_loss = normalized_point_smooth_l1_loss(
            pred_points_img,
            gt_points_img,
            visible,
            image_size=(img_h, img_w),
        )
        if self.weights.reprojection > 0:
            reprojection_loss = homography_reprojection_loss(
                homography.detach(),
                bev_grid_points,
                gt_points_feat,
                visible,
                feature_size=(feat_h, feat_w),
            )
        else:
            reprojection_loss = homography.new_zeros(())

        total = (
            self.weights.heatmap * heatmap_loss
            + self.weights.coord * coord_loss
            + self.weights.reprojection * reprojection_loss
        )
        logs = {
            "loss": total.detach(),
            "loss_heatmap": heatmap_loss.detach(),
            "loss_coord": coord_loss.detach(),
            "loss_reprojection": reprojection_loss.detach(),
        }
        return total, logs


def render_grid_heatmaps(
    points: torch.Tensor,
    visible: torch.Tensor,
    height: int,
    width: int,
    sigma: float,
) -> torch.Tensor:
    b, n, _ = points.shape
    device = points.device
    dtype = points.dtype

    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    heatmaps = points.new_zeros(b, n, height, width)
    for batch_idx in range(b):
        for point_idx in range(n):
            if visible[batch_idx, point_idx] <= 0:
                continue

            x = points[batch_idx, point_idx, 0]
            y = points[batch_idx, point_idx, 1]
            if x < 0 or x > width - 1 or y < 0 or y > height - 1:
                continue

            gaussian = torch.exp(-((xx - x).pow(2) + (yy - y).pow(2)) / (2 * sigma**2))
            heatmaps[batch_idx, point_idx] = gaussian
            heatmaps[batch_idx, point_idx, y.round().long(), x.round().long()] = 1.0

    return heatmaps


def masked_gaussian_focal_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    visible: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    mask = visible[:, :, None, None].to(device=pred.device, dtype=pred.dtype)
    pos_mask = gt.eq(1.0).to(dtype=pred.dtype) * mask
    neg_mask = gt.lt(1.0).to(dtype=pred.dtype) * mask

    pos_loss = (
        -(1.0 - pred).pow(alpha)
        * torch.log(pred.clamp(min=1e-12))
        * pos_mask
    )
    neg_loss = (
        -pred.pow(alpha)
        * (1.0 - gt).pow(beta)
        * torch.log((1.0 - pred).clamp(min=1e-12))
        * neg_mask
    )

    num_pos = pos_mask.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def normalized_point_smooth_l1_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    visible: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    img_h, img_w = image_size
    scale = pred_points.new_tensor([max(img_w - 1, 1), max(img_h - 1, 1)])
    pred_norm = pred_points / scale
    gt_norm = gt_points / scale

    loss = F.smooth_l1_loss(pred_norm, gt_norm, reduction="none").sum(dim=-1)
    loss = loss * visible.to(device=loss.device, dtype=loss.dtype)
    return loss.sum() / visible.sum().clamp(min=1.0)


def homography_reprojection_loss(
    homography: torch.Tensor,
    bev_grid_points: torch.Tensor,
    gt_points_feat: torch.Tensor,
    visible: torch.Tensor,
    feature_size: tuple[int, int],
) -> torch.Tensor:
    feat_h, feat_w = feature_size
    target_norm = pixel_to_norm(gt_points_feat, height=feat_h, width=feat_w)

    src = bev_grid_points.to(device=homography.device, dtype=homography.dtype)
    src = src.unsqueeze(0).expand(homography.shape[0], -1, -1)
    ones = torch.ones(*src.shape[:2], 1, device=src.device, dtype=src.dtype)
    src_h = torch.cat([src, ones], dim=-1)

    projected = src_h @ homography.transpose(1, 2)
    projected_xy = projected[..., :2] / safe_denominator(projected[..., 2:])

    loss = F.smooth_l1_loss(projected_xy, target_norm, reduction="none").sum(dim=-1)
    loss = loss * visible.to(device=loss.device, dtype=loss.dtype)
    return loss.sum() / visible.sum().clamp(min=1.0)


def image_points_to_feature_points(
    points: torch.Tensor,
    image_size: tuple[int, int],
    feature_size: tuple[int, int],
) -> torch.Tensor:
    img_h, img_w = image_size
    feat_h, feat_w = feature_size
    out = points.clone()
    out[..., 0] = out[..., 0] * feat_w / img_w
    out[..., 1] = out[..., 1] * feat_h / img_h
    out[..., 0] = out[..., 0].clamp(0, max(feat_w - 1, 0))
    out[..., 1] = out[..., 1].clamp(0, max(feat_h - 1, 0))
    return out


def pixel_to_norm(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)
    x = points[..., 0] / denom_x * 2.0 - 1.0
    y = points[..., 1] / denom_y * 2.0 - 1.0
    return torch.stack([x, y], dim=-1)


def safe_denominator(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sign = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    return torch.where(x.abs() < eps, sign * eps, x)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: GridToBEVLayer,
    loader: DataLoader,
    criterion: GridBEVLayerLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    num_steps = 0

    if len(loader) == 0:
        raise ValueError("train loader is empty. Implement GridBEVDataset first.")

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        rotation_degree_range = get_rotation_aug_degree_range(epoch)
        if rotation_degree_range is not None:
            images, grid_points, coordinate_valid = rotation_augmentation(
                batch["image"],
                coordinates=batch["grid_points"],
                probability=1.0,
                degree_range=rotation_degree_range,
                return_coordinate_mask=True,
            )
            batch["image"] = images
            batch["grid_points"] = grid_points
            batch["grid_visible"] = batch["grid_visible"] * coordinate_valid.to(
                device=batch["grid_visible"].device,
                dtype=batch["grid_visible"].dtype,
            )

        outputs = model(batch["image"])
        loss, logs = criterion(outputs, batch, model.bev_grid_points)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        num_steps += 1
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value.item())

        if step % log_interval == 0:
            msg = " ".join(f"{key}={float(value.item()):.4f}" for key, value in logs.items())
            print(f"[Epoch {epoch}][Step {step}/{len(loader)}] {msg}")

    return {key: value / max(num_steps, 1) for key, value in totals.items()}


def save_checkpoint(
    model: GridToBEVLayer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_dir: str,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"bev_layer_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return path


def load_checkpoint(
    model: GridToBEVLayer,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GridToBEVLayer.")
    parser.add_argument("--filepaths", default=os.path.join("data", "filepaths_bev.txt"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--num-grid-points", type=int, default=9)
    parser.add_argument("--heatmap-sigma", type=float, default=1.5)
    parser.add_argument("--loss-heatmap", type=float, default=1.0)
    parser.add_argument("--loss-coord", type=float, default=3.0)
    parser.add_argument("--loss-reprojection", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--fpn-out-channels", type=int, default=256)
    parser.add_argument("--backbone-width", type=float, default=0.25)
    parser.add_argument("--backbone-depth", type=float, default=0.33)
    parser.add_argument("--bev-height", type=int, default=60)
    parser.add_argument("--bev-width", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_grid_points < 4:
        raise ValueError("GridToBEVLayer training requires at least 4 grid points.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GridToBEVLayer(
        num_grid_points=args.num_grid_points,
        fpn_out_channels=args.fpn_out_channels,
        backbone_width=args.backbone_width,
        backbone_depth=args.backbone_depth,
        bev_size=(args.bev_height, args.bev_width),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = GridBEVLayerLoss(
        weights=GridBEVLossWeights(
            heatmap=args.loss_heatmap,
            coord=args.loss_coord,
            reprojection=args.loss_reprojection,
        ),
        heatmap_sigma=args.heatmap_sigma,
    )

    start_epoch = 1
    if args.weights is not None:
        start_epoch = load_checkpoint(model, optimizer, args.weights, device)
        print(f"loaded checkpoint: {args.weights} (resume from epoch {start_epoch})")

    train_loader = make_train_loader(
        filepaths_path=args.filepaths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_grid_points=args.num_grid_points,
        max_samples=args.max_samples,
    )

    for epoch in range(start_epoch, start_epoch + args.epochs):
        logs = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            grad_clip_norm=args.grad_clip_norm,
        )
        msg = " ".join(f"{key}={value:.4f}" for key, value in logs.items())
        print(f"[Epoch {epoch}] {msg}")

        if epoch % args.save_interval == 0:
            path = save_checkpoint(model, optimizer, epoch, args.save_dir)
            print(f"saved checkpoint: {path}")


if __name__ == "__main__":
    main()
