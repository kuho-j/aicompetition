import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.model.grid_bev_layer import GridToBEVLayer


class GridBEVDataset(Dataset):
    """
    Placeholder dataset for GridToBEVLayer training.

    Expected item format:
        {
            "image": Tensor [C, H, W],
            "grid_points": Tensor [N, 2],  # input-image pixel coords, (x, y)
            "grid_visible": Tensor [N],    # optional, 1 for valid labeled points
            "homography": Tensor [3, 3],   # optional diagnostic target
        }

    Fill this class later with the real annotation loader.
    """

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Implement GridBEVDataset before training.")


def collate_grid_bev_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(batch) == 0:
        raise ValueError("batch is empty")

    images = torch.stack([item["image"] for item in batch])
    grid_points = torch.stack([item["grid_points"] for item in batch])

    if "grid_visible" in batch[0]:
        grid_visible = torch.stack([item["grid_visible"] for item in batch]).float()
    else:
        grid_visible = torch.ones(grid_points.shape[:2], dtype=torch.float32)

    output = {
        "image": images,
        "grid_points": grid_points.float(),
        "grid_visible": grid_visible,
    }

    if "homography" in batch[0]:
        output["homography"] = torch.stack([item["homography"] for item in batch]).float()

    return output


def make_train_loader(batch_size: int, num_workers: int) -> DataLoader:
    """
    Empty on purpose.

    Replace GridBEVDataset() with your actual dataset implementation once grid
    annotations are ready.
    """

    dataset = GridBEVDataset()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=len(dataset) > 0,
        num_workers=num_workers,
        collate_fn=collate_grid_bev_batch,
    )


@dataclass
class GridBEVLossWeights:
    heatmap: float = 1.0
    coord: float = 3.0
    reprojection: float = 0.5


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
        reprojection_loss = homography_reprojection_loss(
            homography,
            bev_grid_points,
            gt_points_feat,
            visible,
            feature_size=(feat_h, feat_w),
        )

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
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    num_steps = 0

    if len(loader) == 0:
        raise ValueError("train loader is empty. Implement GridBEVDataset first.")

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["image"])
        loss, logs = criterion(outputs, batch, model.bev_grid_points)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
    parser.add_argument("--loss-reprojection", type=float, default=0.5)
    parser.add_argument("--fpn-out-channels", type=int, default=256)
    parser.add_argument("--backbone-width", type=float, default=0.25)
    parser.add_argument("--backbone-depth", type=float, default=0.33)
    parser.add_argument("--bev-height", type=int, default=128)
    parser.add_argument("--bev-width", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
        )
        msg = " ".join(f"{key}={value:.4f}" for key, value in logs.items())
        print(f"[Epoch {epoch}] {msg}")

        if epoch % args.save_interval == 0:
            path = save_checkpoint(model, optimizer, epoch, args.save_dir)
            print(f"saved checkpoint: {path}")


if __name__ == "__main__":
    main()
