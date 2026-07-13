import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset import ImageHeatmapDataset
from src.model.viewpoint_bev_detection import SingleViewBEVDetector
from src.predict import decode_predictions


class ImageHeatmapEvalDataset(Dataset):
    def __init__(
        self,
        data_list: list[dict[str, str]],
        num_classes: int,
        output_size: tuple[int, int],
    ):
        if len(data_list) == 0:
            raise ValueError("data_list is empty")
        self.dataset = ImageHeatmapDataset(data_list, num_classes, output_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        classes, centers = targets_to_classes_and_centers(
            sample["center_mask"],
            sample["center_offset"],
        )

        return {
            "image": sample["image"],
            "classes": classes,
            "centers": centers,
        }


def collate_image_heatmap_eval_fn(batch):
    image_shapes = [tuple(b["image"].shape) for b in batch]
    if len(set(image_shapes)) != 1:
        raise ValueError(f"all image tensors must have the same shape, got {image_shapes}")

    return {
        "images": torch.stack([b["image"] for b in batch]),
        "classes": [b["classes"] for b in batch],
        "centers": [b["centers"] for b in batch],
    }


def make_data_list(
    filepath: str = "data/filepaths_img_and_ht.txt",
    max_samples: int | None = None,
):
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be at least 1, got {max_samples}")

    data_list = []
    skipped_missing = 0

    with open(filepath, "r") as file:
        for line_num, line in enumerate(file, start=1):
            if max_samples is not None and len(data_list) >= max_samples:
                break

            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if len(parts) != 2:
                raise ValueError(
                    f"{filepath}:{line_num} must contain image path and heatmap path, "
                    f"got {line.strip()}"
                )

            image_path, heatmap_path = parts
            if not os.path.exists(image_path) or not os.path.exists(heatmap_path):
                skipped_missing += 1
                continue
            data_list.append(
                {
                    "image_path": image_path,
                    "heatmap_path": heatmap_path,
                }
            )

    print(f"loaded samples: {len(data_list)} from {filepath}")
    if max_samples is not None:
        print(f"sample limit: {max_samples}")
    if skipped_missing > 0:
        print(f"skipped samples with missing image/heatmap files: {skipped_missing}")
    return data_list


def targets_to_classes_and_centers(
    center_mask: torch.Tensor,
    center_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positives = center_mask.nonzero(as_tuple=False)
    if positives.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, 2)

    _, height, width = center_mask.shape
    classes = positives[:, 0].long()
    ys = positives[:, 1]
    xs = positives[:, 2]
    centers_x = (xs.float() + center_offset[0, ys, xs]) / width
    centers_y = (ys.float() + center_offset[1, ys, xs]) / height
    centers = torch.stack([centers_x, centers_y], dim=1)
    return classes, centers


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"model missing keys: {missing}")
    if unexpected:
        print(f"model unexpected keys: {unexpected}")
    print(f"loaded checkpoint: {checkpoint_path}")


def match_predictions(
    predictions: dict[str, torch.Tensor],
    gt_classes: torch.Tensor,
    gt_centers: torch.Tensor,
    center_threshold: float,
) -> tuple[int, int, int]:
    pred_scores = predictions["scores"]
    pred_classes = predictions["classes"]
    pred_centers = predictions["centers"]

    gt_classes = gt_classes.cpu()
    gt_centers = gt_centers.cpu()
    matched_gt: set[int] = set()
    true_positive = 0
    false_positive = 0

    if pred_scores.numel() > 0:
        order = torch.argsort(pred_scores, descending=True)
        pred_classes = pred_classes[order]
        pred_centers = pred_centers[order]

    for pred_cls, pred_center in zip(pred_classes, pred_centers):
        candidate_indices = [
            idx
            for idx, gt_cls in enumerate(gt_classes)
            if idx not in matched_gt and gt_cls.item() == pred_cls.item()
        ]

        if len(candidate_indices) == 0:
            false_positive += 1
            continue

        candidate_centers = gt_centers[candidate_indices]
        distances = torch.linalg.vector_norm(candidate_centers - pred_center, dim=1)
        best_distance, best_pos = distances.min(dim=0)

        if best_distance.item() <= center_threshold:
            true_positive += 1
            matched_gt.add(candidate_indices[best_pos.item()])
        else:
            false_positive += 1

    false_negative = len(gt_classes) - len(matched_gt)
    return true_positive, false_positive, false_negative


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: int,
    score_threshold: float,
    center_threshold: float,
) -> tuple[float, float, float]:
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_images = 0
    elapsed = 0.0

    for batch in loader:
        images = batch["images"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(
            images,
            return_aux=True,
            decode=False,
            use_grid_head=False,
        )
        pred_heatmap = outputs["heatmap"]
        pred_offset = outputs["offset"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start

        decoded = decode_predictions(
            pred_heatmap,
            offset=pred_offset,
            topk=topk,
            score_threshold=score_threshold,
        )

        for pred, gt_classes, gt_centers in zip(
            decoded,
            batch["classes"],
            batch["centers"],
        ):
            tp, fp, fn = match_predictions(
                pred,
                gt_classes,
                gt_centers,
                center_threshold=center_threshold,
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn

        total_images += images.size(0)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    fps = total_images / elapsed if elapsed > 0 else 0.0
    return precision, recall, fps


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SingleViewBEVDetector.")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--filepath",
        "--filename-path",
        dest="filepath",
        default="data/filepaths_img_and_ht.txt",
        help="Path to image/heatmap pairs. Each line must be: image_path heatmap_path.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate at most this many existing image/heatmap pairs.",
    )
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--num-grid-points", type=int, default=9)
    parser.add_argument("--bev-height", type=int, default=60)
    parser.add_argument("--bev-width", type=int, default=80)
    parser.add_argument("--decoder-channels", type=int, default=64)
    parser.add_argument("--center-head-channels", type=int, default=128)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument(
        "--center-threshold",
        type=float,
        default=0.05,
        help="Normalized center-distance threshold used instead of IoU.",
    )
    return parser.parse_args()


def test(
    model: torch.nn.Module | None = None,
    loader: DataLoader | None = None,
    device: torch.device | None = None,
    weights: str | None = None,
    filepath: str = "data/filepaths_img_and_ht.txt",
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: int | None = None,
    num_classes: int = 60,
    num_grid_points: int = 9,
    bev_height: int = 60,
    bev_width: int = 80,
    decoder_channels: int = 64,
    center_head_channels: int = 128,
    topk: int = 100,
    score_threshold: float = 0.3,
    center_threshold: float = 0.05,
    print_result: bool = True,
) -> tuple[float, float, float]:
    if model is None and loader is None and weights is None:
        args = parse_args()
        return test(
            weights=args.weights,
            filepath=args.filepath,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            num_classes=args.num_classes,
            num_grid_points=args.num_grid_points,
            bev_height=args.bev_height,
            bev_width=args.bev_width,
            decoder_channels=args.decoder_channels,
            center_head_channels=args.center_head_channels,
            topk=args.topk,
            score_threshold=args.score_threshold,
            center_threshold=args.center_threshold,
            print_result=print_result,
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if loader is None:
        data_list = make_data_list(filepath, max_samples=max_samples)
        dataset = ImageHeatmapEvalDataset(
            data_list,
            num_classes=num_classes,
            output_size=(60, 80),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_image_heatmap_eval_fn,
        )

    if model is None:
        model = SingleViewBEVDetector(
            num_classes=num_classes,
            img_channels=3,
            fpn_out_channels=256,
            backbone_width=0.25,
            backbone_depth=0.33,
            bev_size=(bev_height, bev_width),
            heatmap_size=(60, 80),
            num_grid_points=num_grid_points,
            decoder_channels=decoder_channels,
            center_head_channels=center_head_channels,
        ).to(device)

    if weights is not None:
        load_model_weights(model, weights, device)

    precision, recall, fps = evaluate(
        model,
        loader,
        device,
        topk=topk,
        score_threshold=score_threshold,
        center_threshold=center_threshold,
    )

    if print_result:
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"FPS: {fps:.2f}")

    return precision, recall, fps


if __name__ == "__main__":
    test()
