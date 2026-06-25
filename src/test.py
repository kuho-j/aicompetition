import argparse
import time

import torch
from torch.utils.data import DataLoader, Dataset

from data.make_filename import make_filename
from src.dataset import format_data
from src.model.multiview_detection import MultiViewDetector
from src.predict import decode_predictions


class MultiViewEvalDataset(Dataset):
    def __init__(self, data_list: list[dict[str, torch.Tensor]]):
        if len(data_list) == 0:
            raise ValueError("data_list is empty")
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        return {
            "images": sample["images"],
            "classes": sample["classes"],
            "centers": sample["centers"],
        }


def collate_eval_fn(batch):
    image_shapes = [tuple(b["images"].shape) for b in batch]
    if len(set(image_shapes)) != 1:
        raise ValueError(f"all image tensors must have the same shape, got {image_shapes}")

    return {
        "images": torch.stack([b["images"] for b in batch]),
        "classes": [b["classes"] for b in batch],
        "centers": [b["centers"] for b in batch],
    }


def make_data_list(filename_path: str, expected_num_views: int = 5):
    data_list = []
    skipped = 0

    with open(filename_path, "r") as file:
        for line in file:
            filename_info = make_filename(line)
            sample = format_data(filename_info, expected_num_views=expected_num_views)
            if sample is None:
                skipped += 1
                continue
            data_list.append(sample)

    print(f"loaded samples: {len(data_list)}, skipped samples: {skipped}")
    return data_list


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
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
        pred_heatmap = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start

        decoded = decode_predictions(
            pred_heatmap,
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
    parser = argparse.ArgumentParser(description="Evaluate MultiViewDetector.")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint.")
    parser.add_argument("--filename-path", default="data/filename.txt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=60)
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
    filename_path: str = "data/filename.txt",
    batch_size: int = 32,
    num_workers: int = 0,
    num_views: int = 5,
    num_classes: int = 60,
    topk: int = 100,
    score_threshold: float = 0.3,
    center_threshold: float = 0.05,
    print_result: bool = True,
) -> tuple[float, float, float]:
    if model is None and loader is None and weights is None:
        args = parse_args()
        return test(
            weights=args.weights,
            filename_path=args.filename_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_views=args.num_views,
            num_classes=args.num_classes,
            topk=args.topk,
            score_threshold=args.score_threshold,
            center_threshold=args.center_threshold,
            print_result=print_result,
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if loader is None:
        data_list = make_data_list(
            filename_path,
            expected_num_views=num_views,
        )
        dataset = MultiViewEvalDataset(data_list)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_eval_fn,
        )

    if model is None:
        model = MultiViewDetector(
            num_views=num_views,
            num_classes=num_classes,
            img_channels=3,
            fpn_out_channels=256,
            backbone_width=0.25,
            backbone_depth=0.33,
            attn_heads=4,
            spatial_ds=2,
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
