import argparse
import os
import pickle
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.model.viewpoint_bev_detection import SingleViewBEVDetector
from src.predict import decode_predictions


cv2 = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize predicted heatmaps and their homography-warped views."
    )
    parser.add_argument("--weights", required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Optional image paths. If multiple paths are passed, one is selected randomly.",
    )
    parser.add_argument(
        "--filepath",
        default="data/filepaths_img_and_ht.txt",
        help="Image/heatmap pair list used when --images is omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sample selection.",
    )
    parser.add_argument(
        "--homography-path",
        default="data/homography_matrix_for_train_dataset.pkl",
        help="Pickle file containing camera homography matrices when --homography-source=file.",
    )
    parser.add_argument(
        "--homography-key",
        default=None,
        help="Optional key inside the homography pickle. Defaults to the view index.",
    )
    parser.add_argument(
        "--view-index",
        type=int,
        default=None,
        help="Camera view index to use. Defaults to camN parsed from the selected image name.",
    )
    parser.add_argument(
        "--homography-source",
        choices=("model", "file"),
        default="model",
        help="Use the model-predicted homography or a matrix loaded from --homography-path.",
    )
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--num-grid-points", type=int, default=9)
    parser.add_argument("--bev-height", type=int, default=60)
    parser.add_argument("--bev-width", type=int, default=80)
    parser.add_argument("--heatmap-height", type=int, default=60)
    parser.add_argument("--heatmap-width", type=int, default=80)
    parser.add_argument("--decoder-channels", type=int, default=64)
    parser.add_argument("--center-head-channels", type=int, default=128)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument(
        "--top-classes",
        type=int,
        default=6,
        help="Number of detected/highest-score class heatmaps to include.",
    )
    parser.add_argument(
        "--out-dir",
        default="heatmap_debug",
        help="Directory where visualization images will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open cv2 windows after saving the visualization files.",
    )
    parser.add_argument(
        "--disable-grid-head",
        action="store_true",
        help="Disable the model grid head during inference.",
    )
    return parser.parse_args()


def require_cv2():
    global cv2
    if cv2 is None:
        try:
            import cv2 as cv2_module
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenCV is required for visualization. Install it with "
                "`pip install opencv-python` in the environment used to run this script."
            ) from exc
        cv2 = cv2_module
    return cv2


def load_model_weights(model: torch.nn.Module, path: str, device: torch.device) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"model not found: {path}")

    loaded = torch.load(path, map_location=device)
    if isinstance(loaded, dict):
        state_dict = loaded.get("model_state_dict", loaded)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"model missing keys: {missing}")
        if unexpected:
            print(f"model unexpected keys: {unexpected}")
    else:
        raise TypeError(f"unsupported checkpoint format: {type(loaded)}")
    print(f"model loaded: {path}")


def format_images(img_list: list[torch.Tensor]) -> torch.Tensor:
    if len(img_list) == 0:
        raise ValueError("img_list is empty")

    formatted = []
    for image in img_list:
        if image.ndim != 3:
            raise ValueError(f"each image tensor must be 3D, got {tuple(image.shape)}")

        image = image.detach().float()
        if image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
            if image.shape[-1] == 3:
                image = image[..., [2, 1, 0]]
            image = image.permute(2, 0, 1)
        if image.max().item() > 1.0:
            image = image / 255.0
        formatted.append(image)
    return torch.stack(formatted)


def read_image(path: str) -> np.ndarray:
    require_cv2()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return image


def load_image_paths_from_file(filepath: str) -> list[str]:
    image_paths = []
    with open(filepath, "r") as file:
        for line_num, line in enumerate(file, start=1):
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if len(parts) < 1:
                raise ValueError(f"{filepath}:{line_num} does not contain an image path")
            image_paths.append(parts[0])

    if len(image_paths) == 0:
        raise ValueError(f"no image paths found in {filepath}")

    existing_paths = [path for path in image_paths if os.path.exists(path)]
    if len(existing_paths) > 0:
        return existing_paths

    print(f"warning: no paths in {filepath} exist on this machine; selecting from all listed paths")
    return image_paths


def infer_view_index(image_path: str) -> int:
    match = re.search(r"cam([1-5])", image_path, flags=re.IGNORECASE)
    if match is None:
        return 0
    return int(match.group(1)) - 1


def choose_random_image(args) -> tuple[str, int]:
    rng = random.Random(args.seed)
    candidates = args.images if args.images is not None else load_image_paths_from_file(args.filepath)
    if len(candidates) == 0:
        raise ValueError("no image candidates were provided")

    selected_path = rng.choice(candidates)
    selected_view_index = infer_view_index(selected_path) if args.view_index is None else args.view_index
    if selected_view_index < 0 or selected_view_index > 4:
        raise ValueError(f"view index must be between 0 and 4, got {selected_view_index}")
    return selected_path, selected_view_index


def load_homography(path: str, key: str | int | None, view_index: int) -> np.ndarray:
    with open(path, "rb") as file:
        data = pickle.load(file)

    if isinstance(data, dict):
        lookup_key = view_index if key is None else _coerce_key(key, data)
        if lookup_key not in data:
            raise KeyError(
                f"homography key {lookup_key!r} not found. Available keys: {list(data.keys())[:20]}"
            )
        matrix = data[lookup_key]
    elif isinstance(data, (list, tuple)):
        lookup_index = view_index if key is None else int(key)
        matrix = data[lookup_index]
    else:
        matrix = data

    if isinstance(matrix, dict):
        for candidate in ("homography", "H", "matrix"):
            if candidate in matrix:
                matrix = matrix[candidate]
                break

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"homography must have shape (3, 3), got {matrix.shape}")
    return matrix


def _coerce_key(key: str, data: dict):
    if key in data:
        return key
    try:
        int_key = int(key)
    except ValueError:
        return key
    return int_key if int_key in data else key


def scale_homography_for_heatmap(
    homography: np.ndarray,
    src_image_shape: tuple[int, int],
    heatmap_shape: tuple[int, int],
) -> np.ndarray:
    image_h, image_w = src_image_shape
    heatmap_h, heatmap_w = heatmap_shape

    heatmap_to_image = np.array(
        [
            [image_w / heatmap_w, 0.0, 0.0],
            [0.0, image_h / heatmap_h, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    image_to_heatmap = np.array(
        [
            [heatmap_w / image_w, 0.0, 0.0],
            [0.0, heatmap_h / image_h, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return image_to_heatmap @ homography @ heatmap_to_image


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)
    return ((array - min_value) / (max_value - min_value) * 255.0).clip(0, 255).astype(np.uint8)


def colorize_heatmap(heatmap: np.ndarray, output_size: tuple[int, int] | None = None) -> np.ndarray:
    heatmap_u8 = normalize_to_uint8(heatmap)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    if output_size is not None:
        colored = cv2.resize(colored, output_size, interpolation=cv2.INTER_NEAREST)
    return colored


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_h, image_w = image_bgr.shape[:2]
    colored = colorize_heatmap(heatmap, output_size=(image_w, image_h))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0.0)


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def make_grid(images: list[np.ndarray], columns: int = 2, pad: int = 8) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("images is empty")

    cell_h = max(image.shape[0] for image in images)
    cell_w = max(image.shape[1] for image in images)
    rows = int(np.ceil(len(images) / columns))
    canvas = np.full(
        (rows * cell_h + (rows - 1) * pad, columns * cell_w + (columns - 1) * pad, 3),
        255,
        dtype=np.uint8,
    )

    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        y = row * (cell_h + pad)
        x = col * (cell_w + pad)
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
    return canvas


def select_class_indices(
    heatmap: torch.Tensor,
    decoded: dict[str, torch.Tensor],
    top_classes: int,
) -> list[int]:
    selected = []
    for cls in decoded["classes"].tolist():
        cls = int(cls)
        if cls not in selected:
            selected.append(cls)
        if len(selected) >= top_classes:
            return selected

    class_scores = heatmap.amax(dim=(1, 2))
    for cls in torch.argsort(class_scores, descending=True).tolist():
        cls = int(cls)
        if cls not in selected:
            selected.append(cls)
        if len(selected) >= top_classes:
            break
    return selected


@torch.no_grad()
def main():
    args = parse_args()
    require_cv2()
    selected_image_path, selected_view_index = choose_random_image(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleViewBEVDetector(
        num_classes=args.num_classes,
        bev_size=(args.bev_height, args.bev_width),
        heatmap_size=(args.heatmap_height, args.heatmap_width),
        num_grid_points=args.num_grid_points,
        decoder_channels=args.decoder_channels,
        center_head_channels=args.center_head_channels,
    ).to(device)
    load_model_weights(model, args.weights, device)
    model.eval()

    images_bgr = [read_image(selected_image_path)]
    model_input = format_images([torch.from_numpy(image) for image in images_bgr]).to(device)
    view_indices = torch.tensor([selected_view_index], device=device)
    viewpoint = F.one_hot(view_indices, num_classes=5).float()

    outputs = model(
        model_input,
        viewpoint=viewpoint,
        return_aux=True,
        decode=False,
        use_grid_head=not args.disable_grid_head,
        warp_center=args.homography_source == "model",
    )
    heatmaps = outputs["heatmap"].detach().cpu()
    offsets = outputs["offset"].detach().cpu()

    view_heatmap = heatmaps[0]
    view_offset = offsets[0:1]
    decoded = decode_predictions(
        view_heatmap.unsqueeze(0),
        offset=view_offset,
        topk=args.topk,
        score_threshold=args.score_threshold,
    )[0]

    heatmap_h, heatmap_w = view_heatmap.shape[-2:]
    merged_heatmap = view_heatmap.amax(dim=0).numpy()

    warped_view_heatmap = None
    if args.homography_source == "model":
        if args.disable_grid_head:
            raise ValueError("--homography-source=model cannot be used with --disable-grid-head")
        if "bev_center_logits" not in outputs:
            raise RuntimeError("model did not return bev_center_logits")
        warped_view_heatmap = outputs["bev_center_logits"][0].sigmoid().detach().cpu()
        warped_merged_heatmap = warped_view_heatmap.amax(dim=0).numpy()
    else:
        homography = load_homography(
            args.homography_path,
            args.homography_key,
            selected_view_index,
        )
        scaled_homography = scale_homography_for_heatmap(
            homography,
            src_image_shape=images_bgr[0].shape[:2],
            heatmap_shape=(heatmap_h, heatmap_w),
        )
        warped_merged_heatmap = cv2.warpPerspective(
            merged_heatmap,
            scaled_homography,
            (heatmap_w, heatmap_h),
            flags=cv2.INTER_LINEAR,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = images_bgr[0]
    overview = make_grid(
        [
            add_label(image_bgr, f"random input view {selected_view_index}: {Path(selected_image_path).name}"),
            add_label(overlay_heatmap(image_bgr, merged_heatmap), "pred heatmap overlay"),
            add_label(colorize_heatmap(merged_heatmap, output_size=(image_bgr.shape[1], image_bgr.shape[0])), "pred heatmap"),
            add_label(colorize_heatmap(warped_merged_heatmap, output_size=(image_bgr.shape[1], image_bgr.shape[0])), "homography warped heatmap"),
        ],
        columns=2,
    )
    overview_path = out_dir / "overview.png"
    cv2.imwrite(str(overview_path), overview)

    class_panels = []
    selected_classes = select_class_indices(view_heatmap, decoded, args.top_classes)
    for cls in selected_classes:
        class_heatmap = view_heatmap[cls].numpy()
        if warped_view_heatmap is None:
            warped_class_heatmap = cv2.warpPerspective(
                class_heatmap,
                scaled_homography,
                (heatmap_w, heatmap_h),
                flags=cv2.INTER_LINEAR,
            )
        else:
            warped_class_heatmap = warped_view_heatmap[cls].numpy()
        label = f"class_{cls}"
        class_panels.extend(
            [
                add_label(
                    colorize_heatmap(class_heatmap, output_size=(image_bgr.shape[1], image_bgr.shape[0])),
                    f"{cls}: {label}",
                ),
                add_label(
                    colorize_heatmap(warped_class_heatmap, output_size=(image_bgr.shape[1], image_bgr.shape[0])),
                    f"warped {cls}: {label}",
                ),
            ]
        )

    class_grid_path = None
    if class_panels:
        class_grid = make_grid(class_panels, columns=2)
        class_grid_path = out_dir / "detected_classes.png"
        cv2.imwrite(str(class_grid_path), class_grid)

    print(f"saved overview: {overview_path}")
    if class_grid_path is not None:
        print(f"saved detected class heatmaps: {class_grid_path}")
    print(f"selected image: {selected_image_path}")
    print(f"selected view index: {selected_view_index}")
    print(f"selected classes: {selected_classes}")
    print(f"detections above threshold: {decoded['scores'].numel()}")
    if decoded["scores"].numel() > 0:
        for score, cls, center in zip(decoded["scores"], decoded["classes"], decoded["centers"]):
            label = f"class_{int(cls)}"
            print(
                f"  score={float(score):.4f}, class={int(cls)} ({label}), "
                f"center=({float(center[0]):.3f}, {float(center[1]):.3f})"
            )

    if args.show:
        cv2.imshow("overview", overview)
        if class_panels:
            cv2.imshow("detected_classes", class_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
