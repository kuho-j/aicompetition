import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.model.grid_bev_layer import GridToBEVLayer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GridToBEVLayer outputs.")
    parser.add_argument("--weights", default=None, help="Optional BEV layer checkpoint path.")
    parser.add_argument("--filepaths", default=Path("data") / "filepaths_bev.txt")
    parser.add_argument("--image", default=None, help="Optional image path to visualize.")
    parser.add_argument("--image-dir", default=None, help="Optional directory to sample an image from.")
    parser.add_argument("--output", default=Path("outputs") / "bev_layer_visualization.png")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-grid-points", type=int, default=9)
    parser.add_argument("--fpn-out-channels", type=int, default=256)
    parser.add_argument("--backbone-width", type=float, default=0.25)
    parser.add_argument("--backbone-depth", type=float, default=0.33)
    parser.add_argument("--bev-height", type=int, default=192)
    parser.add_argument("--bev-width", type=int, default=256)
    parser.add_argument("--random-height", type=int, default=480)
    parser.add_argument("--random-width", type=int, default=640)
    parser.add_argument("--show", action="store_true", help="Open the saved image with the OS viewer.")
    return parser.parse_args()


def load_checkpoint(model: GridToBEVLayer, weights: str, device: torch.device) -> None:
    checkpoint = torch.load(weights, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def read_image_as_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def make_random_image(height: int, width: int) -> torch.Tensor:
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            (xx / max(width - 1, 1)) * 0.65 + 0.18,
            (yy / max(height - 1, 1)) * 0.55 + 0.22,
            ((xx + yy) / max(width + height - 2, 1)) * 0.5 + 0.25,
        ],
        axis=0,
    )
    noise = np.random.normal(0.0, 0.055, size=base.shape)
    image = np.clip(base + noise, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(image)


def random_image_from_dir(image_dir: Path) -> Path | None:
    if not image_dir.exists():
        return None
    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return random.choice(images) if images else None


def random_sample_from_filepaths(filepaths: Path, num_grid_points: int) -> tuple[torch.Tensor, str]:
    if not filepaths.exists():
        raise FileNotFoundError(f"{filepaths} does not exist")

    from data.load_data_for_bev_train import load_data_bev

    lines = [line.strip() for line in filepaths.read_text().splitlines() if line.strip()]
    random.shuffle(lines)
    last_error: Exception | None = None

    for line in lines:
        try:
            sample = load_data_bev(line)
            image = sample["image"]
            image_path = line.split()[-1]
            return image, image_path
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"No readable image in {filepaths}. Last error: {last_error}")


def choose_input_image(args: argparse.Namespace, num_grid_points: int) -> tuple[torch.Tensor, str]:
    if args.image is not None:
        path = Path(args.image)
        return read_image_as_tensor(path), str(path)

    if args.image_dir is not None:
        path = random_image_from_dir(Path(args.image_dir))
        if path is not None:
            return read_image_as_tensor(path), str(path)

    try:
        return random_sample_from_filepaths(Path(args.filepaths), num_grid_points)
    except Exception as exc:
        print(f"Using generated random image because dataset image loading failed: {exc}")
        image = make_random_image(args.random_height, args.random_width)
        return image, "generated random RGB image"


def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().float()
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    array = image.numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def normalize_map(values: torch.Tensor) -> np.ndarray:
    array = values.detach().cpu().float().numpy()
    array = array - float(array.min())
    max_value = float(array.max())
    if max_value > 1e-8:
        array = array / max_value
    return array


def colorize_map(values: torch.Tensor) -> Image.Image:
    heat = normalize_map(values)
    red = np.clip(2.2 * heat, 0, 1)
    green = np.clip(1.8 * (1.0 - np.abs(heat - 0.55) * 1.8), 0, 1)
    blue = np.clip(1.25 * (1.0 - heat), 0, 1)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def resize_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.BILINEAR)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(24, 24, 24),
    )
    draw.text((x, y), text, fill=(245, 245, 245), font=font)


def draw_points(image: Image.Image, points: torch.Tensor, confidence: torch.Tensor) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    points = points.detach().cpu()
    confidence = confidence.detach().cpu()

    for idx, ((x, y), conf) in enumerate(zip(points.tolist(), confidence.tolist()), start=1):
        radius = 5
        color = (255, 62, 62) if conf >= 0.5 else (255, 180, 54)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        draw.text((x + radius + 2, y - radius - 2), str(idx), fill=color)

    return output


def make_visualization(
    image: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    image_source: str,
    output_path: Path,
) -> None:
    input_panel = tensor_to_pil_image(image)
    pred_points = outputs["grid_points"][0]
    pred_conf = outputs["grid_confidence"][0]
    input_panel = draw_points(input_panel, pred_points, pred_conf)

    bev_feature = outputs["bev_feature"][0].abs().mean(dim=0)
    bev_valid_mask = outputs["bev_valid_mask"][0, 0]
    grid_heatmap = outputs["grid_logits"][0].sigmoid().amax(dim=0)

    panel_size = (480, 360)
    panels = [
        (resize_panel(input_panel, panel_size), "input + predicted grid points"),
        (resize_panel(colorize_map(bev_feature), panel_size), "BEV feature energy"),
        (resize_panel(colorize_map(bev_valid_mask), panel_size), "BEV valid mask"),
        (resize_panel(colorize_map(grid_heatmap), panel_size), "max grid-point heatmap"),
    ]

    margin = 18
    label_height = 24
    header_height = 78
    canvas_w = panel_size[0] * 2 + margin * 3
    canvas_h = header_height + (panel_size[1] + label_height) * 2 + margin * 3
    canvas = Image.new("RGB", (canvas_w, canvas_h), (247, 247, 245))
    draw = ImageDraw.Draw(canvas)

    mean_conf = float(pred_conf.mean().item())
    max_conf = float(pred_conf.max().item())
    draw.text((margin, 16), "GridToBEVLayer output visualization", fill=(28, 28, 28))
    draw.text(
        (margin, 40),
        f"source: {image_source} | bev_feature={tuple(outputs['bev_feature'].shape)} "
        f"| mean_conf={mean_conf:.3f} max_conf={max_conf:.3f}",
        fill=(68, 68, 68),
    )

    positions = [
        (margin, header_height + margin),
        (margin * 2 + panel_size[0], header_height + margin),
        (margin, header_height + margin * 2 + panel_size[1] + label_height),
        (
            margin * 2 + panel_size[0],
            header_height + margin * 2 + panel_size[1] + label_height,
        ),
    ]

    for (panel, label), (x, y) in zip(panels, positions):
        canvas.paste(panel, (x, y + label_height))
        draw_label(draw, (x, y), label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GridToBEVLayer(
        num_grid_points=args.num_grid_points,
        fpn_out_channels=args.fpn_out_channels,
        backbone_width=args.backbone_width,
        backbone_depth=args.backbone_depth,
        bev_size=(args.bev_height, args.bev_width),
    ).to(device)

    if args.weights is not None:
        load_checkpoint(model, args.weights, device)
        print(f"Loaded checkpoint: {args.weights}")

    image, image_source = choose_input_image(args, args.num_grid_points)
    model.eval()
    with torch.no_grad():
        outputs = model(image.unsqueeze(0).to(device))

    output_path = Path(args.output)
    make_visualization(image, outputs, image_source, output_path)
    print(f"Saved BEV visualization: {output_path}")

    if args.show:
        Image.open(output_path).show()


if __name__ == "__main__":
    main()
