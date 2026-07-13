import torch


def apply_homography_to_points(
    points: torch.Tensor,
    homography: torch.Tensor,
    invert: bool = False,
) -> torch.Tensor:
    """
    Apply a homography to 2D points.

    points:
        [N, 2] or [B, N, 2]
    homography:
        [3, 3] or [B, 3, 3]

    Coordinates are used as-is. Use image_centers_to_bev() when converting the
    detector's normalized [0, 1] image centers with GridToBEVLayer homographies.
    """
    original_dim = points.dim()
    if original_dim == 2:
        points = points.unsqueeze(0)
    elif original_dim != 3:
        raise ValueError(f"points must have shape [N, 2] or [B, N, 2], got {tuple(points.shape)}")

    if points.shape[-1] != 2:
        raise ValueError(f"points last dimension must be 2, got {tuple(points.shape)}")

    homography = torch.as_tensor(homography, device=points.device, dtype=points.dtype)
    if homography.dim() == 2:
        if homography.shape != (3, 3):
            raise ValueError(f"homography must have shape [3, 3], got {tuple(homography.shape)}")
        homography = homography.unsqueeze(0).expand(points.shape[0], -1, -1)
    elif homography.dim() == 3:
        if homography.shape[1:] != (3, 3):
            raise ValueError(f"homography must have shape [B, 3, 3], got {tuple(homography.shape)}")
        if homography.shape[0] == 1 and points.shape[0] != 1:
            homography = homography.expand(points.shape[0], -1, -1)
        elif homography.shape[0] != points.shape[0]:
            raise ValueError(
                f"homography batch size must be 1 or {points.shape[0]}, got {homography.shape[0]}"
            )
    else:
        raise ValueError(f"homography must have shape [3, 3] or [B, 3, 3], got {tuple(homography.shape)}")

    if invert:
        homography = torch.linalg.inv(homography)

    ones = torch.ones(*points.shape[:2], 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=-1)
    transformed = points_h @ homography.transpose(1, 2)
    transformed_xy = transformed[..., :2] / _safe_denominator(transformed[..., 2:])

    if original_dim == 2:
        return transformed_xy.squeeze(0)
    return transformed_xy


def image_centers_to_bev(
    centers: torch.Tensor,
    homography: torch.Tensor,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Convert detector center coordinates to BEV coordinates.

    centers are normalized image/heatmap coordinates in [0, 1], matching
    decode_predictions(). homography is the GridToBEVLayer matrix that maps BEV
    normalized coordinates [-1, 1] to image-feature normalized coordinates
    [-1, 1]. The inverse is applied here.

    If output_size=(height, width) is supplied, returned centers are BEV pixel
    coordinates ordered as (x, y). Otherwise they remain normalized [0, 1].
    """
    centers = torch.as_tensor(centers, dtype=torch.float32)
    if centers.numel() == 0:
        return centers.clone()

    image_norm = centers * 2.0 - 1.0
    bev_norm = apply_homography_to_points(image_norm, homography, invert=True)
    bev_centers = (bev_norm + 1.0) * 0.5

    if output_size is None:
        return bev_centers

    height, width = output_size
    scale = bev_centers.new_tensor([width, height])
    return bev_centers * scale


def transform_detections_to_bev(
    model_outputs: dict | None = None,
    detections: list[dict[str, torch.Tensor]] | None = None,
    homography: torch.Tensor | None = None,
    centers: torch.Tensor | None = None,
    classes: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
    output_size: tuple[int, int] | None = None,
) -> list[dict[str, torch.Tensor]] | dict[str, torch.Tensor]:
    """
    Transform model detections, or explicitly supplied centers, into BEV space.

    Usage:
        transform_detections_to_bev(model_outputs=outputs)
        transform_detections_to_bev(homography=H, centers=centers, classes=classes)
    """
    if model_outputs is not None:
        if detections is None:
            detections = model_outputs.get("detections")
        if homography is None:
            homography = model_outputs.get("homography")

    if homography is None:
        raise ValueError("homography is required.")

    if centers is not None:
        bev_centers = image_centers_to_bev(
            centers,
            homography,
            output_size=output_size,
        )
        result = {"centers": bev_centers}
        if classes is not None:
            result["classes"] = classes
        if scores is not None:
            result["scores"] = scores
        return result

    if detections is None:
        raise ValueError("detections or centers are required.")

    homography_batch = torch.as_tensor(homography)
    transformed: list[dict[str, torch.Tensor]] = []
    for batch_idx, detection in enumerate(detections):
        if homography_batch.dim() == 3:
            h = homography_batch[min(batch_idx, homography_batch.shape[0] - 1)]
        else:
            h = homography_batch

        bev_centers = image_centers_to_bev(
            detection["centers"],
            h,
            output_size=output_size,
        )
        transformed_detection = {
            key: value
            for key, value in detection.items()
            if key != "centers"
        }
        transformed_detection["centers"] = bev_centers
        transformed.append(transformed_detection)

    return transformed


def _safe_denominator(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sign = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    return torch.where(x.abs() < eps, sign * eps, x)
