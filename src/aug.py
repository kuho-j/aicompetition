import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


HOMOGRAPHY_AUG_PATH = "data/homography_matrix_train_to_video.pkl"
HOMOGRAPHY_AUG_PROB = 0.35
HOMOGRAPHY_AUG_MAX_RETRIES = 8
HOMOGRAPHY_AUG_ALPHA_RANGE = (0.0, 1.0)
HOMOGRAPHY_AUG_STRICT_ALPHA_RANGE = (0.0, 0.75)
HOMOGRAPHY_AUG_MIN_VALID_RATIO = 0.65
HOMOGRAPHY_AUG_STRICT_MIN_VALID_RATIO = 0.78
HOMOGRAPHY_AUG_STRICT_VIEWS = {1, 3}
ROTATION_AUG_PROB = 0.35
ROTATION_AUG_DEGREE_RANGE = (-5.0, 5.0)


def load_homography_augmentation_matrices(
    path: str | Path = HOMOGRAPHY_AUG_PATH,
    num_views: int = 5,
) -> torch.Tensor:
    with open(path, "rb") as f:
        homographies = pickle.load(f)

    if isinstance(homographies, dict):
        matrices = [homographies[i] for i in range(num_views)]
    else:
        matrices = list(homographies)

    if len(matrices) != num_views:
        raise ValueError(f"Expected {num_views} homography matrices, got {len(matrices)}")

    tensor = torch.as_tensor(matrices, dtype=torch.float32)
    if tensor.shape != (num_views, 3, 3):
        raise ValueError(
            f"Expected homographies with shape ({num_views}, 3, 3), got {tuple(tensor.shape)}"
        )

    return tensor


def _clamp_homogeneous_denominator(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sign = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    return torch.where(x.abs() < eps, sign * eps, x)


def _solve_homography_from_points(
    src_points: torch.Tensor,
    dst_points: torch.Tensor,
) -> torch.Tensor:
    x = src_points[:, 0]
    y = src_points[:, 1]
    u = dst_points[:, 0]
    v = dst_points[:, 1]
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)

    rows_x = torch.stack(
        [x, y, ones, zeros, zeros, zeros, -u * x, -u * y],
        dim=1,
    )
    rows_y = torch.stack(
        [zeros, zeros, zeros, x, y, ones, -v * x, -v * y],
        dim=1,
    )
    lhs = torch.empty(8, 8, device=src_points.device, dtype=src_points.dtype)
    lhs[0::2] = rows_x
    lhs[1::2] = rows_y

    rhs = torch.empty(8, device=src_points.device, dtype=src_points.dtype)
    rhs[0::2] = u
    rhs[1::2] = v

    params = torch.linalg.solve(lhs, rhs)
    one = torch.ones(1, device=src_points.device, dtype=src_points.dtype)
    return torch.cat([params, one]).view(3, 3)


def interpolate_homography_by_corners(
    homography: torch.Tensor,
    alpha: float,
    image_h: int,
    image_w: int,
) -> torch.Tensor:
    corners = torch.tensor(
        [
            [0.0, 0.0],
            [image_w - 1.0, 0.0],
            [image_w - 1.0, image_h - 1.0],
            [0.0, image_h - 1.0],
        ],
        device=homography.device,
        dtype=homography.dtype,
    )
    corners_h = torch.cat([corners, torch.ones_like(corners[:, :1])], dim=1)

    dst_h = (homography @ corners_h.T).T
    dst_z = _clamp_homogeneous_denominator(dst_h[:, 2:3])
    dst = dst_h[:, :2] / dst_z
    dst_alpha = corners + alpha * (dst - corners)

    return _solve_homography_from_points(corners, dst_alpha)


def make_homography_sampling_grid(
    homography: torch.Tensor,
    image_h: int,
    image_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    calc_dtype = torch.float32
    homography = homography.to(device=device, dtype=calc_dtype)
    inverse = torch.linalg.inv(homography)

    xs = torch.linspace(0, image_w - 1, image_w, device=device, dtype=calc_dtype)
    ys = torch.linspace(0, image_h - 1, image_h, device=device, dtype=calc_dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    ones = torch.ones_like(grid_x)
    target_points = torch.stack([grid_x, grid_y, ones], dim=0).reshape(3, -1)
    source_points = inverse @ target_points

    z = _clamp_homogeneous_denominator(source_points[2])
    src_x = source_points[0] / z
    src_y = source_points[1] / z

    norm_x = 2.0 * src_x / max(image_w - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(image_h - 1, 1) - 1.0

    return torch.stack([norm_x, norm_y], dim=-1).view(image_h, image_w, 2).to(dtype=dtype)


def make_rotation_homography(
    degrees: float | torch.Tensor,
    image_h: int,
    image_w: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    angle = torch.as_tensor(degrees, device=device, dtype=torch.float32)
    radians = angle * torch.pi / 180.0
    cos = torch.cos(radians)
    sin = torch.sin(radians)
    cx = torch.as_tensor((image_w - 1) * 0.5, device=device, dtype=torch.float32)
    cy = torch.as_tensor((image_h - 1) * 0.5, device=device, dtype=torch.float32)

    homography = torch.eye(3, device=device, dtype=torch.float32)
    homography[0, 0] = cos
    homography[0, 1] = -sin
    homography[0, 2] = cx - cos * cx + sin * cy
    homography[1, 0] = sin
    homography[1, 1] = cos
    homography[1, 2] = cy - sin * cx - cos * cy
    return homography.to(dtype=dtype)


def transform_points_by_homography(
    points: torch.Tensor,
    homography: torch.Tensor,
) -> torch.Tensor:
    original_shape = points.shape
    flat = points.reshape(-1, 2)
    ones = torch.ones(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)
    points_h = torch.cat([flat, ones], dim=1)

    projected = points_h @ homography.to(device=flat.device, dtype=flat.dtype).T
    z = _clamp_homogeneous_denominator(projected[:, 2:3])
    projected_xy = projected[:, :2] / z
    return projected_xy.reshape(original_shape)


def points_in_image_mask(points: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
    return (
        (points[..., 0] >= 0)
        & (points[..., 0] <= image_w - 1)
        & (points[..., 1] >= 0)
        & (points[..., 1] <= image_h - 1)
    )


def _clone_coordinates(coordinates: Any) -> Any:
    if coordinates is None:
        return None
    if torch.is_tensor(coordinates):
        return coordinates.clone()
    if isinstance(coordinates, tuple):
        return [_clone_coordinates(item) for item in coordinates]
    if isinstance(coordinates, list):
        return [_clone_coordinates(item) for item in coordinates]
    return coordinates


def _transform_coordinate_entry(
    entry: Any,
    homography: torch.Tensor,
    image_h: int,
    image_w: int,
    device: torch.device,
) -> tuple[Any, Any]:
    if entry is None:
        return None, None

    if torch.is_tensor(entry):
        points = entry.to(device=device)
        transformed = transform_points_by_homography(points, homography)
        valid = points_in_image_mask(transformed, image_h, image_w)
        return transformed.to(device=entry.device, dtype=entry.dtype), valid.to(device=entry.device)

    points = torch.as_tensor(entry, device=device, dtype=torch.float32)
    transformed = transform_points_by_homography(points, homography)
    valid = points_in_image_mask(transformed, image_h, image_w)
    return transformed.detach().cpu().tolist(), valid.detach().cpu().tolist()


def _apply_coordinate_transform_at(
    coordinates: Any,
    coordinate_valid: Any,
    batch_idx: int,
    view_idx: int,
    is_multiview: bool,
    homography: torch.Tensor,
    image_h: int,
    image_w: int,
    device: torch.device,
) -> None:
    if coordinates is None:
        return

    if torch.is_tensor(coordinates):
        if is_multiview and coordinates.ndim >= 4:
            transformed, valid = _transform_coordinate_entry(
                coordinates[batch_idx, view_idx],
                homography,
                image_h,
                image_w,
                device,
            )
            coordinates[batch_idx, view_idx] = transformed
            if coordinate_valid is not None:
                coordinate_valid[batch_idx, view_idx] = valid
        else:
            transformed, valid = _transform_coordinate_entry(
                coordinates[batch_idx],
                homography,
                image_h,
                image_w,
                device,
            )
            coordinates[batch_idx] = transformed
            if coordinate_valid is not None:
                coordinate_valid[batch_idx] = valid
        return

    if is_multiview:
        transformed, valid = _transform_coordinate_entry(
            coordinates[batch_idx][view_idx],
            homography,
            image_h,
            image_w,
            device,
        )
        coordinates[batch_idx][view_idx] = transformed
        if coordinate_valid is not None:
            coordinate_valid[batch_idx][view_idx] = valid
    else:
        transformed, valid = _transform_coordinate_entry(
            coordinates[batch_idx],
            homography,
            image_h,
            image_w,
            device,
        )
        coordinates[batch_idx] = transformed
        if coordinate_valid is not None:
            coordinate_valid[batch_idx] = valid


def _make_coordinate_valid_like(coordinates: Any) -> Any:
    if coordinates is None:
        return None
    if torch.is_tensor(coordinates):
        return torch.ones(coordinates.shape[:-1], device=coordinates.device, dtype=torch.bool)
    if isinstance(coordinates, tuple):
        return [_make_coordinate_valid_like(item) for item in coordinates]
    if isinstance(coordinates, list):
        return [_make_coordinate_valid_like(item) for item in coordinates]
    return None


def _resolve_homography(
    homographies: torch.Tensor,
    batch_idx: int,
    view_idx: int,
    view_indices: torch.Tensor | None,
    is_multiview: bool,
) -> torch.Tensor:
    if homographies.ndim == 2:
        return homographies
    if homographies.ndim == 3:
        if is_multiview:
            return homographies[view_idx]
        if view_indices is None:
            return homographies[0]
        return homographies[int(view_indices[batch_idx].item())]
    if homographies.ndim == 4:
        if is_multiview:
            return homographies[batch_idx, view_idx]
        return homographies[batch_idx]
    raise ValueError(f"Expected homographies with 2-4 dims, got {tuple(homographies.shape)}")


def homography_augmentation(
    images: torch.Tensor,
    homographies: torch.Tensor | None,
    coordinates: Any = None,
    view_indices: torch.Tensor | None = None,
    probability: float = HOMOGRAPHY_AUG_PROB,
    alpha_range: tuple[float, float] = HOMOGRAPHY_AUG_ALPHA_RANGE,
    augmentation_strength: float | None = None,
    max_retries: int = HOMOGRAPHY_AUG_MAX_RETRIES,
    strict_alpha_range: tuple[float, float] | None = HOMOGRAPHY_AUG_STRICT_ALPHA_RANGE,
    min_valid_ratio: float = HOMOGRAPHY_AUG_MIN_VALID_RATIO,
    strict_min_valid_ratio: float = HOMOGRAPHY_AUG_STRICT_MIN_VALID_RATIO,
    strict_views: set[int] | None = HOMOGRAPHY_AUG_STRICT_VIEWS,
    padding_mode: str = "reflection",
    return_coordinate_mask: bool = False,
    return_applied_homographies: bool = False,
) -> Any:
    """
    Randomly warp images by homographies and optionally move image-space points.

    images:
        [B, C, H, W] for single-view batches, or [B, V, C, H, W].
    homographies:
        [V, 3, 3], [B, 3, 3], [B, V, 3, 3], or [3, 3].
        Each matrix maps source image coordinates to augmented image coordinates.
    coordinates:
        Optional image-space points. Tensor shapes [B, N, 2] or [B, V, N, 2]
        are supported, as are similarly nested Python lists.
    augmentation_strength:
        Multiplies alpha_range. For example 0.5 turns (0, 1) into (0, 0.5).
    """
    if homographies is None or probability <= 0:
        if coordinates is None:
            return images
        outputs = [images, coordinates]
        if return_coordinate_mask:
            outputs.append(_make_coordinate_valid_like(coordinates))
        if return_applied_homographies:
            outputs.append(None)
        return tuple(outputs)

    if images.ndim not in (4, 5):
        raise ValueError(
            f"Expected images with shape [B, C, H, W] or [B, V, C, H, W], got {tuple(images.shape)}"
        )

    alpha_min, alpha_max = alpha_range
    if augmentation_strength is not None:
        if augmentation_strength < 0:
            raise ValueError(f"augmentation_strength must be >= 0, got {augmentation_strength}")
        alpha_min *= augmentation_strength
        alpha_max *= augmentation_strength

    if not (0.0 <= alpha_min <= alpha_max):
        raise ValueError(f"Expected alpha_range as 0 <= min <= max, got {(alpha_min, alpha_max)}")

    if strict_alpha_range is not None:
        strict_alpha_min, strict_alpha_max = strict_alpha_range
        if augmentation_strength is not None:
            strict_alpha_min *= augmentation_strength
            strict_alpha_max *= augmentation_strength
        if not (0.0 <= strict_alpha_min <= strict_alpha_max):
            raise ValueError(
                "Expected strict_alpha_range as 0 <= min <= max, "
                f"got {(strict_alpha_min, strict_alpha_max)}"
            )
    else:
        strict_alpha_min, strict_alpha_max = alpha_min, alpha_max

    is_multiview = images.ndim == 5
    if is_multiview:
        batch_size, num_views, _, image_h, image_w = images.shape
    else:
        batch_size, _, image_h, image_w = images.shape
        num_views = 1

    homographies = homographies.to(device=images.device, dtype=torch.float32)
    augmented = images.clone()
    augmented_coordinates = _clone_coordinates(coordinates)
    coordinate_valid = _make_coordinate_valid_like(augmented_coordinates)
    applied_homographies = torch.eye(
        3,
        device=images.device,
        dtype=torch.float32,
    ).expand(batch_size, num_views, 3, 3).clone()
    mask = torch.ones(1, 1, image_h, image_w, device=images.device, dtype=images.dtype)
    strict_views = strict_views or set()

    for batch_idx in range(batch_size):
        for local_view_idx in range(num_views):
            view_idx = local_view_idx
            if not is_multiview and view_indices is not None:
                view_idx = int(view_indices[batch_idx].item())

            if torch.rand((), device=images.device).item() >= probability:
                continue

            is_strict_view = view_idx in strict_views
            view_alpha_min = strict_alpha_min if is_strict_view else alpha_min
            view_alpha_max = strict_alpha_max if is_strict_view else alpha_max
            view_min_valid_ratio = strict_min_valid_ratio if is_strict_view else min_valid_ratio

            selected_grid = None
            selected_homography = None
            base_homography = _resolve_homography(
                homographies,
                batch_idx,
                local_view_idx,
                view_indices,
                is_multiview,
            )

            for _ in range(max_retries):
                alpha = (
                    view_alpha_min
                    + torch.rand((), device=images.device).item()
                    * (view_alpha_max - view_alpha_min)
                )
                try:
                    interpolated_h = interpolate_homography_by_corners(
                        base_homography,
                        alpha,
                        image_h,
                        image_w,
                    )
                    if not torch.isfinite(interpolated_h).all():
                        continue

                    grid = make_homography_sampling_grid(
                        interpolated_h,
                        image_h,
                        image_w,
                        images.device,
                        images.dtype,
                    )
                    if not torch.isfinite(grid).all():
                        continue
                except RuntimeError:
                    continue

                valid = F.grid_sample(
                    mask,
                    grid.unsqueeze(0),
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=True,
                )
                if valid.mean().item() >= view_min_valid_ratio:
                    selected_grid = grid
                    selected_homography = interpolated_h
                    break

            if selected_grid is None or selected_homography is None:
                continue

            image = (
                images[batch_idx, local_view_idx]
                if is_multiview
                else images[batch_idx]
            )
            warped = F.grid_sample(
                image.unsqueeze(0),
                selected_grid.unsqueeze(0),
                mode="bilinear",
                padding_mode=padding_mode,
                align_corners=True,
            )
            if is_multiview:
                augmented[batch_idx, local_view_idx] = warped.squeeze(0)
            else:
                augmented[batch_idx] = warped.squeeze(0)

            applied_homographies[batch_idx, local_view_idx] = selected_homography
            _apply_coordinate_transform_at(
                augmented_coordinates,
                coordinate_valid,
                batch_idx,
                local_view_idx,
                is_multiview,
                selected_homography,
                image_h,
                image_w,
                images.device,
            )

    outputs = [augmented]
    if coordinates is not None:
        outputs.append(augmented_coordinates)
    if return_coordinate_mask:
        outputs.append(coordinate_valid)
    if return_applied_homographies:
        if not is_multiview:
            applied_homographies = applied_homographies[:, 0]
        outputs.append(applied_homographies)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def homogrphy_augmentation(*args: Any, **kwargs: Any) -> Any:
    return homography_augmentation(*args, **kwargs)


def rotation_augmentation(
    images: torch.Tensor,
    coordinates: Any = None,
    probability: float = ROTATION_AUG_PROB,
    degree_range: tuple[float, float] = ROTATION_AUG_DEGREE_RANGE,
    augmentation_strength: float | None = None,
    padding_mode: str = "reflection",
    return_coordinate_mask: bool = False,
    return_applied_homographies: bool = False,
) -> Any:
    """
    Randomly rotate images around the image center and optionally rotate points.

    images:
        [B, C, H, W] for single-view batches, or [B, V, C, H, W].
    coordinates:
        Optional image-space points. Tensor shapes [B, N, 2] or [B, V, N, 2]
        are supported, as are similarly nested Python lists.
    degree_range:
        Rotation angle range in degrees. The sampled value is used directly.
    augmentation_strength:
        Multiplies degree_range. For example 0.5 turns (-10, 10) into (-5, 5).
    """
    if probability <= 0:
        if coordinates is None:
            return images
        outputs = [images, coordinates]
        if return_coordinate_mask:
            outputs.append(_make_coordinate_valid_like(coordinates))
        if return_applied_homographies:
            outputs.append(None)
        return tuple(outputs)

    if images.ndim not in (4, 5):
        raise ValueError(
            f"Expected images with shape [B, C, H, W] or [B, V, C, H, W], got {tuple(images.shape)}"
        )

    degree_min, degree_max = degree_range
    if augmentation_strength is not None:
        if augmentation_strength < 0:
            raise ValueError(f"augmentation_strength must be >= 0, got {augmentation_strength}")
        degree_min *= augmentation_strength
        degree_max *= augmentation_strength

    if degree_min > degree_max:
        raise ValueError(f"Expected degree_range as min <= max, got {(degree_min, degree_max)}")

    is_multiview = images.ndim == 5
    if is_multiview:
        batch_size, num_views, _, image_h, image_w = images.shape
    else:
        batch_size, _, image_h, image_w = images.shape
        num_views = 1

    augmented = images.clone()
    augmented_coordinates = _clone_coordinates(coordinates)
    coordinate_valid = _make_coordinate_valid_like(augmented_coordinates)
    applied_homographies = torch.eye(
        3,
        device=images.device,
        dtype=torch.float32,
    ).expand(batch_size, num_views, 3, 3).clone()

    for batch_idx in range(batch_size):
        for view_idx in range(num_views):
            if torch.rand((), device=images.device).item() >= probability:
                continue

            degrees = (
                degree_min
                + torch.rand((), device=images.device).item()
                * (degree_max - degree_min)
            )
            homography = make_rotation_homography(
                degrees,
                image_h,
                image_w,
                images.device,
                torch.float32,
            )
            grid = make_homography_sampling_grid(
                homography,
                image_h,
                image_w,
                images.device,
                images.dtype,
            )

            image = images[batch_idx, view_idx] if is_multiview else images[batch_idx]
            warped = F.grid_sample(
                image.unsqueeze(0),
                grid.unsqueeze(0),
                mode="bilinear",
                padding_mode=padding_mode,
                align_corners=True,
            )
            if is_multiview:
                augmented[batch_idx, view_idx] = warped.squeeze(0)
            else:
                augmented[batch_idx] = warped.squeeze(0)

            applied_homographies[batch_idx, view_idx] = homography
            _apply_coordinate_transform_at(
                augmented_coordinates,
                coordinate_valid,
                batch_idx,
                view_idx,
                is_multiview,
                homography,
                image_h,
                image_w,
                images.device,
            )

    outputs = [augmented]
    if coordinates is not None:
        outputs.append(augmented_coordinates)
    if return_coordinate_mask:
        outputs.append(coordinate_valid)
    if return_applied_homographies:
        if not is_multiview:
            applied_homographies = applied_homographies[:, 0]
        outputs.append(applied_homographies)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)
