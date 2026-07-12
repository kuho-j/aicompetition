import pickle
from pathlib import Path

import cv2
import torch

DATA_DIR = Path(__file__).resolve().parent
CALIBRATION_PATHS = (
    DATA_DIR / "homography_matrix_and_bev_grids.pkl",
    DATA_DIR / "homogrphy_matrix_and_bev_grids.pkl",
)

_calibration_cache = None
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def _load_calibration_results():
    global _calibration_cache
    if _calibration_cache is not None:
        return _calibration_cache

    for path in CALIBRATION_PATHS:
        if path.exists():
            with path.open("rb") as f:
                _calibration_cache = pickle.load(f)
            return _calibration_cache

    expected_paths = ", ".join(str(path) for path in CALIBRATION_PATHS)
    raise FileNotFoundError(f"Calibration pkl not found. Expected one of: {expected_paths}")


def _grid_key_from_fileinfo(fileinfo_parts):
    return "_".join(fileinfo_parts[:2])


def _get_calibration_entry(grid_key):
    calibration_results = _load_calibration_results()
    if grid_key not in calibration_results:
        available = ", ".join(sorted(calibration_results.keys()))
        raise KeyError(f"Grid key '{grid_key}' not found in calibration pkl. Available: {available}")

    entry = calibration_results[grid_key]
    if "grid_points" not in entry or "homography" not in entry:
        raise KeyError(f"Calibration entry '{grid_key}' must contain homography and grid_points.")

    return entry


def _make_grid_visible(grid_points):
    x = grid_points[:, 0]
    y = grid_points[:, 1]
    return (
        (x >= 0)
        & (x < IMAGE_WIDTH)
        & (y >= 0)
        & (y < IMAGE_HEIGHT)
    ).to(dtype=torch.float32)


def load_data_bev(fileinfo : str):
    '''
    fileinfo : line of data/filepaths_bev.txt
    '''
    fileinfo = fileinfo.strip().split()
    grid_key = _grid_key_from_fileinfo(fileinfo)
    img_path = fileinfo[-1]
    calibration_entry = _get_calibration_entry(grid_key)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # (H, W, C) -> (C, H, W)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float() / 255.0

    grid_points = torch.as_tensor(calibration_entry["grid_points"], dtype=torch.float32).clone()

    return {
        'image' : img,
        'grid_points' : grid_points,
        'grid_visible' : _make_grid_visible(grid_points),
        'homography' : torch.as_tensor(calibration_entry["homography"], dtype=torch.float32).clone(),
    }
