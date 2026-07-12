import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from src.render_gt_heatmap import render_gaussian_heatmap

class MultiViewDataset(Dataset):
    '''
    input:
    data_list : list[dict[str, torch.Tensor]]
        'images' : [N_views = 5, C, H, W]
        'classes': [N_objects]
        'centers' : [N_objects, 2] # normalized
    num_classes : int
    output_size : tuple[int, int]

    output: dict
        'images' : [N_views = 5, C, H, W]
        'heatmap'
    '''
        
    def __init__(self, data_list : list[dict[str, torch.Tensor]], num_classes : int, output_size : tuple[int, int]):
        if len(data_list) == 0:
            raise ValueError('data_list is empty')

        for idx, sample in enumerate(data_list):
            if sample is None:
                raise ValueError(f'data_list[{idx}] is None')
            if 'images' not in sample:
                raise ValueError(f"data_list[{idx}] does not contain 'images'")

        self.data_list = data_list
        self.num_classes = num_classes
        self.output_h, self.output_w = output_size
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        sample = self.data_list[idx]
        images = sample['images']
        centers = sample['centers']
        classes = sample['classes']

        heatmap, _ = render_gaussian_heatmap(
                centers,
                classes,
                self.num_classes,
                self.output_h,
                self.output_w,
                sigma=1.5,
                )
        return {
            'images' : images,
            'heatmap' : heatmap
        }

def collate_fn(batch):
    image_shapes = [tuple(b['images'].shape) for b in batch]
    if len(set(image_shapes)) != 1:
        raise ValueError(f'all image tensors must have the same shape, got {image_shapes}')

    imgs = torch.stack([b['images'] for b in batch])
    hms = torch.stack([b['heatmap'] for b in batch])

    return imgs, hms


class SingleViewDataset(Dataset):
    """
    Expands each existing 5-view sample into independent single-image samples.

    The label heatmap stays the same top-down target, while each camera image
    becomes its own training input.
    """

    def __init__(
        self,
        data_list: list[dict[str, torch.Tensor]],
        num_classes: int,
        output_size: tuple[int, int],
    ):
        if len(data_list) == 0:
            raise ValueError("data_list is empty")

        self.samples: list[tuple[int, int]] = []
        self.data_list = data_list
        self.num_classes = num_classes
        self.output_h, self.output_w = output_size

        for sample_idx, sample in enumerate(data_list):
            if sample is None:
                raise ValueError(f"data_list[{sample_idx}] is None")
            if "images" not in sample:
                raise ValueError(f"data_list[{sample_idx}] does not contain 'images'")
            images = sample["images"]
            if images.ndim != 4:
                raise ValueError(
                    f"sample images must have shape [N, C, H, W], got {tuple(images.shape)}"
                )
            for view_idx in range(images.shape[0]):
                self.samples.append((sample_idx, view_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_idx, view_idx = self.samples[idx]
        sample = self.data_list[sample_idx]
        image = sample["images"][view_idx]
        centers = sample["centers"]
        classes = sample["classes"]

        heatmap, _ = render_gaussian_heatmap(
            centers,
            classes,
            self.num_classes,
            self.output_h,
            self.output_w,
            sigma=1.5,
        )
        return {
            "image": image,
            "heatmap": heatmap,
        }


def collate_single_view_fn(batch):
    image_shapes = [tuple(b["image"].shape) for b in batch]
    if len(set(image_shapes)) != 1:
        raise ValueError(f"all image tensors must have the same shape, got {image_shapes}")

    imgs = torch.stack([b["image"] for b in batch])
    hms = torch.stack([b["heatmap"] for b in batch])

    return imgs, hms


class ImageHeatmapDataset(Dataset):
    """
    Dataset for image paths paired with precomputed heatmap npz paths.

    input:
    data_list : list[dict]
        'image_path' : jpg path
        'heatmap_path' : npz path containing a 60-channel heatmap

    output:
        'image' : [3, H, W]
        'heatmap' : [num_classes, output_h, output_w]
    """

    def __init__(
        self,
        data_list: list[dict],
        num_classes: int = 60,
        output_size: tuple[int, int] = (60, 80),
    ):
        if len(data_list) == 0:
            raise ValueError("data_list is empty")

        self.data_list = data_list
        self.num_classes = num_classes
        self.output_h, self.output_w = output_size

        for idx, sample in enumerate(data_list):
            if "image_path" not in sample:
                raise ValueError(f"data_list[{idx}] does not contain 'image_path'")
            if "heatmap_path" not in sample:
                raise ValueError(f"data_list[{idx}] does not contain 'heatmap_path'")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        image = self._load_image(sample["image_path"])
        heatmap = self._load_heatmap(sample["heatmap_path"])

        return {
            "image": image,
            "heatmap": heatmap,
        }

    @staticmethod
    def _load_image(image_path):
        img = Image.open(image_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

    def _load_heatmap(self, heatmap_path):
        with np.load(heatmap_path) as npz:
            if "heatmap" in npz:
                heatmap = npz["heatmap"]
            elif "heatmaps" in npz:
                heatmap = npz["heatmaps"]
            elif "arr_0" in npz:
                heatmap = npz["arr_0"]
            elif len(npz.files) == 1:
                heatmap = npz[npz.files[0]]
            else:
                raise ValueError(
                    f"heatmap npz must contain one array or a heatmap key, got keys {npz.files}"
                )

        heatmap = torch.as_tensor(heatmap, dtype=torch.float32)

        if heatmap.ndim == 4 and heatmap.shape[0] == 1:
            heatmap = heatmap.squeeze(0)
        if heatmap.ndim != 3:
            raise ValueError(f"heatmap must have shape [C, H, W] or [H, W, C], got {tuple(heatmap.shape)}")
        if heatmap.shape[0] != self.num_classes and heatmap.shape[-1] == self.num_classes:
            heatmap = heatmap.permute(2, 0, 1)
        heatmap = self._resize_heatmap(heatmap)
        if heatmap.shape != (self.num_classes, self.output_h, self.output_w):
            raise ValueError(
                f"heatmap must have shape {(self.num_classes, self.output_h, self.output_w)}, "
                f"got {tuple(heatmap.shape)}"
            )

        return heatmap

    def _resize_heatmap(self, heatmap):
        _, heatmap_h, heatmap_w = heatmap.shape
        if (heatmap_h, heatmap_w) == (self.output_h, self.output_w):
            return heatmap

        if heatmap_h < self.output_h or heatmap_w < self.output_w:
            raise ValueError(
                f"heatmap spatial size {(heatmap_h, heatmap_w)} is smaller than "
                f"target {(self.output_h, self.output_w)}"
            )

        heatmap = heatmap.unsqueeze(0)
        if heatmap_h % self.output_h == 0 and heatmap_w % self.output_w == 0:
            kernel_size = (heatmap_h // self.output_h, heatmap_w // self.output_w)
            heatmap = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=kernel_size)
        else:
            heatmap = F.adaptive_max_pool2d(heatmap, (self.output_h, self.output_w))

        return heatmap.squeeze(0)


def collate_image_heatmap_fn(batch):
    image_shapes = [tuple(b["image"].shape) for b in batch]
    if len(set(image_shapes)) != 1:
        raise ValueError(f"all image tensors must have the same shape, got {image_shapes}")

    heatmap_shapes = [tuple(b["heatmap"].shape) for b in batch]
    if len(set(heatmap_shapes)) != 1:
        raise ValueError(f"all heatmap tensors must have the same shape, got {heatmap_shapes}")

    imgs = torch.stack([b["image"] for b in batch])
    hms = torch.stack([b["heatmap"] for b in batch])

    return imgs, hms


class GridBEVDataset(Dataset):
    """
    Dataset for GridToBEVLayer training.

    input:
    img_list : list[torch.Tensor] | list[np.ndarray]
        Each image must be [C, H, W] or [H, W, C]. Values are expected to be in
        [0, 1], but uint8-like [0, 255] images are normalized automatically.
    grid_list : list[torch.Tensor] | list[np.ndarray]
        Each grid item must be [N, 2] image pixel coordinates ordered as (x, y).
    grid_visible_list : optional list[torch.Tensor] | list[np.ndarray]
        Each visibility item must be [N]. If omitted, all grid points are visible.

    output: dict
        'image' : [C, H, W]
        'grid_points' : [N, 2]
        'grid_visible' : [N]
    """

    def __init__(
        self,
        img_list: list[torch.Tensor | np.ndarray],
        grid_list: list[torch.Tensor | np.ndarray],
        grid_visible_list: list[torch.Tensor | np.ndarray] | None = None,
    ):
        if len(img_list) == 0:
            raise ValueError("img_list is empty")
        if len(img_list) != len(grid_list):
            raise ValueError(
                f"img_list and grid_list must have the same length, "
                f"got {len(img_list)} and {len(grid_list)}"
            )
        if grid_visible_list is not None and len(grid_visible_list) != len(img_list):
            raise ValueError(
                f"grid_visible_list must have the same length as img_list, "
                f"got {len(grid_visible_list)} and {len(img_list)}"
            )

        self.img_list = img_list
        self.grid_list = grid_list
        self.grid_visible_list = grid_visible_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = self._format_image(self.img_list[idx])
        grid_points = self._format_grid(self.grid_list[idx])

        if self.grid_visible_list is None:
            grid_visible = torch.ones(grid_points.shape[0], dtype=torch.float32)
        else:
            grid_visible = torch.as_tensor(
                self.grid_visible_list[idx],
                dtype=torch.float32,
            )
            if grid_visible.ndim != 1 or grid_visible.shape[0] != grid_points.shape[0]:
                raise ValueError(
                    "each grid_visible item must have shape [N] matching grid_points, "
                    f"got {tuple(grid_visible.shape)} and {tuple(grid_points.shape)}"
                )

        return {
            "image": image,
            "grid_points": grid_points,
            "grid_visible": grid_visible,
        }

    @staticmethod
    def _format_image(image):
        image = torch.as_tensor(image, dtype=torch.float32)
        if image.ndim != 3:
            raise ValueError(f"image must have shape [C, H, W] or [H, W, C], got {tuple(image.shape)}")

        if image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)

        if image.max().item() > 1.0:
            image = image / 255.0

        return image

    @staticmethod
    def _format_grid(grid_points):
        grid_points = torch.as_tensor(grid_points, dtype=torch.float32)
        if grid_points.ndim != 2 or grid_points.shape[1] != 2:
            raise ValueError(f"grid_points must have shape [N, 2], got {tuple(grid_points.shape)}")
        return grid_points


def collate_grid_bev_fn(batch):
    if len(batch) == 0:
        raise ValueError("batch is empty")

    image_shapes = [tuple(b["image"].shape) for b in batch]
    if len(set(image_shapes)) != 1:
        raise ValueError(f"all image tensors must have the same shape, got {image_shapes}")

    grid_shapes = [tuple(b["grid_points"].shape) for b in batch]
    if len(set(grid_shapes)) != 1:
        raise ValueError(f"all grid tensors must have the same shape, got {grid_shapes}")

    images = torch.stack([b["image"] for b in batch])
    grid_points = torch.stack([b["grid_points"] for b in batch])
    grid_visible = torch.stack([b["grid_visible"] for b in batch])

    return {
        "image": images,
        "grid_points": grid_points,
        "grid_visible": grid_visible,
    }

def format_data(filename_info, expected_num_views=5):

    imgfile_lst = filename_info[0]
    label_file = filename_info[2]
    
    img_list = []
    classes = []
    centers = []

    for imgfile in imgfile_lst:
        if not os.path.exists(imgfile):
            return None

        img = Image.open(imgfile).convert('RGB')

        img = np.array(img).astype(np.float32) / 255.0 # normalize

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        img_list.append(torch.from_numpy(img))

    if len(img_list) != expected_num_views:
        return None

    img_list = torch.stack(img_list)
     
    if not os.path.exists(label_file):
        return None

    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            cls = int(parts[0])
            cx, cy = map(float, parts[1:])
            classes.append(cls)
            centers.append([cx, cy])

    classes = torch.tensor(classes, dtype = torch.long)
    centers = torch.tensor(centers, dtype = torch.float32)

    return {
            'images' : img_list,
            'centers' : centers,
            'classes' : classes,
            }


         




