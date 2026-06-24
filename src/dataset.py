import os
import torch
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


         




