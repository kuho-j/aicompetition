import argparse
import pickle
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from data.make_filename import make_filename
from src.dataset import MultiViewDataset, collate_fn
from src.dataset import format_data
from src.loss import gaussian_focal_loss
from src.model.multiview_detection import MultiViewDetector
from src.test import test
from src.test import MultiViewEvalDataset, collate_eval_fn

HOMOGRAPHY_FREEZE_EPOCHS = 50
HOMOGRAPHY_LR_MULT = 0.1
HOMOGRAPHY_REG_WEIGHT = 1e-6
HOMOGRAPHY_AUG_PATH = 'data/homography_matrix_train_to_video.pkl'
HOMOGRAPHY_AUG_PROB = 0.35
HOMOGRAPHY_AUG_MAX_RETRIES = 8
HOMOGRAPHY_AUG_ALPHA_MAX = 1.0
HOMOGRAPHY_AUG_STRICT_ALPHA_MAX = 0.75
HOMOGRAPHY_AUG_MIN_VALID_RATIO = 0.65
HOMOGRAPHY_AUG_STRICT_MIN_VALID_RATIO = 0.78
HOMOGRAPHY_AUG_STRICT_VIEWS = {1, 3}


def load_homography_augmentation_matrices(path=HOMOGRAPHY_AUG_PATH, num_views=5):
    with open(path, 'rb') as f:
        homographies = pickle.load(f)

    if isinstance(homographies, dict):
        matrices = [homographies[i] for i in range(num_views)]
    else:
        matrices = list(homographies)

    if len(matrices) != num_views:
        raise ValueError(
                f'Expected {num_views} homography matrices, got {len(matrices)}'
                )

    tensor = torch.as_tensor(matrices, dtype=torch.float32)
    if tensor.shape != (num_views, 3, 3):
        raise ValueError(
                f'Expected homographies with shape ({num_views}, 3, 3), got {tuple(tensor.shape)}'
                )

    return tensor


def _clamp_homogeneous_denominator(x, eps=1e-6):
    sign = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    return torch.where(x.abs() < eps, sign * eps, x)


def _solve_homography_from_points(src_points, dst_points):
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


def _interpolate_homography_by_corners(homography, alpha, image_h, image_w):
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


def _make_homography_sampling_grid(homography, image_h, image_w, device, dtype):
    calc_dtype = torch.float32
    homography = homography.to(device=device, dtype=calc_dtype)
    inverse = torch.linalg.inv(homography)

    xs = torch.linspace(0, image_w - 1, image_w, device=device, dtype=calc_dtype)
    ys = torch.linspace(0, image_h - 1, image_h, device=device, dtype=calc_dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    ones = torch.ones_like(grid_x)
    target_points = torch.stack([grid_x, grid_y, ones], dim=0).reshape(3, -1)
    source_points = inverse @ target_points

    z = _clamp_homogeneous_denominator(source_points[2])
    src_x = source_points[0] / z
    src_y = source_points[1] / z

    norm_x = 2.0 * src_x / max(image_w - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(image_h - 1, 1) - 1.0

    return torch.stack([norm_x, norm_y], dim=-1).view(image_h, image_w, 2).to(dtype=dtype)


def homogrphy_augmentation(
        images,
        homographies,
        probability=HOMOGRAPHY_AUG_PROB,
        max_retries=HOMOGRAPHY_AUG_MAX_RETRIES,
        alpha_max=HOMOGRAPHY_AUG_ALPHA_MAX,
        strict_alpha_max=HOMOGRAPHY_AUG_STRICT_ALPHA_MAX,
        min_valid_ratio=HOMOGRAPHY_AUG_MIN_VALID_RATIO,
        strict_min_valid_ratio=HOMOGRAPHY_AUG_STRICT_MIN_VALID_RATIO,
        strict_views=HOMOGRAPHY_AUG_STRICT_VIEWS,
        ):
    '''
    Randomly warp training views toward the video-view homographies.

    images: [B, num_views, C, H, W]
    homographies: [num_views, 3, 3], source(train image) -> target(video-like image)
    '''

    if homographies is None or probability <= 0:
        return images

    if images.ndim != 5:
        raise ValueError(f'Expected images with shape [B, V, C, H, W], got {tuple(images.shape)}')

    batch_size, num_views, _, image_h, image_w = images.shape
    if homographies.shape[0] != num_views:
        raise ValueError(
                f'Expected {num_views} homography matrices, got {homographies.shape[0]}'
                )

    homographies = homographies.to(device=images.device, dtype=torch.float32)
    augmented = images.clone()
    mask = torch.ones(1, 1, image_h, image_w, device=images.device, dtype=images.dtype)

    for batch_idx in range(batch_size):
        for view_idx in range(num_views):
            if torch.rand((), device=images.device).item() >= probability:
                continue

            is_strict_view = view_idx in strict_views
            view_alpha_max = strict_alpha_max if is_strict_view else alpha_max
            view_min_valid_ratio = strict_min_valid_ratio if is_strict_view else min_valid_ratio

            selected_grid = None
            for _ in range(max_retries):
                alpha = torch.rand((), device=images.device).item() * view_alpha_max
                try:
                    interpolated_h = _interpolate_homography_by_corners(
                            homographies[view_idx],
                            alpha,
                            image_h,
                            image_w,
                            )
                    if not torch.isfinite(interpolated_h).all():
                        continue

                    grid = _make_homography_sampling_grid(
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
                        mode='nearest',
                        padding_mode='zeros',
                        align_corners=True,
                        )
                if valid.mean().item() >= view_min_valid_ratio:
                    selected_grid = grid
                    break

            if selected_grid is None:
                continue

            warped = F.grid_sample(
                    images[batch_idx, view_idx].unsqueeze(0),
                    selected_grid.unsqueeze(0),
                    mode='bilinear',
                    padding_mode='reflection',
                    align_corners=True,
                    )
            augmented[batch_idx, view_idx] = warped.squeeze(0)

    return augmented

def save_checkpoint(model, optimizer, epoch, save_dir='checkpoints', fold=None):
    os.makedirs(save_dir, exist_ok=True)

    '''
    if fold is None:
        ckpt_path = os.path.join(save_dir, f'epoch_{epoch}.pt')
    else:
        ckpt_path = os.path.join(save_dir, f'fold_{fold}_epoch_{epoch}.pt')
    '''
    
    ckpt_path = os.path.join(save_dir, f'feature1_epoch_{epoch}.pt')

    checkpoint = {
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        }

    if fold is not None:
        checkpoint['fold'] = fold

    torch.save(checkpoint, ckpt_path)

def load_checkpoint(model, optimizer, ckpt_path, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')

    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as exc:
                print(f'optimizer state is incompatible with current parameter groups: {exc}')
                print('continuing with a freshly initialized optimizer')
        else:
            print(f'optimizer state not found in checkpoint: {ckpt_path}')

        start_epoch = checkpoint.get('epoch', 0) + 1
    else:
        model.load_state_dict(checkpoint)
        start_epoch = 1

    print(f'loaded checkpoint: {ckpt_path} (resume from epoch {start_epoch})')

    return start_epoch

def iter_homography_params(model):
    for name, param in model.named_parameters():
        if 'homography_params' in name:
            yield name, param

def has_homography_params(model):
    return any(True for _ in iter_homography_params(model))

def set_homography_trainable(model, trainable):
    for _, param in iter_homography_params(model):
        param.requires_grad = trainable

def make_optimizer(model, lr):
    homography_params = []
    base_params = []

    for name, param in model.named_parameters():
        if 'homography_params' in name:
            homography_params.append(param)
        else:
            base_params.append(param)

    param_groups = [{'params': base_params, 'lr': lr}]
    if homography_params:
        param_groups.append({
            'params': homography_params,
            'lr': lr * HOMOGRAPHY_LR_MULT,
        })

    return torch.optim.Adam(param_groups, lr=lr)

def capture_homography_reference(model, device):
    return {
        name: param.detach().clone().to(device)
        for name, param in iter_homography_params(model)
    }

def homography_regularization(model, reference):
    if not reference:
        return None

    reg = None
    for name, param in iter_homography_params(model):
        if name not in reference:
            continue

        loss = (param - reference[name].to(param.device, param.dtype)).pow(2).mean()
        reg = loss if reg is None else reg + loss

    return reg


def train_one_epoch(
        model,
        loader,
        optimizer,
        device,
        homography_reference=None,
        homography_reg_weight=0.0,
        homography_augmentation_matrices=None,
        ):
    model.train()
    total_loss = 0

    if len(loader) == 0:
        raise ValueError('train loader is empty')

    for images, gt_heatmap in loader:
        images = images.to(device)
        gt_heatmap = gt_heatmap.to(device)
        images = homogrphy_augmentation(
                images,
                homography_augmentation_matrices,
                )

        # forward
        pred_heatmap = model(images) 

        # loss
        loss = gaussian_focal_loss(pred_heatmap, gt_heatmap)
        h_reg = homography_regularization(model, homography_reference)
        if h_reg is not None and homography_reg_weight > 0:
            loss = loss + homography_reg_weight * h_reg

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(loader)

def configure_homography_phase(model, epoch):
    trainable = epoch > HOMOGRAPHY_FREEZE_EPOCHS
    set_homography_trainable(model, trainable)
    return trainable

def make_k_fold_loaders(
        data_list,
        fold,
        n_splits=5,
        batch_size=32,
        num_workers=0,
        num_classes=60,
        output_size=(60, 80),
        random_state=42,
        ):
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2')
    if len(data_list) < n_splits:
        raise ValueError(
                f'n_splits ({n_splits}) cannot be greater than number of samples ({len(data_list)})'
                )
    if fold < 0 or fold >= n_splits:
        raise ValueError(f'fold must be in [0, {n_splits - 1}], got {fold}')

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split_indices = list(kfold.split(data_list))
    train_indices, val_indices = split_indices[fold]

    train_dataset = MultiViewDataset(data_list, num_classes, output_size)
    val_dataset = MultiViewEvalDataset(data_list)

    train_loader = DataLoader(
            Subset(train_dataset, train_indices.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            )

    val_loader = DataLoader(
            Subset(val_dataset, val_indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_eval_fn,
            )

    return train_loader, val_loader

def make_train_loaders(
        data_list,
        test_data_list=None,
        batch_size=32,
        num_workers=0,
        num_classes=60,
        output_size=(60, 80),
        ):
    if test_data_list is None:
        test_data_list = data_list

    train_dataset = MultiViewDataset(data_list, num_classes, output_size)
    test_dataset = MultiViewEvalDataset(test_data_list)

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            )

    test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_eval_fn,
            )

    return train_loader, test_loader

def train(
        model,
        data_list,
        device,
        resume_path=None,
        epochs=50,
        lr=1e-4,
        test_data_list=None,
        test_interval=5,
        batch_size=32,
        num_workers=0,
        num_classes=60,
        output_size=(60, 80),
        topk=100,
        score_threshold=0.3,
        center_threshold=0.05,
        ):
    if test_interval < 1:
        raise ValueError('test_interval must be at least 1')

    model.to(device)

    start_epoch = 1
    homography_augmentation_matrices = load_homography_augmentation_matrices()

    configure_homography_phase(model, start_epoch)
    optimizer = make_optimizer(model, lr)
    homography_reference = capture_homography_reference(model, device)

    if resume_path is not None:
        start_epoch = load_checkpoint(model, optimizer, resume_path, device)
        configure_homography_phase(model, start_epoch)
        homography_reference = capture_homography_reference(model, device)

    train_loader, test_loader = make_train_loaders(
            data_list=data_list,
            test_data_list=test_data_list,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            output_size=output_size,
            )

    for epoch in range(start_epoch, start_epoch + epochs):
        homography_trainable = configure_homography_phase(model, epoch)
        if has_homography_params(model) and (
                epoch == start_epoch or epoch == HOMOGRAPHY_FREEZE_EPOCHS + 1
                ):
            phase = 'trainable' if homography_trainable else 'frozen'
            print(f'[Epoch {epoch}] homography alignment: {phase}')

        loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                homography_reference=homography_reference,
                homography_reg_weight=HOMOGRAPHY_REG_WEIGHT if homography_trainable else 0.0,
                homography_augmentation_matrices=homography_augmentation_matrices,
                )
        print(f'[Epoch {epoch}] loss : {loss:.4f}')
        
        if epoch % test_interval == 0:
            save_checkpoint(model, optimizer, epoch)

            precision, recall, fps = test(
                    model=model,
                    loader=test_loader,
                    device=device,
                    topk=topk,
                    score_threshold=score_threshold,
                    center_threshold=center_threshold,
                    print_result=False,
                    )
            print(
                    f'[Epoch {epoch}]',
                    f'Precision: {precision:.4f},',
                    f'Recall: {recall:.4f},',
                    f'FPS: {fps:.2f}'
                    )

def train_k_fold(
        model,
        data_list,
        device,
        resume_path=None,
        epochs=50,
        lr=1e-4,
        n_splits=5,
        fold_interval=5,
        batch_size=32,
        num_workers=0,
        num_classes=60,
        output_size=(60, 80),
        random_state=42,
        topk=100,
        score_threshold=0.3,
        center_threshold=0.05,
        ):
    if fold_interval < 1:
        raise ValueError('fold_interval must be at least 1')

    model.to(device)

    start_epoch = 1
    homography_augmentation_matrices = load_homography_augmentation_matrices()

    configure_homography_phase(model, start_epoch)
    optimizer = make_optimizer(model, lr)
    homography_reference = capture_homography_reference(model, device)

    if resume_path is not None:
        start_epoch = load_checkpoint(model, optimizer, resume_path, device)
        configure_homography_phase(model, start_epoch)
        homography_reference = capture_homography_reference(model, device)

    active_fold = None
    train_loader = None
    val_loader = None

    for epoch in range(start_epoch, start_epoch + epochs):
        fold = ((epoch - 1) // fold_interval) % n_splits

        if fold != active_fold:
            active_fold = fold
            train_loader, val_loader = make_k_fold_loaders(
                    data_list=data_list,
                    fold=active_fold,
                    n_splits=n_splits,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    num_classes=num_classes,
                    output_size=output_size,
                    random_state=random_state,
                    )
            print(f'[Epoch {epoch}] using fold {active_fold + 1}/{n_splits}')

        homography_trainable = configure_homography_phase(model, epoch)
        if has_homography_params(model) and (
                epoch == start_epoch or epoch == HOMOGRAPHY_FREEZE_EPOCHS + 1
                ):
            phase = 'trainable' if homography_trainable else 'frozen'
            print(f'[Epoch {epoch}] homography alignment: {phase}')

        loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                homography_reference=homography_reference,
                homography_reg_weight=HOMOGRAPHY_REG_WEIGHT if homography_trainable else 0.0,
                homography_augmentation_matrices=homography_augmentation_matrices,
                )
        print(f'[Epoch {epoch}][Fold {active_fold + 1}/{n_splits}] loss : {loss:.4f}')

        if epoch % fold_interval == 0:
            save_checkpoint(model, optimizer, epoch, fold=active_fold + 1)

            precision, recall, fps = test(
                    model=model,
                    loader=val_loader,
                    device=device,
                    topk=topk,
                    score_threshold=score_threshold,
                    center_threshold=center_threshold,
                    print_result=False,
                    )
            print(
                    f'[Epoch {epoch}][Fold {active_fold + 1}/{n_splits}]',
                    f'Precision: {precision:.4f},',
                    f'Recall: {recall:.4f},',
                    f'FPS: {fps:.2f}'
                    )

def parse_args():
    parser = argparse.ArgumentParser(description='Train MultiViewDetector.')
    parser.add_argument(
            '--epochs',
            type=int,
            default=50,
            help='Number of epochs to train. When --weights is used, this is the number of additional epochs.',
            )
    parser.add_argument(
            '--weights',
            default=None,
            help='Path to a checkpoint to resume from.',
            )
    parser.add_argument(
            '--k-folds',
            type=int,
            default=5,
            help='Number of folds for k-fold validation.',
            )
    parser.add_argument(
            '--fold-interval',
            type=int,
            default=5,
            help='Number of epochs to train before checkpointing/testing and moving to the next fold.',
            )
    return parser.parse_args()

def make_data_list():
    data_list = []
    skipped = 0

    with open('data/filename.txt', 'r') as file:
        for line in file:
            filename_info = make_filename(line)
            sample = format_data(filename_info)

            if sample is None:
                skipped += 1
                continue

            data_list.append(sample)

    print(f'loaded samples: {len(data_list)}, skipped samples: {skipped}')

    return data_list

def main(epochs=50, weights=None, k_folds=5, fold_interval=5):
    data_list = make_data_list()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiViewDetector(
            num_views=5,
            num_classes=60,
            img_channels=3,
            fpn_out_channels=256,
            backbone_width=0.25,
            backbone_depth=0.33,
            attn_heads=4,
            spatial_ds=2,
            ).to(device)

    train_k_fold(
            model,
            data_list,
            device,
            resume_path=weights,
            epochs=epochs,
            n_splits=k_folds,
            fold_interval=fold_interval,
            )

if __name__ == '__main__':
    args = parse_args()
    main(
            epochs=args.epochs,
            weights=args.weights,
            k_folds=args.k_folds,
            fold_interval=args.fold_interval,
            )

