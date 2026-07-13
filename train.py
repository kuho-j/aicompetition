import argparse
import torch
import os
import zipfile
import zlib
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split

from src.aug import lighting_augmentation, rotation_augmentation_with_heatmap_targets
from src.dataset import ImageHeatmapDataset, collate_image_heatmap_fn
from src.loss import CenterNetDetectionLoss
from src.model.viewpoint_bev_detection import SingleViewBEVDetector

ROTATION_AUG_PROB = 1.0
ROTATION_AUG_MILD_END_EPOCH = 40
ROTATION_AUG_MILD_DEGREE_RANGE = (-5.0, 5.0)
ROTATION_AUG_STRONG_DEGREE_RANGE = (-15.0, 15.0)
LIGHTING_AUG_PROB = 1.0
LIGHTING_AUG_MILD_END_EPOCH = 40
LIGHTING_AUG_MILD_PERCENT_RANGE = (-10.0, 10.0)
LIGHTING_AUG_STRONG_PERCENT_RANGE = (-20.0, 20.0)
DEFAULT_AUGMENTATION_WARMUP_EPOCHS = 20

def get_rotation_aug_degree_range(
        epoch: int,
        warmup_epochs: int = DEFAULT_AUGMENTATION_WARMUP_EPOCHS,
        ) -> tuple[float, float] | None:
    if epoch <= warmup_epochs:
        return None
    if epoch <= ROTATION_AUG_MILD_END_EPOCH:
        return ROTATION_AUG_MILD_DEGREE_RANGE
    return ROTATION_AUG_STRONG_DEGREE_RANGE

def get_lighting_aug_percent_range(
        epoch: int,
        warmup_epochs: int = DEFAULT_AUGMENTATION_WARMUP_EPOCHS,
        ) -> tuple[float, float] | None:
    if epoch <= warmup_epochs:
        return None
    if epoch <= LIGHTING_AUG_MILD_END_EPOCH:
        return LIGHTING_AUG_MILD_PERCENT_RANGE
    return LIGHTING_AUG_STRONG_PERCENT_RANGE

def save_checkpoint(model, optimizer, epoch, save_dir='checkpoints', fold=None):
    os.makedirs(save_dir, exist_ok=True)

    '''
    if fold is None:
        ckpt_path = os.path.join(save_dir, f'epoch_{epoch}.pt')
    else:
        ckpt_path = os.path.join(save_dir, f'fold_{fold}_epoch_{epoch}.pt')
    '''
    
    ckpt_path = os.path.join(save_dir, f'feature2_epoch_{epoch}.pt')

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
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing:
            print(f'model missing keys: {missing}')
        if unexpected:
            print(f'model unexpected keys: {unexpected}')

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
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing:
            print(f'model missing keys: {missing}')
        if unexpected:
            print(f'model unexpected keys: {unexpected}')
        start_epoch = 1

    print(f'loaded checkpoint: {ckpt_path} (resume from epoch {start_epoch})')

    return start_epoch

def make_optimizer(model, lr):
    trainable_params = [
            param
            for param in model.parameters()
            if param.requires_grad
            ]
    if len(trainable_params) == 0:
        raise ValueError('model has no trainable parameters')

    return torch.optim.Adam(trainable_params, lr=lr)

def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint

def load_bev_layer_checkpoint(model, ckpt_path, device, strict=False):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'BEV layer checkpoint not found: {ckpt_path}')

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = _checkpoint_state_dict(checkpoint)

    prefixed = {
            key[len('grid_to_bev.'):]: value
            for key, value in state_dict.items()
            if key.startswith('grid_to_bev.')
            }
    if prefixed:
        state_dict = prefixed
    state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith('center_head.')
            }

    missing, unexpected = model.grid_to_bev.load_state_dict(state_dict, strict=strict)
    missing = [
            key
            for key in missing
            if not key.startswith('center_head.')
            ]
    print(f'loaded BEV layer checkpoint: {ckpt_path}')
    if missing:
        print(f'BEV layer missing keys: {missing}')
    if unexpected:
        print(f'BEV layer unexpected keys: {unexpected}')

def set_bev_layer_trainable(model, trainable):
    if hasattr(model, 'set_grid_to_bev_trainable'):
        model.set_grid_to_bev_trainable(trainable)
        return

    for param in model.grid_to_bev.parameters():
        param.requires_grad = trainable
    if not trainable:
        model.grid_to_bev.eval()


def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device,
        epoch,
        log_interval=20,
        augmentation_warmup_epochs=DEFAULT_AUGMENTATION_WARMUP_EPOCHS,
        ):
    model.train()
    total_loss = 0

    if len(loader) == 0:
        raise ValueError('train loader is empty')

    for step, (images, gt_heatmap, center_mask, center_offset) in enumerate(loader, start=1):
        images = images.to(device)
        gt_heatmap = gt_heatmap.to(device)
        center_mask = center_mask.to(device)
        center_offset = center_offset.to(device)
        rotation_degree_range = get_rotation_aug_degree_range(
                epoch,
                warmup_epochs=augmentation_warmup_epochs,
                )
        lighting_percent_range = get_lighting_aug_percent_range(
                epoch,
                warmup_epochs=augmentation_warmup_epochs,
                )
        if rotation_degree_range is not None:
            images, gt_heatmap, center_mask, center_offset = rotation_augmentation_with_heatmap_targets(
                    images,
                    gt_heatmap,
                    center_mask,
                    center_offset,
                    probability=ROTATION_AUG_PROB,
                    degree_range=rotation_degree_range,
                    )
        if lighting_percent_range is not None:
            images = lighting_augmentation(
                    images,
                    probability=LIGHTING_AUG_PROB,
                    percent_range=lighting_percent_range,
                    )

        # forward
        outputs = model(images, return_aux=True, decode=False, use_grid_head=False)

        # loss
        loss = criterion(
                outputs["heatmap_logits"],
                gt_heatmap,
                center_mask,
                outputs["offset"],
                center_offset,
                )

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % log_interval == 0:
            print(f'[Epoch {epoch}][Step {step}/{len(loader)}] loss={loss.item():.4f}')
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_loss(
        model,
        loader,
        criterion,
        device,
        ):
    model.eval()
    total_loss = 0

    if len(loader) == 0:
        raise ValueError('eval loader is empty')

    for images, gt_heatmap, center_mask, center_offset in loader:
        images = images.to(device)
        gt_heatmap = gt_heatmap.to(device)
        center_mask = center_mask.to(device)
        center_offset = center_offset.to(device)

        outputs = model(images, return_aux=True, decode=False, use_grid_head=False)
        loss = criterion(
                outputs["heatmap_logits"],
                gt_heatmap,
                center_mask,
                outputs["offset"],
                center_offset,
                )
        total_loss += loss.item()

    return total_loss / len(loader)

def make_k_fold_loaders(
        data_list,
        fold,
        n_splits=5,
        batch_size=32,
        num_workers=4,
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

    train_data = [data_list[idx] for idx in train_indices.tolist()]
    val_data = [data_list[idx] for idx in val_indices.tolist()]

    train_dataset = ImageHeatmapDataset(train_data, num_classes, output_size)
    val_dataset = ImageHeatmapDataset(val_data, num_classes, output_size)

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_image_heatmap_fn,
            )

    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_image_heatmap_fn,
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

    train_dataset = ImageHeatmapDataset(data_list, num_classes, output_size)
    test_dataset = ImageHeatmapDataset(test_data_list, num_classes, output_size)

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_image_heatmap_fn,
            )

    test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_image_heatmap_fn,
            )

    return train_loader, test_loader

def train(
        model,
        data_list,
        device,
        resume_path=None,
        bev_weights=None,
        train_bev_layer=False,
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
        loss_miss_weight=1.0,
        loss_false_positive_weight=0.05,
        loss_empty_confidence_weight=0.01,
        loss_displacement_weight=0.25,
        loss_displacement_radius=4,
        loss_offset_weight=1.0,
        log_interval=20,
        augmentation_warmup_epochs=DEFAULT_AUGMENTATION_WARMUP_EPOCHS,
        ):
    if test_interval < 1:
        raise ValueError('test_interval must be at least 1')
    if log_interval < 1:
        raise ValueError('log_interval must be at least 1')
    if augmentation_warmup_epochs < 0:
        raise ValueError('augmentation_warmup_epochs must be >= 0')

    model.to(device)

    start_epoch = 1

    if bev_weights is not None:
        load_bev_layer_checkpoint(model, bev_weights, device)
    elif not train_bev_layer:
        print('warning: BEV layer is frozen without --bev-weights; using initialized BEV weights')

    set_bev_layer_trainable(model, train_bev_layer)
    optimizer = make_optimizer(model, lr)
    criterion = CenterNetDetectionLoss(
            miss_weight=loss_miss_weight,
            false_positive_weight=loss_false_positive_weight,
            empty_confidence_weight=loss_empty_confidence_weight,
            displacement_weight=loss_displacement_weight,
            displacement_radius=loss_displacement_radius,
            offset_weight=loss_offset_weight,
            )

    if resume_path is not None:
        start_epoch = load_checkpoint(model, optimizer, resume_path, device)
        set_bev_layer_trainable(model, train_bev_layer)

    train_loader, test_loader = make_train_loaders(
            data_list=data_list,
            test_data_list=test_data_list,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            output_size=output_size,
            )

    for epoch in range(start_epoch, start_epoch + epochs):
        if epoch == start_epoch:
            phase = 'trainable' if train_bev_layer else 'frozen'
            print(f'[Epoch {epoch}] BEV layer: {phase}')

        loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch=epoch,
                log_interval=log_interval,
                augmentation_warmup_epochs=augmentation_warmup_epochs,
                )
        print(f'[Epoch {epoch}] loss : {loss:.4f}')
        
        if epoch % test_interval == 0:
            save_checkpoint(model, optimizer, epoch)

            val_loss = evaluate_loss(
                    model=model,
                    loader=test_loader,
                    criterion=criterion,
                    device=device,
                    )
            print(
                    f'[Epoch {epoch}]',
                    f'val_loss: {val_loss:.4f}'
                    )

def train_k_fold(
        model,
        data_list,
        device,
        resume_path=None,
        bev_weights=None,
        train_bev_layer=False,
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
        loss_miss_weight=1.0,
        loss_false_positive_weight=0.05,
        loss_empty_confidence_weight=0.01,
        loss_displacement_weight=0.25,
        loss_displacement_radius=4,
        loss_offset_weight=1.0,
        log_interval=20,
        augmentation_warmup_epochs=DEFAULT_AUGMENTATION_WARMUP_EPOCHS,
        ):
    if fold_interval < 1:
        raise ValueError('fold_interval must be at least 1')
    if log_interval < 1:
        raise ValueError('log_interval must be at least 1')
    if augmentation_warmup_epochs < 0:
        raise ValueError('augmentation_warmup_epochs must be >= 0')

    model.to(device)

    start_epoch = 1

    if bev_weights is not None:
        load_bev_layer_checkpoint(model, bev_weights, device)
    elif not train_bev_layer:
        print('warning: BEV layer is frozen without --bev-weights; using initialized BEV weights')

    set_bev_layer_trainable(model, train_bev_layer)
    optimizer = make_optimizer(model, lr)
    criterion = CenterNetDetectionLoss(
            miss_weight=loss_miss_weight,
            false_positive_weight=loss_false_positive_weight,
            empty_confidence_weight=loss_empty_confidence_weight,
            displacement_weight=loss_displacement_weight,
            displacement_radius=loss_displacement_radius,
            offset_weight=loss_offset_weight,
            )

    if resume_path is not None:
        start_epoch = load_checkpoint(model, optimizer, resume_path, device)
        set_bev_layer_trainable(model, train_bev_layer)

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

        if epoch == start_epoch:
            phase = 'trainable' if train_bev_layer else 'frozen'
            print(f'[Epoch {epoch}] BEV layer: {phase}')

        loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch=epoch,
                log_interval=log_interval,
                augmentation_warmup_epochs=augmentation_warmup_epochs,
                )
        print(f'[Epoch {epoch}][Fold {active_fold + 1}/{n_splits}] loss : {loss:.4f}')

        if epoch % fold_interval == 0:
            save_checkpoint(model, optimizer, epoch, fold=active_fold + 1)

            val_loss = evaluate_loss(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    )
            print(
                    f'[Epoch {epoch}][Fold {active_fold + 1}/{n_splits}]',
                    f'val_loss: {val_loss:.4f}'
                    )

def parse_args():
    parser = argparse.ArgumentParser(description='Train SingleViewBEVDetector.')
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
            '--bev-weights',
            default=None,
            help='Path to a trained GridToBEVLayer checkpoint.',
            )
    parser.add_argument(
            '--train-bev-layer',
            action='store_true',
            help='Also train the geometry/backbone part. By default only the CenterNet head is trained.',
            )
    parser.add_argument(
            '--num-grid-points',
            type=int,
            default=9,
            help='Number of reference grid points used by the trained BEV layer.',
            )
    parser.add_argument(
            '--bev-height',
            type=int,
            default=60,
            help='Height of the BEV feature grid produced by the BEV layer.',
            )
    parser.add_argument(
            '--bev-width',
            type=int,
            default=80,
            help='Width of the BEV feature grid produced by the BEV layer.',
            )
    parser.add_argument(
            '--decoder-channels',
            type=int,
            default=64,
            help='Deprecated compatibility option; BEV U-Net decoder has been removed.',
            )
    parser.add_argument(
            '--center-head-channels',
            type=int,
            default=128,
            help='Hidden channel width of the image-space CenterNet head.',
            )
    parser.add_argument(
            '--loss-miss-weight',
            type=float,
            default=1.0,
            help='Weight for labeled center pixels that are not detected.',
            )
    parser.add_argument(
            '--loss-false-positive-weight',
            type=float,
            default=0.05,
            help='Small penalty for detections away from labeled centers.',
            )
    parser.add_argument(
            '--loss-empty-confidence-weight',
            type=float,
            default=0.01,
            help='Small penalty for confident detections on pixels with no target response.',
            )
    parser.add_argument(
            '--loss-displacement-weight',
            type=float,
            default=0.25,
            help='Weight for local center displacement around labeled centers.',
            )
    parser.add_argument(
            '--loss-displacement-radius',
            type=int,
            default=4,
            help='Local radius in heatmap cells for center displacement loss.',
            )
    parser.add_argument(
            '--loss-offset-weight',
            type=float,
            default=1.0,
            help='Weight for CenterNet local offset L1 loss.',
            )
    parser.add_argument(
            '--k-folds',
            type=int,
            default=5,
            help='Number of folds for k-fold validation.',
            )
    parser.add_argument(
            '--no-k-fold',
            action='store_true',
            help='Disable k-fold validation and use a single train/test split.',
            )
    parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Test split ratio used with --no-k-fold.',
            )
    parser.add_argument(
            '--split-random-state',
            type=int,
            default=42,
            help='Random seed for k-fold and train/test splitting.',
            )
    parser.add_argument(
            '--fold-interval',
            type=int,
            default=5,
            help='Number of epochs to train before checkpointing/testing and moving to the next fold.',
            )
    parser.add_argument(
            '--log-interval',
            type=int,
            default=20,
            help='Number of train steps between progress logs.',
            )
    parser.add_argument(
            '--augmentation-warmup-epochs',
            type=int,
            default=DEFAULT_AUGMENTATION_WARMUP_EPOCHS,
            help='Number of initial epochs to train without rotation augmentation. '
                 'After warmup, train uses mild rotation until epoch 40, then strong rotation.',
            )
    return parser.parse_args()

def _validate_heatmap_npz(heatmap_path):
    with np.load(heatmap_path) as npz:
        if "heatmap" in npz:
            _ = npz["heatmap"]
        elif "heatmaps" in npz:
            _ = npz["heatmaps"]
        elif "arr_0" in npz:
            _ = npz["arr_0"]
        elif len(npz.files) == 1:
            _ = npz[npz.files[0]]
        else:
            raise ValueError(
                    f"heatmap npz must contain one array or a heatmap key, got keys {npz.files}"
                    )


def make_data_list(filepath='data/filepaths_img_and_ht.txt', validate_heatmaps=True):
    data_list = []
    skipped_missing_heatmap = 0
    skipped_invalid_heatmap = 0
    invalid_heatmap_examples = []

    with open(filepath, 'r') as file:
        for line_num, line in enumerate(file, start=1):
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if len(parts) != 2:
                raise ValueError(
                        f'{filepath}:{line_num} must contain image path and heatmap path, got {line.strip()}'
                        )

            image_path, heatmap_path = parts
            if not os.path.exists(heatmap_path):
                skipped_missing_heatmap += 1
                continue
            if validate_heatmaps:
                try:
                    _validate_heatmap_npz(heatmap_path)
                except (OSError, ValueError, KeyError, zipfile.BadZipFile, zlib.error) as exc:
                    skipped_invalid_heatmap += 1
                    if len(invalid_heatmap_examples) < 20:
                        invalid_heatmap_examples.append((line_num, heatmap_path, exc))
                    continue

            data_list.append({
                    'image_path': image_path,
                    'heatmap_path': heatmap_path,
                    })

    print(f'loaded samples: {len(data_list)} from {filepath}')
    if skipped_missing_heatmap > 0:
        print(f'skipped samples with missing heatmap npz: {skipped_missing_heatmap}')
    if skipped_invalid_heatmap > 0:
        print(f'skipped samples with invalid heatmap npz: {skipped_invalid_heatmap}')
        for line_num, heatmap_path, exc in invalid_heatmap_examples:
            print(f'  {filepath}:{line_num}: {heatmap_path} ({type(exc).__name__}: {exc})')

    return data_list

def main(
        epochs=50,
        weights=None,
        bev_weights=None,
        train_bev_layer=False,
        num_grid_points=9,
        bev_height=60,
        bev_width=80,
        decoder_channels=64,
        center_head_channels=128,
        k_folds=5,
        use_k_fold=True,
        test_size=0.2,
        split_random_state=42,
        fold_interval=5,
        loss_miss_weight=1.0,
        loss_false_positive_weight=0.05,
        loss_empty_confidence_weight=0.01,
        loss_displacement_weight=0.25,
        loss_displacement_radius=4,
        loss_offset_weight=1.0,
        log_interval=20,
        augmentation_warmup_epochs=DEFAULT_AUGMENTATION_WARMUP_EPOCHS,
        ):
    data_list = make_data_list()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleViewBEVDetector(
            num_classes=60,
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

    if use_k_fold:
        train_k_fold(
                model,
                data_list,
                device,
                resume_path=weights,
                bev_weights=bev_weights,
                train_bev_layer=train_bev_layer,
                epochs=epochs,
                n_splits=k_folds,
                fold_interval=fold_interval,
                random_state=split_random_state,
                loss_miss_weight=loss_miss_weight,
                loss_false_positive_weight=loss_false_positive_weight,
                loss_empty_confidence_weight=loss_empty_confidence_weight,
                loss_displacement_weight=loss_displacement_weight,
                loss_displacement_radius=loss_displacement_radius,
                loss_offset_weight=loss_offset_weight,
                log_interval=log_interval,
                augmentation_warmup_epochs=augmentation_warmup_epochs,
                )
    else:
        train_data_list, test_data_list = train_test_split(
                data_list,
                test_size=test_size,
                shuffle=True,
                random_state=split_random_state,
                )
        print(
                f'train/test split: train={len(train_data_list)}, '
                f'test={len(test_data_list)}, test_size={test_size}'
                )
        train(
                model,
                train_data_list,
                device,
                resume_path=weights,
                bev_weights=bev_weights,
                train_bev_layer=train_bev_layer,
                epochs=epochs,
                test_data_list=test_data_list,
                test_interval=fold_interval,
                loss_miss_weight=loss_miss_weight,
                loss_false_positive_weight=loss_false_positive_weight,
                loss_empty_confidence_weight=loss_empty_confidence_weight,
                loss_displacement_weight=loss_displacement_weight,
                loss_displacement_radius=loss_displacement_radius,
                loss_offset_weight=loss_offset_weight,
                log_interval=log_interval,
                augmentation_warmup_epochs=augmentation_warmup_epochs,
                )

if __name__ == '__main__':
    args = parse_args()
    main(
            epochs=args.epochs,
            weights=args.weights,
            bev_weights=args.bev_weights,
            train_bev_layer=args.train_bev_layer,
            num_grid_points=args.num_grid_points,
            bev_height=args.bev_height,
            bev_width=args.bev_width,
            decoder_channels=args.decoder_channels,
            center_head_channels=args.center_head_channels,
            k_folds=args.k_folds,
            use_k_fold=not args.no_k_fold,
            test_size=args.test_size,
            split_random_state=args.split_random_state,
            fold_interval=args.fold_interval,
            loss_miss_weight=args.loss_miss_weight,
            loss_false_positive_weight=args.loss_false_positive_weight,
            loss_empty_confidence_weight=args.loss_empty_confidence_weight,
            loss_displacement_weight=args.loss_displacement_weight,
            loss_displacement_radius=args.loss_displacement_radius,
            loss_offset_weight=args.loss_offset_weight,
            log_interval=args.log_interval,
            augmentation_warmup_epochs=args.augmentation_warmup_epochs,
            )
