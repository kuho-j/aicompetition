import argparse
import torch
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from src.aug import (
        HOMOGRAPHY_AUG_ALPHA_RANGE,
        HOMOGRAPHY_AUG_MAX_RETRIES,
        HOMOGRAPHY_AUG_MIN_VALID_RATIO,
        HOMOGRAPHY_AUG_PROB,
        HOMOGRAPHY_AUG_STRICT_ALPHA_RANGE,
        HOMOGRAPHY_AUG_STRICT_MIN_VALID_RATIO,
        HOMOGRAPHY_AUG_STRICT_VIEWS,
        homography_augmentation,
        load_homography_augmentation_matrices,
        )
from src.dataset import ImageHeatmapDataset, collate_image_heatmap_fn
from src.loss import shape_heatmap_loss
from src.model.viewpoint_bev_detection import SingleViewBEVDetector

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

def load_bev_layer_checkpoint(model, ckpt_path, device, strict=True):
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

    missing, unexpected = model.grid_to_bev.load_state_dict(state_dict, strict=strict)
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
        optimizer,
        device,
        homography_augmentation_matrices=None,
        ):
    model.train()
    total_loss = 0

    if len(loader) == 0:
        raise ValueError('train loader is empty')

    for images, gt_heatmap in loader:
        images = images.to(device)
        gt_heatmap = gt_heatmap.to(device)
        images = homography_augmentation(
                images,
                homography_augmentation_matrices,
                probability=HOMOGRAPHY_AUG_PROB,
                max_retries=HOMOGRAPHY_AUG_MAX_RETRIES,
                alpha_range=HOMOGRAPHY_AUG_ALPHA_RANGE,
                strict_alpha_range=HOMOGRAPHY_AUG_STRICT_ALPHA_RANGE,
                min_valid_ratio=HOMOGRAPHY_AUG_MIN_VALID_RATIO,
                strict_min_valid_ratio=HOMOGRAPHY_AUG_STRICT_MIN_VALID_RATIO,
                strict_views=HOMOGRAPHY_AUG_STRICT_VIEWS,
                )

        # forward
        outputs = model(images, return_aux=True)

        # loss
        loss = shape_heatmap_loss(outputs["heatmap_logits"], gt_heatmap)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_loss(
        model,
        loader,
        device,
        ):
    model.eval()
    total_loss = 0

    if len(loader) == 0:
        raise ValueError('eval loader is empty')

    for images, gt_heatmap in loader:
        images = images.to(device)
        gt_heatmap = gt_heatmap.to(device)

        outputs = model(images, return_aux=True)
        loss = shape_heatmap_loss(outputs["heatmap_logits"], gt_heatmap)
        total_loss += loss.item()

    return total_loss / len(loader)

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
        ):
    if test_interval < 1:
        raise ValueError('test_interval must be at least 1')

    model.to(device)

    start_epoch = 1
    homography_augmentation_matrices = load_homography_augmentation_matrices()

    if bev_weights is not None:
        load_bev_layer_checkpoint(model, bev_weights, device)
    elif not train_bev_layer:
        print('warning: BEV layer is frozen without --bev-weights; using initialized BEV weights')

    set_bev_layer_trainable(model, train_bev_layer)
    optimizer = make_optimizer(model, lr)

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
                optimizer,
                device,
                homography_augmentation_matrices=homography_augmentation_matrices,
                )
        print(f'[Epoch {epoch}] loss : {loss:.4f}')
        
        if epoch % test_interval == 0:
            save_checkpoint(model, optimizer, epoch)

            val_loss = evaluate_loss(
                    model=model,
                    loader=test_loader,
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
        ):
    if fold_interval < 1:
        raise ValueError('fold_interval must be at least 1')

    model.to(device)

    start_epoch = 1
    homography_augmentation_matrices = load_homography_augmentation_matrices()

    if bev_weights is not None:
        load_bev_layer_checkpoint(model, bev_weights, device)
    elif not train_bev_layer:
        print('warning: BEV layer is frozen without --bev-weights; using initialized BEV weights')

    set_bev_layer_trainable(model, train_bev_layer)
    optimizer = make_optimizer(model, lr)

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
                optimizer,
                device,
                homography_augmentation_matrices=homography_augmentation_matrices,
                )
        print(f'[Epoch {epoch}][Fold {active_fold + 1}/{n_splits}] loss : {loss:.4f}')

        if epoch % fold_interval == 0:
            save_checkpoint(model, optimizer, epoch, fold=active_fold + 1)

            val_loss = evaluate_loss(
                    model=model,
                    loader=val_loader,
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
            help='Also train the BEV layer. By default only the decoder is trained.',
            )
    parser.add_argument(
            '--num-grid-points',
            type=int,
            default=9,
            help='Number of reference grid points used by the trained BEV layer.',
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

def make_data_list(filepath='data/filepaths_img_and_ht.txt'):
    data_list = []

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
            data_list.append({
                    'image_path': image_path,
                    'heatmap_path': heatmap_path,
                    })

    print(f'loaded samples: {len(data_list)} from {filepath}')

    return data_list

def main(
        epochs=50,
        weights=None,
        bev_weights=None,
        train_bev_layer=False,
        num_grid_points=9,
        k_folds=5,
        fold_interval=5,
        ):
    data_list = make_data_list()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleViewBEVDetector(
            num_classes=60,
            img_channels=3,
            fpn_out_channels=256,
            backbone_width=0.25,
            backbone_depth=0.33,
            heatmap_size=(60, 80),
            num_grid_points=num_grid_points,
            ).to(device)

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
            )

if __name__ == '__main__':
    args = parse_args()
    main(
            epochs=args.epochs,
            weights=args.weights,
            bev_weights=args.bev_weights,
            train_bev_layer=args.train_bev_layer,
            num_grid_points=args.num_grid_points,
            k_folds=args.k_folds,
            fold_interval=args.fold_interval,
            )

