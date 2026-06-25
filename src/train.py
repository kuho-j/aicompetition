import torch
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from src.dataset import MultiViewDataset, collate_fn
from src.loss import gaussian_focal_loss
from src.test import test
from src.test import MultiViewEvalDataset, collate_eval_fn

def save_checkpoint(model, optimizer, epoch, save_dir='checkpoints', fold=None):
    os.makedirs(save_dir, exist_ok=True)

    if fold is None:
        ckpt_path = os.path.join(save_dir, f'epoch_{epoch}.pt')
    else:
        ckpt_path = os.path.join(save_dir, f'fold_{fold}_epoch_{epoch}.pt')

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
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print(f'optimizer state not found in checkpoint: {ckpt_path}')

        start_epoch = checkpoint.get('epoch', 0) + 1
    else:
        model.load_state_dict(checkpoint)
        start_epoch = 1

    print(f'loaded checkpoint: {ckpt_path} (resume from epoch {start_epoch})')

    return start_epoch



def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    if len(loader) == 0:
        raise ValueError('train loader is empty')

    for images, gt_heatmap in loader:
        images = images.to(device)
        gt_heatmap = gt_heatmap.to(device)

        # forward
        pred_heatmap = model(images) 

        # loss
        loss = gaussian_focal_loss(pred_heatmap, gt_heatmap)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1

    if resume_path is not None:
        start_epoch = load_checkpoint(model, optimizer, resume_path, device)

    train_loader, test_loader = make_train_loaders(
            data_list=data_list,
            test_data_list=test_data_list,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            output_size=output_size,
            )

    for epoch in range(start_epoch, start_epoch + epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1

    if resume_path is not None:
        start_epoch = load_checkpoint(model, optimizer, resume_path, device)

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

        loss = train_one_epoch(model, train_loader, optimizer, device)
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

