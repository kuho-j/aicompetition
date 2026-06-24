import torch
import os
from src.loss import gaussian_focal_loss

def save_checkpoint(model, optimizer, epoch, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, f'epoch_{epoch}.pt')

    torch.save({
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        }, ckpt_path)

def load_checkpoint(model, optimizer, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1

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

def train(model, train_loader, device, resume_path = None, epochs=50, lr=1e-4):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0

    if resume_path is not None:
        start_epoch = load_checkpoint(model, optimizer, resume_path, device)

    for epoch in range(start_epoch, epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f'[Epoch {epoch+1}] loss : {loss:.4f}')
        
        if (epoch + 1) % 3 == 0:
            save_checkpoint(model, optimizer, epoch)

