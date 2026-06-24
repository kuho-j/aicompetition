import os
import torch
from data.make_filename import make_filename
from torch.utils.data import Dataset, DataLoader
from src.model.multiview_detection import MultiViewDetector
from src.loss import MultiViewDetectorLoss
from src.render_gt_heatmap import render_gaussian_heatmap
from src.predict import decode_predictions
from src.dataset import MultiViewDataset, format_data, collate_fn
from src.train import train

def make_data_list():
    data_list = []
    with open('data/filename.txt', 'r') as file:
        for line in file:
            filename_info = make_filename(line)
            data_list.append(format_data(filename_info))
    
    return data_list



def main():

    '''
    demo code -----

    model = MultiViewDetector(
        num_views=5,
        num_classes=10,
        img_channels=3,
        fpn_out_channels=128,
        backbone_width=0.25,
        backbone_depth=0.33,
        attn_heads=4,
        spatial_ds=2,
    ).to(device)

    criterion = MultiViewDetectorLoss(w_heatmap=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    B, N, C, H, W = 2, 5, 3, 480, 640
    images = torch.randn(B, N, C, H, W).to(device)

    # forward
    model.train()
    pred = model(images)

    out_h, out_w = pred.shape[2:]
    print(f"Heatmap output : {pred.shape}")

    num_classes = 10
    gt_heatmap = torch.zeros(B, num_classes, out_h, out_w).to(device)
    center_mask = torch.zeros(B, 1, out_h, out_w).to(device)

    for b in range(B):
        centers = torch.rand(2, 2)
        classes = torch.randint(0, num_classes, (2,))
        hm, mask = render_gaussian_heatmap(
            centers, classes, num_classes, out_h, out_w, sigma=2.0
        )
        gt_heatmap[b] = hm.to(device)
        center_mask[b] = mask.to(device)

    losses = criterion(pred, gt_heatmap, center_mask)
    optimizer.zero_grad()
    losses.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    scheduler.step()

    print(f"Heatmap Loss : {losses.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(images)
    detections = decode_predictions(pred, topk=50, score_threshold=0.3)
    for i, det in enumerate(detections):
        print(f"Batch {i}: {len(det['scores'])} detections")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    '''
    data_list = make_data_list()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiViewDetector(
            num_views = 5,
            num_classes = 60,
            img_channels = 3,
            fpn_out_channels = 256,
            backbone_width = 0.25,
            backbone_depth = 0.33,
            attn_heads = 4,
            spatial_ds = 2,
            ).to(device)

    dataset = MultiViewDataset(data_list, 60, (60, 80))
    
    loader = DataLoader(
            dataset,
            batch_size = 32,
            shuffle=True,
            collate_fn = collate_fn,
            )
    
    # TODO : have to check logic of resuming training
    train(model, loader, device)



if __name__ == "__main__":
    main()
