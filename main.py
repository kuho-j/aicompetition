import torch
from torch.utils.data import DataLoader
from data.make_filename import make_filename
from src.model.multiview_detection import MultiViewDetector
from src.dataset import MultiViewDataset, format_data, collate_fn
from src.test import MultiViewEvalDataset, collate_eval_fn
from src.train import train

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



def main(epochs = 50, weights = None):

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

    test_dataset = MultiViewEvalDataset(data_list)

    test_loader = DataLoader(
            test_dataset,
            batch_size = 32,
            shuffle=False,
            collate_fn = collate_eval_fn,
            )
    
    train(model, loader, device, weights, epochs, test_loader=test_loader)



if __name__ == "__main__":
    main()
