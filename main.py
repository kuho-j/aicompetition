import argparse
import torch
from data.make_filename import make_filename
from src.model.multiview_detection import MultiViewDetector
from src.dataset import format_data
from src.train import train_k_fold

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
            num_views = 5,
            num_classes = 60,
            img_channels = 3,
            fpn_out_channels = 256,
            backbone_width = 0.25,
            backbone_depth = 0.33,
            attn_heads = 4,
            spatial_ds = 2,
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



if __name__ == "__main__":
    args = parse_args()
    main(
            epochs=args.epochs,
            weights=args.weights,
            k_folds=args.k_folds,
            fold_interval=args.fold_interval,
            )
