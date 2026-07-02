import os
import csv
import argparse
import numpy as np
import cv2
import torch

from src.model.multiview_detection import MultiViewDetector
from src.predict import decode_predictions


names = [
    'aunt_jemima_original_syrup', 'band_aid_clear_strips', 'bumblebee_albacore', 'cholula_chipotle_hot_sauce', 'crayola_24_crayons', 'hersheys_cocoa',
    'honey_bunches_of_oats_honey_rasted', 'honey_bunches_of_oats_with_almonds', 'hunts_sauce', 'listerine_green', 'mahatma_rice', 'white_rain_body_wash', 'pringles_bbq',
    'cheeze_it', 'hersheys_bar', 'redbull', 'mom_to_mom_sweet_potato_corn_apple', 'a1_steak_sauce', 'jif_creamy_peanut_butter', 'cinnamon_toast_crunch', 'arm_hammer_baking_soda',
    'dr_pepper', 'haribo_gold_bears_gummi_candy', 'bulls_eye_bbq_sauce_original', 'reeses_pieces', 'clif_crunch_peanut_butter', 'mom_to_mom_butternut_squash_pear',
    'pop_tararts_strawberry', 'quaker_big_chewy_chocolate_chip', 'spam', 'coffee_mate_french_vanilla', 'pepperidge_farm_milk_chocolate_macadamia_cookies',
    'kitkat_king_size', 'snickers', 'toblerone_milk_chocolate', 'clif_z_bar_chocolate_chip', 'nature_valley_crunch_oats_n_honey', 'ritz_crackers', 'palmolive_orange',
    'crystal_hot_sauce', 'tapatio_hot_sauce', 'nabisco_nilla_wafers', 'pepperidge_farm_milano_cookies_double_chocolate', 'campbells_chicken_noodle_soup', 'frappuccino_coffee',
    'chewy_dips_chocolate_chip', 'chewy_dips_peanut_butter', 'nature_valley_fruit_and_nut', 'cheerios', 'lindt_excellence_cocoa_dark_chocolate', 'hersheys_symphony',
    'campbells_chunky_classic_chicken_noodle', 'martinellis_apple_juice', 'dove_pink', 'dove_white', 'david_sunflower_seeds', 'monster_energy', 'act_ii_butter_lovers_popcorn',
    'coca_cola_glass_bottle', 'twix'
    ]

prices = np.array([
    3.87, 2.47, 1.58, 5.99, 1.48, 5.22, 6.99, 4.93, 1.54, 8.38,
    2.48, 1.00, 3.49, 3.99, 1.32, 2.12, 3.29, 4.98, 3.27, 3.84,
    1.00, 1.98, 2.49, 3.79, 1.99, 4.45, 3.39, 2.99, 4.12, 3.99,
    3.77, 4.25, 2.19, 1.32, 3.58, 3.95, 3.79, 3.65, 2.75, 1.38,
    1.15, 4.25, 1.49, 3.89, 3.12, 2.78, 2.69, 4.55, 4.92, 2.98,
    3.44, 4.28, 3.99, 2.99, 2.35, 2.35, 1.99, 2.89, 1.75, 1.99,
    ])

def parse_arg():
    parser = argparse.ArgumentParser(description='Read 5 video files and write a csv file')
    parser.add_argument(
        '--weights',
        type = str,
    )
    parser.add_argument(
        '--cam1',
        default=None,
        help='Path of a video file of center'
    )
    parser.add_argument(
        '--cam2',
        default=None,
        help='Path of a video file of left behind'
    )
    parser.add_argument(
        '--cam3',
        default=None,
        help='Path of a video file of left front'
    )
    parser.add_argument(
        '--cam4',
        default=None,
        help='Path of a video file of right behind'
    )
    parser.add_argument(
        '--cam5',
        default=None,
        help='Path of a video file of right front'
    )

    return parser.parse_args()

def load_model(
    model : MultiViewDetector,
    path : str,
    device : torch.device
    ) -> None:

    if not os.path.isfile(path):
        raise FileNotFoundError(f'model not found: {path}')
    
    loaded = torch.load(path, map_location=device)
    
    if isinstance(loaded, dict):
        if 'model_state_dict' in loaded:
            model.load_state_dict(loaded['model_state_dict'])
        else:
            model.load_state_dict(loaded)
    else:
        raise TypeError(f'unsupported file format: {loaded.type()} in {path}')
    
    print(f'model loaded: {path}')


def format_data(img_list : list[torch.Tensor]) -> torch.Tensor:
    '''
    input : list of images, each [H, W, C] BGR or [C, H, W] RGB
    output : model input [1, N_views, C, H, W]
    '''

    if len(img_list) == 0:
        raise ValueError('img_list is empty')

    formatted = []
    for img in img_list:
        if not torch.is_tensor(img):
            raise TypeError(f'unsupported image type: {type(img)}')

        if img.ndim != 3:
            raise ValueError(f'each image tensor must be 3-dimensional, got shape {tuple(img.shape)}')

        img = img.detach().float()

        # HWC BGR -> CHW RGB. CHW input is assumed to already be RGB.
        if img.shape[0] not in (1, 3) and img.shape[-1] in (1, 3):
            if img.shape[-1] == 3:
                img = img[..., [2, 1, 0]]
            img = img.permute(2, 0, 1)

        if img.max().item() > 1.0:
            img = img / 255.0

        formatted.append(img)

    return torch.stack(formatted).unsqueeze(0)


@torch.no_grad()
def evaluate(
    model : MultiViewDetector,
    img : torch.Tensor,
    device : torch.device,
    num_classes : int = 60,
    topk : int = 100,
    score_threshold : float = 0.3,
) -> np.ndarray:
    '''
    input : img [1, N_views, C, H, W]
    output : numbers of each items
    '''

    if img.ndim != 5:
        raise ValueError(f'img must have shape [B, N_views, C, H, W], got {tuple(img.shape)}')

    was_training = model.training
    model.eval()

    with torch.no_grad():
        pred_heatmap = model(img.to(device))
        decoded = decode_predictions(
            pred_heatmap,
            topk=topk,
            score_threshold=score_threshold,
        )

    if was_training:
        model.train()

    classes = decoded[0]['classes'].numpy()
    return np.bincount(classes, minlength=num_classes)[:num_classes]



def main(
    model_path : str,
    cam1 : str,
    cam2 : str,
    cam3 : str,
    cam4 : str,
    cam5 : str,
    num_classes : int = 60,
    ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiViewDetector().to(device)
    load_model(model, model_path, device)
    model.eval()

    # cam1 / cam2 / cam3 / cam4 / cam5
    # 중앙 / 좌측뒤 / 좌측앞 / 우측뒤 / 우측앞
    video_paths = [cam1, cam2, cam3, cam4, cam5]
    caps = [cv2.VideoCapture(path) for path in video_paths]


    # check if 5 videos are opened
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            raise ValueError(f'can not open video of cam{i+1}.\npath: {video_paths[i]}')

    current_item = np.array([0 for _ in range(num_classes)])
    prev_item = np.array([0 for _ in range(num_classes)])
    event_num = 0
    events = [['Product Name', 'Event Number', 'Purchase / Return'] + names + ['Total Inventory Value']]

    while True:
        img_frames = []
        all_success = True
        
        for cap in caps:
            ret, frame = cap.read()
            
            # 프레임이 끝나거나 읽기에 실패했을 때
            if not ret:
                all_success = False
                break
            
            img_frames.append(torch.from_numpy(frame))
        
        if not all_success:
            print('video ends or can not read the frame')
            break

        img = format_data(img_frames)
        current_item = evaluate(model, img, device, num_classes)

        if np.array_equal(current_item, prev_item):
            continue

        # if current != prev
        if event_num == 0:
            # initial setting
            event_num = 1
        else:
            changed_item = current_item - prev_item
            for class_idx, changed in enumerate(changed_item):
                if changed == 0:
                    continue
                
                event = [
                    names[class_idx],
                    event_num,
                    'Purchase' if changed < 0 else 'Return',
                ] + list(current_item) + [sum(current_item * prices)]

                events.append(event)
        prev_item = current_item

    for cap in caps:
        cap.release()
    
    with open('result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(events)

if __name__ == '__main__':
    args = parse_arg()
    main(
        model_path=args.weights,
        cam1=args.cam1,
        cam2=args.cam2,
        cam3=args.cam3,
        cam4=args.cam4,
        cam5=args.cam5,
    )