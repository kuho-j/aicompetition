import torch
import torch.nn.functional as F


@torch.no_grad()
def decode_predictions(
        heatmap : torch.Tensor,
        topk : int = 100,
        score_threshold : float = 0.3,
        ) -> list[dict[str, torch.Tensor]]:
    '''
    heatmap NMS -> extrack top-k peak -> get coordinates

    return : list of dict per batch
        'scores' : [k]    ... confidence
        'classes' : [k]   ... class index
        'centers' : [k, 2] ... normalized coordinates
    '''

    B, C, H, W = heatmap.shape
    results = []

    # Heatmap NMS
    heatmap_nms = heatmap * (
        F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1) == heatmap
        ).float()

    for b in range(B):

        scores_map, cls_map = heatmap_nms[b].max(dim=0) # [H, W]

        # top-k after flatten
        flat_scores = scores_map.flatten()
        topk_val, topk_idx = flat_scores.topk(min(topk, flat_scores.numel()))

        # threshold filter
        keep = topk_val > score_threshold
        topk_val = topk_val[keep]
        topk_idx = topk_idx[keep]

        if topk_idx.numel() == 0:
            results.append({
                'scores' : torch.empty(0),
                'classes' : torch.empty(0, dtype=torch.long),
                'centers' : torch.empty(0, 2)
                })
            continue

        # coordinates of grid
        ys = (topk_idx // W).float()
        xs = (topk_idx % W).float()

        # normalize
        centers = torch.stack([xs / W, ys / H], dim=1)
        cls = cls_map.flatten()[topk_idx]

        results.append({
            'scores' : topk_val.cpu(),
            'classes' : cls.cpu(),
            'centers' : centers.cpu(),
            })

    return results
