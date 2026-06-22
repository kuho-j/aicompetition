import torch

def render_gaussian_heatmap(
        centers : torch.Tensor, # [N, 2], normalized coordinates (x,y)
        classes : torch.Tensor, # [N], class index
        num_classes : int,
        output_h : int,
        output_w : int,
        sigma : float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    make GT heatmap, center_mask

    returns:
        heatmap : [num_classes, H, W]
        center_mask : [1, H, W]
    '''

    heatmap=torch.zeros(num_classes, output_h, output_w)
    center_mask = torch.zeros(1, output_h, output_w)

    ys_grid = torch.arange(output_h, dtype=torch.float32)
    xs_grid = torch.arange(output_w, dtype=torch.float32)

    for (cx, cy), cls_idx in zip(centers, classes):
        # convert into the pixel coordinates
        px = cx.item() * output_w
        py = cy.item() * output_h

        ix, iy = int(px), int(py)

        if not (0 <= ix < output_w and 0 <= iy < output_h):
            continue

        # rendering 2D Gaussian
        x_dist = (xs_grid - px)**2
        y_dist = (ys_grid - py)**2

        gaussian = torch.exp(
                -(y_dist.unsqueeze(1) + x_dist.unsqueeze(0)) / (2 * sigma ** 2)
            ) # [H, W]

        # element-wise max
        heatmap[cls_idx.long()] = torch.maximum(heatmap[cls_idx.long()], gaussian)

        center_mask[0, iy, ix] = 1.0

    return heatmap, center_mask
