import torch
import torch.nn as nn


def gaussian_focal_loss(
        pred : torch.Tensor,
        gt : torch.Tensor,
        alpha : float = 2.0,
        beta : float = 4.0,
        ) -> torch.Tensor:
    '''
    Gaussian Focal Loss of CenterNet
    
    gt : Gaussian-rendered heatmap [B, C, H, W]
    pred : sigmoid output [B, C, H, W]
    '''
    
    pos_mask = gt.eq(1.0).float()
    neg_mask = gt.lt(1.0).float()

    pos_loss = (
            -(1.0 - pred).pow(alpha)
            * torch.log(pred.clamp(min=1e-12))
            * pos_mask
            )
    
    neg_loss = (
            -(pred).pow(alpha)
            * (1. - gt).pow(beta)
            * torch.log((1. - pred).clamp(min=1e-12))
            * neg_mask
            )
    
    num_pos = pos_mask.sum().clamp(min=1.)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos

class MultiViewDetectorLoss(nn.Module):
    def __init__(self, w_heatmap : float = 1.):
        super().__init__()
        self.w_hm = w_heatmap

    def forward(
            self,
            pred : torch.Tensor,
            gt_heatmap : torch.Tensor,
            center_mask : torch.Tensor,
            ) -> torch.Tensor:
        hm_loss = gaussian_focal_loss(pred, gt_heatmap)
        return hm_loss
