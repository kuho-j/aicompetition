import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_focal_loss(
        pred : torch.Tensor,
        gt : torch.Tensor,
        alpha : float = 2.0,
        beta : float = 4.0,
        pos_threshold : float = 0.8,
        ) -> torch.Tensor:
    '''
    Gaussian Focal Loss of CenterNet
    
    gt : Gaussian-rendered heatmap [B, C, H, W]
    pred : sigmoid output [B, C, H, W]
    '''
    
    pos_mask = gt.ge(pos_threshold).float()
    neg_mask = gt.lt(pos_threshold).float()

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


def soft_dice_loss(
        pred : torch.Tensor,
        gt : torch.Tensor,
        eps : float = 1e-6,
        ) -> torch.Tensor:
    pred = pred.flatten(1)
    gt = gt.flatten(1)

    intersection = (pred * gt).sum(dim=1)
    denominator = pred.sum(dim=1) + gt.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def shape_heatmap_loss(
        logits : torch.Tensor,
        gt : torch.Tensor,
        bce_weight : float = 1.0,
        dice_weight : float = 1.0,
        pos_weight : float | None = None,
        ) -> torch.Tensor:
    '''
    Dense heatmap loss for shape/edge-like targets.

    logits : raw model output [B, C, H, W]
    gt     : soft target heatmap [B, C, H, W], expected in [0, 1]
    '''

    gt = gt.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    weight = None
    if pos_weight is not None:
        weight = torch.where(gt > 0, gt.new_full((), pos_weight), gt.new_ones(()))

    bce = F.binary_cross_entropy_with_logits(
            logits,
            gt,
            weight=weight,
            )
    dice = soft_dice_loss(logits.sigmoid(), gt)
    return bce_weight * bce + dice_weight * dice

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
