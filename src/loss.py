import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_focal_loss(
        pred : torch.Tensor,
        gt : torch.Tensor,
        alpha : float = 2.0,
        beta : float = 4.0,
        false_positive_weight : float = 0.05,
        ) -> torch.Tensor:
    '''
    Gaussian Focal Loss of CenterNet
    
    gt : Gaussian-rendered heatmap [B, C, H, W]
    pred : sigmoid output [B, C, H, W]
    '''
    
    pos_mask = gt.eq(1.0).to(dtype=pred.dtype)
    neg_mask = gt.lt(1.0).to(dtype=pred.dtype)

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
    return (pos_loss.sum() + false_positive_weight * neg_loss.sum()) / num_pos


def center_displacement_loss(
        logits : torch.Tensor,
        center_mask : torch.Tensor,
        radius : int = 4,
        temperature : float = 0.25,
        ) -> torch.Tensor:
    '''
    Penalize local center shifts around labeled center pixels.

    For each labeled center, a small local soft-argmax is computed on the matching
    class channel. The loss is intentionally local; missed detections are handled
    by the positive focal term.
    '''
    if radius <= 0:
        return logits.new_zeros(())

    center_mask = center_mask.to(device=logits.device, dtype=torch.bool)
    positives = center_mask.nonzero(as_tuple=False)
    if positives.numel() == 0:
        return logits.new_zeros(())

    _, _, height, width = logits.shape
    losses = []
    temp = max(temperature, 1e-6)

    for batch_idx, class_idx, y, x in positives:
        y0 = max(int(y.item()) - radius, 0)
        y1 = min(int(y.item()) + radius + 1, height)
        x0 = max(int(x.item()) - radius, 0)
        x1 = min(int(x.item()) + radius + 1, width)

        crop = logits[batch_idx, class_idx, y0:y1, x0:x1]
        weights = (crop.flatten() / temp).softmax(dim=0).view_as(crop)

        ys = torch.arange(y0, y1, device=logits.device, dtype=logits.dtype)
        xs = torch.arange(x0, x1, device=logits.device, dtype=logits.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        pred_x = (weights * xx).sum()
        pred_y = (weights * yy).sum()
        target = logits.new_tensor([float(x.item()), float(y.item())])
        pred = torch.stack([pred_x, pred_y])
        losses.append(F.smooth_l1_loss(pred / radius, target / radius, reduction="sum"))

    return torch.stack(losses).mean()


def centernet_detection_loss(
        logits : torch.Tensor,
        gt : torch.Tensor,
        center_mask : torch.Tensor | None = None,
        pred_offset : torch.Tensor | None = None,
        target_offset : torch.Tensor | None = None,
        alpha : float = 2.0,
        beta : float = 4.0,
        miss_weight : float = 1.0,
        false_positive_weight : float = 0.05,
        empty_confidence_weight : float = 0.01,
        displacement_weight : float = 0.25,
        displacement_radius : int = 4,
        offset_weight : float = 1.0,
        ) -> torch.Tensor:
    '''
    Center-focused detection loss for BEV class heatmaps.

    Positive samples are only pixels whose target value is exactly 1.0. Negative
    loss is deliberately weak because missing annotations should not dominate
    training. A small empty-confidence term also suppresses confident detections
    on pixels with no target response at all.
    '''

    gt = gt.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    if center_mask is None:
        center_mask = gt.eq(1.0).to(dtype=logits.dtype)
    else:
        center_mask = center_mask.to(device=logits.device, dtype=logits.dtype)
        if center_mask.shape[1] == 1 and logits.shape[1] != 1:
            center_mask = center_mask.expand(-1, logits.shape[1], -1, -1)

    pred = logits.sigmoid()
    pos_mask = center_mask
    neg_mask = gt.lt(1.0).to(dtype=logits.dtype)

    miss_loss = (
            -(1.0 - pred).pow(alpha)
            * torch.log(pred.clamp(min=1e-12))
            * pos_mask
            )

    false_positive_loss = (
            -pred.pow(alpha)
            * (1.0 - gt).pow(beta)
            * torch.log((1.0 - pred).clamp(min=1e-12))
            * neg_mask
            )
    empty_mask = gt.eq(0.0).to(dtype=logits.dtype)
    empty_confidence_loss = pred.pow(2.0) * empty_mask

    num_pos = pos_mask.sum().clamp(min=1.0)
    num_empty = empty_mask.sum().clamp(min=1.0)
    heatmap_loss = (
            miss_weight * miss_loss.sum()
            + false_positive_weight * false_positive_loss.sum()
            ) / num_pos
    heatmap_loss = heatmap_loss + empty_confidence_weight * (
            empty_confidence_loss.sum() / num_empty
            )

    shift_loss = center_displacement_loss(
            logits,
            pos_mask,
            radius=displacement_radius,
            )
    offset_loss = logits.new_zeros(())
    if pred_offset is not None and target_offset is not None and offset_weight > 0:
        target_offset = target_offset.to(device=logits.device, dtype=logits.dtype)
        offset_loss = centernet_offset_loss(pred_offset, target_offset, pos_mask)

    return heatmap_loss + displacement_weight * shift_loss + offset_weight * offset_loss


def centernet_offset_loss(
        pred_offset : torch.Tensor,
        target_offset : torch.Tensor,
        center_mask : torch.Tensor,
        ) -> torch.Tensor:
    if center_mask.shape[1] != 1:
        center_mask = center_mask.amax(dim=1, keepdim=True)

    center_mask = center_mask.to(device=pred_offset.device, dtype=pred_offset.dtype)
    target_offset = target_offset.to(device=pred_offset.device, dtype=pred_offset.dtype)
    loss = F.l1_loss(pred_offset, target_offset, reduction="none") * center_mask
    return loss.sum() / (center_mask.sum().clamp(min=1.0) * pred_offset.shape[1])


class CenterNetDetectionLoss(nn.Module):
    def __init__(
            self,
            miss_weight : float = 1.0,
            false_positive_weight : float = 0.05,
            empty_confidence_weight : float = 0.01,
            displacement_weight : float = 0.25,
            displacement_radius : int = 4,
            offset_weight : float = 1.0,
            ):
        super().__init__()
        self.miss_weight = miss_weight
        self.false_positive_weight = false_positive_weight
        self.empty_confidence_weight = empty_confidence_weight
        self.displacement_weight = displacement_weight
        self.displacement_radius = displacement_radius
        self.offset_weight = offset_weight

    def forward(
            self,
            logits : torch.Tensor,
            gt_heatmap : torch.Tensor,
            center_mask : torch.Tensor | None = None,
            pred_offset : torch.Tensor | None = None,
            target_offset : torch.Tensor | None = None,
            ) -> torch.Tensor:
        return centernet_detection_loss(
                logits,
                gt_heatmap,
                center_mask=center_mask,
                pred_offset=pred_offset,
                target_offset=target_offset,
                miss_weight=self.miss_weight,
                false_positive_weight=self.false_positive_weight,
                empty_confidence_weight=self.empty_confidence_weight,
                displacement_weight=self.displacement_weight,
                displacement_radius=self.displacement_radius,
                offset_weight=self.offset_weight,
                )


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

