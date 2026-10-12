"""Official RT-DETR set criterion plus box-supervised focal Tversky masks."""

import math

import torch
from torch import nn

from .vendor.matcher import HungarianMatcher
from .vendor.rtdetr_criterion import SetCriterion


def rasterize_boxes(boxes, shape):
    """Rasterize normalized cxcywh boxes by pixel overlap, retaining tiny boxes."""
    height, width = shape
    mask = boxes.new_zeros((height, width))
    for cx, cy, bw, bh in boxes.detach().tolist():
        if bw <= 0 or bh <= 0:
            continue
        x0 = max(0, min(width, math.floor((cx - bw / 2) * width)))
        x1 = max(0, min(width, math.ceil((cx + bw / 2) * width)))
        y0 = max(0, min(height, math.floor((cy - bh / 2) * height)))
        y1 = max(0, min(height, math.ceil((cy + bh / 2) * height)))
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = 1
    return mask


class DyFCLTCriterion(nn.Module):
    def __init__(self, num_classes, mask_weight=4.0, alpha=0.3, beta=0.7,
                 gamma=0.75, eps=1.0):
        super().__init__()
        if num_classes < 1 or mask_weight < 0 or alpha < 0 or beta < 0 or gamma <= 0:
            raise ValueError('invalid criterion parameters')
        matcher = HungarianMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                                   use_focal_loss=True, alpha=0.25, gamma=2.0)
        self.detector = SetCriterion(
            matcher, {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
            ['vfl', 'boxes'], alpha=0.75, gamma=2.0, num_classes=num_classes)
        self.mask_weight = mask_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, outputs, targets):
        losses = self.detector(outputs, targets)
        mask_losses = []
        for mask_logits in outputs.get('mask_logits', []):
            if mask_logits.ndim != 4 or mask_logits.shape[1] != 1:
                raise ValueError('each mask_logits tensor must have shape [B,1,H,W]')
            height, width = mask_logits.shape[-2:]
            foreground = torch.stack([
                rasterize_boxes(t['boxes'], (height, width)) for t in targets
            ]).to(device=mask_logits.device, dtype=mask_logits.dtype).unsqueeze(1)
            probability = mask_logits.float().sigmoid()
            tp = (probability * foreground).sum(dim=(1, 2, 3))
            fp = (probability * (1 - foreground)).sum(dim=(1, 2, 3))
            fn = ((1 - probability) * foreground).sum(dim=(1, 2, 3))
            tversky = (tp + self.eps) / (
                tp + self.alpha * fp + self.beta * fn + self.eps)
            mask_losses.append((1 - tversky).clamp_min(1e-8).pow(self.gamma).mean())
        if mask_losses:
            losses['loss_mask'] = self.mask_weight * torch.stack(mask_losses).mean()
        return losses


def build_criterion(num_classes, mask_weight=4.0, alpha=0.3, beta=0.7,
                    gamma=0.75, eps=1.0):
    return DyFCLTCriterion(num_classes, mask_weight, alpha, beta, gamma, eps)
