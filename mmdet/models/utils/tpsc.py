"""Tiny-aware Prototype Semantic Calibration primitives."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_tpsc_gaussian_targets(gt_bboxes, padded_size, feat_size, device,
                                valid_mask=None):
    """Build max-merged Gaussian foreground targets in Thermal coordinates."""
    input_h, input_w = padded_size
    feat_h, feat_w = feat_size
    if input_h <= 0 or input_w <= 0 or feat_h <= 0 or feat_w <= 0:
        raise ValueError('padded_size and feat_size must be positive')

    target = torch.zeros(
        len(gt_bboxes), 1, feat_h, feat_w, device=device,
        dtype=torch.float32)
    scale_x = feat_w / float(input_w)
    scale_y = feat_h / float(input_h)
    for batch_index, boxes in enumerate(gt_bboxes):
        if boxes is None or not len(boxes):
            continue
        for x1, y1, x2, y2 in boxes.detach().cpu().tolist():
            width = max((float(x2) - float(x1)) * scale_x, 0.0)
            height = max((float(y2) - float(y1)) * scale_y, 0.0)
            center_x = 0.5 * (float(x1) + float(x2)) * scale_x
            center_y = 0.5 * (float(y1) + float(y2)) * scale_y
            radius = max(1, int(math.ceil(0.5 * math.sqrt(width * height))))
            sigma = (2.0 * radius + 1.0) / 6.0
            center_col = int(math.floor(center_x))
            center_row = int(math.floor(center_y))
            col0 = max(0, center_col - radius)
            col1 = min(feat_w, center_col + radius + 1)
            row0 = max(0, center_row - radius)
            row1 = min(feat_h, center_row + radius + 1)
            if col0 >= col1 or row0 >= row1:
                continue
            rows = torch.arange(
                row0, row1, device=device, dtype=torch.float32)
            cols = torch.arange(
                col0, col1, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
            gaussian = torch.exp(-(
                (grid_x - center_x).square() +
                (grid_y - center_y).square()) / (2.0 * sigma * sigma))
            gaussian = gaussian / gaussian.max().clamp_min(1e-12)
            current = target[batch_index, 0, row0:row1, col0:col1]
            target[batch_index, 0, row0:row1, col0:col1] = torch.maximum(
                current, gaussian)

    if valid_mask is not None:
        if tuple(valid_mask.shape) != tuple(target.shape):
            raise ValueError(
                'valid_mask must have the same shape as Gaussian targets')
        target = target * valid_mask.to(target.dtype)
    return target


def valid_average_pool(feature, valid_mask, kernel_size=3):
    """Average a BCHW tensor without including invalid or padded cells."""
    if feature.ndim != 4:
        raise ValueError('feature must be a BCHW tensor')
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('kernel_size must be a positive odd integer')
    expected_mask = (feature.shape[0], 1, *feature.shape[-2:])
    if tuple(valid_mask.shape) != expected_mask:
        raise ValueError('valid_mask must have shape Bx1xHxW')
    valid = valid_mask.to(feature.dtype)
    padding = kernel_size // 2
    feature_mean = F.avg_pool2d(
        feature * valid, kernel_size, stride=1, padding=padding)
    valid_mean = F.avg_pool2d(
        valid, kernel_size, stride=1, padding=padding)
    return torch.where(
        valid_mean > 0,
        feature_mean / valid_mean.clamp_min(1e-12),
        torch.zeros_like(feature_mean))


class TPSCDescriptor(nn.Module):
    """Create detail-preserving P3 and semantic P4 prototype descriptors."""

    def __init__(self, channels, prototype_dim):
        super().__init__()
        if channels < 1 or prototype_dim < 1:
            raise ValueError('descriptor dimensions must be positive')
        self.channels = int(channels)
        self.prototype_dim = int(prototype_dim)
        self.p4_projection = nn.Conv2d(
            self.channels, self.prototype_dim, 1, bias=False)
        self.p3_merge = nn.Conv2d(
            2 * self.channels + self.prototype_dim,
            self.prototype_dim, 1, bias=False)

    @staticmethod
    def _validate(feature, valid_mask, channels, name):
        if feature.ndim != 4 or feature.shape[1] != channels:
            raise ValueError(
                '{} must be a BCHW tensor with descriptor channels'.format(
                    name))
        expected = (feature.shape[0], 1, *feature.shape[-2:])
        if tuple(valid_mask.shape) != expected:
            raise ValueError('{} valid mask has the wrong shape'.format(name))

    def forward(self, p3, p4, valid_p3, valid_p4):
        self._validate(p3, valid_p3, self.channels, 'P3')
        self._validate(p4, valid_p4, self.channels, 'P4')
        if p3.shape[0] != p4.shape[0]:
            raise ValueError('P3 and P4 batch sizes must match')

        valid3 = valid_p3.to(p3.dtype)
        valid4 = valid_p4.to(p4.dtype)
        detail = p3 - valid_average_pool(p3, valid_p3, kernel_size=3)
        p4_descriptor = self.p4_projection(p4 * valid4)
        semantic = F.interpolate(
            p4_descriptor, size=p3.shape[-2:], mode='bilinear',
            align_corners=False)
        p3_descriptor = self.p3_merge(torch.cat((
            p3, detail, semantic), dim=1))
        return p3_descriptor * valid3, p4_descriptor * valid4
