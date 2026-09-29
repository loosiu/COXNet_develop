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


class _ChannelLayerNorm(nn.Module):
    """Apply LayerNorm over channels independently at every spatial point."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, feature):
        return self.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _masked_spatial_softmax(logits, valid_mask):
    """Normalize each slot over valid positions and keep all-invalid rows zero."""
    batch, slots, height, width = logits.shape
    expected = (batch, 1, height, width)
    if tuple(valid_mask.shape) != expected:
        raise ValueError('valid_mask must have shape Bx1xHxW')
    valid = valid_mask.to(dtype=torch.bool).expand(-1, slots, -1, -1)
    flat_logits = logits.flatten(2)
    flat_valid = valid.flatten(2)
    masked_logits = flat_logits.masked_fill(
        ~flat_valid, torch.finfo(flat_logits.dtype).min)
    attention = torch.softmax(masked_logits, dim=-1) * flat_valid.to(
        flat_logits.dtype)
    attention = attention / attention.sum(
        dim=-1, keepdim=True).clamp_min(1e-12)
    return attention.reshape(batch, slots, height, width)


class SharedSlotPrototypeExtractor(nn.Module):
    """Extract modality-specific prototypes using shared semantic queries."""

    def __init__(self, prototype_dim=64, num_slots=8):
        super().__init__()
        if prototype_dim < 1 or num_slots < 1:
            raise ValueError('prototype_dim and num_slots must be positive')
        self.prototype_dim = int(prototype_dim)
        self.num_slots = int(num_slots)
        self.rgb_projection = nn.Sequential(
            nn.Conv2d(self.prototype_dim, self.prototype_dim, 1, bias=False),
            _ChannelLayerNorm(self.prototype_dim))
        self.thermal_projection = nn.Sequential(
            nn.Conv2d(self.prototype_dim, self.prototype_dim, 1, bias=False),
            _ChannelLayerNorm(self.prototype_dim))
        self.slot_queries = nn.Parameter(torch.empty(
            self.num_slots, self.prototype_dim))
        nn.init.normal_(self.slot_queries, std=self.prototype_dim ** -0.5)

    def _extract(self, feature, projection, valid_mask):
        if feature.ndim != 4 or feature.shape[1] != self.prototype_dim:
            raise ValueError(
                'slot input must be BCHW with prototype_dim channels')
        embedded = projection(feature)
        logits = torch.einsum(
            'kd,bdhw->bkhw', self.slot_queries, embedded)
        logits = logits / math.sqrt(self.prototype_dim)
        attention = _masked_spatial_softmax(logits, valid_mask)
        prototypes = torch.einsum('bkhw,bdhw->bkd', attention, embedded)
        return prototypes, attention

    def forward(self, rgb, thermal, valid_mask):
        if tuple(rgb.shape) != tuple(thermal.shape):
            raise ValueError('RGB and Thermal slot inputs must have equal shapes')
        rgb_prototypes, rgb_attention = self._extract(
            rgb, self.rgb_projection, valid_mask)
        thermal_prototypes, thermal_attention = self._extract(
            thermal, self.thermal_projection, valid_mask)
        return dict(
            rgb_prototypes=rgb_prototypes,
            thermal_prototypes=thermal_prototypes,
            rgb_attention=rgb_attention,
            thermal_attention=thermal_attention)


def prototype_coverage_loss(thermal_attention, target, valid_mask):
    """Symmetric KL between mean Thermal slot coverage and GT Gaussians."""
    if thermal_attention.ndim != 4:
        raise ValueError('thermal_attention must have shape BxKxHxW')
    expected = (
        thermal_attention.shape[0], 1, *thermal_attention.shape[-2:])
    if tuple(target.shape) != expected or tuple(valid_mask.shape) != expected:
        raise ValueError('target and valid_mask must have shape Bx1xHxW')

    valid = valid_mask.to(thermal_attention.dtype)
    coverage = thermal_attention.mean(dim=1, keepdim=True) * valid
    target = target.to(thermal_attention.dtype) * valid
    losses = []
    eps = torch.finfo(thermal_attention.dtype).eps
    for sample_coverage, sample_target in zip(coverage, target):
        target_mass = sample_target.sum()
        coverage_mass = sample_coverage.sum()
        if target_mass.detach().item() <= 0 or coverage_mass.detach().item() <= 0:
            continue
        probability = sample_coverage.flatten() / coverage_mass
        target_probability = sample_target.flatten() / target_mass
        probability = probability.clamp_min(eps)
        target_probability = target_probability.clamp_min(eps)
        probability = probability / probability.sum()
        target_probability = target_probability / target_probability.sum()
        kl_tc = torch.sum(target_probability * (
            target_probability.log() - probability.log()))
        kl_ct = torch.sum(probability * (
            probability.log() - target_probability.log()))
        losses.append(0.5 * (kl_tc + kl_ct))
    if not losses:
        return thermal_attention.sum() * 0.0
    return torch.stack(losses).mean()


def prototype_diversity_loss(thermal_attention, valid_mask):
    """Penalize cosine overlap between different Thermal slot maps."""
    if thermal_attention.ndim != 4:
        raise ValueError('thermal_attention must have shape BxKxHxW')
    batch, slots, height, width = thermal_attention.shape
    expected = (batch, 1, height, width)
    if tuple(valid_mask.shape) != expected:
        raise ValueError('valid_mask must have shape Bx1xHxW')
    if slots < 2:
        return thermal_attention.sum() * 0.0

    losses = []
    for sample_attention, sample_valid in zip(
            thermal_attention, valid_mask):
        if not sample_valid.any():
            continue
        masked = sample_attention * sample_valid.to(
            sample_attention.dtype)
        flattened = F.normalize(masked.flatten(1), dim=-1, eps=1e-12)
        cosine = flattened @ flattened.transpose(0, 1)
        off_diagonal = ~torch.eye(
            slots, device=cosine.device, dtype=torch.bool)
        losses.append(cosine[off_diagonal].mean())
    if not losses:
        return thermal_attention.sum() * 0.0
    return torch.stack(losses).mean()
