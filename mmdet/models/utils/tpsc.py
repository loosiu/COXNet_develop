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


class PrototypeRelationBlock(nn.Module):
    """A small pre-norm Transformer block over prototype nodes."""

    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        if dim < 1 or num_heads < 1 or dim % num_heads:
            raise ValueError('dim must be positive and divisible by num_heads')
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, nodes):
        normalized = self.norm1(nodes)
        update, weights = self.attention(
            normalized, normalized, normalized, need_weights=True,
            average_attn_weights=False)
        nodes = nodes + update
        nodes = nodes + self.mlp(self.norm2(nodes))
        return nodes, weights


def _attention_entropy(attention):
    probability = attention.flatten(2)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
    return entropy.mean()


def _prototype_pairwise_cosine(prototypes):
    slots = prototypes.shape[1]
    if slots < 2:
        return prototypes.sum() * 0.0
    normalized = F.normalize(prototypes, dim=-1, eps=1e-12)
    cosine = normalized @ normalized.transpose(1, 2)
    off_diagonal = ~torch.eye(
        slots, device=cosine.device, dtype=torch.bool)
    return cosine[:, off_diagonal].mean()


def _prototype_effective_rank(prototypes):
    ranks = []
    for sample in prototypes:
        centered = sample - sample.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        probability = singular / singular.sum().clamp_min(1e-12)
        ranks.append(torch.exp(-(
            probability * probability.clamp_min(1e-12).log()).sum()))
    return torch.stack(ranks).mean()


class TinyAwarePrototypeSemanticCalibration(nn.Module):
    """Use P3/P4 semantic prototypes to calibrate only RGB channels."""

    def __init__(self, channels, prototype_dim=64, num_slots=8,
                 num_heads=4, relation_depth=1, modulation_bound=0.1,
                 init_std=1e-2, use_relation=True):
        super().__init__()
        if channels < 1 or relation_depth < 0:
            raise ValueError('channels must be positive and depth non-negative')
        if not 0 < modulation_bound <= 1:
            raise ValueError('modulation_bound must be in (0, 1]')
        self.channels = int(channels)
        self.prototype_dim = int(prototype_dim)
        self.num_slots = int(num_slots)
        self.modulation_bound = float(modulation_bound)
        self.use_relation = bool(use_relation)

        self.rgb_descriptor = TPSCDescriptor(channels, prototype_dim)
        self.thermal_descriptor = TPSCDescriptor(channels, prototype_dim)
        self.slot_extractors = nn.ModuleList([
            SharedSlotPrototypeExtractor(prototype_dim, num_slots)
            for _ in range(2)
        ])
        self.relation_blocks = nn.ModuleList([
            PrototypeRelationBlock(prototype_dim, num_heads)
            for _ in range(relation_depth)
        ])
        self.conditioners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * prototype_dim, prototype_dim),
                nn.GELU(),
                nn.Linear(prototype_dim, prototype_dim))
            for _ in range(2)
        ])
        self.modulation_heads = nn.ModuleList([
            nn.Linear(num_slots * prototype_dim, channels)
            for _ in range(2)
        ])
        for head in self.modulation_heads:
            nn.init.normal_(head.weight, std=init_std)
            nn.init.zeros_(head.bias)

        self._gradient_norms = {
            name: 0.0 for name in (
                'rgb_projection', 'thermal_projection', 'relation',
                'conditioner', 'modulation')
        }
        self._register_gradient_diagnostics()

    def _cache_gradient(self, name):
        def hook(gradient):
            self._gradient_norms[name] = float(
                gradient.detach().float().norm().item())
            return gradient
        return hook

    def _register_gradient_diagnostics(self):
        parameters = {
            'rgb_projection':
                self.slot_extractors[0].rgb_projection[0].weight,
            'thermal_projection':
                self.slot_extractors[0].thermal_projection[0].weight,
            'conditioner': self.conditioners[0][0].weight,
            'modulation': self.modulation_heads[0].weight,
        }
        if self.relation_blocks:
            parameters['relation'] = (
                self.relation_blocks[0].attention.in_proj_weight)
        for name, parameter in parameters.items():
            parameter.register_hook(self._cache_gradient(name))

    def gradient_diagnostics(self):
        return {
            'grad_norm_' + name: value
            for name, value in self._gradient_norms.items()
        }

    @staticmethod
    def _validate_levels(rgb_feats, thermal_feats, valid_masks):
        if len(rgb_feats) != 2 or len(thermal_feats) != 2:
            raise ValueError('TPSC expects exactly two feature levels')
        if valid_masks is not None and len(valid_masks) != 2:
            raise ValueError('TPSC expects exactly two valid-mask levels')
        for level, (rgb, thermal) in enumerate(zip(rgb_feats, thermal_feats)):
            if tuple(rgb.shape) != tuple(thermal.shape):
                raise ValueError(
                    'RGB/Thermal shape mismatch at level {}'.format(level))

    @staticmethod
    def _relation_masses(weights, num_slots):
        if weights is None:
            zero = None
            return zero, zero
        device = weights.device
        group = torch.arange(4, device=device).repeat_interleave(num_slots)
        modality = group.remainder(2)
        scale = torch.div(group, 2, rounding_mode='floor')
        cross_modal = modality[:, None] != modality[None, :]
        cross_scale = scale[:, None] != scale[None, :]
        return (
            (weights * cross_modal.to(weights.dtype)).sum(-1).mean(),
            (weights * cross_scale.to(weights.dtype)).sum(-1).mean())

    def _condition(self, rgb_prototypes, thermal_prototypes, level):
        difference = thermal_prototypes - rgb_prototypes
        update = self.conditioners[level](torch.cat((
            rgb_prototypes, thermal_prototypes, difference), dim=-1))
        return rgb_prototypes + update

    def forward(self, rgb_feats, thermal_feats, valid_masks=None,
                coverage_targets=None, return_aux=False,
                disable_modulation=False, shuffle_thermal_prototypes=False,
                disable_relation=False):
        self._validate_levels(rgb_feats, thermal_feats, valid_masks)
        if valid_masks is None:
            valid_masks = tuple(torch.ones(
                feature.shape[0], 1, *feature.shape[-2:],
                device=feature.device, dtype=torch.bool)
                for feature in rgb_feats)
        if coverage_targets is not None and len(coverage_targets) != 2:
            raise ValueError('TPSC expects exactly two coverage target levels')

        rgb_descriptors = self.rgb_descriptor(
            rgb_feats[0], rgb_feats[1], valid_masks[0], valid_masks[1])
        thermal_descriptors = self.thermal_descriptor(
            thermal_feats[0], thermal_feats[1],
            valid_masks[0], valid_masks[1])
        slots = [
            extractor(rgb_descriptor, thermal_descriptor, valid_mask)
            for extractor, rgb_descriptor, thermal_descriptor, valid_mask in
            zip(self.slot_extractors, rgb_descriptors,
                thermal_descriptors, valid_masks)
        ]
        rgb_prototypes = [item['rgb_prototypes'] for item in slots]
        thermal_prototypes = [item['thermal_prototypes'] for item in slots]
        if shuffle_thermal_prototypes and thermal_prototypes[0].shape[0] > 1:
            thermal_prototypes = [
                prototype.roll(1, dims=0) for prototype in thermal_prototypes]

        cosine_before = torch.stack([
            F.cosine_similarity(rgb, thermal, dim=-1).mean()
            for rgb, thermal in zip(rgb_prototypes, thermal_prototypes)
        ]).mean()
        relation_weights = None
        relation_enabled = (
            self.use_relation and not disable_relation and
            len(self.relation_blocks) > 0)
        if relation_enabled:
            nodes = torch.cat((
                rgb_prototypes[0], thermal_prototypes[0],
                rgb_prototypes[1], thermal_prototypes[1]), dim=1)
            weights = []
            for block in self.relation_blocks:
                nodes, block_weights = block(nodes)
                weights.append(block_weights)
            relation_weights = torch.stack(weights).mean(dim=0)
            k = self.num_slots
            rgb_prototypes = [nodes[:, :k], nodes[:, 2 * k:3 * k]]
            thermal_prototypes = [
                nodes[:, k:2 * k], nodes[:, 3 * k:4 * k]]

        conditioned = [
            self._condition(rgb, thermal, level)
            for level, (rgb, thermal) in enumerate(zip(
                rgb_prototypes, thermal_prototypes))
        ]
        cosine_after = torch.stack([
            F.cosine_similarity(rgb, thermal, dim=-1).mean()
            for rgb, thermal in zip(conditioned, thermal_prototypes)
        ]).mean()

        scales = [
            self.modulation_bound * torch.tanh(head(prototype.flatten(1)))
            for head, prototype in zip(self.modulation_heads, conditioned)
        ]
        if disable_modulation:
            calibrated = tuple(rgb_feats)
        else:
            calibrated = tuple(
                feature * (1.0 + scale[:, :, None, None])
                for feature, scale in zip(rgb_feats, scales))
        if not return_aux:
            return calibrated

        zero = rgb_feats[0].sum() * 0.0
        if coverage_targets is None:
            coverage = zero
        else:
            coverage = torch.stack([
                prototype_coverage_loss(
                    item['thermal_attention'], target, valid_mask)
                for item, target, valid_mask in zip(
                    slots, coverage_targets, valid_masks)
            ]).mean()
        diversity = torch.stack([
            prototype_diversity_loss(
                item['thermal_attention'], valid_mask)
            for item, valid_mask in zip(slots, valid_masks)
        ]).mean()
        cross_modal, cross_scale = self._relation_masses(
            relation_weights, self.num_slots)
        if cross_modal is None:
            cross_modal = zero
            cross_scale = zero
        delta_norm = torch.stack([
            (output - source).float().norm()
            for source, output in zip(rgb_feats, calibrated)
        ]).sum()
        source_norm = torch.stack([
            source.float().norm() for source in rgb_feats
        ]).sum().clamp_min(1e-12)
        thermal_stack = torch.cat(thermal_prototypes, dim=1)
        gradient_monitors = {
            name: rgb_feats[0].new_tensor(value)
            for name, value in self.gradient_diagnostics().items()
        }
        aux = dict(
            coverage_loss=coverage,
            diversity_loss=diversity,
            attention_entropy_rgb=torch.stack([
                _attention_entropy(item['rgb_attention'])
                for item in slots]).mean().detach(),
            attention_entropy_thermal=torch.stack([
                _attention_entropy(item['thermal_attention'])
                for item in slots]).mean().detach(),
            prototype_pairwise_cosine=(
                _prototype_pairwise_cosine(thermal_stack).detach()),
            prototype_effective_rank=(
                _prototype_effective_rank(thermal_stack).detach()),
            proto_cos_before=cosine_before.detach(),
            proto_cos_after=cosine_after.detach(),
            cross_modal_attention_mass=cross_modal.detach(),
            cross_scale_attention_mass=cross_scale.detach(),
            channel_scale_abs_mean=torch.stack([
                scale.abs().mean() for scale in scales]).mean().detach(),
            channel_scale_abs_max=torch.stack([
                scale.abs().max() for scale in scales]).max().detach(),
            modulation_ratio=(delta_norm / source_norm).detach(),
            **gradient_monitors)
        return calibrated, aux
