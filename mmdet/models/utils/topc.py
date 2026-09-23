"""Thermal-Anchored Object Prototype Calibration.

TOPC uses Thermal P3 objects as semantic anchors for local RGB calibration
before the original COXNet AAM.  It does not align or modify Thermal features.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trpc import ChannelLayerNorm


def build_topc_gaussian_targets(gt_boxes, padded_size, feat_size, device,
                                valid_mask=None, max_radius=2):
    """Build object-wise normalized Gaussian centers in Thermal coordinates."""
    input_h, input_w = padded_size
    feat_h, feat_w = feat_size
    if input_h <= 0 or input_w <= 0 or feat_h <= 0 or feat_w <= 0:
        raise ValueError('padded_size and feat_size must be positive')
    if max_radius < 1:
        raise ValueError('max_radius must be positive')

    target = torch.zeros(
        len(gt_boxes), 1, feat_h, feat_w, device=device,
        dtype=torch.float32)
    scale_x = feat_w / float(input_w)
    scale_y = feat_h / float(input_h)
    for batch_index, boxes in enumerate(gt_boxes):
        if boxes is None or not len(boxes):
            continue
        for x1, y1, x2, y2 in boxes.detach().cpu().tolist():
            width = max((float(x2) - float(x1)) * scale_x, 0.0)
            height = max((float(y2) - float(y1)) * scale_y, 0.0)
            center_x = 0.5 * (float(x1) + float(x2)) * scale_x
            center_y = 0.5 * (float(y1) + float(y2)) * scale_y
            radius = max(1, min(
                int(max_radius),
                int(math.ceil(0.5 * math.sqrt(width * height)))))
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
            target_slice = target[
                batch_index, 0, row0:row1, col0:col1]
            target[batch_index, 0, row0:row1, col0:col1] = torch.maximum(
                target_slice, gaussian)

    if valid_mask is not None:
        if tuple(valid_mask.shape) != tuple(target.shape):
            raise ValueError(
                'valid_mask must have the same shape as Gaussian targets')
        target = target * valid_mask.to(target.dtype)
    return target


class ThermalAnchoredObjectPrototypeCalibration(nn.Module):
    """Discover sparse Thermal object candidates for RGB calibration."""

    def __init__(self, channels=256, calibration_dim=64,
                 candidate_prior=0.01, candidate_threshold=0.05,
                 max_candidates=100, object_kernel=3, search_radius=2,
                 search_temperature=0.2, residual_init_std=1e-2,
                 focal_gamma=2.0):
        super().__init__()
        if channels < 1 or calibration_dim < 1:
            raise ValueError('channel dimensions must be positive')
        if not 0.0 < candidate_prior < 1.0:
            raise ValueError('candidate_prior must be in (0, 1)')
        if not 0.0 <= candidate_threshold <= 1.0:
            raise ValueError('candidate_threshold must be in [0, 1]')
        if max_candidates < 1:
            raise ValueError('max_candidates must be positive')
        if object_kernel < 1 or object_kernel % 2 == 0:
            raise ValueError('object_kernel must be a positive odd integer')
        if search_radius < 0 or search_temperature <= 0:
            raise ValueError('search parameters are invalid')
        if residual_init_std <= 0:
            raise ValueError('residual_init_std must be positive')

        self.channels = int(channels)
        self.calibration_dim = int(calibration_dim)
        self.candidate_threshold = float(candidate_threshold)
        self.max_candidates = int(max_candidates)
        self.object_kernel = int(object_kernel)
        self.search_radius = int(search_radius)
        self.search_temperature = float(search_temperature)
        self.residual_init_std = float(residual_init_std)
        self.focal_gamma = float(focal_gamma)

        self.objectness_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1))
        prior_bias = math.log(candidate_prior / (1.0 - candidate_prior))
        nn.init.constant_(self.objectness_head[-1].bias, prior_bias)

        # One projection defines the common calibration space.  Reusing both
        # this convolution and its normalization is intentional: modality-
        # specific projections would make the prototype discrepancy
        # coordinate-system dependent.
        self.shared_projection = nn.Conv2d(
            channels, calibration_dim, 1, bias=False)
        self.shared_norm = ChannelLayerNorm(calibration_dim)

    def objectness_logits(self, thermal):
        if thermal.ndim != 4 or thermal.shape[1] != self.channels:
            raise ValueError('thermal must be a BCHW tensor with TOPC channels')
        return self.objectness_head(thermal)

    def _shared_features(self, rgb, thermal):
        if tuple(rgb.shape) != tuple(thermal.shape):
            raise ValueError('RGB and Thermal features must have equal shapes')
        if rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError('features must be BCHW tensors with TOPC channels')
        rgb_shared = self.shared_norm(self.shared_projection(rgb))
        thermal_shared = self.shared_norm(self.shared_projection(thermal))
        return rgb_shared, thermal_shared

    def _select_candidates(self, logits, valid):
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError('candidate logits must have shape (B,1,H,W)')
        if tuple(valid.shape) != tuple(logits.shape):
            raise ValueError('candidate valid mask must match logits')
        valid = valid.bool()
        probability = logits.sigmoid()
        eligible = valid & (probability >= self.candidate_threshold)
        flat_probability = probability.flatten(1)
        flat_eligible = eligible.flatten(1)
        spatial_size = flat_probability.shape[1]
        select_count = min(self.max_candidates, spatial_size)
        ranking = flat_probability.masked_fill(~flat_eligible, -1.0)
        values, indices = ranking.topk(select_count, dim=1)
        active = values >= self.candidate_threshold
        scores = flat_probability.gather(1, indices) * active.to(
            flat_probability.dtype)

        if select_count < self.max_candidates:
            pad = self.max_candidates - select_count
            indices = torch.cat([
                indices,
                indices.new_zeros(indices.shape[0], pad)], dim=1)
            scores = torch.cat([
                scores,
                scores.new_zeros(scores.shape[0], pad)], dim=1)
            active = torch.cat([
                active,
                active.new_zeros(active.shape[0], pad)], dim=1)

        dense_mask = torch.zeros_like(flat_eligible)
        dense_mask.scatter_(1, indices, active)
        dense_mask = dense_mask.reshape_as(valid)
        return dict(
            indices=indices,
            scores=scores,
            active=active,
            dense_mask=dense_mask,
            probability=probability)

    def _thermal_prototypes(self, thermal_shared, probability, selection,
                            valid):
        """Pool one heatmap-weighted local Thermal prototype per candidate."""
        batch, dim, height, width = thermal_shared.shape
        if tuple(probability.shape) != (batch, 1, height, width):
            raise ValueError('probability must match the shared feature map')
        if tuple(valid.shape) != tuple(probability.shape):
            raise ValueError('prototype valid mask must match probability')
        indices = selection['indices']
        selected_active = selection['active'].bool()
        if indices.ndim != 2 or tuple(indices.shape) != tuple(
                selected_active.shape):
            raise ValueError('candidate indices and active mask must be BxK')

        kernel = self.object_kernel
        kernel_area = kernel * kernel
        spatial_size = height * width
        feature_patches = F.unfold(
            thermal_shared, kernel, padding=kernel // 2)
        feature_patches = feature_patches.reshape(
            batch, dim, kernel_area, spatial_size).permute(0, 3, 1, 2)
        feature_patches = torch.gather(
            feature_patches, 1,
            indices[:, :, None, None].expand(-1, -1, dim, kernel_area))

        weights = probability * valid.to(probability.dtype)
        weight_patches = F.unfold(
            weights, kernel, padding=kernel // 2).transpose(1, 2)
        weight_patches = torch.gather(
            weight_patches, 1,
            indices[:, :, None].expand(-1, -1, kernel_area))
        mass = weight_patches.sum(-1)
        active = selected_active & (mass > 1e-12)
        prototypes = (
            feature_patches * weight_patches[:, :, None, :]).sum(-1)
        prototypes = prototypes / mass.clamp_min(1e-12).unsqueeze(-1)
        prototypes = prototypes * active.unsqueeze(-1).to(prototypes.dtype)
        return prototypes, mass * selected_active.to(mass.dtype), active

    def _local_rgb_match(self, rgb_shared, prototypes, indices, active,
                         valid):
        """Use each Thermal prototype to search a local RGB neighborhood."""
        batch, dim, height, width = rgb_shared.shape
        if prototypes.ndim != 3 or prototypes.shape[:2] != indices.shape:
            raise ValueError('prototype and candidate dimensions do not match')
        if prototypes.shape[2] != dim:
            raise ValueError('prototype dimension does not match RGB space')
        if tuple(valid.shape) != (batch, 1, height, width):
            raise ValueError('matching valid mask must match RGB feature map')

        window = 2 * self.search_radius + 1
        option_count = window * window
        spatial_size = height * width
        rgb_windows = F.unfold(
            rgb_shared, window, padding=self.search_radius)
        rgb_windows = rgb_windows.reshape(
            batch, dim, option_count, spatial_size).permute(0, 3, 1, 2)
        rgb_windows = torch.gather(
            rgb_windows, 1,
            indices[:, :, None, None].expand(-1, -1, dim, option_count))

        valid_windows = F.unfold(
            valid.to(rgb_shared.dtype), window,
            padding=self.search_radius).transpose(1, 2).bool()
        valid_options = torch.gather(
            valid_windows, 1,
            indices[:, :, None].expand(-1, -1, option_count))
        valid_options = valid_options & active[:, :, None].bool()

        normalized_query = F.normalize(prototypes.float(), dim=-1)
        normalized_options = F.normalize(rgb_windows.float(), dim=2)
        similarity = (
            normalized_query[:, :, :, None] * normalized_options).sum(2)
        logits = similarity / self.search_temperature
        attention = torch.softmax(
            logits.masked_fill(~valid_options, -1e4), dim=-1)
        attention = attention * valid_options.to(attention.dtype)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(
            1e-12)

        rgb_prototypes = (
            rgb_windows.float() * attention[:, :, None, :]).sum(-1)
        rgb_prototypes = rgb_prototypes.to(rgb_shared.dtype)
        confidence = attention.max(-1).values.to(rgb_shared.dtype)
        similarity = similarity.to(rgb_shared.dtype)

        coordinate_range = torch.arange(
            -self.search_radius, self.search_radius + 1,
            device=rgb_shared.device)
        offset_y, offset_x = torch.meshgrid(
            coordinate_range, coordinate_range, indexing='ij')
        offsets = torch.stack([offset_y.flatten(), offset_x.flatten()], dim=-1)
        return dict(
            rgb_prototypes=rgb_prototypes,
            attention=attention.to(rgb_shared.dtype),
            similarity=similarity,
            confidence=confidence,
            offsets=offsets,
            valid_options=valid_options)
