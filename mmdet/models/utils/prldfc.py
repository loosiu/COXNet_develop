import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class PRLDFCSeedHead(nn.Module):
    """Predict Thermal seed existence, sub-cell offset, and object scale."""

    def __init__(self, channels, seed_prior=0.01):
        super().__init__()
        if not 0.0 < seed_prior < 1.0:
            raise ValueError('seed_prior must be between zero and one')
        self.channels = int(channels)
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.channels, self.channels, 3, padding=1,
                groups=self.channels, bias=False),
            nn.GELU())
        self.prediction = nn.Conv2d(self.channels, 5, 1)
        nn.init.zeros_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)
        prior_bias = math.log(seed_prior / (1.0 - seed_prior))
        with torch.no_grad():
            self.prediction.bias[0] = prior_bias

    def forward(self, thermal):
        if thermal.ndim != 4 or thermal.shape[1] != self.channels:
            raise ValueError(
                'thermal must be a BCHW tensor with PRLDFC channels')
        prediction = self.prediction(self.stem(thermal))
        logits = prediction[:, :1]
        offsets = 0.5 * torch.tanh(prediction[:, 1:3])
        log_scales = prediction[:, 3:5]
        return logits, offsets, log_scales


def _validate_seed_inputs(seed_logits, seed_offsets, seed_log_scales,
                          gt_bboxes, valid_mask):
    if seed_logits.ndim != 4 or seed_logits.shape[1] != 1:
        raise ValueError('seed_logits must have shape (B,1,H,W)')
    batch, _, height, width = seed_logits.shape
    if tuple(seed_offsets.shape) != (batch, 2, height, width):
        raise ValueError('seed_offsets must have shape (B,2,H,W)')
    if tuple(seed_log_scales.shape) != (batch, 2, height, width):
        raise ValueError('seed_log_scales must have shape (B,2,H,W)')
    if tuple(valid_mask.shape) != (batch, 1, height, width):
        raise ValueError('valid_mask must match seed_logits')
    if len(gt_bboxes) != batch:
        raise ValueError('gt_bboxes length must match the batch size')


def build_prldfc_seed_targets(seed_logits, seed_offsets, seed_log_scales,
                              gt_bboxes, padded_size, valid_mask,
                              level_scale_range, matching_radius=2):
    """Build prediction-aware one-to-one seed targets for one FPN level.

    Hungarian matching is intentionally non-differentiable. Predictions are
    detached only while deciding the assignment; the returned targets are
    consumed by differentiable losses in the calibration module.
    """
    _validate_seed_inputs(
        seed_logits, seed_offsets, seed_log_scales, gt_bboxes, valid_mask)
    if len(padded_size) != 2 or min(padded_size) <= 0:
        raise ValueError('padded_size must be a positive (height, width) pair')
    if len(level_scale_range) != 2:
        raise ValueError('level_scale_range must contain lower and upper')
    lower, upper = (float(value) for value in level_scale_range)
    if lower < 0 or upper <= lower:
        raise ValueError('level_scale_range must be increasing and nonnegative')
    if matching_radius < 0:
        raise ValueError('matching_radius must be nonnegative')

    batch, _, height, width = seed_logits.shape
    padded_height, padded_width = (float(value) for value in padded_size)
    stride_y = padded_height / height
    stride_x = padded_width / width
    device = seed_logits.device
    dtype = seed_logits.dtype

    valid = valid_mask.bool()
    seed_target = seed_logits.new_zeros(seed_logits.shape)
    loss_valid = valid.clone()
    positive_mask = torch.zeros_like(valid)
    offset_target = seed_offsets.new_zeros(seed_offsets.shape)
    scale_target = seed_log_scales.new_zeros(seed_log_scales.shape)
    eligible_total = 0
    unassigned_total = 0

    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing='ij')
    cell_centers = torch.stack((grid_x + 0.5, grid_y + 0.5), dim=-1)
    flat_centers = cell_centers.reshape(-1, 2)

    detached_logits = seed_logits.detach().float().flatten(2)[:, 0]
    detached_offsets = seed_offsets.detach().float().flatten(2).transpose(1, 2)
    detached_scales = (
        seed_log_scales.detach().float().flatten(2).transpose(1, 2))

    for batch_index, boxes in enumerate(gt_bboxes):
        boxes = boxes.to(device=device, dtype=dtype)
        if boxes.numel() == 0:
            continue
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError('each gt_bboxes tensor must have shape (N,4)')

        sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
        geometric_scale = torch.sqrt(sizes[:, 0] * sizes[:, 1])
        eligible = (geometric_scale >= lower) & (geometric_scale < upper)
        boxes = boxes[eligible]
        sizes = sizes[eligible]
        eligible_total += int(eligible.sum().item())
        if boxes.numel() == 0:
            continue

        centers_pixels = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        centers_feature = torch.stack((
            centers_pixels[:, 0] / stride_x,
            centers_pixels[:, 1] / stride_y), dim=-1)
        target_scales = torch.log(torch.stack((
            sizes[:, 0] / stride_x,
            sizes[:, 1] / stride_y), dim=-1).clamp_min(1e-6))

        candidate_rows = []
        candidate_union = torch.zeros(
            height * width, dtype=torch.bool, device=device)
        flat_valid = valid[batch_index, 0].flatten()
        for center in centers_feature:
            delta = (flat_centers - center).abs()
            candidates = (
                (delta[:, 0] <= matching_radius) &
                (delta[:, 1] <= matching_radius) & flat_valid)
            candidate_rows.append(candidates)
            candidate_union |= candidates

        union_indices = candidate_union.nonzero(as_tuple=False).flatten()
        if union_indices.numel() == 0:
            unassigned_total += boxes.shape[0]
            continue
        candidate_matrix = torch.stack(candidate_rows)[:, union_indices]
        candidate_centers = flat_centers[union_indices].float()
        predicted_centers = (
            candidate_centers +
            detached_offsets[batch_index, union_indices])
        center_cost = torch.cdist(
            centers_feature.float(), predicted_centers, p=1)
        score_cost = F.softplus(
            -detached_logits[batch_index, union_indices]).unsqueeze(0)
        scale_cost = torch.cdist(
            target_scales.float(),
            detached_scales[batch_index, union_indices], p=1)
        cost = center_cost + 0.25 * score_cost + 0.25 * scale_cost
        large_cost = cost.new_tensor(1e6)
        cost = cost.masked_fill(~candidate_matrix, large_cost)
        # Stable, negligible column-order tie breaker.
        cost = cost + union_indices.float().unsqueeze(0) * 1e-7

        row_indices, column_indices = linear_sum_assignment(
            cost.cpu().numpy())
        assigned_gt = set()
        for gt_index, union_column in zip(row_indices, column_indices):
            if cost[gt_index, union_column].item() >= 5e5:
                continue
            assigned_gt.add(int(gt_index))
            flat_index = int(union_indices[union_column].item())
            y_index, x_index = divmod(flat_index, width)
            positive_mask[batch_index, 0, y_index, x_index] = True
            seed_target[batch_index, 0, y_index, x_index] = 1.0
            target_offset = (
                centers_feature[gt_index] -
                cell_centers[y_index, x_index]).clamp(-0.5, 0.5)
            offset_target[batch_index, :, y_index, x_index] = target_offset
            scale_target[batch_index, :, y_index, x_index] = (
                target_scales[gt_index])

        unassigned_total += boxes.shape[0] - len(assigned_gt)

        if candidate_matrix.shape[0] > 1:
            shared = candidate_matrix.sum(0) > 1
            geometry = torch.cdist(
                centers_feature.float(), candidate_centers, p=1)
            geometry = geometry.masked_fill(~candidate_matrix, large_cost)
            two_best = geometry.topk(2, dim=0, largest=False).values
            ambiguous = shared & ((two_best[1] - two_best[0]).abs() < 0.25)
            ambiguous_indices = union_indices[ambiguous]
            if ambiguous_indices.numel() > 0:
                flat_loss_valid = loss_valid[batch_index, 0].flatten()
                flat_loss_valid[ambiguous_indices] = False
                flat_positive = positive_mask[batch_index, 0].flatten()
                flat_loss_valid[flat_positive] = True

    return dict(
        seed_target=seed_target,
        loss_valid=loss_valid,
        positive_mask=positive_mask,
        offset_target=offset_target,
        scale_target=scale_target,
        eligible_gt_count=seed_logits.new_tensor(float(eligible_total)),
        unassigned_gt_count=seed_logits.new_tensor(float(unassigned_total)),
    )
