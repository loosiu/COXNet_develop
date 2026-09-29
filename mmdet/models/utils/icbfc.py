import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt(
        tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an orthonormal one-level Haar transform to a BCHW tensor."""
    if tensor.ndim != 4:
        raise ValueError('Haar DWT expects a 4D tensor')
    if tensor.shape[-2] % 2 or tensor.shape[-1] % 2:
        raise ValueError('Haar DWT expects even spatial dimensions')

    x00 = tensor[..., 0::2, 0::2]
    x01 = tensor[..., 0::2, 1::2]
    x10 = tensor[..., 1::2, 0::2]
    x11 = tensor[..., 1::2, 1::2]
    return (
        (x00 + x01 + x10 + x11) * 0.5,
        (-x00 - x01 + x10 + x11) * 0.5,
        (-x00 + x01 - x10 + x11) * 0.5,
        (x00 - x01 - x10 + x11) * 0.5,
    )


def haar_idwt(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor,
              hh: torch.Tensor) -> torch.Tensor:
    """Invert :func:`haar_dwt` without changing dtype or device."""
    shapes = {tuple(component.shape) for component in (ll, lh, hl, hh)}
    if len(shapes) != 1:
        raise ValueError('Haar IDWT components must share one shape')

    output = ll.new_empty(
        (*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2))
    output[..., 0::2, 0::2] = (ll - lh - hl + hh) * 0.5
    output[..., 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    output[..., 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    output[..., 1::2, 1::2] = (ll + lh + hl + hh) * 0.5
    return output


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project logits onto the probability simplex (Martins & Astudillo)."""
    if logits.numel() == 0:
        return logits
    dim = dim if dim >= 0 else logits.ndim + dim
    shifted = logits - logits.amax(dim=dim, keepdim=True)
    sorted_logits, _ = torch.sort(shifted, dim=dim, descending=True)
    ranks_shape = [1] * logits.ndim
    ranks_shape[dim] = logits.shape[dim]
    ranks = torch.arange(
        1, logits.shape[dim] + 1, device=logits.device,
        dtype=logits.dtype).view(ranks_shape)
    cumulative = sorted_logits.cumsum(dim)
    support = (1 + ranks * sorted_logits) > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau_sum = cumulative.gather(dim, support_size - 1)
    tau = (tau_sum - 1) / support_size.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0)


def _valid_mask(img_metas: Sequence[dict], output_size: Tuple[int, int],
                device: torch.device) -> torch.Tensor:
    out_h, out_w = output_size
    masks = []
    for meta in img_metas:
        pad_h, pad_w = tuple(meta.get(
            'batch_input_shape', meta['pad_shape'][:2]))[:2]
        img_h, img_w = meta['img_shape'][:2]
        valid_h = min(out_h, int(math.ceil(img_h * out_h / pad_h)))
        valid_w = min(out_w, int(math.ceil(img_w * out_w / pad_w)))
        mask = torch.zeros((1, out_h, out_w), dtype=torch.bool,
                           device=device)
        mask[:, :valid_h, :valid_w] = True
        masks.append(mask)
    return torch.stack(masks, dim=0)


def _draw_gaussian(heatmap: torch.Tensor, center_x: int, center_y: int,
                   radius: int) -> None:
    radius = max(int(radius), 0)
    if radius == 0:
        heatmap[center_y, center_x] = 1
        return
    diameter = 2 * radius + 1
    coordinates = torch.arange(
        diameter, device=heatmap.device, dtype=heatmap.dtype) - radius
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    sigma = diameter / 6.0
    gaussian = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))

    height, width = heatmap.shape
    left = min(center_x, radius)
    right = min(width - center_x - 1, radius)
    top = min(center_y, radius)
    bottom = min(height - center_y - 1, radius)
    heat_region = heatmap[
        center_y - top:center_y + bottom + 1,
        center_x - left:center_x + right + 1]
    gaussian_region = gaussian[
        radius - top:radius + bottom + 1,
        radius - left:radius + right + 1]
    torch.maximum(heat_region, gaussian_region, out=heat_region)


def build_instance_targets(gt_bboxes: Sequence[torch.Tensor],
                           img_metas: Sequence[dict],
                           output_size: Tuple[int, int],
                           device: torch.device,
                           dtype: torch.dtype = torch.float32
                           ) -> Dict[str, torch.Tensor]:
    """Build class-agnostic center, offset, and scale targets."""
    if len(gt_bboxes) != len(img_metas):
        raise ValueError('gt_bboxes and img_metas must have equal length')
    out_h, out_w = output_size
    batch_size = len(gt_bboxes)
    center = torch.zeros(
        (batch_size, 1, out_h, out_w), device=device, dtype=dtype)
    offset = torch.zeros(
        (batch_size, 2, out_h, out_w), device=device, dtype=dtype)
    log_scale = torch.zeros_like(offset)
    regression_mask = torch.zeros(
        (batch_size, 1, out_h, out_w), device=device, dtype=torch.bool)
    valid = _valid_mask(img_metas, output_size, device)

    for batch_index, (boxes, meta) in enumerate(zip(gt_bboxes, img_metas)):
        pad_h, pad_w = tuple(meta.get(
            'batch_input_shape', meta['pad_shape'][:2]))[:2]
        scale_x = out_w / float(pad_w)
        scale_y = out_h / float(pad_h)
        for box in boxes.to(device=device, dtype=dtype):
            center_x = 0.5 * (box[0] + box[2]) * scale_x
            center_y = 0.5 * (box[1] + box[3]) * scale_y
            cell_x = int(torch.floor(center_x).item())
            cell_y = int(torch.floor(center_y).item())
            if not (0 <= cell_x < out_w and 0 <= cell_y < out_h):
                continue
            if not valid[batch_index, 0, cell_y, cell_x]:
                continue
            width = torch.clamp((box[2] - box[0]) * scale_x, min=1e-6)
            height = torch.clamp((box[3] - box[1]) * scale_y, min=1e-6)
            radius = int(torch.floor(torch.minimum(width, height) * 0.5).item())
            _draw_gaussian(center[batch_index, 0], cell_x, cell_y, radius)
            offset[batch_index, 0, cell_y, cell_x] = center_x - cell_x
            offset[batch_index, 1, cell_y, cell_x] = center_y - cell_y
            log_scale[batch_index, 0, cell_y, cell_x] = torch.log(width)
            log_scale[batch_index, 1, cell_y, cell_x] = torch.log(height)
            regression_mask[batch_index, 0, cell_y, cell_x] = True

    return dict(
        center=center,
        offset=offset,
        log_scale=log_scale,
        regression_mask=regression_mask,
        valid_mask=valid)


def extract_instance_candidates(center_logits: torch.Tensor,
                                offsets: torch.Tensor,
                                log_scales: torch.Tensor,
                                valid_mask: torch.Tensor,
                                score_threshold: float = 0.1,
                                chunk_size: int = 256
                                ) -> List[Dict[str, torch.Tensor]]:
    """Extract every valid local maximum above threshold without top-k."""
    if center_logits.ndim != 4 or center_logits.shape[1] != 1:
        raise ValueError('center_logits must have shape [B, 1, H, W]')
    if offsets.shape[:2] != (center_logits.shape[0], 2):
        raise ValueError('offsets must have shape [B, 2, H, W]')
    if log_scales.shape != offsets.shape:
        raise ValueError('log_scales must match offsets')
    if valid_mask.shape != center_logits.shape:
        raise ValueError('valid_mask must match center_logits')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')

    probabilities = center_logits.sigmoid()
    pooled = F.max_pool2d(probabilities, kernel_size=3, stride=1, padding=1)
    peaks = ((probabilities == pooled) &
             (probabilities >= score_threshold) & valid_mask.bool())
    results = []
    for batch_index in range(center_logits.shape[0]):
        positions = peaks[batch_index, 0].nonzero(as_tuple=False)
        if positions.numel() == 0:
            empty_two = offsets.new_empty((0, 2))
            results.append(dict(
                centers=empty_two,
                scales=empty_two.clone(),
                scores=probabilities.new_empty((0,)),
                chunks=torch.empty(
                    (0, 2), device=center_logits.device, dtype=torch.long)))
            continue
        ys, xs = positions[:, 0], positions[:, 1]
        point_offsets = offsets[batch_index, :, ys, xs].transpose(0, 1)
        centers = torch.stack((xs, ys), dim=1).to(offsets.dtype)
        centers = centers + point_offsets
        scales = log_scales[
            batch_index, :, ys, xs].transpose(0, 1).clamp(-4, 4).exp()
        scores = probabilities[batch_index, 0, ys, xs]
        chunk_ranges = [
            (start, min(start + chunk_size, positions.shape[0]))
            for start in range(0, positions.shape[0], chunk_size)
        ]
        chunks = torch.tensor(
            chunk_ranges, device=center_logits.device, dtype=torch.long)
        results.append(dict(
            centers=centers,
            scales=scales,
            scores=scores,
            chunks=chunks))
    return results


def _center_focal_loss(logits: torch.Tensor, target: torch.Tensor,
                       valid_mask: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid().clamp(min=1e-6, max=1 - 1e-6)
    positive = (target == 1) & valid_mask
    negative = (target < 1) & valid_mask
    negative_weight = (1 - target).pow(4)
    positive_loss = -torch.log(probabilities) * (1 - probabilities).pow(2)
    negative_loss = (-torch.log(1 - probabilities) * probabilities.pow(2) *
                     negative_weight)
    normalizer = positive.sum().clamp(min=1).to(logits.dtype)
    return ((positive_loss * positive).sum() +
            (negative_loss * negative).sum()) / normalizer


class ThermalInstancePrior(nn.Module):
    """Predict a stride-4 Thermal center prior and materialize instances."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 score_threshold: float = 0.1,
                 candidate_chunk_size: int = 256,
                 center_prior: float = 0.01):
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0:
            raise ValueError('channel counts must be positive')
        if not 0 < center_prior < 1:
            raise ValueError('center_prior must be in (0, 1)')
        groups = math.gcd(hidden_channels, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True))
        self.center_head = nn.Conv2d(hidden_channels, 1, 1)
        self.offset_head = nn.Conv2d(hidden_channels, 2, 1)
        self.scale_head = nn.Conv2d(hidden_channels, 2, 1)
        self.score_threshold = float(score_threshold)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self._initialize(center_prior)

    def _initialize(self, center_prior: float) -> None:
        nn.init.kaiming_normal_(
            self.stem[0].weight, mode='fan_out', nonlinearity='relu')
        for head in (self.center_head, self.offset_head, self.scale_head):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)
        center_bias = math.log(center_prior / (1 - center_prior))
        nn.init.constant_(self.center_head.bias, center_bias)

    def _gt_instances(self, gt_bboxes: Sequence[torch.Tensor],
                      img_metas: Sequence[dict],
                      output_size: Tuple[int, int],
                      device: torch.device,
                      dtype: torch.dtype) -> List[Dict[str, torch.Tensor]]:
        out_h, out_w = output_size
        instances = []
        for batch_index, (boxes, meta) in enumerate(zip(gt_bboxes, img_metas)):
            pad_h, pad_w = tuple(meta.get(
                'batch_input_shape', meta['pad_shape'][:2]))[:2]
            scale_x = out_w / float(pad_w)
            scale_y = out_h / float(pad_h)
            boxes = boxes.to(device=device, dtype=dtype)
            if boxes.numel() == 0:
                centers = boxes.new_empty((0, 2))
                scales = boxes.new_empty((0, 2))
            else:
                centers = torch.stack((
                    0.5 * (boxes[:, 0] + boxes[:, 2]) * scale_x,
                    0.5 * (boxes[:, 1] + boxes[:, 3]) * scale_y,
                ), dim=1)
                scales = torch.stack((
                    (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6) * scale_x,
                    (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6) * scale_y,
                ), dim=1)
                valid = ((centers[:, 0] >= 0) & (centers[:, 0] < out_w) &
                         (centers[:, 1] >= 0) & (centers[:, 1] < out_h))
                centers, scales = centers[valid], scales[valid]
            count = centers.shape[0]
            instances.append(dict(
                centers=centers,
                scales=scales,
                scores=torch.ones(count, device=device, dtype=dtype),
                batch_index=torch.full(
                    (count,), batch_index, device=device, dtype=torch.long),
                chunks=torch.tensor(
                    [(start, min(start + self.candidate_chunk_size, count))
                     for start in range(0, count,
                                        self.candidate_chunk_size)],
                    device=device, dtype=torch.long).reshape(-1, 2)))
        return instances

    @staticmethod
    def _masked_regression_loss(prediction: torch.Tensor,
                                target: torch.Tensor,
                                mask: torch.Tensor,
                                smooth: bool = False) -> torch.Tensor:
        expanded_mask = mask.expand_as(prediction)
        if smooth:
            element_loss = F.smooth_l1_loss(
                prediction, target, reduction='none')
        else:
            element_loss = F.l1_loss(prediction, target, reduction='none')
        normalizer = expanded_mask.sum().clamp(min=1).to(prediction.dtype)
        return (element_loss * expanded_mask).sum() / normalizer

    def forward(self,
                thermal_s4: torch.Tensor,
                gt_bboxes: Sequence[torch.Tensor] = None,
                img_metas: Sequence[dict] = None,
                return_loss: bool = False
                ) -> Tuple[List[Dict[str, torch.Tensor]],
                           Dict[str, torch.Tensor]]:
        if thermal_s4.ndim != 4:
            raise ValueError('thermal_s4 must be a BCHW tensor')
        batch_size, _, height, width = thermal_s4.shape
        if img_metas is None:
            img_metas = [dict(
                img_shape=(height, width, 1),
                pad_shape=(height, width, 1),
                batch_input_shape=(height, width)) for _ in range(batch_size)]
        if len(img_metas) != batch_size:
            raise ValueError('img_metas must match the batch size')

        hidden = self.stem(thermal_s4)
        center_logits = self.center_head(hidden)
        offsets = self.offset_head(hidden)
        log_scales = self.scale_head(hidden)
        valid_mask = _valid_mask(
            img_metas, (height, width), thermal_s4.device)
        aux = dict(
            center_logits=center_logits,
            offsets=offsets,
            log_scales=log_scales,
            valid_mask=valid_mask)

        if gt_bboxes is not None:
            if len(gt_bboxes) != batch_size:
                raise ValueError('gt_bboxes must match the batch size')
            instances = self._gt_instances(
                gt_bboxes, img_metas, (height, width), thermal_s4.device,
                thermal_s4.dtype)
        else:
            instances = extract_instance_candidates(
                center_logits, offsets, log_scales, valid_mask,
                score_threshold=self.score_threshold,
                chunk_size=self.candidate_chunk_size)
            for batch_index, candidate in enumerate(instances):
                candidate['batch_index'] = torch.full(
                    (candidate['centers'].shape[0],), batch_index,
                    device=thermal_s4.device, dtype=torch.long)

        counts = center_logits.new_tensor(
            [instance['centers'].shape[0] for instance in instances])
        aux['candidate_count'] = counts.mean()
        score_values = [
            instance['scores'] for instance in instances
            if instance['scores'].numel()
        ]
        aux['candidate_score_mean'] = (
            torch.cat(score_values).mean() if score_values else
            center_logits.new_zeros(()))

        if return_loss:
            if gt_bboxes is None:
                raise ValueError('gt_bboxes are required when return_loss=True')
            targets = build_instance_targets(
                gt_bboxes, img_metas, (height, width), thermal_s4.device,
                thermal_s4.dtype)
            aux['loss_icbfc_center'] = _center_focal_loss(
                center_logits, targets['center'], targets['valid_mask'])
            aux['loss_icbfc_offset'] = self._masked_regression_loss(
                offsets, targets['offset'], targets['regression_mask'])
            aux['loss_icbfc_scale'] = self._masked_regression_loss(
                log_scales, targets['log_scale'],
                targets['regression_mask'], smooth=True)
        return instances, aux
