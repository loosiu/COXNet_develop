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


class DynamicFrequencyBank(nn.Module):
    """Split a feature map into learnable radial frequency bands."""

    def __init__(self, channels, frequency_dim=64, num_bands=3,
                 temperature=0.02, min_band_width=0.05):
        super().__init__()
        self.channels = int(channels)
        self.frequency_dim = int(frequency_dim)
        self.num_bands = int(num_bands)
        self.temperature = float(temperature)
        self.min_band_width = float(min_band_width)
        if self.channels <= 0 or self.frequency_dim <= 0:
            raise ValueError('frequency channels must be positive')
        if self.num_bands <= 0:
            raise ValueError('num_bands must be positive')
        if self.temperature <= 0:
            raise ValueError('frequency temperature must be positive')
        if not 0 <= self.min_band_width < 0.5 / self.num_bands:
            raise ValueError('min_band_width is incompatible with num_bands')

        self.projection = nn.Conv2d(
            self.channels, self.frequency_dim, 1, bias=False)
        if self.num_bands == 3:
            initial_widths = torch.tensor([0.125, 0.125, 0.25])
        else:
            initial_widths = torch.full(
                (self.num_bands,), 0.5 / self.num_bands)
        free_width = 0.5 - self.num_bands * self.min_band_width
        probabilities = (
            (initial_widths - self.min_band_width) / free_width)
        self.band_logits = nn.Parameter(probabilities.clamp_min(1e-8).log())

    def _boundaries(self):
        free_width = 0.5 - self.num_bands * self.min_band_width
        widths = (
            self.min_band_width +
            free_width * torch.softmax(self.band_logits, dim=0))
        zero = widths.new_zeros(1)
        return torch.cat((zero, widths.cumsum(0)), dim=0)

    @staticmethod
    def _radial_grid(height, width, device):
        fy = torch.fft.fftfreq(height, device=device)
        fx = torch.fft.rfftfreq(width, device=device)
        radial = torch.sqrt(
            (fy[:, None] / 0.5).square() +
            (fx[None, :] / 0.5).square())
        return (0.5 / math.sqrt(2.0)) * radial

    def forward(self, feature, valid_mask):
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                'feature must be a BCHW tensor with frequency-bank channels')
        batch, _, height, width = feature.shape
        if tuple(valid_mask.shape) != (batch, 1, height, width):
            raise ValueError('valid_mask must match the frequency feature')

        projected = self.projection(feature)
        projected = projected * valid_mask.to(projected.dtype)
        fft_input = projected.float()
        spectrum = torch.fft.rfft2(fft_input, norm='ortho')
        radial = self._radial_grid(height, width, spectrum.device)
        boundaries = self._boundaries().to(spectrum.device)

        masks = []
        for band_index in range(self.num_bands):
            lower = boundaries[band_index]
            upper = boundaries[band_index + 1]
            mask = (
                torch.sigmoid((radial - lower) / self.temperature) -
                torch.sigmoid((radial - upper) / self.temperature))
            masks.append(mask.clamp_min(0.0))
        masks = torch.stack(masks, dim=0)
        masks = masks / masks.sum(0, keepdim=True).clamp_min(1e-12)

        bands = []
        for mask in masks:
            spatial = torch.fft.irfft2(
                spectrum * mask[None, None], s=(height, width),
                norm='ortho')
            bands.append(spatial.to(projected.dtype))
        bands = torch.stack(bands, dim=1)
        return bands, dict(
            boundaries=boundaries,
            masks=masks,
            projected=projected,
        )


class _ChannelLayerNorm(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, feature):
        return self.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _valid_average_pool(feature, valid_mask, kernel_size):
    valid = valid_mask.to(feature.dtype)
    area = float(kernel_size * kernel_size)
    numerator = F.avg_pool2d(
        feature * valid, kernel_size, stride=1,
        padding=kernel_size // 2) * area
    denominator = F.avg_pool2d(
        valid, kernel_size, stride=1,
        padding=kernel_size // 2) * area
    pooled = numerator / denominator.clamp_min(1e-6)
    return pooled * valid


def _sample_with_offsets(feature, offsets):
    batch, _, height, width = feature.shape
    if tuple(offsets.shape) != (batch, 2, height, width):
        raise ValueError('offsets must match the sampled feature map')
    y, x = torch.meshgrid(
        torch.arange(height, device=feature.device, dtype=feature.dtype),
        torch.arange(width, device=feature.device, dtype=feature.dtype),
        indexing='ij')
    sample_x = x[None] + offsets[:, 0].to(feature.dtype)
    sample_y = y[None] + offsets[:, 1].to(feature.dtype)
    if width > 1:
        sample_x = 2.0 * sample_x / (width - 1) - 1.0
    else:
        sample_x = torch.zeros_like(sample_x)
    if height > 1:
        sample_y = 2.0 * sample_y / (height - 1) - 1.0
    else:
        sample_y = torch.zeros_like(sample_y)
    grid = torch.stack((sample_x, sample_y), dim=-1)
    return F.grid_sample(
        feature, grid, mode='bilinear', padding_mode='zeros',
        align_corners=True)


def _masked_focal_loss(logits, target, valid, gamma=2.0):
    probability = logits.sigmoid()
    target = target.to(logits.dtype)
    valid = valid.bool()
    positive = valid & (target > 0.5)
    negative = valid & ~positive
    losses = []
    if positive.any():
        loss = -(
            (1.0 - probability[positive]).pow(gamma) *
            probability[positive].clamp_min(1e-6).log())
        losses.append(loss.mean())
    if negative.any():
        loss = -(
            probability[negative].pow(gamma) *
            (1.0 - probability[negative]).clamp_min(1e-6).log())
        losses.append(loss.mean())
    if not losses:
        return logits.sum() * 0.0
    return sum(losses) / len(losses)


class PrototypeRoutedLocalDynamicFrequencyCalibration(nn.Module):
    """Thermal-seeded frequency routing that calibrates RGB only."""

    def __init__(self, channels, frequency_dim=64, prototype_dim=64,
                 num_bands=3, search_radius=2, seed_prior=0.01,
                 seed_threshold=0.10, seed_temperature=0.25,
                 frequency_temperature=0.02, min_band_width=0.05,
                 residual_epsilon=0.1, matching_radius=2,
                 focal_gamma=2.0, sparse_inference=False):
        super().__init__()
        self.channels = int(channels)
        self.frequency_dim = int(frequency_dim)
        self.prototype_dim = int(prototype_dim)
        self.num_bands = int(num_bands)
        self.search_radius = int(search_radius)
        self.seed_threshold = float(seed_threshold)
        self.seed_temperature = float(seed_temperature)
        self.residual_epsilon = float(residual_epsilon)
        self.matching_radius = int(matching_radius)
        self.focal_gamma = float(focal_gamma)
        self.sparse_inference = bool(sparse_inference)
        if self.channels <= 0 or self.prototype_dim <= 0:
            raise ValueError('PRLDFC channel dimensions must be positive')
        if self.search_radius < 0 or self.matching_radius < 0:
            raise ValueError('PRLDFC radii must be nonnegative')
        if not 0.0 < self.seed_threshold < 1.0:
            raise ValueError('seed_threshold must be between zero and one')
        if self.seed_temperature <= 0 or self.residual_epsilon <= 0:
            raise ValueError('temperatures and residual_epsilon must be positive')

        self.seed_head = PRLDFCSeedHead(self.channels, seed_prior=seed_prior)
        self.frequency_bank = DynamicFrequencyBank(
            self.channels, frequency_dim=self.frequency_dim,
            num_bands=self.num_bands,
            temperature=frequency_temperature,
            min_band_width=min_band_width)
        self.shared_projection = nn.Conv2d(
            self.channels, self.prototype_dim, 1, bias=False)
        self.shared_norm = _ChannelLayerNorm(self.prototype_dim)
        self.prototype_mlp = nn.Sequential(
            nn.Conv2d(3 * self.prototype_dim + 2, self.prototype_dim, 1),
            nn.GELU(),
            nn.Conv2d(self.prototype_dim, self.prototype_dim, 1))
        self.broad_query = nn.Conv2d(
            self.prototype_dim, self.prototype_dim, 1, bias=False)
        self.broad_key = nn.Conv2d(
            self.prototype_dim, self.prototype_dim, 1, bias=False)
        self.broad_value = nn.Conv2d(
            self.prototype_dim, self.prototype_dim, 1, bias=False)

        self.band_queries = nn.ModuleList()
        self.band_keys = nn.ModuleList()
        self.band_values = nn.ModuleList()
        self.residual_mlps = nn.ModuleList()
        for _ in range(self.num_bands):
            self.band_queries.append(nn.Conv2d(
                self.prototype_dim + self.frequency_dim,
                self.frequency_dim, 1, bias=False))
            self.band_keys.append(nn.Conv2d(
                self.frequency_dim, self.frequency_dim, 1, bias=False))
            self.band_values.append(nn.Conv2d(
                self.frequency_dim, self.frequency_dim, 1, bias=False))
            residual = nn.Sequential(
                nn.Conv2d(
                    3 * self.frequency_dim, self.frequency_dim, 1),
                nn.GELU(),
                nn.Conv2d(self.frequency_dim, self.channels, 1))
            nn.init.normal_(residual[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(residual[-1].bias)
            self.residual_mlps.append(residual)

        router_inputs = 3 * self.prototype_dim + 6
        self.band_router = nn.Sequential(
            nn.Conv2d(router_inputs, self.prototype_dim, 1),
            nn.GELU(),
            nn.Conv2d(self.prototype_dim, self.num_bands, 1))
        self.reliability_router = nn.Sequential(
            nn.Conv2d(router_inputs, self.prototype_dim, 1),
            nn.GELU(),
            nn.Conv2d(self.prototype_dim, 1, 1))
        nn.init.zeros_(self.band_router[-1].weight)
        nn.init.zeros_(self.band_router[-1].bias)
        nn.init.zeros_(self.reliability_router[-1].weight)
        nn.init.zeros_(self.reliability_router[-1].bias)

    def _shared_features(self, rgb, thermal, valid_mask):
        valid = valid_mask.to(rgb.dtype)
        rgb_embedding = self.shared_norm(self.shared_projection(rgb)) * valid
        thermal_embedding = (
            self.shared_norm(self.shared_projection(thermal)) * valid)
        return rgb_embedding, thermal_embedding

    def _local_attention(self, query, key, value, valid_mask,
                         normalize_query_key=False):
        output_dtype = value.dtype
        batch, dim, height, width = key.shape
        window = 2 * self.search_radius + 1
        options = window * window
        key_windows = F.unfold(
            key, window, padding=self.search_radius).reshape(
                batch, dim, options, height * width)
        value_windows = F.unfold(
            value, window, padding=self.search_radius).reshape(
                batch, value.shape[1], options, height * width)
        query_flat = query.reshape(
            batch, query.shape[1], height * width).float()
        key_windows = key_windows.float()
        if normalize_query_key:
            query_flat = F.normalize(query_flat, dim=1)
            key_windows = F.normalize(key_windows, dim=1)
        score = (
            query_flat[:, :, None] * key_windows).sum(1) / math.sqrt(dim)
        valid_windows = F.unfold(
            valid_mask.to(score.dtype), window,
            padding=self.search_radius).bool()
        attention = torch.softmax(
            score.masked_fill(~valid_windows, -1e4), dim=1)
        attention = attention * valid_windows.to(attention.dtype)
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-12)
        read = (
            attention[:, None] * value_windows.float()).sum(2).reshape(
                batch, value.shape[1], height, width)
        entropy = -(
            attention.clamp_min(1e-12).log() * attention).sum(1, keepdim=True)
        if options > 1:
            entropy = entropy / math.log(options)
        entropy = entropy.reshape(batch, 1, height, width)
        attention = attention.reshape(batch, options, height, width)
        return (
            read.to(output_dtype), attention.to(output_dtype),
            entropy.to(output_dtype))

    def _seed_losses(self, logits, offsets, log_scales, valid_mask,
                     gt_bboxes, padded_size, level_scale_range):
        zero = logits.sum() * 0.0
        if gt_bboxes is None or padded_size is None:
            return dict(
                seed_loss=zero, offset_loss=zero, scale_loss=zero,
                cardinality_loss=zero,
                eligible_gt_count=zero.detach(),
                unassigned_gt_count=zero.detach())
        targets = build_prldfc_seed_targets(
            logits, offsets, log_scales, gt_bboxes, padded_size,
            valid_mask, level_scale_range,
            matching_radius=self.matching_radius)
        positive = targets['positive_mask'].expand_as(offsets)
        if positive.any():
            offset_loss = F.smooth_l1_loss(
                offsets[positive], targets['offset_target'][positive])
            scale_loss = F.smooth_l1_loss(
                log_scales[positive], targets['scale_target'][positive])
        else:
            offset_loss = zero
            scale_loss = zero
        probability_mass = (
            logits.sigmoid() * valid_mask.to(logits.dtype)).flatten(1).sum(1)
        eligible_per_image = targets['eligible_gt_per_image']
        cardinality_loss = (
            (probability_mass - eligible_per_image).abs() /
            eligible_per_image.clamp_min(1.0)).mean()
        eligible = targets['eligible_gt_count']
        return dict(
            seed_loss=_masked_focal_loss(
                logits, targets['seed_target'], targets['loss_valid'],
                gamma=self.focal_gamma),
            offset_loss=offset_loss,
            scale_loss=scale_loss,
            cardinality_loss=cardinality_loss,
            eligible_gt_count=eligible.detach(),
            unassigned_gt_count=targets['unassigned_gt_count'].detach())

    def _aggregate(self, residual, attention, gate, valid_mask):
        batch, channels, height, width = residual.shape
        window = 2 * self.search_radius + 1
        options = window * window
        weight = gate * attention
        mass = F.fold(
            weight.reshape(batch, options, height * width),
            output_size=(height, width), kernel_size=window,
            padding=self.search_radius)
        numerator_columns = (
            residual[:, :, None] * weight[:, None]).reshape(
                batch, channels * options, height * width)
        numerator = F.fold(
            numerator_columns, output_size=(height, width),
            kernel_size=window, padding=self.search_radius)
        average = numerator / mass.clamp_min(1e-6)
        support = 1.0 - torch.exp(-mass)
        delta = (
            self.residual_epsilon * support * torch.tanh(average) *
            valid_mask.to(average.dtype))
        return delta, mass, support

    def forward(self, rgb, thermal, valid_mask=None, gt_bboxes=None,
                padded_size=None, level_scale_range=(0, 32),
                return_aux=False):
        if tuple(rgb.shape) != tuple(thermal.shape):
            raise ValueError(
                'PRLDFC requires RGB and Thermal features with equal shapes')
        if rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError('features must be BCHW tensors with PRLDFC channels')
        batch, _, height, width = rgb.shape
        if valid_mask is None:
            valid_mask = torch.ones(
                batch, 1, height, width, dtype=torch.bool,
                device=rgb.device)
        if tuple(valid_mask.shape) != (batch, 1, height, width):
            raise ValueError('valid_mask must match PRLDFC features')
        valid_mask = valid_mask.bool()

        masked_thermal = thermal * valid_mask.to(thermal.dtype)
        logits, offsets, log_scales = self.seed_head(masked_thermal)
        probability = logits.sigmoid()
        threshold_logit = math.log(
            self.seed_threshold / (1.0 - self.seed_threshold))
        if self.sparse_inference and not self.training:
            gate = (probability >= self.seed_threshold).to(probability.dtype)
        else:
            gate = torch.sigmoid(
                (logits - threshold_logit) / self.seed_temperature)
        gate = gate * valid_mask.to(gate.dtype)

        rgb_embedding, thermal_embedding = self._shared_features(
            rgb, thermal, valid_mask)
        thermal_avg3 = _valid_average_pool(
            thermal_embedding, valid_mask, 3)
        thermal_detail = thermal_embedding - thermal_avg3
        thermal_context = thermal_avg3 - _valid_average_pool(
            thermal_embedding, valid_mask, 7)
        rgb_avg3 = _valid_average_pool(rgb_embedding, valid_mask, 3)
        rgb_detail = rgb_embedding - rgb_avg3
        rgb_context = rgb_avg3 - _valid_average_pool(
            rgb_embedding, valid_mask, 7)

        thermal_prototype = self.prototype_mlp(torch.cat((
            _sample_with_offsets(thermal_embedding, offsets),
            _sample_with_offsets(thermal_detail, offsets),
            _sample_with_offsets(thermal_context, offsets),
            log_scales), dim=1))
        rgb_prototype, broad_attention, broad_entropy = self._local_attention(
            self.broad_query(thermal_prototype),
            self.broad_key(rgb_embedding), self.broad_value(rgb_embedding),
            valid_mask, normalize_query_key=True)

        rgb_bands, frequency_diagnostics = self.frequency_bank(
            rgb, valid_mask)
        thermal_bands, _ = self.frequency_bank(thermal, valid_mask)
        band_attentions = []
        band_residuals = []
        band_entropies = []
        for band_index in range(self.num_bands):
            thermal_object = _sample_with_offsets(
                thermal_bands[:, band_index], offsets)
            query = self.band_queries[band_index](torch.cat((
                thermal_prototype, thermal_object), dim=1))
            rgb_read, attention, entropy = self._local_attention(
                query,
                self.band_keys[band_index](rgb_bands[:, band_index]),
                self.band_values[band_index](rgb_bands[:, band_index]),
                valid_mask)
            residual = self.residual_mlps[band_index](torch.cat((
                thermal_object, rgb_read, thermal_object - rgb_read), dim=1))
            band_attentions.append(attention)
            band_entropies.append(entropy)
            band_residuals.append(residual)

        contrast_thermal = torch.sqrt(
            thermal_detail.float().square().mean(1, keepdim=True) + 1e-12)
        contrast_rgb = torch.sqrt(
            rgb_detail.float().square().mean(1, keepdim=True) + 1e-12)
        router_input = torch.cat((
            thermal_prototype,
            rgb_prototype,
            thermal_prototype - rgb_prototype,
            probability,
            log_scales,
            broad_entropy,
            _sample_with_offsets(contrast_thermal, offsets).to(rgb.dtype),
            contrast_rgb.to(rgb.dtype)), dim=1)
        band_weights = torch.softmax(self.band_router(router_input), dim=1)
        reliability = torch.sigmoid(self.reliability_router(router_input))

        stacked_residuals = torch.stack(band_residuals, dim=1)
        residual = reliability * (
            band_weights[:, :, None] * stacked_residuals).sum(1)
        stacked_attention = torch.stack(band_attentions, dim=1)
        combined_attention = (
            band_weights[:, :, None] * stacked_attention).sum(1)
        delta, mass, support = self._aggregate(
            residual, combined_attention, gate, valid_mask)
        output = rgb + delta.to(rgb.dtype)

        losses = self._seed_losses(
            logits, offsets, log_scales, valid_mask, gt_bboxes,
            padded_size, level_scale_range)
        aux = dict(
            **losses,
            seed_logits=logits,
            seed_probability=probability,
            seed_gate=gate,
            seed_mass=(probability * valid_mask.to(probability.dtype)).sum() /
            max(batch, 1),
            threshold_candidate_count=(
                (probability >= self.seed_threshold) & valid_mask).float().sum() /
            max(batch, 1),
            frequency_boundaries=frequency_diagnostics['boundaries'],
            band_weights=band_weights,
            reliability=reliability,
            broadband_attention_entropy=broad_entropy.mean(),
            band_attention_entropy=torch.stack(band_entropies).mean(),
            support_ratio=(support > 1e-4).float().mean(),
            overlap_mass=mass.mean(),
            raw_residual_rms=residual.float().square().mean().sqrt(),
            delta_ratio=(
                delta.float().square().mean().sqrt() /
                rgb.float().square().mean().sqrt().clamp_min(1e-12)),
            residual_cap_fraction=(
                delta.abs() >= 0.99 * self.residual_epsilon).float().mean(),
            spatial_support=support,
            delta=delta,
        )
        if return_aux:
            return output, aux
        return output


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
    eligible_per_image = seed_logits.new_zeros(batch)
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
        eligible_count = int(eligible.sum().item())
        eligible_per_image[batch_index] = float(eligible_count)
        eligible_total += eligible_count
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
        eligible_gt_per_image=eligible_per_image,
        eligible_gt_count=seed_logits.new_tensor(float(eligible_total)),
        unassigned_gt_count=seed_logits.new_tensor(float(unassigned_total)),
    )
