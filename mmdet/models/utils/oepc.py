"""Object-centric RGB calibration before COXNet AAM.

OEPC replaces CLFM at the same RGB/Thermal FPN stage. RGB and Thermal propose
candidates independently, then use local soft retrieval without treating it as
an identity match. Separate foreground heads build local object-minus-context
descriptors; GT boxes supervise only fixed-coordinate center, foreground/bag,
and contrastive objectives. The original Thermal feature is never modified.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trpc import ChannelLayerNorm, balanced_binary_focal_loss


def build_center_targets(gt_boxes, padded_size, feat_size, device):
    """Build one Thermal-coordinate center target per GT box."""
    input_h, input_w = padded_size
    feat_h, feat_w = feat_size
    scale_x = feat_w / float(input_w)
    scale_y = feat_h / float(input_h)
    target = torch.zeros(
        len(gt_boxes), 1, feat_h, feat_w, device=device)
    for batch_idx, boxes in enumerate(gt_boxes):
        if boxes is None or not len(boxes):
            continue
        for box in boxes.detach().cpu().tolist():
            center_x = 0.5 * (float(box[0]) + float(box[2]))
            center_y = 0.5 * (float(box[1]) + float(box[3]))
            col = max(0, min(int(math.floor(center_x * scale_x)), feat_w - 1))
            row = max(0, min(int(math.floor(center_y * scale_y)), feat_h - 1))
            target[batch_idx, 0, row, col] = 1.0
    return target


def _balanced_masked_mean(loss, foreground, valid):
    """Balance foreground and background contributions for every image."""
    foreground = foreground.bool() & valid.bool()
    background = (~foreground) & valid.bool()
    per_image = []
    for batch_idx in range(loss.shape[0]):
        parts = []
        if foreground[batch_idx].any():
            parts.append(loss[batch_idx][foreground[batch_idx]].mean())
        if background[batch_idx].any():
            parts.append(loss[batch_idx][background[batch_idx]].mean())
        if parts:
            per_image.append(torch.stack(parts).mean())
    if not per_image:
        return loss.sum() * 0.0
    return torch.stack(per_image).mean()


def _masked_mean(value, mask):
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _dirichlet_kl_to_uniform(alpha):
    """KL(Dir(alpha) || Dir(1)) for a two-class prediction."""
    alpha = alpha.float()
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    classes = alpha.shape[1]
    log_normalizer = (
        torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(torch.tensor(float(classes), device=alpha.device)))
    digamma_term = ((alpha - 1.0) * (
        torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(
            dim=1, keepdim=True)
    return log_normalizer + digamma_term


def evidential_binary_loss(evidence, target, valid, kl_weight=1e-3):
    """Expected CE plus incorrect-evidence KL for BG/FG Dirichlet output."""
    evidence = evidence.float().clamp_min(0.0)
    alpha = evidence + 1.0
    target = target.float()
    labels = torch.cat([1.0 - target, target], dim=1)
    strength = alpha.sum(dim=1, keepdim=True)
    expected_ce = (labels * (
        torch.digamma(strength) - torch.digamma(alpha))).sum(
            dim=1, keepdim=True)
    adjusted_alpha = labels + (1.0 - labels) * alpha
    per_cell = expected_ce + float(kl_weight) * _dirichlet_kl_to_uniform(
        adjusted_alpha)
    return _balanced_masked_mean(per_cell, target > 0.5, valid.bool())


class ObjectCentricEvidentialCalibration(nn.Module):
    """Calibrate RGB from sparse dual-modality object-context evidence."""

    def __init__(self, channels=256, embed_dim=64, contrast_dim=32,
                 object_kernel=3, context_kernel=7, search_radius=2,
                 search_temperature=0.2, distance_prior_weight=0.1,
                 candidate_prior=0.1, candidate_threshold=0.05,
                 max_candidates=100, peak_kernel=3, support_kernel=3,
                 residual_scale=0.2, feature_scale_floor=0.1,
                 modulation_init_std=1e-2, edl_kl_weight=1e-3,
                 focal_gamma=2.0, use_rgb_candidates=True, use_edl=True,
                 use_detector_utility=False):
        super().__init__()
        if channels < 1 or embed_dim < 1 or contrast_dim < 1:
            raise ValueError('channel dimensions must be positive')
        for name, kernel in (
                ('object_kernel', object_kernel),
                ('context_kernel', context_kernel),
                ('peak_kernel', peak_kernel),
                ('support_kernel', support_kernel)):
            if kernel < 1 or kernel % 2 == 0:
                raise ValueError(f'{name} must be a positive odd integer')
        if context_kernel <= object_kernel:
            raise ValueError('context_kernel must exceed object_kernel')
        if search_radius < 0 or max_candidates < 1:
            raise ValueError('search_radius/max_candidates is invalid')
        if search_temperature <= 0:
            raise ValueError('search_temperature must be positive')
        if residual_scale <= 0 or feature_scale_floor <= 0:
            raise ValueError('residual cap parameters must be positive')
        if modulation_init_std <= 0 or distance_prior_weight < 0:
            raise ValueError('initialization/prior parameters are invalid')
        if not 0.0 <= candidate_threshold <= 1.0:
            raise ValueError('candidate_threshold must be in [0, 1]')

        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        self.contrast_dim = int(contrast_dim)
        self.object_kernel = int(object_kernel)
        self.context_kernel = int(context_kernel)
        self.search_radius = int(search_radius)
        self.search_kernel = 2 * self.search_radius + 1
        self.search_temperature = float(search_temperature)
        self.distance_prior_weight = float(distance_prior_weight)
        self.candidate_threshold = float(candidate_threshold)
        self.max_candidates = int(max_candidates)
        self.peak_kernel = int(peak_kernel)
        self.support_kernel = int(support_kernel)
        self.edl_kl_weight = float(edl_kl_weight)
        self.focal_gamma = float(focal_gamma)
        self.feature_scale_floor = float(feature_scale_floor)
        self.use_rgb_candidates = bool(use_rgb_candidates)
        self.use_edl = bool(use_edl)
        self.use_detector_utility = bool(use_detector_utility)
        self.register_buffer(
            'residual_scale', torch.tensor(float(residual_scale)))

        axis = torch.arange(-self.search_radius,
                            self.search_radius + 1).float()
        offset_y, offset_x = torch.meshgrid(axis, axis)
        distance = (offset_x.square() + offset_y.square()).sqrt()
        if self.search_radius > 0:
            distance = distance / float(self.search_radius)
        self.register_buffer(
            'distance_penalty', distance.reshape(1, -1, 1))

        object_axis = torch.arange(object_kernel).float()
        object_axis = object_axis - (object_kernel - 1) / 2.0
        object_y, object_x = torch.meshgrid(object_axis, object_axis)
        sigma = max(object_kernel / 3.0, 0.5)
        center_prior = torch.exp(
            -(object_x.square() + object_y.square()) / (2.0 * sigma * sigma))
        self.register_buffer(
            'center_prior', center_prior.reshape(
                1, 1, object_kernel, object_kernel))
        context_ring = torch.ones(context_kernel, context_kernel)
        inner_start = (context_kernel - object_kernel) // 2
        context_ring[inner_start:inner_start + object_kernel,
                     inner_start:inner_start + object_kernel] = 0.0
        self.register_buffer(
            'context_ring', context_ring.reshape(
                1, 1, context_kernel, context_kernel))

        self.rgb_embed = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.thermal_embed = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.rgb_embed_norm = ChannelLayerNorm(embed_dim)
        self.thermal_embed_norm = ChannelLayerNorm(embed_dim)
        self.thermal_value = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)

        def make_prediction_head():
            return nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1,
                          groups=embed_dim, bias=False),
                nn.GELU(),
                nn.Conv2d(embed_dim, 1, 1))

        self.thermal_candidate_head = make_prediction_head()
        self.rgb_candidate_head = make_prediction_head()
        self.thermal_foreground_head = make_prediction_head()
        self.rgb_foreground_head = make_prediction_head()
        prior = min(max(float(candidate_prior), 1e-4), 1.0 - 1e-4)
        prior_bias = math.log(prior / (1.0 - prior))
        for head in (self.thermal_candidate_head, self.rgb_candidate_head,
                     self.thermal_foreground_head, self.rgb_foreground_head):
            nn.init.constant_(head[-1].bias, prior_bias)

        self.rgb_evidence_head = nn.Conv2d(embed_dim, 2, 1)
        self.thermal_evidence_head = nn.Conv2d(embed_dim, 2, 1)
        self.rgb_contrast = nn.Conv2d(embed_dim, contrast_dim, 1, bias=False)
        self.thermal_contrast = nn.Conv2d(
            embed_dim, contrast_dim, 1, bias=False)

        hidden = max(embed_dim, 32)
        # RGB/Thermal descriptors plus p_fg(T/R), uncertainty(T/R), matching
        # entropy, modality disagreement/confidence, candidate origins, and
        # modality-specific context availability.
        self.router = nn.Sequential(
            nn.Conv2d(2 * embed_dim + 11, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, 2, 1))
        nn.init.zeros_(self.router[-1].weight)
        # Preserve is initially preferred, but transfer remains non-zero so
        # the residual branch receives detection gradients from iteration one.
        nn.init.constant_(self.router[-1].bias[0], 1.0)
        nn.init.constant_(self.router[-1].bias[1], -1.0)

        self.film_hidden = nn.Sequential(
            nn.Conv2d(2 * embed_dim, hidden, 1), nn.GELU())
        self.film_out = nn.Conv2d(hidden, 2 * channels, 1)
        nn.init.normal_(self.film_out.weight, 0.0,
                        float(modulation_init_std))
        nn.init.zeros_(self.film_out.bias)
        self.rgb_norm = ChannelLayerNorm(channels)

    @staticmethod
    def _as_valid_mask(valid_mask, feature):
        b, _, h, w = feature.shape
        if valid_mask is None:
            return torch.ones(
                b, 1, h, w, dtype=torch.bool, device=feature.device)
        if tuple(valid_mask.shape) == (b, h, w):
            valid_mask = valid_mask[:, None]
        if tuple(valid_mask.shape) != (b, 1, h, w):
            raise ValueError(
                f'valid mask {tuple(valid_mask.shape)} does not match '
                f'{(b, 1, h, w)}')
        return valid_mask.bool()

    @staticmethod
    def _probability_uncertainty(evidence):
        alpha = evidence.float() + 1.0
        strength = alpha.sum(dim=1, keepdim=True).clamp_min(2.0)
        return alpha[:, 1:2] / strength, 2.0 / strength

    @staticmethod
    def _weighted_filter(feature, weight, kernel):
        channels = feature.shape[1]
        expanded = kernel.to(feature).expand(channels, 1, -1, -1)
        numerator = F.conv2d(
            feature.float() * weight.float(), expanded,
            padding=kernel.shape[-1] // 2, groups=channels)
        denominator = F.conv2d(
            weight.float(), kernel.to(weight),
            padding=kernel.shape[-1] // 2)
        return numerator / denominator.clamp_min(1e-6), denominator

    def _object_context(self, feature, foreground_probability, uncertainty,
                        valid):
        """Predicted weighted object/context prototypes for the main path."""
        valid_float = valid.to(feature.dtype)
        object_weight = foreground_probability * valid_float
        object_proto, object_mass = self._weighted_filter(
            feature, object_weight, self.center_prior)
        reliable_background = (
            (1.0 - foreground_probability) * (1.0 - uncertainty) * valid_float)
        context_proto, context_mass = self._weighted_filter(
            feature, reliable_background, self.context_ring)
        object_available = object_mass > 1e-6
        context_available = context_mass > 1e-6
        descriptor_available = object_available & context_available
        difference = object_proto - context_proto
        difference = torch.where(
            descriptor_available.expand_as(difference), difference,
            torch.zeros_like(difference))
        return (object_proto, context_proto, difference, object_mass,
                context_mass, descriptor_available)

    def _local_attention(self, query, key, valid):
        """Read a soft local key bag for every query-grid location."""
        b, channels, h, w = query.shape
        locations = h * w
        candidates = F.unfold(
            key.float(), kernel_size=self.search_kernel,
            padding=self.search_radius)
        candidates = candidates.reshape(
            b, channels, self.search_kernel ** 2, locations)
        query_flat = F.normalize(query.float(), dim=1).reshape(
            b, channels, 1, locations)
        similarity = (query_flat * F.normalize(
            candidates, dim=1)).sum(dim=1)
        logits = (similarity - self.distance_prior_weight *
                  self.distance_penalty.to(similarity.dtype))
        logits = logits / self.search_temperature

        candidate_valid = F.unfold(
            valid.float(), kernel_size=self.search_kernel,
            padding=self.search_radius)
        candidate_valid = candidate_valid.reshape(
            b, self.search_kernel ** 2, locations) > 0.5
        logits = logits.masked_fill(~candidate_valid, -1e4)
        attention = logits.softmax(dim=1)
        attention = attention * valid.flatten(2).to(attention.dtype)
        confidence = attention.max(dim=1, keepdim=True).values
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum(
            dim=1, keepdim=True)
        valid_options = candidate_valid.sum(dim=1, keepdim=True)
        entropy_scale = valid_options.clamp_min(2).to(entropy.dtype).log()
        entropy = torch.where(
            valid_options > 1, entropy / entropy_scale,
            torch.zeros_like(entropy))
        return (attention, confidence.reshape(b, 1, h, w),
                entropy.reshape(b, 1, h, w))

    def _select(self, value, attention):
        """Softly read RGB values for Thermal-coordinate queries."""
        b, channels, h, w = value.shape
        candidates = F.unfold(
            value.float(), kernel_size=self.search_kernel,
            padding=self.search_radius)
        candidates = candidates.reshape(
            b, channels, self.search_kernel ** 2, h * w)
        selected = (candidates * attention[:, None]).sum(dim=2)
        return selected.reshape(b, channels, h, w)

    def _sparse_support(self, probability, valid):
        """Select peaks/top-k strictly in Thermal coordinates."""
        local_max = F.max_pool2d(
            probability, self.peak_kernel, stride=1,
            padding=self.peak_kernel // 2)
        is_peak = (probability >= local_max - 1e-7) & valid
        scores = probability * is_peak.to(probability.dtype)
        batch, _, height, width = scores.shape
        flat = scores.flatten(1)
        count = min(self.max_candidates, flat.shape[1])
        top_values, top_indices = flat.topk(count, dim=1)
        selected = top_values >= self.candidate_threshold
        peak_flat = torch.zeros_like(flat)
        peak_flat.scatter_(1, top_indices, selected.to(flat.dtype))
        peak = peak_flat.reshape(batch, 1, height, width).detach()
        return peak * valid.to(peak.dtype)

    @staticmethod
    def _shift_slices(length, offset):
        if offset >= 0:
            return slice(0, length - offset), slice(offset, length)
        return slice(-offset, length), slice(0, length + offset)

    def _scatter_to_rgb(self, value, attention, peak, valid):
        """Scatter Thermal candidate values into their soft RGB local bags."""
        batch, channels, height, width = value.shape
        numerator = value.new_zeros(batch, channels, height, width).float()
        mass_out = value.new_zeros(batch, 1, height, width).float()
        index = 0
        for offset_y in range(-self.search_radius, self.search_radius + 1):
            src_y, dst_y = self._shift_slices(height, offset_y)
            for offset_x in range(-self.search_radius,
                                  self.search_radius + 1):
                src_x, dst_x = self._shift_slices(width, offset_x)
                mass = (peak * attention[:, index:index + 1].reshape(
                    batch, 1, height, width))[:, :, src_y, src_x]
                numerator[:, :, dst_y, dst_x] += (
                    value[:, :, src_y, src_x].float() * mass)
                mass_out[:, :, dst_y, dst_x] += mass
                index += 1
        numerator = numerator * valid.to(numerator.dtype)
        mass_out = mass_out * valid.to(mass_out.dtype)
        return numerator, mass_out

    @staticmethod
    def _window_sum(tensor, kernel):
        return F.avg_pool2d(
            tensor, kernel, stride=1, padding=kernel // 2) * kernel * kernel

    def _candidate_fields(self, peak, candidate_probability, descriptor,
                          foreground_probability, uncertainty, confidence,
                          entropy, context_available, attention, valid):
        """Map Thermal-origin candidates into possible RGB locations."""
        score = peak * candidate_probability
        packed = torch.cat([
            descriptor, foreground_probability, uncertainty,
            confidence, entropy, candidate_probability,
            context_available.to(descriptor.dtype)], dim=1)
        numerator, mass = self._scatter_to_rgb(
            packed, attention, score, valid)
        numerator = self._window_sum(numerator, self.support_kernel)
        mass = self._window_sum(mass, self.support_kernel)
        packed_rgb = numerator / mass.clamp_min(1e-6)
        support = (mass > 0).to(packed_rgb.dtype) * valid.to(packed_rgb.dtype)
        descriptor_rgb = packed_rgb[:, :self.embed_dim]
        scalar = packed_rgb[:, self.embed_dim:]
        return dict(
            thermal_condition=descriptor_rgb,
            foreground_thermal=scalar[:, 0:1],
            uncertainty_thermal=scalar[:, 1:2],
            matching_confidence=scalar[:, 2:3],
            matching_entropy=scalar[:, 3:4],
            candidate_score=scalar[:, 4:5],
            context_thermal=scalar[:, 5:6],
            origin_thermal=support,
            origin_rgb=torch.zeros_like(support),
            weight=mass,
            support=support)

    def _rgb_candidate_fields(self, peak, candidate_probability,
                              thermal_condition, foreground_thermal,
                              uncertainty_thermal, confidence, entropy,
                              context_thermal, valid):
        """Expand RGB-origin candidates without mixing coordinate systems."""
        score = peak * candidate_probability
        packed = torch.cat([
            thermal_condition, foreground_thermal, uncertainty_thermal,
            confidence, entropy, candidate_probability,
            context_thermal.to(thermal_condition.dtype)], dim=1)
        numerator = self._window_sum(packed * score, self.support_kernel)
        mass = self._window_sum(score, self.support_kernel)
        packed_rgb = numerator / mass.clamp_min(1e-6)
        support = (mass > 0).to(packed_rgb.dtype) * valid.to(packed_rgb.dtype)
        descriptor_rgb = packed_rgb[:, :self.embed_dim]
        scalar = packed_rgb[:, self.embed_dim:]
        return dict(
            thermal_condition=descriptor_rgb,
            foreground_thermal=scalar[:, 0:1],
            uncertainty_thermal=scalar[:, 1:2],
            matching_confidence=scalar[:, 2:3],
            matching_entropy=scalar[:, 3:4],
            candidate_score=scalar[:, 4:5],
            context_thermal=scalar[:, 5:6],
            origin_thermal=torch.zeros_like(support),
            origin_rgb=support,
            weight=mass,
            support=support)

    @staticmethod
    def _merge_candidate_fields(thermal_fields, rgb_fields, valid):
        """Union candidate regions after each modality has defined its map."""
        if rgb_fields is None:
            return thermal_fields
        weight_t = thermal_fields['weight']
        weight_r = rgb_fields['weight']
        total = (weight_t + weight_r).clamp_min(1e-6)
        merged = {}
        for key in (
                'thermal_condition', 'foreground_thermal',
                'uncertainty_thermal', 'matching_confidence',
                'matching_entropy', 'candidate_score', 'context_thermal'):
            merged[key] = (
                thermal_fields[key] * weight_t + rgb_fields[key] * weight_r
            ) / total
        merged['origin_thermal'] = thermal_fields['support']
        merged['origin_rgb'] = rgb_fields['support']
        merged['weight'] = weight_t + weight_r
        merged['support'] = (
            (merged['weight'] > 0).to(total.dtype) * valid.to(total.dtype))
        return merged

    def _route_and_modulate(self, rgb, rgb_descriptor,
                            foreground_rgb, uncertainty_rgb,
                            context_rgb, fields):
        support = fields['support']
        foreground_thermal = fields['foreground_thermal']
        uncertainty_thermal = fields['uncertainty_thermal']
        disagreement = (foreground_thermal - foreground_rgb).abs()
        route_input = torch.cat([
            rgb_descriptor, fields['thermal_condition'],
            foreground_thermal, foreground_rgb,
            uncertainty_thermal, uncertainty_rgb,
            fields['matching_entropy'], disagreement,
            fields['matching_confidence'], fields['origin_thermal'],
            fields['origin_rgb'], fields['context_thermal'],
            context_rgb.to(rgb_descriptor.dtype)], dim=1)
        route = self.router(route_input.float()).softmax(dim=1)
        transfer_weight = route[:, 1:2]
        film_input = torch.cat(
            [rgb_descriptor, fields['thermal_condition']], dim=1)
        film_raw = self.film_out(self.film_hidden(film_input.float()))
        scale, shift = film_raw.chunk(2, dim=1)
        modulation = (
            scale.tanh() * self.rgb_norm(rgb).float() + shift.tanh())
        correction_support = support * (
            fields['context_thermal'] > 0.5).to(support.dtype)
        trial_delta = self._cap_residual(
            modulation * correction_support, rgb)
        delta = self._cap_residual(
            transfer_weight * trial_delta * correction_support, rgb)
        return (delta, trial_delta, transfer_weight, film_raw,
                correction_support)

    def _cap_residual(self, residual, rgb):
        """Cap every spatial correction vector relative to RGB magnitude."""
        rgb_norm = torch.linalg.vector_norm(
            rgb.float(), dim=1, keepdim=True)
        scale = rgb_norm.clamp_min(self.feature_scale_floor)
        maximum = self.residual_scale.float() * scale
        residual_norm = torch.linalg.vector_norm(
            residual.float(), dim=1, keepdim=True).clamp_min(1e-12)
        factor = torch.minimum(
            torch.ones_like(residual_norm), maximum / residual_norm)
        return residual.float() * factor

    @staticmethod
    def _sample_peak(peak, score):
        """Sample one Thermal candidate globally for detector utility."""
        weighted = (peak * score).flatten()
        sample = torch.zeros_like(weighted)
        available = weighted > 0
        if available.any():
            selected = torch.multinomial(weighted.detach().clamp_min(0), 1)
            sample[selected] = 1.0
        return sample.reshape_as(peak).detach(), bool(available.any())

    def _contrastive_loss(self, thermal_descriptor, rgb_descriptor,
                          center_target, safe_thermal_background,
                          safe_rgb_background, valid, attention):
        """Misalignment-tolerant local object-context contrast."""
        thermal_z = F.normalize(
            self.thermal_contrast(thermal_descriptor).float(), dim=1)
        rgb_z = F.normalize(
            self.rgb_contrast(rgb_descriptor).float(), dim=1)
        # The correspondence is detached here: a learned attention map must
        # not move a GT-positive label to make the auxiliary loss easier.
        rgb_positive = F.normalize(
            self._select(rgb_z, attention.detach()), dim=1)
        positive_similarity = (
            thermal_z * rgb_positive).sum(dim=1, keepdim=True)
        per_image = []
        for batch_idx in range(thermal_z.shape[0]):
            centers = center_target[batch_idx, 0] > 0.5
            thermal_background = (
                safe_thermal_background[batch_idx, 0] &
                valid[batch_idx, 0])
            rgb_background = (
                safe_rgb_background[batch_idx, 0] & valid[batch_idx, 0])
            if (not centers.any() or not thermal_background.any() or
                    not rgb_background.any()):
                continue
            rgb_background_proto = F.normalize(
                rgb_z[batch_idx, :, rgb_background].mean(dim=1), dim=0)
            thermal_background_proto = F.normalize(
                thermal_z[batch_idx, :, thermal_background].mean(dim=1),
                dim=0)
            anchors = thermal_z[batch_idx, :, centers].transpose(0, 1)
            positives = positive_similarity[batch_idx, 0, centers]
            rgb_negative = anchors @ rgb_background_proto
            positive_rgb = rgb_positive[
                batch_idx, :, centers].transpose(0, 1)
            thermal_negative = positive_rgb @ thermal_background_proto
            loss = 0.5 * (
                F.softplus((rgb_negative - positives) /
                           self.search_temperature) +
                F.softplus((thermal_negative - positives) /
                           self.search_temperature))
            per_image.append(loss.mean())
        if not per_image:
            return thermal_descriptor.sum() * 0.0
        return torch.stack(per_image).mean()

    def _weak_bag_loss(self, logits, center_target, safe_background, valid):
        """Fixed local bag supervision that cannot move labels via attention."""
        probability = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        candidates = F.unfold(
            probability, kernel_size=self.search_kernel,
            padding=self.search_radius)
        candidates = candidates.reshape(
            logits.shape[0], self.search_kernel ** 2,
            logits.shape[-2], logits.shape[-1])
        bag_probability = candidates.amax(dim=1, keepdim=True)
        positive_mask = (center_target > 0.5) & valid
        positive = _masked_mean(-bag_probability.log(), positive_mask)
        background_mask = safe_background & valid
        if not background_mask.any():
            return positive
        background = _masked_mean(
            -torch.log1p(-probability), background_mask)
        return 0.5 * (positive + background)

    def _evidential_loss(self, thermal_evidence, rgb_evidence,
                         foreground_target, center_target,
                         safe_rgb_background, valid):
        """Thermal pixel supervision plus weak RGB bag/background supervision."""
        loss_thermal = evidential_binary_loss(
            thermal_evidence, foreground_target, valid, self.edl_kl_weight)
        rgb_probability, _ = self._probability_uncertainty(rgb_evidence)
        candidates = F.unfold(
            rgb_probability, kernel_size=self.search_kernel,
            padding=self.search_radius)
        candidates = candidates.reshape(
            rgb_probability.shape[0], self.search_kernel ** 2,
            rgb_probability.shape[-2], rgb_probability.shape[-1])
        bag_probability = candidates.amax(
            dim=1, keepdim=True).clamp(1e-6, 1.0 - 1e-6)
        positive_mask = (center_target > 0.5) & valid
        positive = _masked_mean(-bag_probability.log(), positive_mask)
        background_mask = safe_rgb_background & valid
        if background_mask.any():
            background = evidential_binary_loss(
                rgb_evidence, torch.zeros_like(foreground_target),
                background_mask, self.edl_kl_weight)
            loss_rgb = 0.5 * (positive + background)
        else:
            loss_rgb = positive
        return 0.5 * (loss_thermal + loss_rgb)

    def forward(self, rgb, thermal, valid_mask=None, center_target=None,
                foreground_target=None, return_aux=False):
        if rgb.shape != thermal.shape:
            raise ValueError(
                'OEPC requires equal same-stage RGB/Thermal shapes, got '
                f'{tuple(rgb.shape)} and {tuple(thermal.shape)}')
        if rgb.shape[1] != self.channels:
            raise ValueError('OEPC channel mismatch')

        valid = self._as_valid_mask(valid_mask, rgb)
        rgb_base = self.rgb_embed_norm(self.rgb_embed(rgb))
        thermal_base = self.thermal_embed_norm(self.thermal_embed(thermal))
        thermal_candidate_logits = self.thermal_candidate_head(thermal_base)
        rgb_candidate_logits = self.rgb_candidate_head(rgb_base)
        thermal_candidate_probability = thermal_candidate_logits.sigmoid()
        rgb_candidate_probability = rgb_candidate_logits.sigmoid()

        rgb_foreground_logits = self.rgb_foreground_head(rgb_base)
        thermal_foreground_logits = self.thermal_foreground_head(thermal_base)
        foreground_rgb = rgb_foreground_logits.sigmoid()
        foreground_thermal = thermal_foreground_logits.sigmoid()

        rgb_evidence_dense = None
        thermal_evidence_dense = None
        if self.use_edl:
            rgb_evidence_dense = F.softplus(self.rgb_evidence_head(rgb_base))
            thermal_evidence_dense = F.softplus(
                self.thermal_evidence_head(self.thermal_value(thermal_base)))
            _, uncertainty_rgb = self._probability_uncertainty(
                rgb_evidence_dense)
            _, uncertainty_thermal = self._probability_uncertainty(
                thermal_evidence_dense)
        else:
            uncertainty_rgb = 1.0 - (2.0 * foreground_rgb - 1.0).abs()
            uncertainty_thermal = (
                1.0 - (2.0 * foreground_thermal - 1.0).abs())

        (_, _, rgb_difference, _, _, context_rgb) = self._object_context(
            rgb, foreground_rgb, uncertainty_rgb, valid)
        (_, _, thermal_difference, _, _, context_thermal) = (
            self._object_context(
                thermal, foreground_thermal, uncertainty_thermal, valid))
        rgb_descriptor = self.rgb_embed_norm(self.rgb_embed(rgb_difference))
        thermal_descriptor = self.thermal_embed_norm(
            self.thermal_embed(thermal_difference))
        thermal_descriptor = self.thermal_value(thermal_descriptor)

        # Candidate existence is decided separately in each modality. Thermal
        # candidates are then mapped to possible RGB bags; RGB candidates stay
        # in RGB coordinates and read local Thermal conditions.
        thermal_peak = self._sparse_support(
            thermal_candidate_probability, valid)
        rgb_peak = self._sparse_support(
            rgb_candidate_probability, valid) if self.use_rgb_candidates \
            else torch.zeros_like(thermal_peak)
        thermal_to_rgb, confidence_t, entropy_t = self._local_attention(
            thermal_descriptor, rgb_descriptor, valid)
        rgb_to_thermal, confidence_r, entropy_r = self._local_attention(
            rgb_descriptor, thermal_descriptor, valid)

        thermal_fields = self._candidate_fields(
            thermal_peak, thermal_candidate_probability, thermal_descriptor,
            foreground_thermal, uncertainty_thermal,
            confidence_t, entropy_t, context_thermal,
            thermal_to_rgb, valid)
        rgb_fields = None
        if self.use_rgb_candidates:
            thermal_at_rgb = self._select(
                thermal_descriptor, rgb_to_thermal)
            foreground_thermal_at_rgb = self._select(
                foreground_thermal, rgb_to_thermal)
            uncertainty_thermal_at_rgb = self._select(
                uncertainty_thermal, rgb_to_thermal)
            context_thermal_at_rgb = self._select(
                context_thermal.to(thermal_descriptor.dtype),
                rgb_to_thermal)
            rgb_fields = self._rgb_candidate_fields(
                rgb_peak, rgb_candidate_probability, thermal_at_rgb,
                foreground_thermal_at_rgb, uncertainty_thermal_at_rgb,
                confidence_r, entropy_r, context_thermal_at_rgb, valid)
        fields = self._merge_candidate_fields(
            thermal_fields, rgb_fields, valid)
        (delta, _, transfer_weight, film_raw,
         correction_support) = self._route_and_modulate(
            rgb, rgb_descriptor, foreground_rgb, uncertainty_rgb,
            context_rgb, fields)
        rgb_calibrated = rgb + delta.to(rgb.dtype)

        if not return_aux:
            return rgb_calibrated

        support_bool = correction_support > 0
        with torch.no_grad():
            base_norm = rgb.float().pow(2).mean(
                dim=(1, 2, 3)).sqrt().clamp_min(1e-6)
            delta_norm = delta.pow(2).mean(dim=(1, 2, 3)).sqrt()
            valid_count = valid.sum().clamp_min(1)
            cap_denominator = self.residual_scale * torch.linalg.vector_norm(
                rgb.float(), dim=1).clamp_min(self.feature_scale_floor)
            capped_ratio = (
                torch.linalg.vector_norm(delta, dim=1) /
                cap_denominator).amax()

        aux = dict(
            candidate_logits=thermal_candidate_logits,
            candidate_probability=thermal_candidate_probability.detach(),
            candidate_peaks=thermal_peak.detach(),
            thermal_candidate_logits=thermal_candidate_logits,
            rgb_candidate_logits=rgb_candidate_logits,
            thermal_candidate_probability=(
                thermal_candidate_probability.detach()),
            rgb_candidate_probability=rgb_candidate_probability.detach(),
            thermal_candidate_peaks=thermal_peak.detach(),
            rgb_candidate_peaks=rgb_peak.detach(),
            candidate_support=fields['support'].detach(),
            spatial_support=correction_support.detach(),
            transfer_weight=transfer_weight.detach(),
            foreground_rgb=foreground_rgb.detach(),
            foreground_thermal=fields['foreground_thermal'].detach(),
            uncertainty_rgb=uncertainty_rgb.detach(),
            uncertainty_thermal=fields['uncertainty_thermal'].detach(),
            matching_confidence=fields['matching_confidence'].detach(),
            matching_entropy=fields['matching_entropy'].detach(),
            local_attention_entropy=_masked_mean(
                fields['matching_entropy'], support_bool).detach(),
            matching_confidence_mean=_masked_mean(
                fields['matching_confidence'], support_bool).detach(),
            candidate_mean=_masked_mean(
                fields['candidate_score'], support_bool).detach(),
            candidate_count=(
                thermal_peak.sum(dim=(1, 2, 3)) +
                rgb_peak.sum(dim=(1, 2, 3))).mean().detach(),
            candidate_count_thermal=thermal_peak.sum(
                dim=(1, 2, 3)).mean().detach(),
            candidate_count_rgb=rgb_peak.sum(
                dim=(1, 2, 3)).mean().detach(),
            candidate_support_ratio=(
                fields['support'].sum() / valid_count).detach(),
            support_ratio=(correction_support.sum() / valid_count).detach(),
            transfer_mean=_masked_mean(
                transfer_weight, support_bool).detach(),
            foreground_rgb_mean=_masked_mean(
                foreground_rgb, support_bool).detach(),
            foreground_thermal_mean=_masked_mean(
                fields['foreground_thermal'], support_bool).detach(),
            uncertainty_rgb_mean=_masked_mean(
                uncertainty_rgb, support_bool).detach(),
            uncertainty_thermal_mean=_masked_mean(
                fields['uncertainty_thermal'], support_bool).detach(),
            context_rgb_ratio=_masked_mean(
                context_rgb.float(), fields['support'] > 0).detach(),
            context_thermal_ratio=_masked_mean(
                fields['context_thermal'], fields['support'] > 0).detach(),
            film_raw_rms=_masked_mean(
                film_raw.float().square().mean(dim=1, keepdim=True).sqrt(),
                support_bool).detach(),
            delta_ratio=(delta_norm / base_norm).mean().detach(),
            residual_cap_ratio=capped_ratio.detach(),
            residual_scale=self.residual_scale.detach(),
            utility_payload=None,
            utility_sampled=rgb.new_tensor(0.0))

        if center_target is not None or foreground_target is not None:
            if center_target is None or foreground_target is None:
                raise ValueError(
                    'center_target and foreground_target must be provided together')
            expected = tuple(thermal_candidate_logits.shape)
            if tuple(center_target.shape) != expected:
                raise ValueError('center_target must match candidate logits')
            if tuple(foreground_target.shape) != expected:
                raise ValueError('foreground_target must match candidate logits')
            safe_thermal_background = valid & ~(foreground_target > 0.5)
            expanded_objects = F.max_pool2d(
                foreground_target.float(), self.search_kernel, stride=1,
                padding=self.search_radius) > 0.5
            safe_rgb_background = valid & ~expanded_objects
            thermal_candidate_loss = balanced_binary_focal_loss(
                thermal_candidate_logits, center_target, valid,
                gamma=self.focal_gamma)
            rgb_candidate_loss = self._weak_bag_loss(
                rgb_candidate_logits, center_target,
                safe_rgb_background, valid)
            aux['candidate_loss'] = 0.5 * (
                thermal_candidate_loss + rgb_candidate_loss)
            thermal_foreground_loss = balanced_binary_focal_loss(
                thermal_foreground_logits, foreground_target, valid,
                gamma=self.focal_gamma)
            rgb_foreground_loss = self._weak_bag_loss(
                rgb_foreground_logits, center_target,
                safe_rgb_background, valid)
            aux['foreground_loss'] = 0.5 * (
                thermal_foreground_loss + rgb_foreground_loss)
            aux['contrastive_loss'] = self._contrastive_loss(
                thermal_descriptor, rgb_descriptor, center_target,
                safe_thermal_background, safe_rgb_background, valid,
                thermal_to_rgb)
            if self.use_edl:
                aux['edl_loss'] = self._evidential_loss(
                    thermal_evidence_dense, rgb_evidence_dense,
                    foreground_target, center_target,
                    safe_rgb_background, valid)
            else:
                aux['edl_loss'] = rgb.sum() * 0.0

            sample_peak, has_sample = self._sample_peak(
                thermal_peak, thermal_candidate_probability)
            if self.use_detector_utility and has_sample:
                other_peak = (thermal_peak - sample_peak).clamp_min(0.0)
                other_fields = self._candidate_fields(
                    other_peak, thermal_candidate_probability,
                    thermal_descriptor,
                    foreground_thermal, uncertainty_thermal,
                    confidence_t, entropy_t, context_thermal,
                    thermal_to_rgb, valid)
                other_fields = self._merge_candidate_fields(
                    other_fields, rgb_fields, valid)
                other_delta, _, _, _, _ = self._route_and_modulate(
                    rgb, rgb_descriptor, foreground_rgb, uncertainty_rgb,
                    context_rgb, other_fields)
                sample_fields = self._candidate_fields(
                    sample_peak, thermal_candidate_probability,
                    thermal_descriptor,
                    foreground_thermal, uncertainty_thermal,
                    confidence_t, entropy_t, context_thermal,
                    thermal_to_rgb, valid)
                _, sample_trial_delta, sample_transfer, _, _ = (
                    self._route_and_modulate(
                        rgb, rgb_descriptor, foreground_rgb, uncertainty_rgb,
                        context_rgb, sample_fields))
                combined_trial = self._cap_residual(
                    other_delta + sample_trial_delta, rgb)
                flat_index = sample_peak.flatten().nonzero().squeeze(1)[0]
                spatial_size = rgb.shape[-2] * rgb.shape[-1]
                sample_batch = int(flat_index.item() // spatial_size)
                sample_support = sample_fields['support'] > 0
                aux['utility_payload'] = dict(
                    batch_index=sample_batch,
                    rgb_keep=(rgb + other_delta.to(rgb.dtype))[
                        sample_batch:sample_batch + 1],
                    rgb_trial=(rgb + combined_trial.to(rgb.dtype))[
                        sample_batch:sample_batch + 1],
                    thermal=thermal[sample_batch:sample_batch + 1],
                    route_prediction=_masked_mean(
                        sample_transfer, sample_support),
                    residual_penalty=_masked_mean(
                        sample_trial_delta.square().mean(dim=1, keepdim=True),
                        sample_support),
                    level=None)
                aux['utility_sampled'] = rgb.new_tensor(1.0)
        return rgb_calibrated, aux
