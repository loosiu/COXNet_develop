"""Misalignment-tolerant object-centric RGB calibration before COXNet AAM.

OEPC is a complete same-stage replacement for CLFM. Each RGB location retrieves
semantic evidence only from a local Thermal neighborhood; neither feature map
is warped. Retrieved Thermal candidate scores are converted to sparse peaks in
RGB coordinates, and correction is restricted to their local supports. The
original Thermal tensor is passed unchanged to AAM/HOFM.

GT boxes supervise targetness, prototype contrast, evidential reliability, and
keep-versus-trial utility only. They never choose calibration regions, so the
calibration path is identical during training and inference.
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
    return _balanced_masked_mean(
        per_cell, target > 0.5, valid.bool())


class ObjectCentricEvidentialCalibration(nn.Module):
    """Calibrate RGB at sparse candidate ROIs using nearby Thermal evidence."""

    def __init__(self, channels=256, embed_dim=64, object_kernel=3,
                 context_kernel=7, search_radius=2, search_temperature=0.2,
                 distance_prior_weight=0.1, candidate_prior=0.1,
                 candidate_threshold=0.05, max_candidates=100,
                 peak_kernel=3, support_kernel=3, residual_scale=0.2,
                 modulation_init_std=1e-2, edl_kl_weight=1e-3,
                 utility_temperature=0.5, edl_route_weight=0.5,
                 focal_gamma=2.0):
        super().__init__()
        if channels < 1 or embed_dim < 1:
            raise ValueError('channels and embed_dim must be positive')
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
        if search_temperature <= 0 or utility_temperature <= 0:
            raise ValueError('temperatures must be positive')
        if residual_scale <= 0 or modulation_init_std <= 0:
            raise ValueError('residual parameters must be positive')
        if distance_prior_weight < 0 or edl_route_weight < 0:
            raise ValueError('loss/prior weights must be non-negative')
        if not 0.0 <= candidate_threshold <= 1.0:
            raise ValueError('candidate_threshold must be in [0, 1]')

        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
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
        self.utility_temperature = float(utility_temperature)
        self.edl_route_weight = float(edl_route_weight)
        self.focal_gamma = float(focal_gamma)
        self.register_buffer(
            'residual_scale', torch.tensor(float(residual_scale)))

        axis = torch.arange(-self.search_radius, self.search_radius + 1).float()
        offset_y, offset_x = torch.meshgrid(axis, axis)
        distance = (offset_x.square() + offset_y.square()).sqrt()
        if self.search_radius > 0:
            distance = distance / float(self.search_radius)
        self.register_buffer(
            'distance_penalty', distance.reshape(1, -1, 1))

        self.rgb_embed = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.thermal_embed = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.rgb_embed_norm = ChannelLayerNorm(embed_dim)
        self.thermal_embed_norm = ChannelLayerNorm(embed_dim)
        self.thermal_value = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)

        self.candidate_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1,
                      groups=embed_dim, bias=False),
            nn.GELU(),
            nn.Conv2d(embed_dim, 1, 1))
        prior = min(max(float(candidate_prior), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(
            self.candidate_head[-1].bias, math.log(prior / (1.0 - prior)))

        self.rgb_evidence_head = nn.Conv2d(embed_dim, 2, 1)
        self.thermal_evidence_head = nn.Conv2d(embed_dim, 2, 1)

        hidden = max(embed_dim, 32)
        self.router = nn.Sequential(
            nn.Conv2d(2 * embed_dim + 4, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, 2, 1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        self.film_hidden = nn.Sequential(
            nn.Conv2d(2 * embed_dim, hidden, 1), nn.GELU())
        self.film_out = nn.Conv2d(hidden, 2 * channels, 1)
        nn.init.normal_(self.film_out.weight, 0.0,
                        float(modulation_init_std))
        nn.init.zeros_(self.film_out.bias)
        self.rgb_norm = ChannelLayerNorm(channels)
        self.utility_head = nn.Conv2d(embed_dim, 1, 1)

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
    def _window_sum(tensor, kernel):
        return F.avg_pool2d(
            tensor, kernel, stride=1, padding=kernel // 2) * (kernel * kernel)

    def _object_context(self, feature, valid, context_exclusion=None):
        """Return local object, background-ring, and difference prototypes."""
        valid_float = valid.to(feature.dtype)
        object_sum = self._window_sum(
            feature * valid_float, self.object_kernel)
        object_count = self._window_sum(valid_float, self.object_kernel)
        object_proto = object_sum / object_count.clamp_min(1.0)

        context_valid = valid
        if context_exclusion is not None:
            context_valid = valid & ~(context_exclusion > 0.5)
        context_valid_float = context_valid.to(feature.dtype)
        outer_sum = self._window_sum(
            feature * context_valid_float, self.context_kernel)
        inner_sum = self._window_sum(
            feature * context_valid_float, self.object_kernel)
        outer_count = self._window_sum(
            context_valid_float, self.context_kernel)
        inner_count = self._window_sum(
            context_valid_float, self.object_kernel)
        context_count = (outer_count - inner_count).clamp_min(1.0)
        context_proto = (outer_sum - inner_sum) / context_count
        return object_proto, context_proto, object_proto - context_proto

    def _local_attention(self, query, key, valid):
        """Retrieve local key evidence without moving either feature map."""
        b, channels, h, w = query.shape
        locations = h * w
        kernel = self.search_kernel
        candidates = F.unfold(
            key.float(), kernel_size=kernel, padding=self.search_radius)
        candidates = candidates.reshape(
            b, channels, kernel * kernel, locations)
        query_flat = F.normalize(query.float(), dim=1).reshape(
            b, channels, 1, locations)
        similarity = (query_flat * F.normalize(
            candidates, dim=1)).sum(dim=1)
        logits = (similarity - self.distance_prior_weight *
                  self.distance_penalty.to(similarity.dtype))
        logits = logits / self.search_temperature

        candidate_valid = F.unfold(
            valid.float(), kernel_size=kernel, padding=self.search_radius)
        candidate_valid = candidate_valid.reshape(
            b, kernel * kernel, locations) > 0.5
        logits = logits.masked_fill(~candidate_valid, -1e4)
        attention = logits.softmax(dim=1)
        query_valid = valid.flatten(2).to(attention.dtype)
        attention = attention * query_valid

        confidence = attention.max(dim=1, keepdim=True).values
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum(
            dim=1, keepdim=True)
        valid_options = candidate_valid.sum(dim=1, keepdim=True)
        entropy_scale = valid_options.clamp_min(2).to(entropy.dtype).log()
        entropy = torch.where(
            valid_options > 1, entropy / entropy_scale, torch.zeros_like(entropy))
        return (attention, confidence.reshape(b, 1, h, w),
                entropy.reshape(b, 1, h, w))

    def _select(self, value, attention):
        b, channels, h, w = value.shape
        locations = h * w
        kernel = self.search_kernel
        candidates = F.unfold(
            value.float(), kernel_size=kernel, padding=self.search_radius)
        candidates = candidates.reshape(
            b, channels, kernel * kernel, locations)
        selected = (candidates * attention[:, None]).sum(dim=2)
        return selected.reshape(b, channels, h, w)

    def _sparse_support(self, probability, valid):
        """Select local maxima/top-k in RGB coordinates and make ROI support."""
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
        peak = peak * valid.to(peak.dtype)
        support = F.max_pool2d(
            peak, self.support_kernel, stride=1,
            padding=self.support_kernel // 2)
        support = (support > 0).to(probability.dtype) * valid.to(
            probability.dtype)
        return peak, support

    def _broadcast_candidates(self, value, peak, peak_score):
        """Broadcast each candidate prototype only inside its ROI support."""
        weight = peak * peak_score
        normalizer = self._window_sum(weight, self.support_kernel)
        value_sum = self._window_sum(
            value.float() * weight, self.support_kernel)
        return value_sum / normalizer.clamp_min(1e-6)

    @staticmethod
    def _uncertainty(evidence):
        alpha = evidence.float() + 1.0
        return 2.0 / alpha.sum(dim=1, keepdim=True).clamp_min(2.0)

    def _contrastive_loss(self, t_difference, r_difference,
                          t_object, r_object, t_context, r_context,
                          target, valid):
        """Use the same object as positive and background context as negative."""
        t_query = self.thermal_embed_norm(self.thermal_embed(t_difference))
        r_key = self.rgb_embed_norm(self.rgb_embed(r_difference))
        t_to_r, _, _ = self._local_attention(t_query, r_key, valid)
        z_t_object = F.normalize(
            self.thermal_embed(t_object).float(), dim=1)
        z_t_context = F.normalize(
            self.thermal_embed(t_context).float(), dim=1)
        z_r_object = F.normalize(
            self._select(self.rgb_embed(r_object), t_to_r), dim=1)
        z_r_context = F.normalize(
            self._select(self.rgb_embed(r_context), t_to_r), dim=1)
        positive = (z_t_object * z_r_object).sum(dim=1, keepdim=True)
        negative_t = (z_t_object * z_r_context).sum(dim=1, keepdim=True)
        negative_r = (z_r_object * z_t_context).sum(dim=1, keepdim=True)
        per_cell = 0.5 * (
            F.softplus((negative_t - positive) / self.search_temperature) +
            F.softplus((negative_r - positive) / self.search_temperature))
        foreground = (target > 0.5) & valid
        if not foreground.any():
            return per_cell.sum() * 0.0
        return per_cell[foreground].mean()

    def _evidential_loss(self, t_embed, r_embed, target, valid):
        t_to_r, _, _ = self._local_attention(t_embed, r_embed, valid)
        selected_rgb = self._select(r_embed, t_to_r)
        thermal_evidence = F.softplus(
            self.thermal_evidence_head(self.thermal_value(t_embed)))
        rgb_evidence = F.softplus(self.rgb_evidence_head(selected_rgb))
        loss_t = evidential_binary_loss(
            thermal_evidence, target, valid, self.edl_kl_weight)
        loss_r = evidential_binary_loss(
            rgb_evidence, target, valid, self.edl_kl_weight)
        return 0.5 * (loss_t + loss_r)

    def _utility_loss(self, rgb, rgb_trial, r_candidate, peak, peak_score,
                      support, transfer_weight, uncertainty_rgb,
                      uncertainty_thermal, target, r_to_t, valid):
        """Supervise keep/transfer using full-correction trial utility."""
        _, _, trial_difference = self._object_context(rgb_trial, valid)
        trial_embed = self.rgb_embed_norm(self.rgb_embed(trial_difference))
        trial_candidate = self._broadcast_candidates(
            trial_embed, peak, peak_score)
        keep_logit = self.utility_head(r_candidate)
        trial_logit = self.utility_head(trial_candidate)

        target_rgb = self._select(target.float(), r_to_t).clamp(0.0, 1.0)
        target_candidate = self._broadcast_candidates(
            target_rgb, peak, peak_score).clamp(0.0, 1.0)
        utility_valid = (support > 0) & valid
        keep_ce = F.binary_cross_entropy_with_logits(
            keep_logit, target_candidate, reduction='none')
        trial_ce = F.binary_cross_entropy_with_logits(
            trial_logit, target_candidate, reduction='none')
        classification = _balanced_masked_mean(
            0.5 * (keep_ce + trial_ce), target_candidate > 0.5,
            utility_valid)

        utility_target = torch.sigmoid(
            (keep_ce.detach() - trial_ce.detach()) /
            self.utility_temperature)
        route_utility = _masked_mean(
            F.binary_cross_entropy(
                transfer_weight.clamp(1e-6, 1.0 - 1e-6),
                utility_target, reduction='none'), utility_valid)
        edl_target = ((1.0 - uncertainty_thermal) *
                      uncertainty_rgb).detach().clamp(0.0, 1.0)
        route_edl = _masked_mean(
            F.binary_cross_entropy(
                transfer_weight.clamp(1e-6, 1.0 - 1e-6),
                edl_target, reduction='none'), utility_valid)
        loss = classification + route_utility + self.edl_route_weight * route_edl
        diagnostics = dict(
            utility_keep_ce=_masked_mean(keep_ce, utility_valid).detach(),
            utility_trial_ce=_masked_mean(trial_ce, utility_valid).detach(),
            utility_target_mean=_masked_mean(
                utility_target, utility_valid).detach(),
            edl_transfer_target_mean=_masked_mean(
                edl_target, utility_valid).detach())
        return loss, diagnostics

    def forward(self, rgb, thermal, valid_mask=None, target=None,
                context_exclusion=None, return_aux=False):
        if rgb.shape != thermal.shape:
            raise ValueError(
                'OEPC requires equal same-stage RGB/Thermal shapes, got '
                f'{tuple(rgb.shape)} and {tuple(thermal.shape)}')
        if rgb.shape[1] != self.channels:
            raise ValueError('OEPC channel mismatch')

        valid = self._as_valid_mask(valid_mask, rgb)
        r_object, r_context, r_difference = self._object_context(rgb, valid)
        t_object, t_context, t_difference = self._object_context(
            thermal, valid)
        r_embed = self.rgb_embed_norm(self.rgb_embed(r_difference))
        t_embed = self.thermal_embed_norm(self.thermal_embed(t_difference))
        candidate_logits = self.candidate_head(t_embed)

        r_to_t, match_confidence_dense, match_entropy_dense = (
            self._local_attention(r_embed, t_embed, valid))
        thermal_condition_dense = self._select(
            self.thermal_value(t_embed), r_to_t)
        candidate_probability_dense = self._select(
            candidate_logits.sigmoid(), r_to_t).clamp(0.0, 1.0)

        peak, support = self._sparse_support(
            candidate_probability_dense, valid)
        r_candidate = self._broadcast_candidates(
            r_embed, peak, candidate_probability_dense)
        thermal_condition = self._broadcast_candidates(
            thermal_condition_dense, peak, candidate_probability_dense)
        match_confidence = self._broadcast_candidates(
            match_confidence_dense, peak, candidate_probability_dense)
        match_entropy = self._broadcast_candidates(
            match_entropy_dense, peak, candidate_probability_dense)
        candidate_score = self._broadcast_candidates(
            candidate_probability_dense, peak, candidate_probability_dense)

        rgb_evidence = F.softplus(self.rgb_evidence_head(r_candidate))
        thermal_evidence = F.softplus(
            self.thermal_evidence_head(thermal_condition))
        uncertainty_rgb = self._uncertainty(rgb_evidence)
        uncertainty_thermal = self._uncertainty(thermal_evidence)
        route_input = torch.cat([
            r_candidate, thermal_condition,
            uncertainty_rgb.to(r_candidate.dtype),
            uncertainty_thermal.to(r_candidate.dtype),
            match_confidence.to(r_candidate.dtype),
            candidate_score.to(r_candidate.dtype)], dim=1)
        route = self.router(route_input.float()).softmax(dim=1)
        transfer_weight = route[:, 1:2]

        film_input = torch.cat([r_candidate, thermal_condition], dim=1)
        film_raw = self.film_out(self.film_hidden(film_input.float()))
        scale, shift = film_raw.chunk(2, dim=1)
        modulation = (
            scale.tanh() * self.rgb_norm(rgb).float() + shift.tanh())
        trial_delta = self.residual_scale.float() * support * modulation
        delta = transfer_weight * trial_delta
        rgb_trial = rgb + trial_delta.to(rgb.dtype)
        rgb_calibrated = rgb + delta.to(rgb.dtype)

        if not return_aux:
            return rgb_calibrated

        support_bool = support > 0
        with torch.no_grad():
            base_norm = rgb.float().pow(2).mean(
                dim=(1, 2, 3)).sqrt().clamp_min(1e-6)
            delta_norm = delta.pow(2).mean(dim=(1, 2, 3)).sqrt()
            valid_count = valid.sum().clamp_min(1)

        aux = dict(
            candidate_logits=candidate_logits,
            candidate_probability=candidate_probability_dense.detach(),
            candidate_peaks=peak.detach(),
            spatial_support=support.detach(),
            transfer_weight=transfer_weight.detach(),
            uncertainty_rgb=uncertainty_rgb.detach(),
            uncertainty_thermal=uncertainty_thermal.detach(),
            matching_confidence=match_confidence.detach(),
            matching_entropy=match_entropy.detach(),
            local_attention_entropy=_masked_mean(
                match_entropy, support_bool).detach(),
            matching_confidence_mean=_masked_mean(
                match_confidence, support_bool).detach(),
            candidate_mean=_masked_mean(
                candidate_score, support_bool).detach(),
            candidate_count=peak.sum(dim=(1, 2, 3)).mean().detach(),
            support_ratio=(support.sum() / valid_count).detach(),
            transfer_mean=_masked_mean(
                transfer_weight, support_bool).detach(),
            uncertainty_rgb_mean=_masked_mean(
                uncertainty_rgb, support_bool).detach(),
            uncertainty_thermal_mean=_masked_mean(
                uncertainty_thermal, support_bool).detach(),
            film_raw_rms=_masked_mean(
                film_raw.float().square().mean(dim=1, keepdim=True).sqrt(),
                support_bool).detach(),
            delta_ratio=(delta_norm / base_norm).mean().detach(),
            residual_scale=self.residual_scale.detach())

        if target is not None:
            if tuple(target.shape) != tuple(candidate_logits.shape):
                raise ValueError(
                    f'target {tuple(target.shape)} does not match candidate '
                    f'logits {tuple(candidate_logits.shape)}')
            if context_exclusion is None:
                context_exclusion = target
            if tuple(context_exclusion.shape) != tuple(target.shape):
                raise ValueError('context_exclusion must match target shape')

            r_obj_aux, r_ctx_aux, r_diff_aux = self._object_context(
                rgb, valid, context_exclusion=context_exclusion)
            t_obj_aux, t_ctx_aux, t_diff_aux = self._object_context(
                thermal, valid, context_exclusion=context_exclusion)
            r_embed_aux = self.rgb_embed_norm(self.rgb_embed(r_diff_aux))
            t_embed_aux = self.thermal_embed_norm(self.thermal_embed(t_diff_aux))

            aux['candidate_loss'] = balanced_binary_focal_loss(
                candidate_logits, target, valid, gamma=self.focal_gamma)
            aux['contrastive_loss'] = self._contrastive_loss(
                t_diff_aux, r_diff_aux,
                t_obj_aux, r_obj_aux, t_ctx_aux, r_ctx_aux,
                target, valid)
            aux['edl_loss'] = self._evidential_loss(
                t_embed_aux, r_embed_aux, target, valid)
            utility_loss, utility_diagnostics = self._utility_loss(
                rgb, rgb_trial, r_candidate, peak,
                candidate_probability_dense, support, transfer_weight,
                uncertainty_rgb, uncertainty_thermal,
                target, r_to_t, valid)
            aux['utility_loss'] = utility_loss
            aux.update(utility_diagnostics)
        return rgb_calibrated, aux
