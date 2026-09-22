"""Object-centric evidential RGB calibration for same-stage RGB-T fusion.

The module is a complete replacement for COXNet's CLFM input slot.  RGB and
Thermal features must have the same shape.  Thermal is used as read-only
conditioning evidence; only RGB is changed before the original AAM/HOFM.

Object and context prototypes are computed in sliding local windows.  This
keeps training and inference identical (no GT ROI is used in the calibration
path), while GT box support is used only for auxiliary targetness,
object-context contrastive, and evidential losses.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trpc import ChannelLayerNorm, balanced_binary_focal_loss


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


def _dirichlet_kl_to_uniform(alpha):
    """KL(Dir(alpha) || Dir(1)) for a two-class evidential prediction."""
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

    # Preserve evidence for the true class and penalize unsupported evidence
    # assigned to the other class.
    adjusted_alpha = labels + (1.0 - labels) * alpha
    per_cell = expected_ce + float(kl_weight) * _dirichlet_kl_to_uniform(
        adjusted_alpha)
    return _balanced_masked_mean(
        per_cell, target > 0.5, valid.bool())


class ObjectCentricEvidentialCalibration(nn.Module):
    """Enhance same-stage RGB using local Thermal object evidence only.

    The spatial windows are dense ROI prototypes: ``object_kernel`` pools the
    candidate object and ``context_kernel`` forms a surrounding ring.  Local
    cross-modal search handles small displacement without globally warping
    either modality.  A two-way keep/transfer router is used instead of a
    product of confidence gates.
    """

    def __init__(self, channels=256, embed_dim=64, object_kernel=3,
                 context_kernel=7, search_radius=2, search_temperature=0.2,
                 candidate_prior=0.1, residual_scale=0.2,
                 modulation_init_std=1e-2, edl_kl_weight=1e-3,
                 focal_gamma=2.0):
        super().__init__()
        if channels < 1 or embed_dim < 1:
            raise ValueError('channels and embed_dim must be positive')
        if object_kernel < 1 or object_kernel % 2 == 0:
            raise ValueError('object_kernel must be a positive odd integer')
        if context_kernel <= object_kernel or context_kernel % 2 == 0:
            raise ValueError(
                'context_kernel must be odd and larger than object_kernel')
        if search_radius < 0:
            raise ValueError('search_radius must be non-negative')
        if search_temperature <= 0 or residual_scale <= 0:
            raise ValueError('temperatures and scales must be positive')
        if modulation_init_std <= 0:
            raise ValueError('modulation_init_std must be positive')

        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        self.object_kernel = int(object_kernel)
        self.context_kernel = int(context_kernel)
        self.search_radius = int(search_radius)
        self.search_kernel = 2 * self.search_radius + 1
        self.search_temperature = float(search_temperature)
        self.edl_kl_weight = float(edl_kl_weight)
        self.focal_gamma = float(focal_gamma)
        self.register_buffer(
            'residual_scale', torch.tensor(float(residual_scale)))

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
            nn.Conv2d(2 * embed_dim + 2, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, 2, 1))
        # Start from an unbiased keep/transfer decision.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

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
    def _window_sum(tensor, kernel):
        return F.avg_pool2d(
            tensor, kernel, stride=1, padding=kernel // 2) * (kernel * kernel)

    def _object_context(self, feature, valid):
        valid_float = valid.to(feature.dtype)
        inner_sum = self._window_sum(feature * valid_float,
                                     self.object_kernel)
        inner_count = self._window_sum(valid_float, self.object_kernel)
        outer_sum = self._window_sum(feature * valid_float,
                                     self.context_kernel)
        outer_count = self._window_sum(valid_float, self.context_kernel)
        object_proto = inner_sum / inner_count.clamp_min(1.0)
        context_count = (outer_count - inner_count).clamp_min(1.0)
        context_proto = (outer_sum - inner_sum) / context_count
        return object_proto, context_proto, object_proto - context_proto

    def _local_attention(self, query, key, valid):
        """Attend from each query cell to a local neighborhood in ``key``."""
        b, d, h, w = query.shape
        n = h * w
        k = self.search_kernel
        candidates = F.unfold(
            key.float(), kernel_size=k, padding=self.search_radius)
        candidates = candidates.reshape(b, d, k * k, n)
        query_flat = F.normalize(query.float(), dim=1).reshape(
            b, d, 1, n)
        candidates_norm = F.normalize(candidates, dim=1)
        logits = (query_flat * candidates_norm).sum(dim=1)
        logits = logits / self.search_temperature

        candidate_valid = F.unfold(
            valid.float(), kernel_size=k, padding=self.search_radius)
        candidate_valid = candidate_valid.reshape(b, k * k, n) > 0.5
        logits = logits.masked_fill(~candidate_valid, -1e4)
        attention = logits.softmax(dim=1)
        query_valid = valid.flatten(2).to(attention.dtype)
        attention = attention * query_valid
        return attention

    def _select(self, value, attention):
        b, channels, h, w = value.shape
        n = h * w
        k = self.search_kernel
        candidates = F.unfold(
            value.float(), kernel_size=k, padding=self.search_radius)
        candidates = candidates.reshape(b, channels, k * k, n)
        selected = (candidates * attention[:, None]).sum(dim=2)
        return selected.reshape(b, channels, h, w)

    @staticmethod
    def _uncertainty(evidence):
        alpha = evidence.float() + 1.0
        return 2.0 / alpha.sum(dim=1, keepdim=True).clamp_min(2.0)

    def _contrastive_loss(self, t_difference, r_difference,
                          t_object, r_object, t_context, r_context,
                          target, valid):
        """Same-object cross-modal positive; surrounding context negative."""
        # Thermal GT coordinates anchor the auxiliary loss. RGB candidates are
        # selected locally so exact cross-modal ROI coincidence is not assumed.
        t_query = self.thermal_embed_norm(self.thermal_embed(t_difference))
        r_key = self.rgb_embed_norm(self.rgb_embed(r_difference))
        t_to_r = self._local_attention(t_query, r_key, valid)

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

    def _evidential_losses(self, t_embed, r_embed, target, valid):
        # Thermal coordinates anchor supervision. Select the corresponding RGB
        # local evidence before applying its evidential head.
        t_to_r = self._local_attention(t_embed, r_embed, valid)
        selected_rgb = self._select(r_embed, t_to_r)
        thermal_evidence = F.softplus(
            self.thermal_evidence_head(self.thermal_value(t_embed)))
        rgb_evidence = F.softplus(self.rgb_evidence_head(selected_rgb))
        loss_t = evidential_binary_loss(
            thermal_evidence, target, valid, self.edl_kl_weight)
        loss_r = evidential_binary_loss(
            rgb_evidence, target, valid, self.edl_kl_weight)
        return 0.5 * (loss_t + loss_r)

    def forward(self, rgb, thermal, valid_mask=None, target=None,
                return_aux=False):
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

        # Output remains in RGB coordinates: each RGB cell reads only a local
        # Thermal neighborhood. AAM remains responsible for final alignment.
        r_to_t = self._local_attention(r_embed, t_embed, valid)
        thermal_condition = self._select(
            self.thermal_value(t_embed), r_to_t)
        candidate_probability = self._select(
            candidate_logits.sigmoid(), r_to_t).clamp(0.0, 1.0)

        rgb_evidence = F.softplus(self.rgb_evidence_head(r_embed))
        thermal_evidence = F.softplus(
            self.thermal_evidence_head(thermal_condition))
        uncertainty_rgb = self._uncertainty(rgb_evidence)
        uncertainty_thermal = self._uncertainty(thermal_evidence)
        route_input = torch.cat([
            r_embed, thermal_condition,
            uncertainty_rgb.to(r_embed.dtype),
            uncertainty_thermal.to(r_embed.dtype)], dim=1)
        route = self.router(route_input.float()).softmax(dim=1)
        transfer_weight = route[:, 1:2]

        film_input = torch.cat([r_embed, thermal_condition], dim=1)
        film_raw = self.film_out(self.film_hidden(film_input.float()))
        scale, shift = film_raw.chunk(2, dim=1)
        modulation = (
            scale.tanh() * self.rgb_norm(rgb).float() + shift.tanh())
        support = candidate_probability * valid.to(candidate_probability.dtype)
        delta = (self.residual_scale.float() * support * transfer_weight *
                 modulation)
        rgb_calibrated = rgb + delta.to(rgb.dtype)

        if not return_aux:
            return rgb_calibrated

        with torch.no_grad():
            base_norm = rgb.float().pow(2).mean(
                dim=(1, 2, 3)).sqrt().clamp_min(1e-6)
            delta_norm = delta.pow(2).mean(dim=(1, 2, 3)).sqrt()
            valid_count = valid.sum().clamp_min(1)
            entropy = -(r_to_t * r_to_t.clamp_min(1e-12).log()).sum(dim=1)
            if self.search_kernel * self.search_kernel > 1:
                entropy = entropy / math.log(self.search_kernel ** 2)
            entropy = entropy.reshape(rgb.shape[0], rgb.shape[2], rgb.shape[3])
            entropy = (
                entropy * valid[:, 0].to(entropy.dtype)).sum() / valid_count

        aux = dict(
            candidate_logits=candidate_logits,
            candidate_probability=candidate_logits.sigmoid().detach(),
            spatial_support=support.detach(),
            transfer_weight=transfer_weight.detach(),
            uncertainty_rgb=uncertainty_rgb.detach(),
            uncertainty_thermal=uncertainty_thermal.detach(),
            local_attention_entropy=entropy.detach(),
            candidate_mean=candidate_probability.mean().detach(),
            transfer_mean=transfer_weight.mean().detach(),
            uncertainty_rgb_mean=uncertainty_rgb.mean().detach(),
            uncertainty_thermal_mean=uncertainty_thermal.mean().detach(),
            film_raw_rms=film_raw.float().pow(2).mean().sqrt().detach(),
            delta_ratio=(delta_norm / base_norm).mean().detach(),
            residual_scale=self.residual_scale.detach())

        if target is not None:
            if tuple(target.shape) != tuple(candidate_logits.shape):
                raise ValueError(
                    f'target {tuple(target.shape)} does not match candidate '
                    f'logits {tuple(candidate_logits.shape)}')
            aux['candidate_loss'] = balanced_binary_focal_loss(
                candidate_logits, target, valid, gamma=self.focal_gamma)
            aux['contrastive_loss'] = self._contrastive_loss(
                t_difference, r_difference,
                t_object, r_object, t_context, r_context,
                target, valid)
            aux['edl_loss'] = self._evidential_losses(
                t_embed, r_embed, target, valid)
        return rgb_calibrated, aux
