"""Thermal-Referenced Prototype Calibration (TRPC).

TRPC replaces COXNet's CLFM before the original AAM while deliberately leaving
spatial alignment to AAM. The primary same-stage variant treats Thermal
prototypes as scene-conditioning tokens and applies spatial FiLM modulation to
RGB. The legacy mutual-matching variant is kept to reproduce the first
experiment.

    RGB, Thermal -> learned prototypes -> mutual semantic matching
                 -> Thermal-guided RGB prototype residual
                 -> RGB-attention reconstruction -> AAM(RGB_cal, Thermal)

Thermal decides *what* semantic residual is useful.  The RGB assignment map
decides *where* that residual is written, so a displaced thermal response is
never copied to the same image coordinate.  The thermal feature itself is not
modified.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet_process import TransBasicConv2d


def make_padding_mask(image_shapes, padded_size, feat_size, device):
    """Return a (B,1,H,W) mask for feature cells inside each unpadded image."""
    h_in, w_in = padded_size
    fh, fw = feat_size
    mask = torch.zeros(len(image_shapes), 1, fh, fw, dtype=torch.bool, device=device)
    for batch_idx, (height, width) in enumerate(image_shapes):
        valid_h = min(fh, int(math.ceil(height * fh / float(h_in))))
        valid_w = min(fw, int(math.ceil(width * fw / float(w_in))))
        mask[batch_idx, 0, :valid_h, :valid_w] = True
    return mask


def _box_cells(box, scale_x, scale_y, feat_h, feat_w):
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    col0 = int(math.floor(x1 * scale_x))
    col1 = max(int(math.ceil(x2 * scale_x)) - 1, col0)
    row0 = int(math.floor(y1 * scale_y))
    row1 = max(int(math.ceil(y2 * scale_y)) - 1, row0)
    col0, col1 = max(0, min(col0, feat_w - 1)), max(0, min(col1, feat_w - 1))
    row0, row1 = max(0, min(row0, feat_h - 1)), max(0, min(row1, feat_h - 1))
    return row0, row1, col0, col1


def build_box_targets(gt_boxes, image_shapes, padded_size, feat_size, device,
                      ignore_boxes=None):
    """Build thermal-coordinate box-support targets and a valid loss mask."""
    input_h, input_w = padded_size
    feat_h, feat_w = feat_size
    scale_x, scale_y = feat_w / float(input_w), feat_h / float(input_h)
    target = torch.zeros(len(gt_boxes), 1, feat_h, feat_w, device=device)
    valid = make_padding_mask(image_shapes, padded_size, feat_size, device)
    for batch_idx, boxes in enumerate(gt_boxes):
        if boxes is not None and len(boxes):
            for box in boxes.detach().cpu().tolist():
                row0, row1, col0, col1 = _box_cells(
                    box, scale_x, scale_y, feat_h, feat_w)
                target[batch_idx, 0, row0:row1 + 1, col0:col1 + 1] = 1.0
        if (ignore_boxes is not None and ignore_boxes[batch_idx] is not None and
                len(ignore_boxes[batch_idx])):
            for box in ignore_boxes[batch_idx].detach().cpu().tolist():
                row0, row1, col0, col1 = _box_cells(
                    box, scale_x, scale_y, feat_h, feat_w)
                valid[batch_idx, 0, row0:row1 + 1, col0:col1 + 1] = False
    return target, valid


def balanced_binary_focal_loss(logits, target, valid, gamma=2.0):
    """Focal BCE balanced between foreground and background for each image."""
    logits = logits.float()
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    focal = cross_entropy * (1.0 - p_t).pow(gamma)
    foreground = (target > 0.5) & valid
    background = (target <= 0.5) & valid
    per_image = []
    for batch_idx in range(logits.shape[0]):
        parts = []
        if foreground[batch_idx].any():
            parts.append(focal[batch_idx][foreground[batch_idx]].mean())
        if background[batch_idx].any():
            parts.append(focal[batch_idx][background[batch_idx]].mean())
        if parts:
            per_image.append(torch.stack(parts).mean())
    if not per_image:
        return logits.sum() * 0.0
    return torch.stack(per_image).mean()


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at every spatial position."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ObjectAwarePrototypeExtractor(nn.Module):
    """Extract K semantic prototypes and retain RGB/Thermal spatial assignments.

    ``pool_attention`` is normalized over space for prototype extraction.
    ``where_attention`` is normalized over prototypes and gated by objectness;
    TRPC reuses the RGB version for spatial reconstruction.
    """

    def __init__(self, channels, embed_dim, num_prototypes,
                 objectness_prior=0.1, objectness_bias=1.0):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.objectness_bias = float(objectness_bias)
        self.embed = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.embed_norm = ChannelLayerNorm(embed_dim)
        self.assignment = nn.Conv2d(embed_dim, num_prototypes, 1)
        self.objectness_dw = nn.Conv2d(embed_dim, embed_dim, 3, padding=1,
                                       groups=embed_dim, bias=False)
        self.objectness_out = nn.Conv2d(embed_dim, 1, 1)
        prior = min(max(float(objectness_prior), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.objectness_out.bias, math.log(prior / (1.0 - prior)))

    def forward(self, feat, valid_mask=None):
        b, _, h, w = feat.shape
        n = h * w
        token_map = self.embed_norm(self.embed(feat))
        assign_logits = self.assignment(token_map).flatten(2)       # B,K,N
        objectness_logits = self.objectness_out(
            F.gelu(self.objectness_dw(token_map)))                  # B,1,H,W
        objectness = objectness_logits.sigmoid().flatten(2)         # B,1,N

        if valid_mask is None:
            valid = torch.ones(b, 1, n, device=feat.device, dtype=torch.bool)
        else:
            if tuple(valid_mask.shape) == (b, 1, h, w):
                valid_mask = valid_mask[:, 0]
            if tuple(valid_mask.shape) != (b, h, w):
                raise ValueError(
                    f'prototype valid mask {tuple(valid_mask.shape)} does not match {(b, h, w)}')
            valid = valid_mask.reshape(b, 1, n).bool()

        # One base map is shared by extraction and reconstruction.  Semantic
        # membership is normalized over prototypes at every location, then the
        # objectness prior suppresses background.  Prototype pooling only adds
        # a per-prototype normalization over space; RGB reconstruction reuses
        # ``where_attention`` itself without introducing a thermal coordinate.
        semantic_membership = assign_logits.float().softmax(dim=1)
        if self.objectness_bias == 0:
            foreground_prior = torch.ones_like(objectness, dtype=torch.float32)
        else:
            foreground_prior = objectness.float().clamp_min(1e-6).pow(self.objectness_bias)
        where_attention = semantic_membership * foreground_prior
        where_attention = where_attention * valid.to(where_attention.dtype)
        pool_attention = where_attention / where_attention.sum(
            dim=-1, keepdim=True).clamp_min(1e-6)

        tokens = token_map.float().flatten(2).transpose(1, 2)       # B,N,D
        prototypes = torch.bmm(pool_attention, tokens)              # B,K,D

        # Reusing this RGB map later is the explicit "Thermal what / RGB where"
        # part of TRPC.
        return prototypes, pool_attention, where_attention, objectness_logits


class TRPC(nn.Module):
    """Legacy RGB/Thermal prototype-matching TRPC."""

    def __init__(self, channels=256, embed_dim=64, num_prototypes=8,
                 match_temperature=0.2, match_topk=1, mutual_matching=True,
                 objectness_prior=0.1, objectness_bias=1.0,
                 residual_scale=0.2, learn_residual_scale=True,
                 same_stage=False):
        super().__init__()
        if num_prototypes < 1:
            raise ValueError('num_prototypes must be >= 1')
        if embed_dim < 1:
            raise ValueError('embed_dim must be >= 1')
        if match_temperature <= 0:
            raise ValueError('match_temperature must be > 0')
        if match_topk < 1 or match_topk > num_prototypes:
            raise ValueError('match_topk must be in [1, num_prototypes]')

        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        self.num_prototypes = int(num_prototypes)
        self.match_temperature = float(match_temperature)
        self.match_topk = int(match_topk)
        self.mutual_matching = bool(mutual_matching)
        self.same_stage = bool(same_stage)

        # Same-stage TRPC removes the complete CLFM path, including DeConv.
        # The legacy cross-stage variant remains available to reproduce the
        # first experiment and its checkpoints.
        self.visible_upsample = (
            nn.Identity() if self.same_stage
            else TransBasicConv2d(channels, channels))
        self.rgb_extractor = ObjectAwarePrototypeExtractor(
            channels, embed_dim, num_prototypes, objectness_prior, objectness_bias)
        self.thermal_extractor = ObjectAwarePrototypeExtractor(
            channels, embed_dim, num_prototypes, objectness_prior, objectness_bias)

        # alpha_i = sigmoid(MLP[p_r, g_t, |p_r-g_t|]); phi(g_t-p_r)
        hidden = max(embed_dim, 32)
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim))
        self.delta_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim))
        self.delta_to_rgb = nn.Linear(embed_dim, channels, bias=False)
        # Start from the exact incoming RGB feature (or legacy DeConv output).
        # Only this actuator is zeroed:
        # its first update receives a detector gradient, then gradients can
        # propagate into the matching/gating path without perturbing AAM at
        # iteration zero.
        nn.init.zeros_(self.delta_to_rgb.weight)

        scale = torch.tensor(float(residual_scale))
        if learn_residual_scale:
            scale = scale.clamp(1e-4, 1.0 - 1e-4)
            self.residual_scale_logit = nn.Parameter(torch.log(scale / (1.0 - scale)))
            self.register_buffer('_fixed_residual_scale', torch.tensor(-1.0), persistent=False)
        else:
            self.register_parameter('residual_scale_logit', None)
            self.register_buffer('_fixed_residual_scale', scale)

    @classmethod
    def from_legacy_clfm(cls, legacy_clfm, **kwargs):
        """Build TRPC while retaining only CLFM's resolution-matching DeConv."""
        module = cls(**kwargs)
        if module.same_stage:
            raise ValueError('same-stage TRPC cannot retain a legacy DeConv')
        if not hasattr(legacy_clfm, 'deconv'):
            raise ValueError("TRPC expects a legacy DWTC(mode='up_new') with a deconv block")
        module.visible_upsample = copy.deepcopy(legacy_clfm.deconv)
        return module

    @property
    def residual_scale(self):
        if self.residual_scale_logit is None:
            return self._fixed_residual_scale
        return self.residual_scale_logit.sigmoid()

    def _match(self, rgb_prototypes, thermal_prototypes):
        """Cosine top-k correspondence, optionally retaining only mutual pairs."""
        pr = F.normalize(rgb_prototypes.float(), dim=-1, eps=1e-6)
        pt = F.normalize(thermal_prototypes.float(), dim=-1, eps=1e-6)
        similarity = torch.bmm(pr, pt.transpose(1, 2))              # B,K_r,K_t
        soft_weights = (similarity / self.match_temperature).softmax(dim=-1)

        k = min(self.match_topk, self.num_prototypes)
        rgb_topk = similarity.topk(k, dim=-1).indices
        rgb_mask = torch.zeros_like(similarity, dtype=torch.bool)
        rgb_mask.scatter_(-1, rgb_topk, True)
        if self.mutual_matching:
            thermal_topk = similarity.transpose(1, 2).topk(k, dim=-1).indices
            thermal_mask = torch.zeros_like(similarity.transpose(1, 2), dtype=torch.bool)
            thermal_mask.scatter_(-1, thermal_topk, True)
            pair_mask = rgb_mask & thermal_mask.transpose(1, 2)
        else:
            pair_mask = rgb_mask

        selected_mass = (soft_weights * pair_mask).sum(dim=-1)      # confidence before renorm
        weights = soft_weights * pair_mask
        denom = weights.sum(dim=-1, keepdim=True)
        weights = weights / denom.clamp_min(1e-6)
        weights = torch.where(denom > 0, weights, torch.zeros_like(weights))
        guidance = torch.bmm(weights, thermal_prototypes.float())
        match_confidence = selected_mass * pair_mask.any(dim=-1).to(selected_mass.dtype)
        return guidance, weights, similarity, match_confidence

    @staticmethod
    def prototype_diversity_loss(prototypes):
        k = prototypes.shape[1]
        if k <= 1:
            return prototypes.new_tensor(0.0)
        p = F.normalize(prototypes.float(), dim=-1, eps=1e-6)
        gram = torch.bmm(p, p.transpose(1, 2))
        off_diag = ~torch.eye(k, device=gram.device, dtype=torch.bool).unsqueeze(0)
        return gram.square().masked_select(off_diag.expand_as(gram)).mean()

    @staticmethod
    def normalized_attention_entropy(attention):
        """Mean entropy in [0,1]; low values expose spatial attention collapse."""
        a = attention.float().clamp_min(0)
        entropy = -(a * a.clamp_min(1e-12).log()).sum(dim=-1)
        support = (a > 0).sum(dim=-1).clamp_min(2).to(a.dtype)
        return (entropy / support.log()).mean()

    @staticmethod
    def effective_prototype_usage(where_attention):
        """Entropy effective number of prototypes used per image (1..K)."""
        mass = where_attention.float().sum(dim=-1)
        prob = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return (-(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)).exp().mean()

    def forward(self, rgb, thermal, valid_mask=None, return_aux=False):
        rgb_up = self.visible_upsample(rgb)
        if rgb_up.shape[-2:] != thermal.shape[-2:]:
            if self.same_stage:
                raise ValueError(
                    'same-stage TRPC requires equal RGB/Thermal feature '
                    f'shapes, got {tuple(rgb_up.shape)} and '
                    f'{tuple(thermal.shape)}')
            # Legacy cross-stage fallback for odd FPN dimensions only. AAM
            # still owns cross-modal displacement correction.
            rgb_up = F.interpolate(
                rgb_up, size=thermal.shape[-2:], mode='bilinear',
                align_corners=False)
        if rgb_up.shape[1] != self.channels or thermal.shape[1] != self.channels:
            raise ValueError('TRPC channel mismatch')

        p_rgb, a_rgb_pool, a_rgb_where, o_rgb = self.rgb_extractor(rgb_up, valid_mask)
        p_thermal, a_t_pool, a_t_where, o_thermal = self.thermal_extractor(thermal, valid_mask)

        # Thermal is a one-way semantic reference for calibration.  The raw
        # thermal feature still receives the detector gradient through HOFM,
        # and its extractor receives targetness/diversity supervision, but the
        # RGB-calibration gradient cannot pull the reference toward RGB.
        p_thermal_ref = p_thermal.detach()
        guidance, match_weights, similarity, confidence = self._match(
            p_rgb, p_thermal_ref)

        difference = guidance - p_rgb.float()
        alpha = torch.sigmoid(self.gate_mlp(torch.cat(
            [p_rgb.float(), guidance, difference.abs()], dim=-1)))
        alpha = alpha * confidence.unsqueeze(-1)
        delta_prototype = alpha * self.delta_mlp(difference)         # B,K,D
        calibrated_prototype = p_rgb.float() + delta_prototype

        delta_rgb = self.delta_to_rgb(delta_prototype)              # B,K,C
        delta_map = torch.einsum('bkn,bkc->bcn', a_rgb_where, delta_rgb)
        delta_map = delta_map.reshape_as(rgb_up).to(rgb_up.dtype)
        rgb_calibrated = rgb_up + self.residual_scale.to(rgb_up.dtype) * delta_map

        if not return_aux:
            return rgb_calibrated
        with torch.no_grad():
            base = rgb_up.float().pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-6)
            prototype_residual_rms = delta_prototype.float().pow(2).mean().sqrt()
            projected_prototype_rms = delta_rgb.float().pow(2).mean().sqrt()
            delta_map_norm = delta_map.float().pow(2).mean(
                dim=(1, 2, 3)).sqrt()
            scaled_delta = self.residual_scale.float() * delta_map.float()
            scaled_delta_norm = scaled_delta.pow(2).mean(
                dim=(1, 2, 3)).sqrt()
            change = (rgb_calibrated.float() - rgb_up.float()).pow(2).mean(
                dim=(1, 2, 3)).sqrt()
            delta_map_ratio = (delta_map_norm / base).mean()
            scaled_delta_ratio = (scaled_delta_norm / base).mean()
            delta_ratio = (change / base).mean()
            matched = match_weights.sum(dim=-1) > 0
            before = F.cosine_similarity(p_rgb.float(), guidance, dim=-1)
            after = F.cosine_similarity(calibrated_prototype, guidance, dim=-1)
            n_matched = matched.sum().clamp_min(1)
            # Prototype-space diagnostics only.  ``after`` is measured before
            # projection, spatial reconstruction, and residual scaling, so it
            # does not establish calibration of the actual AAM input.
            proto_cos_before = (before * matched).sum() / n_matched
            proto_cos_after = (after * matched).sum() / n_matched
            entropy_rgb = self.normalized_attention_entropy(a_rgb_pool)
            entropy_thermal = self.normalized_attention_entropy(a_t_pool)
            usage_rgb = self.effective_prototype_usage(a_rgb_where)
            usage_thermal = self.effective_prototype_usage(a_t_where)
        aux = dict(
            thermal_objectness_logits=o_thermal,
            rgb_objectness=o_rgb.sigmoid().detach(),
            thermal_objectness=o_thermal.sigmoid().detach(),
            rgb_pool_attention=a_rgb_pool.detach(),
            thermal_pool_attention=a_t_pool.detach(),
            rgb_where_attention=a_rgb_where.detach(),
            rgb_prototypes=p_rgb.detach(),
            thermal_prototypes=p_thermal.detach(),
            calibrated_rgb_prototypes=calibrated_prototype.detach(),
            match_weights=match_weights.detach(),
            match_similarity=similarity.detach(),
            match_rate=(match_weights.sum(-1) > 0).float().mean().detach(),
            match_confidence=confidence.mean().detach(),
            gate_mean=alpha.mean().detach(),
            residual_scale=self.residual_scale.detach(),
            prototype_residual_rms=prototype_residual_rms.detach(),
            projected_prototype_rms=projected_prototype_rms.detach(),
            delta_map_ratio=delta_map_ratio.detach(),
            scaled_delta_ratio=scaled_delta_ratio.detach(),
            delta_ratio=delta_ratio.detach(),
            proto_rgb_cos_before=proto_cos_before.detach(),
            proto_rgb_cos_after=proto_cos_after.detach(),
            attention_entropy_rgb=entropy_rgb.detach(),
            attention_entropy_thermal=entropy_thermal.detach(),
            prototype_usage_rgb=usage_rgb.detach(),
            prototype_usage_thermal=usage_thermal.detach(),
            prototype_usage=((usage_rgb + usage_thermal) * 0.5).detach(),
            diversity_loss=(self.prototype_diversity_loss(p_rgb) +
                            self.prototype_diversity_loss(p_thermal)) * 0.5)
        return rgb_calibrated, aux


class ThermalConditionedTRPC(nn.Module):
    """Same-stage Thermal-token conditioning with spatial RGB modulation.

    Thermal prototypes summarize the current scene; they are not interpreted
    as object correspondences. Every RGB location softly reads those tokens and
    predicts feature-wise scale and shift terms. No hard matching, confidence
    gate, RGB objectness gate, stop-gradient, CLFM, or DeConv is used.
    """

    def __init__(self, channels=256, embed_dim=64, num_prototypes=8,
                 objectness_prior=0.1, objectness_bias=1.0,
                 epsilon=0.1, modulation_init_std=1e-3):
        super().__init__()
        if channels < 1 or embed_dim < 1 or num_prototypes < 1:
            raise ValueError('channels, embed_dim and num_prototypes must be >= 1')
        if epsilon <= 0:
            raise ValueError('epsilon must be > 0')
        if modulation_init_std <= 0:
            raise ValueError('modulation_init_std must be > 0')

        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        self.num_prototypes = int(num_prototypes)
        self.register_buffer('epsilon', torch.tensor(float(epsilon)))

        self.thermal_extractor = ObjectAwarePrototypeExtractor(
            channels, embed_dim, num_prototypes,
            objectness_prior, objectness_bias)
        self.rgb_query = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.rgb_query_norm = ChannelLayerNorm(embed_dim)
        self.thermal_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.thermal_value = nn.Linear(embed_dim, embed_dim, bias=False)

        hidden = max(embed_dim, 32)
        self.film_hidden = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU())
        self.film_out = nn.Linear(hidden, 2 * channels)
        nn.init.normal_(self.film_out.weight, mean=0.0,
                        std=float(modulation_init_std))
        nn.init.zeros_(self.film_out.bias)
        self.rgb_norm = ChannelLayerNorm(channels)

    @staticmethod
    def _valid_flat(valid_mask, shape, device):
        b, _, h, w = shape
        if valid_mask is None:
            return torch.ones(b, h * w, dtype=torch.bool, device=device)
        if tuple(valid_mask.shape) == (b, 1, h, w):
            valid_mask = valid_mask[:, 0]
        if tuple(valid_mask.shape) != (b, h, w):
            raise ValueError(
                f'conditioning valid mask {tuple(valid_mask.shape)} does not '
                f'match {(b, h, w)}')
        return valid_mask.reshape(b, h * w).bool()

    def forward(self, rgb, thermal, valid_mask=None, return_aux=False):
        if rgb.shape != thermal.shape:
            raise ValueError(
                'Thermal-conditioned same-stage TRPC requires equal RGB/'
                f'Thermal shapes, got {tuple(rgb.shape)} and '
                f'{tuple(thermal.shape)}')
        if rgb.shape[1] != self.channels:
            raise ValueError('TRPC channel mismatch')

        b, _, h, w = rgb.shape
        valid = self._valid_flat(valid_mask, rgb.shape, rgb.device)
        p_thermal, a_t_pool, a_t_where, o_thermal = self.thermal_extractor(
            thermal, valid_mask)

        query = self.rgb_query_norm(self.rgb_query(rgb)).float()
        query = query.flatten(2).transpose(1, 2)                    # B,N,D
        key = self.thermal_key(p_thermal.float())                  # B,K,D
        value = self.thermal_value(p_thermal.float())              # B,K,D
        logits = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(
            self.embed_dim)
        attention = logits.softmax(dim=-1)
        attention = attention * valid.unsqueeze(-1).to(attention.dtype)
        conditioning = torch.bmm(attention, value)                 # B,N,D

        film_raw = self.film_out(self.film_hidden(conditioning))   # B,N,2C
        scale_raw, shift_raw = film_raw.chunk(2, dim=-1)
        scale = scale_raw.tanh().transpose(1, 2).reshape(b, self.channels, h, w)
        shift = shift_raw.tanh().transpose(1, 2).reshape(b, self.channels, h, w)
        valid_map = valid.reshape(b, 1, h, w).to(scale.dtype)
        modulation = (
            scale * self.rgb_norm(rgb).float() + shift) * valid_map
        delta = self.epsilon.float() * modulation
        rgb_calibrated = rgb + delta.to(rgb.dtype)

        if not return_aux:
            return rgb_calibrated

        with torch.no_grad():
            base = rgb.float().pow(2).mean(
                dim=(1, 2, 3)).sqrt().clamp_min(1e-6)
            modulation_norm = modulation.pow(2).mean(
                dim=(1, 2, 3)).sqrt()
            delta_norm = delta.pow(2).mean(dim=(1, 2, 3)).sqrt()
            realized_delta_norm = (
                rgb_calibrated.float() - rgb.float()).pow(2).mean(
                    dim=(1, 2, 3)).sqrt()
            valid_count = valid.sum().clamp_min(1)
            attention_entropy = -(attention * attention.clamp_min(
                1e-12).log()).sum(dim=-1)
            if self.num_prototypes > 1:
                attention_entropy = attention_entropy / math.log(
                    self.num_prototypes)
            attention_entropy = (
                attention_entropy * valid.to(attention_entropy.dtype)
            ).sum() / valid_count
            mass = attention.sum(dim=1)
            probability = mass / mass.sum(
                dim=-1, keepdim=True).clamp_min(1e-12)
            conditioning_usage = (-(
                probability * probability.clamp_min(1e-12).log()
            ).sum(dim=-1)).exp().mean()

        aux = dict(
            thermal_objectness_logits=o_thermal,
            thermal_objectness=o_thermal.sigmoid().detach(),
            thermal_pool_attention=a_t_pool.detach(),
            thermal_prototypes=p_thermal.detach(),
            conditioning_attention=attention.detach(),
            conditioning=conditioning.detach(),
            epsilon=self.epsilon.detach(),
            conditioning_attention_entropy=attention_entropy.detach(),
            conditioning_prototype_usage=conditioning_usage.detach(),
            attention_entropy_thermal=TRPC.normalized_attention_entropy(
                a_t_pool).detach(),
            prototype_usage_thermal=TRPC.effective_prototype_usage(
                a_t_where).detach(),
            conditioning_rms=conditioning.float().pow(2).mean().sqrt().detach(),
            film_raw_rms=film_raw.float().pow(2).mean().sqrt().detach(),
            film_scale_abs_mean=scale.float().abs().mean().detach(),
            film_shift_abs_mean=shift.float().abs().mean().detach(),
            modulation_ratio=(modulation_norm / base).mean().detach(),
            delta_ratio=(delta_norm / base).mean().detach(),
            realized_delta_ratio=(realized_delta_norm / base).mean().detach(),
            diversity_loss=TRPC.prototype_diversity_loss(p_thermal))
        return rgb_calibrated, aux
