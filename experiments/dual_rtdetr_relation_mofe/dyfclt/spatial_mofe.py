"""Spatial routing of the existing four frequency experts at projected P3.

Both independent banks read the original pair. The experts, projections and
router convolutions retain B's parameter names and initialization. Global
pooling alone is removed; mixing is dense at each cell. The fixed multiplier
scales the learned residual gamma, and does not constrain its eventual value.
This is an RT-DETR adaptation, not the complete S3AM/RCFA implementation.
"""
import math

import torch
from torch import nn

from .frequency_utility import valid_mask, validate_features
from .s3am_mofe import S3AMFrequencyMixture, build_frequency_pairs


class SpatialFrequencyMixture(S3AMFrequencyMixture):
    def __init__(self, **options):
        super().__init__(**options)
        self.router[2] = nn.Identity()

    def build_cache(self, rgb, thermal, valid_masks=None):
        validate_features((rgb, thermal), self.channels)
        valid_masks = (None, None) if valid_masks is None else valid_masks
        if len(valid_masks) != 2:
            raise ValueError('Expected RGB and thermal valid masks')
        rm, tm = valid_mask(valid_masks[0], rgb), valid_mask(valid_masks[1], thermal)
        mask = rm & tm
        pairs = build_frequency_pairs(*self.prepare_bands(rgb, thermal, rm, tm))
        candidates = torch.stack([expert(pair) for expert, pair in zip(self.experts, pairs)], 1)
        logits = self.router(torch.cat(pairs, 1)).masked_fill(~mask, 0)
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            logits = logits.float()
            weights = logits.softmax(dim=1)
        return dict(rgb=rgb, thermal=thermal, candidates=candidates,
                    w=weights, s=logits, valid_mask=mask)

    def mix(self, cache, gates=None):
        weights = cache['w'] if gates is None else gates
        _validate_weights(weights, cache['w'], expert_dim=1)
        update = (weights.to(cache['candidates'].dtype).unsqueeze(2) * cache['candidates']).sum(1)
        return cache['rgb'] + update * cache['valid_mask'], cache['thermal']

    def forward(self, rgb, thermal, valid_masks=None, gates=None):
        cache = self.build_cache(rgb, thermal, valid_masks)
        outputs = self.mix(cache, gates)
        weights = cache['w'] if gates is None else gates
        return *outputs, dict(routing_weights=weights, routing_logits=cache['s'])


def _validate_weights(weights, reference, expert_dim):
    if weights.shape != reference.shape or weights.device != reference.device:
        raise ValueError('Routing override must match the spatial weights shape and device')
    if (not torch.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any()
            or not torch.allclose(weights.float().sum(expert_dim),
                                  torch.ones_like(weights.select(expert_dim, 0), dtype=torch.float32), atol=1e-6)):
        raise ValueError('Routing weights must be finite probabilities summing over experts to one')


class SpatialResidualMoFE(nn.Module):
    gate_mode = 'spatial_mofe'

    def __init__(self, channels=256, dim=64, expert_hidden=64, router_hidden=64,
                 residual_scale=.1, residual_multiplier=.5):
        super().__init__()
        if isinstance(residual_scale, bool) or not math.isfinite(residual_scale):
            raise ValueError('residual_scale must be a finite scalar')
        if (isinstance(residual_multiplier, bool) or not math.isfinite(residual_multiplier)
                or residual_multiplier <= 0):
            raise ValueError('residual_multiplier must be a finite positive scalar')
        options = dict(channels=channels, dim=dim, expert_hidden=expert_hidden, router_hidden=router_hidden)
        self.rgb_from_thermal = SpatialFrequencyMixture(**options)
        self.thermal_from_rgb = SpatialFrequencyMixture(**options)
        self.gamma_rgb = nn.Parameter(torch.tensor(float(residual_scale)))
        self.gamma_thermal = nn.Parameter(torch.tensor(float(residual_scale)))
        self.residual_multiplier = float(residual_multiplier)

    def build_cache(self, rgb, thermal, valid_masks=None):
        rgb_cache = self.rgb_from_thermal.build_cache(rgb, thermal, valid_masks)
        swapped_masks = None if valid_masks is None else (valid_masks[1], valid_masks[0])
        thermal_cache = self.thermal_from_rgb.build_cache(thermal, rgb, swapped_masks)
        return dict(rgb_from_thermal=rgb_cache, thermal_from_rgb=thermal_cache,
                    w=torch.stack((rgb_cache['w'], thermal_cache['w']), dim=1))

    def mix(self, cache, gates=None):
        weights = cache['w'] if gates is None else gates
        _validate_weights(weights, cache['w'], expert_dim=2)
        results = []
        for direction, name, gamma in [(0, 'rgb_from_thermal', self.gamma_rgb),
                                       (1, 'thermal_from_rgb', self.gamma_thermal)]:
            bank = cache[name]
            update = (weights[:, direction].to(bank['candidates'].dtype).unsqueeze(2)
                      * bank['candidates']).sum(1)
            results.append(bank['rgb'] + self.residual_multiplier * gamma * update * bank['valid_mask'])
        return tuple(results)

    def forward(self, rgb, thermal, valid_masks=None, gates=None):
        cache = self.build_cache(rgb, thermal, valid_masks)
        rgb_out, thermal_out = self.mix(cache, gates)
        weights = cache['w'] if gates is None else gates
        with torch.no_grad():
            means, stds, entropies, candidate_rms, relative_updates, valid_fractions = [], [], [], [], [], []
            for direction, name, source, output in [
                (0, 'rgb_from_thermal', rgb, rgb_out), (1, 'thermal_from_rgb', thermal, thermal_out),
            ]:
                mask = cache[name]['valid_mask'].float()
                count = mask.sum((2, 3)).clamp_min(1)
                w = weights[:, direction].detach().float()
                mean = (w * mask).sum((2, 3)) / count
                means.append(mean)
                stds.append((((w - mean[:, :, None, None]).square() * mask).sum((2, 3)) / count).sqrt())
                entropy = -(w * w.clamp_min(1e-12).log()).sum(1, keepdim=True)
                entropies.append(((entropy * mask).sum((2, 3)) / count).squeeze(1))
                c = cache[name]['candidates'].detach().float()
                candidate_rms.append(((c.square() * mask.unsqueeze(1)).sum((2, 3, 4))
                                      / (count * c.shape[2])).sqrt())
                delta_rms = (((output-source).float().square() * mask).sum((1, 2, 3))
                             / (count.squeeze(1) * source.shape[1])).sqrt()
                source_rms = ((source.float().square() * mask).sum((1, 2, 3))
                              / (count.squeeze(1) * source.shape[1])).sqrt()
                relative_updates.append(delta_rms / source_rms.clamp_min(1e-12))
                valid_fractions.append(mask.mean((1, 2, 3)))
            gamma = torch.stack((self.gamma_rgb, self.gamma_thermal)).detach().unsqueeze(0).expand(rgb.shape[0], -1)
            diagnostics = dict(routing_weight=torch.stack(means, 1), weight_spatial_std=torch.stack(stds, 1),
                routing_entropy=torch.stack(entropies, 1), residual_scale=gamma,
                residual_multiplier=torch.full_like(gamma, self.residual_multiplier),
                effective_residual_scale=self.residual_multiplier * gamma,
                candidate_rms=torch.stack(candidate_rms, 1), relative_update=torch.stack(relative_updates, 1),
                valid_fraction=torch.stack(valid_fractions, 1))
        return rgb_out, thermal_out, dict(routing_weights=weights,
            routing_logits=torch.stack((cache['rgb_from_thermal']['s'], cache['thermal_from_rgb']['s']), 1),
            diagnostics=diagnostics)
