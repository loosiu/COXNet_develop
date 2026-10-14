"""Original image-level MoFE banks with independent, scaled P3 residuals.

The four experts, stride-one Haar analysis and softmax router are reused
unchanged. Both directions read the original features. Only the detector
connection changes: independent banks and learnable scalar residual scales.
This is a detection adaptation, not the complete S3AM/RCFA architecture.
"""
import math

import torch
from torch import nn

from .s3am_mofe import S3AMFrequencyMixture


class BidirectionalResidualMoFE(nn.Module):
    gate_mode = 'residual_mofe'

    def __init__(self, channels=256, dim=64, expert_hidden=64,
                 router_hidden=64, residual_scale=.1):
        super().__init__()
        if isinstance(residual_scale, bool) or not math.isfinite(residual_scale):
            raise ValueError('residual_scale must be a finite scalar')
        options = dict(channels=channels, dim=dim, expert_hidden=expert_hidden,
                       router_hidden=router_hidden)
        self.rgb_from_thermal = S3AMFrequencyMixture(**options)
        self.thermal_from_rgb = S3AMFrequencyMixture(**options)
        self.gamma_rgb = nn.Parameter(torch.tensor(float(residual_scale)))
        self.gamma_thermal = nn.Parameter(torch.tensor(float(residual_scale)))

    def build_cache(self, rgb, thermal, valid_masks=None):
        rgb_cache = self.rgb_from_thermal.build_cache(rgb, thermal, valid_masks)
        swapped_masks = None if valid_masks is None else (valid_masks[1], valid_masks[0])
        thermal_cache = self.thermal_from_rgb.build_cache(thermal, rgb, swapped_masks)
        return dict(rgb_from_thermal=rgb_cache, thermal_from_rgb=thermal_cache,
                    w=torch.stack((rgb_cache['w'], thermal_cache['w']), dim=1))

    def mix(self, cache, gates=None):
        weights = cache['w'] if gates is None else gates
        if weights.shape != cache['w'].shape or weights.device != cache['w'].device:
            raise ValueError('Routing override must match [B,2,4] and device')
        if (not torch.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any()
                or not torch.allclose(weights.float().sum(-1),
                                      torch.ones_like(weights[:, :, 0], dtype=torch.float32), atol=1e-6)):
            raise ValueError('Routing weights must be finite probabilities summing to one')
        results = []
        for direction, name, gamma in [(0, 'rgb_from_thermal', self.gamma_rgb),
                                       (1, 'thermal_from_rgb', self.gamma_thermal)]:
            bank = cache[name]
            update = (weights[:, direction].to(bank['candidates'].dtype)[:, :, None, None, None]
                      * bank['candidates']).sum(1)
            results.append(bank['rgb'] + gamma * update * bank['valid_mask'])
        return tuple(results)

    def forward(self, rgb, thermal, valid_masks=None, gates=None):
        cache = self.build_cache(rgb, thermal, valid_masks)
        rgb_out, thermal_out = self.mix(cache, gates)
        weights = cache['w'] if gates is None else gates
        with torch.no_grad():
            diagnostics = dict(
                routing_weight=weights.detach().float(),
                routing_entropy=-(weights.float() * weights.float().clamp_min(1e-12).log()).sum(-1),
                residual_scale=torch.stack((self.gamma_rgb, self.gamma_thermal)).detach()
                    .unsqueeze(0).expand(rgb.shape[0], -1),
                candidate_rms=torch.stack([
                    cache[name]['candidates'].float().square().mean((2, 3, 4)).sqrt()
                    for name in ['rgb_from_thermal', 'thermal_from_rgb']], dim=1),
                relative_update=torch.stack([
                    (out-source).float().square().mean((1, 2, 3)).sqrt()
                    / source.float().square().mean((1, 2, 3)).sqrt().clamp_min(1e-12)
                    for out, source in [(rgb_out, rgb), (thermal_out, thermal)]], dim=1))
        aux = dict(routing_weights=weights,
                   routing_logits=torch.stack((cache['rgb_from_thermal']['s'],
                                               cache['thermal_from_rgb']['s']), dim=1),
                   diagnostics=diagnostics)
        return rgb_out, thermal_out, aux
