"""S3AM-style four frequency pairs, adapted to the unchanged dual RT-DETR P3.

Pair composition, simplex routing and direct RGB residual follow paper eqs.2-7.
Haar boundaries, widths, convolutional experts and router layers are disclosed
adaptation choices; the author's exact expert implementation is not public.
"""
import torch
from torch import nn

from .frequency_utility import haar_analysis, positive_integer, valid_mask, validate_features


PAIR_NAMES = ('L_L', 'L_H', 'H_L', 'H_H')


def build_frequency_pairs(rgb_bands, thermal_bands):
    if len(rgb_bands) != 4 or len(thermal_bands) != 4:
        raise ValueError('Each modality must supply L, LH, HL, HH')
    validate_features(tuple(rgb_bands) + tuple(thermal_bands), rgb_bands[0].shape[1])
    rgb_low, thermal_low = rgb_bands[0], thermal_bands[0]
    rgb_high = torch.cat(rgb_bands[1:], dim=1)
    thermal_high = torch.cat(thermal_bands[1:], dim=1)
    return (torch.cat((rgb_low, thermal_low), 1),
            torch.cat((rgb_low, thermal_high), 1),
            torch.cat((rgb_high, thermal_low), 1),
            torch.cat((rgb_high, thermal_high), 1))


class S3AMFrequencyMixture(nn.Module):
    gate_mode = 's3am_mofe'

    def __init__(self, channels=256, dim=64, expert_hidden=64, router_hidden=64):
        super().__init__()
        for name, value in [('channels', channels), ('dim', dim),
                            ('expert_hidden', expert_hidden), ('router_hidden', router_hidden)]:
            positive_integer(value, name)
        self.channels, self.dim = channels, dim
        self.project = nn.Conv2d(channels, dim, 1, bias=False)
        nn.init.xavier_uniform_(self.project.weight)
        self.experts = nn.ModuleList(nn.Sequential(
            nn.Conv2d(multiplier * dim, expert_hidden, 1, bias=False), nn.GELU(),
            nn.Conv2d(expert_hidden, expert_hidden, 3, padding=1,
                      groups=expert_hidden, bias=False), nn.GELU(),
            nn.Conv2d(expert_hidden, channels, 1, bias=False))
            for multiplier in (2, 4, 4, 6))
        self.router = nn.Sequential(
            nn.Conv2d(16 * dim, router_hidden, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(router_hidden, 4, 1))
        nn.init.normal_(self.router[-1].weight, std=.01)
        nn.init.zeros_(self.router[-1].bias)

    def build_cache(self, rgb, thermal, valid_masks=None):
        validate_features((rgb, thermal), self.channels)
        valid_masks = (None, None) if valid_masks is None else valid_masks
        if len(valid_masks) != 2:
            raise ValueError('Expected RGB and thermal valid masks')
        rm, tm = valid_mask(valid_masks[0], rgb), valid_mask(valid_masks[1], thermal)
        mask = rm & tm
        pairs = build_frequency_pairs(haar_analysis(self.project(rgb) * rm),
                                      haar_analysis(self.project(thermal) * tm))
        candidates = torch.stack([expert(pair) for expert, pair in zip(self.experts, pairs)], 1)
        hidden = self.router[:2](torch.cat(pairs, 1))
        if valid_masks[0] is None and valid_masks[1] is None:
            pooled = self.router[2](hidden)
        else:
            pooled = (hidden * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True).clamp_min(1)
        logits = self.router[-1](pooled).flatten(1)
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            logits = logits.float()
            weights = logits.softmax(dim=1)
        return dict(rgb=rgb, thermal=thermal, candidates=candidates,
                    w=weights, s=logits, valid_mask=mask)

    def mix(self, cache, gates=None):
        weights = cache['w'] if gates is None else gates
        if weights.shape != cache['w'].shape or weights.device != cache['w'].device:
            raise ValueError('Routing override must match [B,4] and device')
        if not torch.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any() or not torch.allclose(
                weights.float().sum(1), torch.ones(weights.shape[0], device=weights.device), atol=1e-6):
            raise ValueError('Routing weights must be finite probabilities summing to one')
        update = (weights.to(cache['candidates'].dtype)[:, :, None, None, None] * cache['candidates']).sum(1)
        return cache['rgb'] + update * cache['valid_mask'], cache['thermal']

    def forward(self, rgb, thermal, valid_masks=None, gates=None):
        cache = self.build_cache(rgb, thermal, valid_masks)
        rgb_out, thermal_out = self.mix(cache, gates)
        with torch.no_grad():
            candidate_rms = cache['candidates'].float().square().mean((2, 3, 4)).sqrt()
            relative_update = ((rgb_out-rgb).float().square().mean((1, 2, 3)).sqrt()
                               / rgb.float().square().mean((1, 2, 3)).sqrt().clamp_min(1e-12))
        aux = dict(routing_weights=cache['w'], routing_logits=cache['s'],
                   candidate_rms=candidate_rms, relative_update=relative_update)
        return rgb_out, thermal_out, aux
