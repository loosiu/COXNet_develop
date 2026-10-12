"""D2: the same frequency candidates, with detection-trained independent sigmoid gates."""
import math

import torch
from torch import nn

from .frequency_utility import (FrequencyUtilityEnhancer, ChannelLN, haar_analysis,
                                validate_features, valid_mask)


class DetectionGate(nn.Module):
    def __init__(self, dim, hidden, feature_layers=None):
        super().__init__()
        layers = (list(feature_layers) if feature_layers is not None else [
            nn.Conv2d(3*dim, hidden, 1), ChannelLN(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU()])
        self.head = nn.Sequential(*layers, nn.Conv2d(hidden, 1, 1))
        nn.init.normal_(self.head[-1].weight, std=.01)
        nn.init.constant_(self.head[-1].bias, -math.log(3))

    def forward(self, q, context, candidate):
        # Detection gradients reach the score head and its input features.
        score = self.head(torch.cat((q, context, candidate), 1))[:, 0]
        with torch.autocast(device_type=score.device.type, enabled=False):
            score = score.float()
            return score, torch.sigmoid(score)


class DetectionFrequencyEnhancer(FrequencyUtilityEnhancer):
    gate_mode = 'sigmoid_detection'

    def __init__(self, channels=256, dim=64, heads=4, window_size=7,
                 residual_scale=.1, evaluator_hidden=64, chunk_size=256):
        # Preserve every original producer weight and score hidden-layer initialization.
        super().__init__(channels, dim, heads, window_size, residual_scale, evaluator_hidden, chunk_size)
        with torch.random.fork_rng(devices=[]):
            for direction in (self.rgb_direction, self.thermal_direction):
                for expert in direction.experts:
                    expert.evaluator = DetectionGate(dim, evaluator_hidden, list(expert.evaluator.head.children())[:-1])
                    expert.attention.detach_context = False

    @staticmethod
    def direction_cache(direction, receiver, donor, bands, donor_mask):
        q = direction.query(direction.query_norm(receiver))
        candidates, scores, weights = [], [], []
        for expert, band in zip(direction.experts, bands):
            candidate, context = expert(q, donor, band, donor_mask)
            score, weight = expert.evaluator(q, context, candidate)
            candidates.append(candidate); scores.append(score); weights.append(weight)
        return torch.stack(candidates, 1), torch.stack(scores, 1), torch.stack(weights, 1)

    def build_cache(self, rgb, thermal, valid_masks=None):
        validate_features((rgb, thermal), self.channels)
        valid_masks = (None, None) if valid_masks is None else valid_masks
        if len(valid_masks) != 2: raise ValueError('Expected RGB and thermal valid masks')
        masks = valid_mask(valid_masks[0], rgb), valid_mask(valid_masks[1], thermal)
        zr, zt = self.project_rgb(rgb)*masks[0], self.project_thermal(thermal)*masks[1]
        r, rs, rw = self.direction_cache(self.rgb_direction, zr, zt, haar_analysis(zt), masks[1])
        t, ts, tw = self.direction_cache(self.thermal_direction, zt, zr, haar_analysis(zr), masks[0])
        return dict(rgb=rgb, thermal=thermal, candidates=torch.stack((r, t), 1),
                    s=torch.stack((rs, ts), 1), w=torch.stack((rw, tw), 1),
                    valid_masks=torch.stack([mask[:, 0] for mask in masks], 1))

    def mix(self, cache, gates=None):
        weights = cache['w'] if gates is None else gates
        if weights.shape != cache['w'].shape or weights.device != cache['w'].device:
            raise ValueError('Gate override must match [B,2,4,H,W] and device')
        if not torch.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
            raise ValueError('Gate values must be finite within [0,1]')
        candidates = cache['candidates']
        # Independent gates, mean4 unchanged; neither normalization nor detach.
        message = (weights.to(candidates.dtype).unsqueeze(3)*candidates).mean(2)
        message = message*cache['valid_masks'].unsqueeze(2)
        return (cache['rgb']+self.residual_scale*self.output_rgb(message[:, 0]),
                cache['thermal']+self.residual_scale*self.output_thermal(message[:, 1]))

    def forward(self, rgb, thermal, valid_masks=None, gates=None):
        cache = self.build_cache(rgb, thermal, valid_masks)
        r, t = self.mix(cache, gates)
        return r, t, {key: cache[key] for key in ('s', 'w', 'valid_masks')}
