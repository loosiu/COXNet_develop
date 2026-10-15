"""Receiver-value frequency experts conditioned on cross-modal band relations.

L/H pairs, signed SWT, spatial router and residual mixing follow the existing
spatial MoFE. Donor bands affect relation gates and routing, never the value
path. This is a local relation gate, not DFCA FFT correlation or alignment.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .frequency_utility import valid_mask, validate_features
from .s3am_mofe import build_frequency_pairs
from .spatial_mofe import SpatialFrequencyMixture, SpatialResidualMoFE


class SelfBandRelationExpert(nn.Module):
    def __init__(self, hidden, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(), nn.Conv2d(hidden, hidden, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.output = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, receiver_relation, donor_relation, receiver_value):
        descriptor = torch.cat((receiver_relation * donor_relation,
                                (receiver_relation-donor_relation).abs()), dim=1)
        gate = 2 * self.gate(descriptor).sigmoid()
        return self.output(gate * receiver_value), gate


class SelfBandSpatialMixture(SpatialFrequencyMixture):
    pairs = ((0, 0), (0, 1), (1, 0), (1, 1))

    def __init__(self, channels=256, dim=64, expert_hidden=64, router_hidden=64):
        super().__init__(channels=channels, dim=dim, expert_hidden=expert_hidden,
                         router_hidden=router_hidden)
        # Preserve the existing project/router initialization and next bank RNG.
        with torch.random.fork_rng(devices=[]):
            widths = (dim, 3 * dim)
            self.receiver_adapters = nn.ModuleList(nn.Conv2d(c, expert_hidden, 1, bias=False) for c in widths)
            self.donor_adapters = nn.ModuleList(nn.Conv2d(c, expert_hidden, 1, bias=False) for c in widths)
            self.values = nn.ModuleList(nn.Conv2d(c, expert_hidden, 1, bias=False) for c in widths)
            self.experts = nn.ModuleList(SelfBandRelationExpert(expert_hidden, channels) for _ in self.pairs)

    @staticmethod
    def normalize_channels(value):
        return F.layer_norm(value.permute(0, 2, 3, 1), (value.shape[1],), eps=1e-5).permute(0, 3, 1, 2)

    def build_cache(self, rgb, thermal, valid_masks=None):
        validate_features((rgb, thermal), self.channels)
        valid_masks = (None, None) if valid_masks is None else valid_masks
        if len(valid_masks) != 2:
            raise ValueError('Expected RGB and thermal valid masks')
        rm, tm = valid_mask(valid_masks[0], rgb), valid_mask(valid_masks[1], thermal)
        mask = rm & tm
        receiver_bands, donor_bands = self.prepare_bands(rgb, thermal, rm, tm)
        pairs = build_frequency_pairs(receiver_bands, donor_bands)
        receiver = (receiver_bands[0], torch.cat(receiver_bands[1:], dim=1))
        donor = (donor_bands[0], torch.cat(donor_bands[1:], dim=1))
        u = [self.normalize_channels(adapter(band)) for adapter, band in zip(self.receiver_adapters, receiver)]
        d = [self.normalize_channels(adapter(band)) for adapter, band in zip(self.donor_adapters, donor)]
        values = [projection(band) for projection, band in zip(self.values, receiver)]
        results = [expert(u[a], d[b], values[a]) for expert, (a, b) in zip(self.experts, self.pairs)]
        candidates = torch.stack([item[0] for item in results], dim=1)
        # Detached diagnostics only. Detection gradients flow through candidates.
        relation_gates = torch.stack([item[1].detach() for item in results], dim=1)
        logits = self.router(torch.cat(pairs, dim=1)).masked_fill(~mask, 0)
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            logits = logits.float()
            weights = logits.softmax(dim=1)
        return dict(rgb=rgb, thermal=thermal, candidates=candidates, w=weights,
                    s=logits, valid_mask=mask, relation_gates=relation_gates)


class SelfBandResidualMoFE(SpatialResidualMoFE):
    gate_mode = 'self_band_mofe'

    def __init__(self, channels=256, dim=64, expert_hidden=64, router_hidden=64,
                 residual_scale=0., residual_multiplier=1.):
        nn.Module.__init__(self)
        if isinstance(residual_scale, bool) or not math.isfinite(residual_scale):
            raise ValueError('residual_scale must be finite')
        if isinstance(residual_multiplier, bool) or not math.isfinite(residual_multiplier) or residual_multiplier <= 0:
            raise ValueError('residual_multiplier must be positive and finite')
        options = dict(channels=channels, dim=dim, expert_hidden=expert_hidden, router_hidden=router_hidden)
        self.rgb_from_thermal = SelfBandSpatialMixture(**options)
        self.thermal_from_rgb = SelfBandSpatialMixture(**options)
        self.gamma_rgb = nn.Parameter(torch.tensor(float(residual_scale)))
        self.gamma_thermal = nn.Parameter(torch.tensor(float(residual_scale)))
        self.residual_multiplier = float(residual_multiplier)
