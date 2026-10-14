"""SWT-pair relation experts with direct P3 replacement or residual control.

Inspired by DFCA's Q/K frequency correlation and sigmoid-modulated V. There
is one SWT decomposition; FFT here computes correlation, with no extra radial
frequency partition. RMS normalization and joint Q/attended-V projection are
explicit adaptation choices. This is neither full DFCA nor local token QK
softmax attention. Direct mode has no original-feature skip or gamma parameter.
"""
import math

import torch
from torch import nn

from .frequency import frequency_correlation
from .frequency_utility import positive_integer, valid_mask, validate_features
from .s3am_mofe import build_frequency_pairs
from .spatial_mofe import SpatialFrequencyMixture, SpatialResidualMoFE, _validate_weights


def _validate_eps(eps):
    if (isinstance(eps, bool) or not math.isfinite(eps)
            or not torch.finfo(torch.float32).tiny <= eps <= torch.finfo(torch.float32).max):
        raise ValueError('correlation_eps must be positive, finite, normal float32')


def normalized_frequency_relation(query, key, eps=1e-4):
    """Channelwise circular correlation, normalized by per-image/channel RMS."""
    _validate_eps(eps)
    validate_features((query, key), query.shape[1])
    with torch.autocast(device_type=query.device.type, enabled=False):
        relation = frequency_correlation(torch.fft.fft2(query.float()), torch.fft.fft2(key.float()))
        scale = relation.detach().abs().amax((-2, -1), keepdim=True).clamp_min(eps)
        unit = relation / scale
        return unit / (unit.square().mean((-2, -1), keepdim=True) + (eps / scale).square()).sqrt()


class RelationExpert(nn.Module):
    def __init__(self, receiver_channels, donor_channels, hidden_dim=64,
                 channels=256, correlation_eps=1e-4):
        super().__init__()
        for name, value in [('receiver_channels', receiver_channels), ('donor_channels', donor_channels),
                            ('hidden_dim', hidden_dim), ('channels', channels)]:
            positive_integer(value, name)
        _validate_eps(correlation_eps)
        self.receiver_channels, self.donor_channels = receiver_channels, donor_channels
        self.correlation_eps = float(correlation_eps)

        def project(input_channels):
            return nn.Sequential(nn.Conv2d(input_channels, hidden_dim, 1, bias=False),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False))

        self.query = project(receiver_channels)
        self.key = project(donor_channels)
        self.value = project(donor_channels)
        self.modulator = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm = nn.LayerNorm(2 * hidden_dim)
        self.output = nn.Conv2d(2 * hidden_dim, channels, 1, bias=False)

    def forward_with_diagnostics(self, receiver, donor, mask=None):
        validate_features((receiver,), self.receiver_channels)
        validate_features((donor,), self.donor_channels)
        if (receiver.shape[0] != donor.shape[0] or receiver.shape[-2:] != donor.shape[-2:]
                or receiver.device != donor.device or receiver.dtype != donor.dtype):
            raise ValueError('Expert inputs must share batch, spatial dimensions, dtype and device')
        mask = valid_mask(mask, receiver)
        query, key, value = self.query(receiver) * mask, self.key(donor) * mask, self.value(donor) * mask
        relation = normalized_frequency_relation(query, key, self.correlation_eps)
        attention = self.modulator(relation).sigmoid()
        # Learned joint transformation retains receiver information through Q,
        # without adding the original C-channel feature as an identity shortcut.
        joint = torch.cat((query, attention.to(value.dtype) * value), dim=1)
        joint = self.norm(joint.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return self.output(joint), dict(attention=attention, relation=relation)

    def forward(self, receiver, donor):
        return self.forward_with_diagnostics(receiver, donor)[0]


class RelationFrequencyMixture(SpatialFrequencyMixture):
    def __init__(self, original, expert_hidden, correlation_eps):
        nn.Module.__init__(self)
        self.channels, self.dim = original.channels, original.dim
        self.project, self.router = original.project, original.router
        self.experts = nn.ModuleList(RelationExpert(r * self.dim, t * self.dim,
            expert_hidden, self.channels, correlation_eps) for r, t in [(1, 1), (1, 3), (3, 1), (3, 3)])

    def build_cache(self, rgb, thermal, valid_masks=None):
        validate_features((rgb, thermal), self.channels)
        valid_masks = (None, None) if valid_masks is None else valid_masks
        if len(valid_masks) != 2:
            raise ValueError('Expected RGB and thermal valid masks')
        rm, tm = valid_mask(valid_masks[0], rgb), valid_mask(valid_masks[1], thermal)
        mask = rm & tm
        rgb_bands, thermal_bands = self.prepare_bands(rgb, thermal, rm, tm)
        r_low, t_low = rgb_bands[0], thermal_bands[0]
        r_high, t_high = torch.cat(rgb_bands[1:], 1), torch.cat(thermal_bands[1:], 1)
        inputs = [(r_low, t_low), (r_low, t_high), (r_high, t_low), (r_high, t_high)]
        candidates, attentions, relations = [], [], []
        for expert, (receiver, donor) in zip(self.experts, inputs):
            output, stats = expert.forward_with_diagnostics(receiver, donor, mask)
            candidates.append(output); attentions.append(stats['attention']); relations.append(stats['relation'])
        logits = self.router(torch.cat(build_frequency_pairs(rgb_bands, thermal_bands), 1)).masked_fill(~mask, 0)
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            logits = logits.float(); weights = logits.softmax(dim=1)
        return dict(rgb=rgb, thermal=thermal, valid_mask=mask, candidates=torch.stack(candidates, 1),
                    w=weights, s=logits, attention=torch.stack(attentions, 1), relation=torch.stack(relations, 1))

    def mix(self, cache, gates=None):
        weights = cache['w'] if gates is None else gates
        _validate_weights(weights, cache['w'], 1)
        mixed = (weights.to(cache['candidates'].dtype).unsqueeze(2) * cache['candidates']).sum(1)
        return torch.where(cache['valid_mask'], mixed.to(cache['rgb'].dtype), cache['rgb']), cache['thermal']


class BidirectionalRelationMoFE(nn.Module):
    gate_mode = 'relation_mofe'

    def __init__(self, channels=256, dim=64, expert_hidden=64, router_hidden=64,
                 output_mode='direct', correlation_eps=1e-4,
                 residual_scale=0., residual_multiplier=.5):
        super().__init__()
        if output_mode not in ('direct', 'residual'):
            raise ValueError('output_mode must be direct or residual')
        _validate_eps(correlation_eps)
        # Preserve original project/router draws and the caller's RNG. The
        # replaced CNN experts are discarded, not registered with the optimizer.
        original = SpatialResidualMoFE(channels, dim, expert_hidden, router_hidden,
                                       residual_scale, residual_multiplier)
        with torch.random.fork_rng(devices=[]):
            self.rgb_from_thermal = RelationFrequencyMixture(original.rgb_from_thermal, expert_hidden, correlation_eps)
            self.thermal_from_rgb = RelationFrequencyMixture(original.thermal_from_rgb, expert_hidden, correlation_eps)
        self.output_mode = output_mode
        if output_mode == 'residual':
            self.gamma_rgb, self.gamma_thermal = original.gamma_rgb, original.gamma_thermal
            self.residual_multiplier = float(residual_multiplier)

    def build_cache(self, rgb, thermal, valid_masks=None):
        rgb_cache = self.rgb_from_thermal.build_cache(rgb, thermal, valid_masks)
        swapped = None if valid_masks is None else (valid_masks[1], valid_masks[0])
        thermal_cache = self.thermal_from_rgb.build_cache(thermal, rgb, swapped)
        return dict(rgb_from_thermal=rgb_cache, thermal_from_rgb=thermal_cache,
                    w=torch.stack((rgb_cache['w'], thermal_cache['w']), 1))

    def mix(self, cache, gates=None):
        weights = cache['w'] if gates is None else gates
        _validate_weights(weights, cache['w'], 2)
        outputs = []
        for direction, name in enumerate(['rgb_from_thermal', 'thermal_from_rgb']):
            bank = cache[name]
            mixed = (weights[:, direction].to(bank['candidates'].dtype).unsqueeze(2)
                     * bank['candidates']).sum(1).to(bank['rgb'].dtype)
            if self.output_mode == 'residual':
                gamma = self.gamma_rgb if direction == 0 else self.gamma_thermal
                mixed = bank['rgb'] + self.residual_multiplier * gamma * mixed
            outputs.append(torch.where(bank['valid_mask'], mixed, bank['rgb']))
        return tuple(outputs)

    def forward(self, rgb, thermal, valid_masks=None, gates=None):
        cache = self.build_cache(rgb, thermal, valid_masks)
        outputs = self.mix(cache, gates); weights = cache['w'] if gates is None else gates
        with torch.no_grad():
            values = {k: [] for k in ['routing_weight', 'weight_spatial_std', 'routing_entropy',
                'candidate_rms', 'relation_attention', 'relation_saturation', 'relative_update',
                'output_rms', 'source_rms', 'valid_fraction']}
            for direction, name in enumerate(['rgb_from_thermal', 'thermal_from_rgb']):
                bank = cache[name]; source, output = bank['rgb'], outputs[direction]
                mask = bank['valid_mask'].float(); count = mask.sum((2, 3)).clamp_min(1)
                w = weights[:, direction].detach().float(); mean = (w * mask).sum((2, 3)) / count
                values['routing_weight'].append(mean)
                values['weight_spatial_std'].append((((w-mean[:, :, None, None]).square()*mask).sum((2, 3))/count).sqrt())
                entropy = -(w*w.clamp_min(1e-12).log()).sum(1, keepdim=True)
                values['routing_entropy'].append(((entropy*mask).sum((2, 3))/count).squeeze(1))
                for key, tensor in [('candidate_rms', bank['candidates'].float())]:
                    values[key].append(((tensor.square()*mask.unsqueeze(1)).sum((2, 3, 4))/(count*tensor.shape[2])).sqrt())
                attention = bank['attention'].float()
                values['relation_attention'].append((attention*mask.unsqueeze(1)).sum((2, 3, 4))/(count*attention.shape[2]))
                saturation = ((attention < .05) | (attention > .95)).float()
                values['relation_saturation'].append((saturation*mask.unsqueeze(1)).sum((2, 3, 4))/(count*attention.shape[2]))
                rms = []
                for tensor in [source, output, output-source]:
                    rms.append(((tensor.float().square()*mask).sum((1, 2, 3))/(count.squeeze(1)*tensor.shape[1])).sqrt())
                values['source_rms'].append(rms[0]); values['output_rms'].append(rms[1])
                values['relative_update'].append(rms[2]/rms[0].clamp_min(1e-12))
                values['valid_fraction'].append(mask.mean((1, 2, 3)))
            diagnostics = {key: torch.stack(value, 1) for key, value in values.items()}
            if self.output_mode == 'residual':
                gamma = torch.stack((self.gamma_rgb, self.gamma_thermal)).detach().unsqueeze(0).expand(rgb.shape[0], -1)
                diagnostics.update(residual_scale=gamma,
                    residual_multiplier=torch.full_like(gamma, self.residual_multiplier),
                    effective_residual_scale=self.residual_multiplier * gamma)
        return *outputs, dict(routing_weights=weights,
            routing_logits=torch.stack((cache['rgb_from_thermal']['s'], cache['thermal_from_rgb']['s']), 1),
            diagnostics=diagnostics, output_mode=self.output_mode)
