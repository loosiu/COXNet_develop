"""Post-encoder donor-conditioned CNN experts and separate spatial injection strength."""
import math

import torch
from torch import nn

from .spatial_mofe import SpatialFrequencyMixture, SpatialResidualMoFE, _validate_weights


class DonorConditionedExpert(nn.Module):
    """Receiver conditions a gate; only donor values enter the output projection."""
    def __init__(self, receiver_channels, donor_channels, hidden, channels):
        super().__init__()
        self.receiver_channels, self.donor_channels = receiver_channels, donor_channels
        self.value = nn.Sequential(nn.Conv2d(donor_channels, hidden, 1, bias=False), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False), nn.GELU())
        self.gate = nn.Sequential(nn.Conv2d(receiver_channels + donor_channels, hidden, 1),
            nn.GELU(), nn.Conv2d(hidden, hidden, 1), nn.Sigmoid())
        self.output = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, pair):
        _, donor = pair.split((self.receiver_channels, self.donor_channels), dim=1)
        return self.output(self.gate(pair) * self.value(donor))


class DonorSpatialMixture(SpatialFrequencyMixture):
    def __init__(self, channels=256, dim=64, expert_hidden=64, router_hidden=64):
        super().__init__(channels=channels, dim=dim, expert_hidden=expert_hidden, router_hidden=router_hidden)
        # Replacing only experts must leave projection/router and the next bank's RNG intact.
        with torch.random.fork_rng(devices=[]):
            self.experts = nn.ModuleList(DonorConditionedExpert(r * dim, t * dim, expert_hidden, channels)
                                        for r, t in ((1, 1), (1, 3), (3, 1), (3, 3)))


class DonorResidualMoFE(SpatialResidualMoFE):
    gate_mode = 'donor_mofe'

    def __init__(self, channels=256, dim=64, expert_hidden=64, router_hidden=64,
                 residual_scale=0., residual_multiplier=1.):
        nn.Module.__init__(self)
        if isinstance(residual_scale, bool) or not math.isfinite(residual_scale):
            raise ValueError('residual_scale must be finite')
        if isinstance(residual_multiplier, bool) or not math.isfinite(residual_multiplier) or residual_multiplier <= 0:
            raise ValueError('residual_multiplier must be positive and finite')
        options = dict(channels=channels, dim=dim, expert_hidden=expert_hidden, router_hidden=router_hidden)
        self.rgb_from_thermal = DonorSpatialMixture(**options)
        self.thermal_from_rgb = DonorSpatialMixture(**options)
        self.gamma_rgb = nn.Parameter(torch.tensor(float(residual_scale)))
        self.gamma_thermal = nn.Parameter(torch.tensor(float(residual_scale)))
        self.residual_multiplier = float(residual_multiplier)


class DonorStrengthResidualMoFE(DonorResidualMoFE):
    gate_mode = 'donor_strength_mofe'

    def __init__(self, channels=256, dim=64, expert_hidden=64, router_hidden=64,
                 residual_scale=0., residual_multiplier=1.):
        super().__init__(channels, dim, expert_hidden, router_hidden, residual_scale, residual_multiplier)
        with torch.random.fork_rng(devices=[]):
            self.beta_rgb = nn.Sequential(nn.Conv2d(3 * dim, router_hidden, 1), nn.GELU(),
                                          nn.Conv2d(router_hidden, 1, 1))
            self.beta_thermal = nn.Sequential(nn.Conv2d(3 * dim, router_hidden, 1), nn.GELU(),
                                              nn.Conv2d(router_hidden, 1, 1))
            for head in (self.beta_rgb, self.beta_thermal):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def _mix_with_strength(self, cache, gates=None):
        weights = cache['w'] if gates is None else gates
        _validate_weights(weights, cache['w'], expert_dim=2)
        outputs, betas = [], []
        for direction, name, bank_module, head, gamma in (
            (0, 'rgb_from_thermal', self.rgb_from_thermal, self.beta_rgb, self.gamma_rgb),
            (1, 'thermal_from_rgb', self.thermal_from_rgb, self.beta_thermal, self.gamma_thermal),
        ):
            bank = cache[name]
            update = (weights[:, direction].to(bank['candidates'].dtype).unsqueeze(2) * bank['candidates']).sum(1)
            context = torch.cat((bank_module.project(bank['rgb']), bank_module.project(bank['thermal']),
                                 bank_module.project(update)), 1)
            logits = head(context)
            with torch.autocast(device_type=logits.device.type, enabled=False):
                beta = 2 * logits.float().sigmoid()
            outputs.append(bank['rgb'] + self.residual_multiplier * gamma * beta.to(update.dtype) * update * bank['valid_mask'])
            betas.append(beta)
        return tuple(outputs), torch.cat(betas, dim=1)

    def mix(self, cache, gates=None):
        return self._mix_with_strength(cache, gates)[0]

    def forward(self, rgb, thermal, valid_masks=None, gates=None):
        cache = self.build_cache(rgb, thermal, valid_masks)
        (rgb_out, thermal_out), beta = self._mix_with_strength(cache, gates)
        weights = cache['w'] if gates is None else gates
        values = {key: [] for key in ('routing_weight', 'weight_spatial_std', 'routing_entropy',
                                     'candidate_rms', 'relative_update', 'valid_fraction',
                                     'injection_strength', 'injection_strength_spatial_std')}
        with torch.no_grad():
            for direction, name, source, output in ((0, 'rgb_from_thermal', rgb, rgb_out),
                                                   (1, 'thermal_from_rgb', thermal, thermal_out)):
                mask = cache[name]['valid_mask'].float()
                count = mask.sum((2, 3)).clamp_min(1)
                w = weights[:, direction].detach().float()
                mean = (w * mask).sum((2, 3)) / count
                values['routing_weight'].append(mean)
                values['weight_spatial_std'].append((((w-mean[:, :, None, None]).square()*mask).sum((2, 3))/count).sqrt())
                entropy = -(w*w.clamp_min(1e-12).log()).sum(1, keepdim=True)
                values['routing_entropy'].append(((entropy*mask).sum((2, 3))/count).squeeze(1))
                c = cache[name]['candidates'].detach().float()
                values['candidate_rms'].append(((c.square()*mask.unsqueeze(1)).sum((2, 3, 4))/(count*c.shape[2])).sqrt())
                delta_rms = (((output-source).float().square()*mask).sum((1, 2, 3))/(count.squeeze(1)*source.shape[1])).sqrt()
                source_rms = ((source.float().square()*mask).sum((1, 2, 3))/(count.squeeze(1)*source.shape[1])).sqrt()
                values['relative_update'].append(delta_rms/source_rms.clamp_min(1e-12))
                values['valid_fraction'].append(mask.mean((1, 2, 3)))
                strength = beta[:, direction:direction+1].detach()
                bmean = (strength*mask).sum((2, 3))/count
                values['injection_strength'].append(bmean.squeeze(1))
                values['injection_strength_spatial_std'].append((((strength-bmean[:, :, None, None]).square()*mask).sum((2, 3))/count).sqrt().squeeze(1))
            diagnostics = {key: torch.stack(value, 1) for key, value in values.items()}
            gamma = torch.stack((self.gamma_rgb, self.gamma_thermal)).detach().unsqueeze(0).expand(rgb.shape[0], -1)
            diagnostics.update(residual_scale=gamma, residual_multiplier=torch.full_like(gamma, self.residual_multiplier),
                               effective_residual_scale=gamma*self.residual_multiplier)
        return rgb_out, thermal_out, dict(routing_weights=weights, injection_strength=beta,
            routing_logits=torch.stack((cache['rgb_from_thermal']['s'], cache['thermal_from_rgb']['s']), 1),
            diagnostics=diagnostics)
