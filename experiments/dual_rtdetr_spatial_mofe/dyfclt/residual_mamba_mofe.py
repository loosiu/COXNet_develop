"""B-preserving Mamba input refinement; no expert/router or loss redesign.

Each independent directional bank has one context shared by its recipient and
donor inputs (their projection is also shared in B). Zero-initialized context
output projections preserve B exactly at initialization. No staged training:
output projections learn on the first backward, SS2D after they become nonzero.
"""
import torch
from torch import nn
from .residual_mofe import BidirectionalResidualMoFE
from .s3am_mofe import S3AMFrequencyMixture
from .ss2d_context import FrequencyContext


class _MambaFrequencyBank(S3AMFrequencyMixture):
    def __init__(self, base, state_dim, eta):
        # Adopt initialized B layers, avoiding reinitialization or RNG changes.
        nn.Module.__init__(self)
        self.channels, self.dim = base.channels, base.dim
        self.project, self.experts, self.router = base.project, base.experts, base.router
        self.context = FrequencyContext(self.dim, state_dim=state_dim, eta=eta)
        nn.init.zeros_(self.context.modulation.weight)
        nn.init.zeros_(self.context.modulation.bias)
        for local in self.context.locals:
            nn.init.zeros_(local[-1].weight)
            nn.init.zeros_(local[-1].bias)

    def prepare_bands(self, rgb, thermal, rm, tm):
        originals = super().prepare_bands(rgb, thermal, rm, tm)
        result = []
        for bands, mask in zip(originals, (rm, tm)):
            refined, _ = self.context(bands)
            # Preserve original Haar boundary values outside valid input.
            result.append(tuple(band + mask * (value-band)
                                for band, value in zip(bands, refined)))
        return tuple(result)


class ResidualMambaMoFE(BidirectionalResidualMoFE):
    gate_mode = 'residual_mamba_mofe'

    def __init__(self, channels=256, dim=64, expert_hidden=64, router_hidden=64,
                 residual_scale=.1, state_dim=8, eta=.05):
        super().__init__(channels, dim, expert_hidden, router_hidden, residual_scale)
        with torch.random.fork_rng(devices=[]):
            self.rgb_from_thermal = _MambaFrequencyBank(self.rgb_from_thermal, state_dim, eta)
            self.thermal_from_rgb = _MambaFrequencyBank(self.thermal_from_rgb, state_dim, eta)

    def forward(self, *args, **kwargs):
        rgb, thermal, aux = super().forward(*args, **kwargs)
        aux['diagnostics']['context_eta'] = torch.stack((
            self.rgb_from_thermal.context.eta.detach(),
            self.thermal_from_rgb.context.eta.detach())).unsqueeze(0).expand(rgb.shape[0], -1, -1)
        return rgb, thermal, aux
