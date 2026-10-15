"""Small four-direction selective-scan context, no spatial downsampling.

SS2D equations/direction ordering follow VMamba v0; use the installed official
Mamba scan kernel on CUDA and its differentiable reference on CPU. This adapter
uses inner width=width and averages the four restored directions.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .frequency_utility import ChannelLN, positive_integer


def scan_sequences(x):
    forward = torch.stack((x.flatten(2), x.transpose(2, 3).flatten(2)), dim=1)
    return torch.cat((forward, forward.flip(-1)), dim=1)


def merge_sequences(y, height, width):
    reverse = y[:, 2:].flip(-1)
    row = y[:, 0] + reverse[:, 0]
    column = (y[:, 1] + reverse[:, 1]).reshape(y.shape[0], -1, width, height)
    return (row.reshape(y.shape[0], -1, height, width) + column.transpose(2, 3)) / 4


class SS2DContext(nn.Module):
    def __init__(self, dim, state_dim=8):
        super().__init__()
        positive_integer(dim, 'dim'); positive_integer(state_dim, 'state_dim')
        self.dim, self.state_dim, self.rank = dim, state_dim, math.ceil(dim/16)
        self.in_proj = nn.Conv2d(dim, 2*dim, 1, bias=False)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.x_weight = nn.Parameter(torch.stack([
            nn.Linear(dim, self.rank+2*state_dim, bias=False).weight.detach() for _ in range(4)]))
        self.dt_weight = nn.Parameter(torch.empty(4, dim, self.rank))
        nn.init.uniform_(self.dt_weight, -self.rank**-.5, self.rank**-.5)
        dt = torch.exp(torch.rand(4, dim)*(math.log(.1)-math.log(.001))+math.log(.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, state_dim+1).float().log().repeat(4*dim, 1))
        self.D = nn.Parameter(torch.ones(4*dim))
        self.norm = ChannelLN(dim)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x, reference=False):
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
        u, z = self.in_proj(x).chunk(2, dim=1)
        u = F.silu(self.local(u))
        seq = scan_sequences(u).float()
        packed = torch.einsum('bkdl,knd->bknl', seq, self.x_weight.float())
        dt, b, c = torch.split(packed, [self.rank, self.state_dim, self.state_dim], dim=2)
        dt = torch.einsum('bkrl,kdr->bkdl', dt, self.dt_weight.float())
        fn = selective_scan_ref if not x.is_cuda or reference else selective_scan_fn
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = fn(seq.flatten(1, 2).contiguous(), dt.flatten(1, 2).contiguous(),
                   -self.A_log.float().exp(), b.contiguous(), c.contiguous(), self.D.float(),
                   delta_bias=self.dt_bias.float().flatten(), delta_softplus=True)
        y = merge_sequences(y.reshape(x.shape[0], 4, self.dim, -1), *x.shape[-2:])
        return self.out_proj(self.norm(y.to(x.dtype))*F.silu(z))


class FrequencyContext(nn.Module):
    def __init__(self, dim, state_dim=8, eta=.05):
        super().__init__()
        if not math.isfinite(eta): raise ValueError('eta must be finite')
        self.band_norms = nn.ModuleList(nn.Sequential(ChannelLN(dim), nn.Conv2d(dim, dim, 1))
                                        for _ in range(4))
        self.reduce = nn.Conv2d(4*dim+4, dim, 1)
        self.scan = SS2DContext(dim, state_dim)
        self.modulation = nn.Conv2d(dim, 4*dim, 1)
        nn.init.normal_(self.modulation.weight, std=.01); nn.init.zeros_(self.modulation.bias)
        self.locals = nn.ModuleList(nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                                                 nn.GELU(), nn.Conv2d(dim, dim, 1)) for _ in range(4))
        self.eta = nn.Parameter(torch.full((4,), float(eta)))

    def forward(self, bands):
        energy = [torch.log1p((band.square().mean(1, keepdim=True)+1e-12).sqrt()) for band in bands]
        context = self.scan(self.reduce(torch.cat([norm(band) for norm, band in zip(self.band_norms, bands)]+energy, 1)))
        modulation = self.modulation(context).chunk(4, 1)
        corrected = tuple((1+self.eta[k]*torch.tanh(modulation[k]))*band
                          + self.eta[k]*self.locals[k](band) for k, band in enumerate(bands))
        return corrected, context
