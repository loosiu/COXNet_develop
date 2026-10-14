"""Signed undecimated Haar cross-modal candidates and utility evidence gates."""
import math

import torch
from torch import nn
from torch.nn import functional as F


BANDS = ('LL', 'LH', 'HL', 'HH')


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('{} must be a positive integer'.format(name))


def validate_features(features, channels):
    first = features[0]
    if first.ndim != 4 or first.shape[1] != channels or min(first.shape) <= 0:
        raise ValueError('Expected nonempty NCHW features with configured channels')
    if not first.is_floating_point() or any(
            x.shape != first.shape or x.dtype != first.dtype or x.device != first.device
            for x in features):
        raise ValueError('Features must share shape, floating dtype and device')


def valid_mask(mask, feature):
    if mask is None:
        return torch.ones(feature.shape[0], 1, *feature.shape[-2:],
                          device=feature.device, dtype=torch.bool)
    if mask.dtype != torch.bool or mask.device != feature.device:
        raise ValueError('Valid masks must be boolean and on the feature device')
    if tuple(mask.shape) != (feature.shape[0], 1, *feature.shape[-2:]):
        raise ValueError('Valid mask shape must be [B,1,H,W]')
    return mask


def haar_analysis(x):
    """One-level stride-1 analysis, with signed bands and replicate right/bottom."""
    if x.ndim != 4 or not x.is_floating_point() or min(x.shape) <= 0:
        raise ValueError('Haar input must be a nonempty floating NCHW tensor')
    h, w = x.shape[-2:]
    padded = F.pad(x, (0, 1, 0, 1), mode='replicate')
    a, b = padded[:, :, :h, :w], padded[:, :, :h, 1:w+1]
    c, d = padded[:, :, 1:h+1, :w], padded[:, :, 1:h+1, 1:w+1]
    return ((a+b+c+d)/2, (a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2)


class ChannelLN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class LocalFrequencyAttention(nn.Module):
    """Chunked row-major neighbours; FP32 scores and softmax, masked boundaries."""
    def __init__(self, dim=64, heads=4, window_size=7, chunk_size=256):
        super().__init__()
        for key, value in [('dim', dim), ('heads', heads), ('window_size', window_size), ('chunk_size', chunk_size)]:
            positive_integer(value, key)
        if dim % heads or window_size % 2 != 1:
            raise ValueError('dim must divide heads and window_size must be odd')
        self.dim, self.heads, self.head_dim = dim, heads, dim//heads
        self.window_size, self.chunk_size = window_size, chunk_size
        self.detach_context = True
        self.relative_bias = nn.Parameter(torch.zeros(heads, window_size**2))

    def forward(self, q, k, v, donor_mask=None):
        validate_features((q, k, v), self.dim)
        mask = valid_mask(donor_mask, v)
        batch, _, height, width = q.shape
        n = height * width
        original_dtype = v.dtype
        q, k, v = [x.reshape(batch, self.heads, self.head_dim, n).permute(0, 1, 3, 2)
                   for x in (q, k, v)]
        radius = self.window_size//2
        offsets = torch.arange(-radius, radius+1, device=q.device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')
        dy, dx = dy.reshape(1, -1), dx.reshape(1, -1)
        values, contexts = [], []
        with torch.autocast(device_type=q.device.type, enabled=False):
            for start in range(0, n, self.chunk_size):
                stop = min(n, start+self.chunk_size)
                idx = torch.arange(start, stop, device=q.device)
                yy = torch.div(idx, width, rounding_mode='floor')[:, None] + dy
                xx = idx.remainder(width)[:, None] + dx
                inside = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
                neighbours = yy.clamp(0, height-1)*width + xx.clamp(0, width-1)
                allowed = inside[None] & mask.flatten(2)[:, 0, neighbours]
                local_k, local_v = k[:, :, neighbours].float(), v[:, :, neighbours].float()
                scores = (q[:, :, start:stop, None].float()*local_k).sum(-1)/math.sqrt(self.head_dim)
                scores = scores + self.relative_bias.float()[None, :, None]
                scores = scores.masked_fill(~allowed[:, None], float('-inf'))
                # Avoid NaN from an all -inf softmax, then explicitly zero all invalid probabilities.
                scores = torch.where(allowed.any(-1)[:, None, :, None], scores, torch.zeros_like(scores))
                probability = scores.softmax(-1) * allowed[:, None]
                values.append((probability[..., None]*local_v).sum(-2))
                if self.detach_context:
                    with torch.no_grad():
                        contexts.append((probability.detach()[..., None]*local_k.detach()).sum(-2))
                else:
                    contexts.append((probability[..., None]*local_k).sum(-2))
        def restore(chunks):
            return torch.cat(chunks, 2).permute(0, 1, 3, 2).reshape(batch, self.dim, height, width).to(original_dtype)
        return restore(values), restore(contexts)


class EvidenceEvaluator(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3*dim, hidden, 1), ChannelLN(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, 2, 1))
        nn.init.normal_(self.head[-1].weight, std=.01)
        nn.init.constant_(self.head[-1].bias, .5413248546)

    def forward(self, q, context, candidate):
        logits = self.head(torch.cat((q, context, candidate), 1).detach())
        with torch.autocast(device_type=logits.device.type, enabled=False):
            evidence = F.softplus(logits.float())
            a, b = 1+evidence[:, 0], 1+evidence[:, 1]
            total = a+b
            return a, b, a/total, 2/total, evidence[:, 0]/total


class FrequencyExpert(nn.Module):
    def __init__(self, dim, heads, window_size, chunk_size, evaluator_hidden):
        super().__init__()
        self.donor_norm, self.band_norm = ChannelLN(dim), ChannelLN(dim)
        self.key = nn.Conv2d(2*dim, dim, 1, bias=False)
        self.value = nn.Conv2d(dim, dim, 1, bias=False)
        self.output = nn.Conv2d(dim, dim, 1, bias=False)
        self.attention = LocalFrequencyAttention(dim, heads, window_size, chunk_size)
        self.evaluator = EvidenceEvaluator(dim, evaluator_hidden)
        for conv in [self.key, self.value, self.output]:
            nn.init.xavier_uniform_(conv.weight)

    def forward(self, q, donor, band, donor_mask):
        k = self.key(torch.cat((self.donor_norm(donor), self.band_norm(band)), 1))
        retrieved, context = self.attention(q, k, self.value(band), donor_mask)
        return self.output(retrieved), context


class FrequencyDirection(nn.Module):
    def __init__(self, dim, heads, window_size, chunk_size, evaluator_hidden):
        super().__init__()
        self.query_norm, self.query = ChannelLN(dim), nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.xavier_uniform_(self.query.weight)
        self.experts = nn.ModuleList(FrequencyExpert(dim, heads, window_size, chunk_size, evaluator_hidden)
                                     for _ in BANDS)

    def forward(self, receiver, donor, donor_bands, donor_mask):
        q = self.query(self.query_norm(receiver))
        candidates, statistics = [], []
        for expert, band in zip(self.experts, donor_bands):
            candidate, context = expert(q, donor, band, donor_mask)
            candidates.append(candidate)
            statistics.append(expert.evaluator(q, context, candidate))
        return torch.stack(candidates, 1), [torch.stack([s[i] for s in statistics], 1) for i in range(5)]


class FrequencyUtilityEnhancer(nn.Module):
    """Two original streams -> independent dense frequency gates -> two residuals."""
    ema_constant_buffers = ('residual_scale',)

    def __init__(self, channels=256, dim=64, heads=4, window_size=7,
                 residual_scale=.1, evaluator_hidden=64, chunk_size=256):
        super().__init__()
        for key, value in [('channels', channels), ('dim', dim), ('heads', heads),
                           ('evaluator_hidden', evaluator_hidden), ('chunk_size', chunk_size)]:
            positive_integer(value, key)
        if isinstance(residual_scale, bool) or not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError('residual_scale must be finite and nonnegative')
        self.channels, self.dim = channels, dim
        self.register_buffer('residual_scale', torch.tensor(float(residual_scale)))
        self.project_rgb, self.project_thermal = [nn.Conv2d(channels, dim, 1, bias=False) for _ in range(2)]
        self.output_rgb, self.output_thermal = [nn.Conv2d(dim, channels, 1, bias=False) for _ in range(2)]
        for conv in [self.project_rgb, self.project_thermal, self.output_rgb, self.output_thermal]:
            nn.init.xavier_uniform_(conv.weight)
        self.rgb_direction = FrequencyDirection(dim, heads, window_size, chunk_size, evaluator_hidden)
        self.thermal_direction = FrequencyDirection(dim, heads, window_size, chunk_size, evaluator_hidden)

    def build_cache(self, rgb, thermal, valid_masks=None):
        validate_features((rgb, thermal), self.channels)
        if valid_masks is None:
            valid_masks = (None, None)
        if len(valid_masks) != 2:
            raise ValueError('Expected RGB and thermal valid masks')
        masks = (valid_mask(valid_masks[0], rgb), valid_mask(valid_masks[1], thermal))
        zr, zt = self.project_rgb(rgb)*masks[0], self.project_thermal(thermal)*masks[1]
        r, rs = self.rgb_direction(zr, zt, haar_analysis(zt), masks[1])
        t, ts = self.thermal_direction(zt, zr, haar_analysis(zr), masks[0])
        cache = dict(rgb=rgb, thermal=thermal, candidates=torch.stack((r, t), 1),
                     valid_masks=torch.stack([m[:, 0] for m in masks], 1))
        for i, key in enumerate(('a', 'b', 'p', 'u', 'w')):
            cache[key] = torch.stack((rs[i], ts[i]), 1)
        return cache

    def mix(self, cache, gates=None):
        weights = cache['w'] if gates is None else gates
        if weights.shape != cache['w'].shape or weights.device != cache['w'].device:
            raise ValueError('Gate override must match [B,2,4,H,W] and device')
        if not torch.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
            raise ValueError('Gate values must be finite within [0,1]')
        candidates = cache['candidates']
        message = (weights.detach().to(candidates.dtype).unsqueeze(3)*candidates).mean(2)
        message = message * cache['valid_masks'].unsqueeze(2)
        return (cache['rgb'] + self.residual_scale*self.output_rgb(message[:, 0]),
                cache['thermal'] + self.residual_scale*self.output_thermal(message[:, 1]))

    def forward(self, rgb, thermal, valid_masks=None, gates=None):
        cache = self.build_cache(rgb, thermal, valid_masks)
        rgb_out, thermal_out = self.mix(cache, gates)
        aux = {k: cache[k] for k in ('a', 'b', 'p', 'u', 'w', 'valid_masks')}
        return rgb_out, thermal_out, aux
