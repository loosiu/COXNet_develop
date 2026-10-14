"""Dynamic radial-frequency cross-modal attention from DyFCLT, Eqs. 1–5.

The paper specifies binary masks with trainable boundaries but does not specify
how gradients pass through thresholding. We use hard-forward sigmoid surrogate
masks; their temperature is an explicit reproduction assumption.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn


def frequency_correlation(query_spectrum: Tensor, key_spectrum: Tensor) -> Tensor:
    """Circular cross-correlation: IFFT(Q * conjugate(K)), as in Eq. 3."""
    return torch.fft.ifft2(query_spectrum * key_spectrum.conj(), dim=(-2, -1)).real


class FrequencyBandDecomposition(nn.Module):
    """Shared hard radial bands with differentiable boundary surrogates."""

    def __init__(self, num_bands: int = 3, temperature: float = 0.01):
        super().__init__()
        if num_bands < 1:
            raise ValueError("num_bands must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.num_bands = num_bands
        self.temperature = float(temperature)
        # Octave initial partition: for B=3, [0, 1/8, 1/4, 1/2].
        boundaries = [0.0] + [2.0 ** (index - num_bands) for index in range(num_bands)]
        increments = [boundaries[index + 1] - boundaries[index] for index in range(num_bands)]
        self.band_logits = nn.Parameter(torch.tensor(increments).log())

    def boundaries(self) -> Tensor:
        increments = self.band_logits.softmax(dim=0) * 0.5
        inner = increments.cumsum(dim=0)[:-1]
        # Construct fixed endpoints explicitly: a rounded cumulative sum can
        # otherwise exceed .5 and incorrectly include Nyquist-edge FFT bins.
        return torch.cat((increments.new_zeros(1), inner, increments.new_full((1,), .5)))

    def masks(self, height: int, width: int) -> Tensor:
        """Return B×H×W masks; the radial Nyquist edge and corners are excluded."""
        if height < 1 or width < 1:
            raise ValueError("FFT dimensions must be positive")
        device = self.band_logits.device
        vertical = torch.fft.fftfreq(height, device=device).float()
        horizontal = torch.fft.fftfreq(width, device=device).float()
        radius = torch.sqrt(vertical[:, None].square() + horizontal[None, :].square())
        edges = self.boundaries()
        hard = ((radius[None] >= edges[:-1, None, None]) &
                (radius[None] < edges[1:, None, None])).to(radius.dtype)
        # The hard forward path follows Eq. 2 exactly. The sigmoid path supplies
        # a gradient for the otherwise non-differentiable learned boundaries.
        lower = torch.sigmoid((radius[None] - edges[:-1, None, None]) / self.temperature)
        upper = torch.sigmoid((edges[1:, None, None] - radius[None]) / self.temperature)
        soft = lower * upper
        return soft + (hard - soft).detach()


class DFCA(nn.Module):
    """One directional branch with optional grouped gate and RMS stabilization.

    Defaults retain the original Eq. 3-4 correlation and dense spatial gate.
    RMS is an experimental extension, not an operation specified by the paper.
    """

    def __init__(self, channels: int, num_bands: int = 3, temperature: float = 0.01,
                 inner_channels: int | None = None, gate_groups: int = 1,
                 correlation_normalization: str = "none", correlation_eps: float = 1e-4):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if inner_channels is not None and (
            isinstance(inner_channels, bool) or not isinstance(inner_channels, int)
            or inner_channels < 1
        ):
            raise ValueError("inner_channels must be a positive integer or None")
        self.channels = channels
        self.inner_channels = channels if inner_channels is None else inner_channels
        if (isinstance(gate_groups, bool) or not isinstance(gate_groups, int)
                or gate_groups < 1 or self.inner_channels % gate_groups):
            raise ValueError("gate_groups must be a positive integer dividing inner_channels")
        if correlation_normalization not in ("none", "rms"):
            raise ValueError("correlation_normalization must be 'none' or 'rms'")
        if (isinstance(correlation_eps, bool) or not isinstance(correlation_eps, Real)
                or not math.isfinite(correlation_eps)
                or not torch.finfo(torch.float32).tiny <= correlation_eps <= torch.finfo(torch.float32).max):
            raise ValueError("correlation_eps must be positive, finite and representable as normal float32")
        self.gate_groups = gate_groups
        self.correlation_normalization = correlation_normalization
        self.correlation_eps = float(correlation_eps)
        self.bands = FrequencyBandDecomposition(num_bands, temperature)
        self.query = self._projection(channels, self.inner_channels)
        self.key = self._projection(channels, self.inner_channels)
        self.value = self._projection(channels, self.inner_channels)
        self.spatial_modulator = nn.Conv2d(self.inner_channels, self.inner_channels,
                                          kernel_size=3, padding=1, groups=gate_groups)
        self.norm = nn.LayerNorm(self.inner_channels)
        self.projection = nn.Conv2d(self.inner_channels, channels, kernel_size=1)

    @staticmethod
    def _projection(channels: int, out_channels: int | None = None) -> nn.Sequential:
        out_channels = channels if out_channels is None else out_channels
        return nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      groups=out_channels),
        )

    def _normalize_correlation(self, correlation: Tensor) -> Tensor:
        if self.correlation_normalization == "none":
            return correlation
        # Per-band/image/channel spatial RMS, without centering. Rescaling
        # prevents c.square() overflowing for large finite float32 FFT results.
        # Detached scale cancels algebraically; the gradient still equals that
        # of c / sqrt(mean_HW(c**2) + eps**2), including at all-zero bands.
        scale = correlation.detach().abs().amax(dim=(-2, -1), keepdim=True)
        scale = scale.clamp_min(self.correlation_eps)
        unit = correlation / scale
        denominator = (unit.square().mean(dim=(-2, -1), keepdim=True)
                       + (self.correlation_eps / scale).square()).sqrt()
        return unit / denominator

    def forward(self, query_source: Tensor, target: Tensor) -> Tensor:
        if query_source.shape != target.shape or query_source.ndim != 4:
            raise ValueError("query_source and target must have equal NCHW shapes")
        if target.shape[1] != self.channels:
            raise ValueError("input channel count differs from DFCA channels")
        query = self.query(query_source)
        key = self.key(target)
        value = self.value(target)
        # FFT and complex multiplication must remain float32 under AMP. This
        # also supports odd, non-square maps without an rFFT symmetry shortcut.
        with torch.autocast(device_type=target.device.type, enabled=False):
            q_fft = torch.fft.fft2(query.float(), dim=(-2, -1))
            k_fft = torch.fft.fft2(key.float(), dim=(-2, -1))
            v_fft = torch.fft.fft2(value.float(), dim=(-2, -1))
            masks = self.bands.masks(*target.shape[-2:])
            response = torch.zeros_like(query.float())
            for band in masks:
                gate = band[None, None]
                correlation = frequency_correlation(q_fft * gate, k_fft * gate)
                correlation = self._normalize_correlation(correlation)
                attention = torch.sigmoid(self.spatial_modulator(correlation))
                band_value = torch.fft.ifft2(v_fft * gate, dim=(-2, -1)).real
                response = response + attention * band_value
            response = self.norm(response.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            response = self.projection(response)
        return response.to(dtype=target.dtype)
