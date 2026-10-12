"""Selective Smoothing Enhancement from DyFCLT, Eqs. 6–15.

The paper omits the channel-reduction ratio, dynamic-kernel width, and the
binary mask estimator. Defaults here are documented reproduction assumptions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SSE(nn.Module):
    """Top-down foreground-guided refinement of a 2× finer feature map."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        kernel_size: int = 3,
        fusion_block: nn.Module | None = None,
        use_fref: bool = True,
    ):
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")
        compressed = max(1, channels // reduction)
        self.channels = channels
        self.kernel_size = kernel_size
        self.use_fref = bool(use_fref)
        self.mask_predictor = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.background_compress = nn.Conv2d(channels, compressed, kernel_size=3, padding=1, bias=False)
        self.background_recover = nn.Conv2d(compressed, channels, kernel_size=3, padding=1, bias=False)
        self.kernel_predictor = (nn.Conv2d(channels, kernel_size * kernel_size, kernel_size=3, padding=1)
                                 if self.use_fref else None)
        if fusion_block is None:
            from .vendor.hybrid_encoder import CSPRepLayer

            fusion_block = CSPRepLayer(2 * channels, channels, num_blocks=3)
        self.fusion_block = fusion_block

    def forward(self, fine: Tensor, coarse: Tensor) -> tuple[Tensor, Tensor]:
        if fine.ndim != 4 or coarse.ndim != 4:
            raise ValueError("fine and coarse features must be NCHW tensors")
        if fine.shape[:2] != coarse.shape[:2] or fine.shape[1] != self.channels:
            raise ValueError("fine and coarse batch/channel dimensions must match SSE")
        if fine.shape[-2] != 2 * coarse.shape[-2] or fine.shape[-1] != 2 * coarse.shape[-1]:
            raise ValueError("fine spatial dimensions must be exactly twice coarse dimensions")

        mask_logits = self.mask_predictor(fine)
        mask_prob = torch.sigmoid(mask_logits)
        hard_mask = (mask_prob >= 0.5).to(mask_prob.dtype)
        mask = mask_prob + (hard_mask - mask_prob).detach()
        foreground = mask * fine
        background = (1 - mask) * fine
        smoothed_background = self.background_recover(self.background_compress(background))
        background_suppressed = foreground + smoothed_background

        if not self.use_fref:
            # Table4 IBS-only interpretation: retain ordinary baseline FPN
            # nearest upsampling when guided/bilinear FREF is removed.
            upsampled = F.interpolate(coarse, size=fine.shape[-2:], mode="nearest")
            return self.fusion_block(torch.cat((upsampled, background_suppressed), dim=1)), mask_logits

        kernel_logits = self.kernel_predictor(background_suppressed)
        kernels = kernel_logits.softmax(dim=1)
        # PixelUnshuffle returns channels ordered [kernel, phase]. Reorder to
        # [phase, kernel] before each of the four coarse-grid convolutions.
        coarse_height, coarse_width = coarse.shape[-2:]
        batch = coarse.shape[0]
        kernel_elements = self.kernel_size * self.kernel_size
        phase_kernels = F.pixel_unshuffle(kernels, 2)
        phase_kernels = phase_kernels.reshape(batch, kernel_elements, 4, coarse_height, coarse_width)
        phase_kernels = phase_kernels.permute(0, 2, 1, 3, 4)
        patches = F.unfold(coarse, self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.reshape(batch, self.channels, kernel_elements, coarse_height, coarse_width)
        per_phase = (phase_kernels[:, :, None] * patches[:, None]).sum(dim=3)
        guided = F.pixel_shuffle(per_phase.permute(0, 2, 1, 3, 4).reshape(
            batch, 4 * self.channels, coarse_height, coarse_width), 2)
        bilinear = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fusion_block(torch.cat((guided + bilinear, background_suppressed), dim=1))
        return fused, mask_logits
