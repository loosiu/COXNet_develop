"""Paper-based two-branch DyFCLT with the official RT-DETR decoder."""

from pathlib import Path

import torch
from torch import nn

from .frequency import DFCA
from .sse import SSE
from .dual_encoder import DualRTDETREncoder
from .vendor.hybrid_encoder import (
    CSPRepLayer, ConvNormLayer, HybridEncoder, TransformerEncoder,
    TransformerEncoderLayer,
)
from .vendor.presnet import PResNet, donwload_url
from .vendor.rtdetr_decoder import RTDETRTransformer


class DyFCLTEncoder(nn.Module):
    def __init__(self, hidden_dim=256, num_bands=3, sse_reduction=4,
                 sse_kernel_size=3, pan=False, band_temperature=0.01):
        super().__init__()
        self.pan = pan
        channels = (512, 1024, 2048)
        self.rgb_proj = nn.ModuleList(self._projection(c, hidden_dim) for c in channels)
        self.thermal_proj = nn.ModuleList(self._projection(c, hidden_dim) for c in channels)
        layer = TransformerEncoderLayer(hidden_dim, nhead=8, dim_feedforward=1024,
                                        dropout=0.0, activation='gelu')
        self.rgb_top_encoder = TransformerEncoder(layer, 1)
        self.thermal_top_encoder = TransformerEncoder(
            TransformerEncoderLayer(hidden_dim, nhead=8, dim_feedforward=1024,
                                    dropout=0.0, activation='gelu'), 1)
        self.rgb_dfca = nn.ModuleList(DFCA(hidden_dim, num_bands=num_bands, temperature=band_temperature) for _ in range(2))
        self.thermal_dfca = nn.ModuleList(DFCA(hidden_dim, num_bands=num_bands, temperature=band_temperature) for _ in range(2))
        self.rgb_sse = nn.ModuleList(
            SSE(hidden_dim, reduction=sse_reduction, kernel_size=sse_kernel_size,
                fusion_block=CSPRepLayer(2 * hidden_dim, hidden_dim)) for _ in range(2))
        self.thermal_sse = nn.ModuleList(
            SSE(hidden_dim, reduction=sse_reduction, kernel_size=sse_kernel_size,
                fusion_block=CSPRepLayer(2 * hidden_dim, hidden_dim)) for _ in range(2))
        self.modal_fusion = nn.ModuleList(
            CSPRepLayer(2 * hidden_dim, hidden_dim) for _ in range(3))
        if pan:
            self.downsample_convs = nn.ModuleList(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act='silu') for _ in range(2))
            self.pan_blocks = nn.ModuleList(
                CSPRepLayer(2 * hidden_dim, hidden_dim) for _ in range(2))

    @staticmethod
    def _projection(in_channels, hidden_dim):
        return nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                             nn.BatchNorm2d(hidden_dim))

    @staticmethod
    def position_embedding(width, height, channels):
        # Upstream meshgrid(width, height) flattens x-major; NCHW tokens are
        # flattened row-major. Restore y-major order for each spatial token.
        embedding = HybridEncoder.build_2d_sincos_position_embedding(
            width, height, channels)
        return embedding.reshape(1, width, height, channels).transpose(1, 2).reshape(
            1, height * width, channels)

    @staticmethod
    def _attend_top(feat, encoder):
        batch, channels, height, width = feat.shape
        pos = DyFCLTEncoder.position_embedding(
            width, height, channels).to(device=feat.device, dtype=feat.dtype)
        tokens = feat.flatten(2).transpose(1, 2)
        return encoder(tokens, pos_embed=pos).transpose(1, 2).reshape(
            batch, channels, height, width)

    def forward(self, rgb_feats, thermal_feats):
        rgb = [proj(feat) for proj, feat in zip(self.rgb_proj, rgb_feats)]
        thermal = [proj(feat) for proj, feat in zip(self.thermal_proj, thermal_feats)]
        rgb[-1] = self._attend_top(rgb[-1], self.rgb_top_encoder)
        thermal[-1] = self._attend_top(thermal[-1], self.thermal_top_encoder)
        mask_logits = []
        for index in (1, 0):
            rgb_fine = self.rgb_dfca[index](thermal[index], rgb[index])
            thermal_fine = self.thermal_dfca[index](rgb[index], thermal[index])
            rgb[index], rgb_mask = self.rgb_sse[index](rgb_fine, rgb[index + 1])
            thermal[index], thermal_mask = self.thermal_sse[index](thermal_fine, thermal[index + 1])
            mask_logits.extend((rgb_mask, thermal_mask))
        fused = [block(torch.cat((r, t), dim=1))
                 for block, r, t in zip(self.modal_fusion, rgb, thermal)]
        if self.pan:
            outputs = [fused[0]]
            for index in range(2):
                down = self.downsample_convs[index](outputs[-1])
                outputs.append(self.pan_blocks[index](torch.cat((down, fused[index + 1]), dim=1)))
            fused = outputs
        return fused, mask_logits


class DyFCLT(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=300,
                 num_decoder_layers=6, num_denoising=100, pretrained=False,
                 pan=False, num_bands=3, sse_reduction=4, sse_kernel_size=3,
                 band_temperature=0.01, architecture="dyfclt",
                 use_dfca=None, use_sse=None, use_fref=True, dfca_hidden_dim=None,
                 dfca_gate_groups=1, dfca_correlation_normalization="none",
                 dfca_correlation_eps=1e-4, frequency_utility=None):
        super().__init__()
        if architecture not in ("dyfclt", "dual_rtdetr"):
            raise ValueError(f"Unknown architecture: {architecture}")
        if frequency_utility is not None:
            if not isinstance(frequency_utility, dict):
                raise ValueError('frequency_utility must be an option mapping')
            if architecture != 'dual_rtdetr' or use_dfca or use_sse:
                raise ValueError('frequency_utility requires the plain A dual-stream scaffold')
        if architecture == "dual_rtdetr" and pan:
            raise ValueError("dual_rtdetr already includes two PANs; postfusion pan is unsupported")
        custom_dfca = (isinstance(dfca_gate_groups, bool)
                       or not isinstance(dfca_gate_groups, int)
                       or dfca_gate_groups != 1
                       or dfca_correlation_normalization != "none"
                       or dfca_correlation_eps != 1e-4)
        if architecture == 'dyfclt' and custom_dfca:
            raise ValueError('DFCA gate/correlation options require architecture=dual_rtdetr')
        if architecture == 'dual_rtdetr' and not use_dfca and custom_dfca:
            raise ValueError('Nondefault DFCA gate/correlation options require use_dfca=True')
        if architecture == 'dyfclt' and (use_dfca is not None or use_sse is not None
                                        or not use_fref or dfca_hidden_dim is not None):
            raise ValueError('Module ablation switches and DFCA width require architecture=dual_rtdetr')
        self.architecture = architecture
        self.use_dfca = architecture == 'dyfclt' or bool(use_dfca)
        self.use_sse = architecture == 'dyfclt' or bool(use_sse)
        self.use_fref = self.use_sse and bool(use_fref)
        if hidden_dim % 8:
            raise ValueError('hidden_dim must be divisible by 8')
        self.rgb_backbone = PResNet(depth=50, variant='d', return_idx=[1, 2, 3],
                                    freeze_at=0, freeze_norm=True, pretrained=False)
        self.thermal_backbone = PResNet(depth=50, variant='d', return_idx=[1, 2, 3],
                                        freeze_at=0, freeze_norm=True, pretrained=False)
        if pretrained:
            if isinstance(pretrained, (str, Path)):
                path = Path(pretrained)
                if not path.is_absolute():
                    path = Path(__file__).resolve().parents[1] / path
                if not path.is_file():
                    raise FileNotFoundError(f'Backbone checkpoint not found: {path}')
                state = torch.load(path, map_location='cpu')
            elif pretrained is True:
                weight_dir = Path(__file__).resolve().parents[1] / 'weights'
                weight_dir.mkdir(exist_ok=True)
                state = torch.hub.load_state_dict_from_url(
                    donwload_url[50], model_dir=str(weight_dir), map_location='cpu')
            else:
                raise TypeError('pretrained must be False, True, or a checkpoint path')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            self.rgb_backbone.load_state_dict(state, strict=True)
            self.thermal_backbone.load_state_dict(state, strict=True)
        if architecture == 'dual_rtdetr':
            self.encoder = DualRTDETREncoder(
                hidden_dim, use_dfca=self.use_dfca, use_sse=self.use_sse,
                use_fref=use_fref, num_bands=num_bands, band_temperature=band_temperature,
                sse_reduction=sse_reduction, sse_kernel_size=sse_kernel_size,
                dfca_hidden_dim=dfca_hidden_dim, dfca_gate_groups=dfca_gate_groups,
                dfca_correlation_normalization=dfca_correlation_normalization,
                dfca_correlation_eps=dfca_correlation_eps, frequency_utility=frequency_utility)
        else:
            self.encoder = DyFCLTEncoder(hidden_dim, num_bands, sse_reduction,
                                         sse_kernel_size, pan, band_temperature)
        self.decoder = RTDETRTransformer(
            num_classes=num_classes, hidden_dim=hidden_dim, num_queries=num_queries,
            feat_channels=[hidden_dim] * 3, feat_strides=[8, 16, 32],
            num_decoder_layers=num_decoder_layers, num_denoising=num_denoising)

    @staticmethod
    def _validate_images(rgb, thermal):
        if rgb.ndim != 4 or thermal.ndim != 4 or rgb.shape[1] != 3 or thermal.shape[1] != 3:
            raise ValueError('rgb and thermal must be NCHW tensors with 3 channels')
        if rgb.shape[-2] % 32 or rgb.shape[-1] % 32:
            raise ValueError('image height and width must be divisible by 32')
        if rgb.shape != thermal.shape:
            raise ValueError('rgb and thermal tensors must have the same shape')

    def prepare_frequency(self, rgb, thermal, valid_masks=None):
        """Teacher cache before modal fusion: original streams and one candidate bank."""
        self._validate_images(rgb, thermal)
        if self.training:
            raise ValueError('Frequency replay cache requires an eval-mode teacher')
        if self.architecture != 'dual_rtdetr' or self.encoder.frequency_module is None:
            raise ValueError('Frequency replay requires frequency_utility')
        r, t = self.encoder.encode_frequency_streams(self.rgb_backbone(rgb), self.thermal_backbone(thermal))
        masks = self.encoder.resize_valid_masks(valid_masks, r[0])
        cache = self.encoder.frequency_module.build_cache(r[0], t[0], masks)
        return dict(rgb_features=r, thermal_features=t, frequency=cache)

    def replay_frequency(self, cache, gates=None, forced_indices=None, return_probe=False, return_diagnostics=False):
        """Recompute fusion/decoder only, retaining candidate/encoder/backbone cache."""
        if self.training:
            raise ValueError('Frequency replay requires an eval-mode teacher')
        rgb, thermal = list(cache['rgb_features']), list(cache['thermal_features'])
        rgb[0], thermal[0] = self.encoder.frequency_module.mix(cache['frequency'], gates)
        fused = self.encoder.fuse_streams(rgb, thermal)
        outputs = self.decoder(fused,
                               forced_indices=forced_indices, return_query_metadata=return_probe)
        outputs['mask_logits'] = []
        if return_diagnostics:
            outputs['diagnostic_features'] = dict(rgb_p3=rgb[0], thermal_p3=thermal[0], fused_p3=fused[0])
        return outputs

    def forward(self, rgb, thermal, targets=None, return_frequency_aux=False, valid_masks=None):
        self._validate_images(rgb, thermal)
        if self.architecture != 'dual_rtdetr' and (return_frequency_aux or valid_masks is not None):
            raise ValueError('Frequency options require architecture=dual_rtdetr')
        if self.architecture == 'dual_rtdetr':
            encoded = self.encoder(self.rgb_backbone(rgb), self.thermal_backbone(thermal),
                                   return_frequency_aux=return_frequency_aux, valid_masks=valid_masks)
            features, mask_logits = encoded[:2]
        else:
            features, mask_logits = self.encoder(self.rgb_backbone(rgb), self.thermal_backbone(thermal))
        tokens = sum(feat.shape[-2] * feat.shape[-1] for feat in features)
        if self.decoder.num_queries > tokens:
            raise ValueError(f'num_queries={self.decoder.num_queries} exceeds {tokens} feature tokens')
        if self.training and self.decoder.num_denoising > 0 and targets is None:
            raise ValueError('targets are required during training with denoising')
        outputs = self.decoder(features, targets)
        outputs['mask_logits'] = mask_logits
        if return_frequency_aux:
            outputs['frequency_aux'] = encoded[2]
        return outputs
