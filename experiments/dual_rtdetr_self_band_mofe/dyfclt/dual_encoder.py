"""Controlled dual RT-DETR scaffold for supplement 6.1 and module ablations.

The supplement defines the complete baseline, but does not resolve whether the
final proposal retains lateral/PAN stages. We preserve them in every variant,
using the same FPN CSP weights and positional encoding. This is an explicit
reconstruction choice, not a claim of recovered author topology.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .frequency import DFCA
from .sse import SSE
from .vendor.hybrid_encoder import HybridEncoder, CSPRepLayer
from .frequency_utility import FrequencyUtilityEnhancer
from .detection_gate import DetectionFrequencyEnhancer
from .s3am_mofe import S3AMFrequencyMixture
from .residual_mofe import BidirectionalResidualMoFE
from .staged_mofe import DonorResidualMoFE, DonorStrengthResidualMoFE
from .self_band_mofe import SelfBandResidualMoFE
from .residual_mamba_mofe import ResidualMambaMoFE
from .spatial_mofe import SpatialResidualMoFE
from .relation_mofe import BidirectionalRelationMoFE


class DualRTDETREncoder(nn.Module):
    def __init__(self, hidden_dim=256, use_dfca=False, use_sse=False,
                 use_fref=True, num_bands=3, band_temperature=.01,
                 sse_reduction=4, sse_kernel_size=3, dfca_hidden_dim=None,
                 dfca_gate_groups=1, dfca_correlation_normalization="none",
                 dfca_correlation_eps=1e-4, frequency_utility=None):
        super().__init__()
        custom_dfca = (isinstance(dfca_gate_groups, bool)
                       or not isinstance(dfca_gate_groups, int)
                       or dfca_gate_groups != 1
                       or dfca_correlation_normalization != "none"
                       or dfca_correlation_eps != 1e-4)
        if not use_dfca and custom_dfca:
            raise ValueError("Nondefault DFCA gate/correlation options require use_dfca=True")
        self.use_dfca = bool(use_dfca)
        self.use_sse = bool(use_sse)
        self.use_fref = bool(use_sse and use_fref)
        if frequency_utility is not None and (use_dfca or use_sse):
            raise ValueError('frequency_utility requires plain dual-stream RT-DETR')
        self.rgb_encoder = HybridEncoder(hidden_dim=hidden_dim, eval_spatial_size=None)
        self.thermal_encoder = HybridEncoder(hidden_dim=hidden_dim, eval_spatial_size=None)
        self.modal_fusion = nn.ModuleList(
            CSPRepLayer(2 * hidden_dim, hidden_dim) for _ in range(3))
        # Additions must not shift initialization of the shared decoder or each
        # other. All layers are constructed on CPU; preserve its RNG separately.
        with torch.random.fork_rng(devices=[]):
            self.rgb_dfca = nn.ModuleList(
                DFCA(hidden_dim, num_bands, band_temperature, inner_channels=dfca_hidden_dim,
                     gate_groups=dfca_gate_groups,
                     correlation_normalization=dfca_correlation_normalization,
                     correlation_eps=dfca_correlation_eps)
                for _ in range(2 if use_dfca else 0))
            self.thermal_dfca = nn.ModuleList(
                DFCA(hidden_dim, num_bands, band_temperature, inner_channels=dfca_hidden_dim,
                     gate_groups=dfca_gate_groups,
                     correlation_normalization=dfca_correlation_normalization,
                     correlation_eps=dfca_correlation_eps)
                for _ in range(2 if use_dfca else 0))
        def build_sse():
            # Consume the same initialization draws even in IBS-only ablations,
            # then remove the unused kernel. Later IBS modules keep identical
            # initial weights; no discarded parameter remains registered.
            module = SSE(hidden_dim, sse_reduction, sse_kernel_size,
                         fusion_block=nn.Identity(), use_fref=True)
            if not use_fref:
                module.use_fref = False
                module.kernel_predictor = None
            return module

        with torch.random.fork_rng(devices=[]):
            self.rgb_sse = nn.ModuleList(build_sse() for _ in range(2 if use_sse else 0))
            self.thermal_sse = nn.ModuleList(build_sse() for _ in range(2 if use_sse else 0))
            self.frequency_module = None
            self.frequency_insertion_point = 'post_encoder'
            if frequency_utility is not None:
                frequency_options = dict(frequency_utility)
                self.frequency_insertion_point = frequency_options.pop('insertion_point', 'post_encoder')
                if self.frequency_insertion_point not in ('post_encoder', 'projected_p3'):
                    raise ValueError('Unknown frequency insertion point')
                mode = frequency_options.pop('gate_mode', 'beta_utility')
                if self.frequency_insertion_point == 'projected_p3' and mode not in ('residual_mofe', 'residual_mamba_mofe', 'spatial_mofe', 'relation_mofe'):
                    raise ValueError('Projected P3 experiments require the existing B/E banks')
                enhancers = {'beta_utility': FrequencyUtilityEnhancer,
                             'sigmoid_detection': DetectionFrequencyEnhancer,
                             's3am_mofe': S3AMFrequencyMixture,
                             'residual_mofe': BidirectionalResidualMoFE,
                             'residual_mamba_mofe': ResidualMambaMoFE,
                             'spatial_mofe': SpatialResidualMoFE,
                             'donor_mofe': DonorResidualMoFE,
                             'donor_strength_mofe': DonorStrengthResidualMoFE,
                             'self_band_mofe': SelfBandResidualMoFE,
                             'relation_mofe': BidirectionalRelationMoFE}
                if mode not in enhancers:
                    raise ValueError('Unknown frequency gate_mode: '+str(mode))
                enhancer = enhancers[mode]
                self.frequency_module = enhancer(channels=hidden_dim, **frequency_options)

    def encode_frequency_streams(self, rgb_feats, thermal_feats):
        if self.frequency_module is None:
            raise ValueError('No frequency module is configured')
        if self.frequency_insertion_point == 'projected_p3':
            return (self.rgb_encoder.project_features(rgb_feats),
                    self.thermal_encoder.project_features(thermal_feats))
        return list(self.rgb_encoder(rgb_feats)), list(self.thermal_encoder(thermal_feats))

    def fuse_frequency_streams(self, rgb, thermal):
        """Complete the pyramid after pre-pyramid enhancement, then RGB/T fusion."""
        if self.frequency_insertion_point == 'projected_p3':
            rgb = self.rgb_encoder.forward_projected(rgb)
            thermal = self.thermal_encoder.forward_projected(thermal)
        return self.fuse_streams(rgb, thermal)

    @staticmethod
    def resize_valid_masks(masks, feature):
        if masks is None:
            return None
        if len(masks) != 2:
            raise ValueError('Expected two valid masks')
        result = []
        for mask in masks:
            if mask.ndim != 4 or mask.shape[:2] != (feature.shape[0], 1) or mask.dtype != torch.bool or mask.device != feature.device:
                raise ValueError('Valid mask must be boolean [B,1,H,W] on the feature device')
            result.append(F.interpolate(mask.float(), size=feature.shape[-2:], mode='nearest').bool())
        return result

    def fuse_streams(self, rgb, thermal):
        return [block(torch.cat((r, t), dim=1))
                for block, r, t in zip(self.modal_fusion, rgb, thermal)]

    @staticmethod
    def _project_and_attend(encoder, feats):
        projected = [layer(feat) for layer, feat in zip(encoder.input_proj, feats)]
        if encoder.num_encoder_layers > 0:
            for index, level in enumerate(encoder.use_encoder_idx):
                height, width = projected[level].shape[-2:]
                tokens = projected[level].flatten(2).permute(0, 2, 1)
                # Preserve official positional ordering for every ablation.
                position = encoder.build_2d_sincos_position_embedding(
                    width, height, encoder.hidden_dim, encoder.pe_temperature).to(tokens.device)
                memory = encoder.encoder[index](tokens, pos_embed=position)
                projected[level] = memory.permute(0, 2, 1).reshape(
                    -1, encoder.hidden_dim, height, width).contiguous()
        return projected

    @staticmethod
    def _pan(encoder, inner):
        outputs = [inner[0]]
        for stage in range(len(inner) - 1):
            down = encoder.downsample_convs[stage](outputs[-1])
            outputs.append(encoder.pan_blocks[stage](torch.cat((down, inner[stage + 1]), dim=1)))
        return outputs

    def forward(self, rgb_feats, thermal_feats, return_frequency_aux=False, valid_masks=None):
        frequency_aux = None
        if (return_frequency_aux or valid_masks is not None) and self.frequency_module is None:
            raise ValueError('Frequency diagnostics/masks require frequency_utility')
        if self.frequency_module is not None and self.frequency_insertion_point == 'projected_p3':
            rgb, thermal = self.encode_frequency_streams(rgb_feats, thermal_feats)
            rgb[0], thermal[0], frequency_aux = self.frequency_module(
                rgb[0], thermal[0], self.resize_valid_masks(valid_masks, rgb[0]))
            fused = self.fuse_frequency_streams(rgb, thermal)
            return (fused, [], frequency_aux) if return_frequency_aux else (fused, [])
        if not self.use_dfca and not self.use_sse:
            rgb = self.rgb_encoder(rgb_feats)
            thermal = self.thermal_encoder(thermal_feats)
            mask_logits = []
        else:
            rgb = self._project_and_attend(self.rgb_encoder, rgb_feats)
            thermal = self._project_and_attend(self.thermal_encoder, thermal_feats)
            rgb_inner, thermal_inner = [rgb[-1]], [thermal[-1]]
            mask_logits = []
            for stage, level in enumerate((1, 0)):
                rgb_high = self.rgb_encoder.lateral_convs[stage](rgb_inner[0])
                thermal_high = self.thermal_encoder.lateral_convs[stage](thermal_inner[0])
                # PAN consumes these lateral-transformed skip tensors upstream.
                rgb_inner[0], thermal_inner[0] = rgb_high, thermal_high
                rgb_low, thermal_low = rgb[level], thermal[level]
                if self.use_dfca:
                    # Both directions see the original same-level pair.
                    rgb_low, thermal_low = (
                        self.rgb_dfca[level](thermal[level], rgb[level]),
                        self.thermal_dfca[level](rgb[level], thermal[level]))
                if self.use_sse:
                    # Identity fusion inside SSE exposes Eq.15's concatenated
                    # input. The baseline CSP remains registered only once.
                    rgb_input, rgb_mask = self.rgb_sse[level](rgb_low, rgb_high)
                    thermal_input, thermal_mask = self.thermal_sse[level](thermal_low, thermal_high)
                    mask_logits.extend((rgb_mask, thermal_mask))
                else:
                    rgb_input = torch.cat((F.interpolate(rgb_high, scale_factor=2., mode='nearest'), rgb_low), dim=1)
                    thermal_input = torch.cat((F.interpolate(thermal_high, scale_factor=2., mode='nearest'), thermal_low), dim=1)
                rgb_inner.insert(0, self.rgb_encoder.fpn_blocks[stage](rgb_input))
                thermal_inner.insert(0, self.thermal_encoder.fpn_blocks[stage](thermal_input))
            rgb = self._pan(self.rgb_encoder, rgb_inner)
            thermal = self._pan(self.thermal_encoder, thermal_inner)
        if self.frequency_module is not None:
            rgb, thermal = list(rgb), list(thermal)
            rgb[0], thermal[0], frequency_aux = self.frequency_module(
                rgb[0], thermal[0], self.resize_valid_masks(valid_masks, rgb[0]))
        fused = self.fuse_streams(rgb, thermal)
        if return_frequency_aux:
            return fused, mask_logits, frequency_aux
        return fused, mask_logits
