import math

import torch
import torch.nn as nn
from mmcv import Config

from mmdet.models.detectors.fusionnet_xo import FusionNetXO
from mmdet.models.utils.fusion_strategy import FusionLayer
from mmdet.models.utils.prldfc import (
    DynamicFrequencyBank,
    PRLDFCSeedHead,
    PrototypeRoutedLocalDynamicFrequencyCalibration,
    build_prldfc_seed_targets,
)


def _predictions(batch=1, height=4, width=4):
    logits = torch.zeros(batch, 1, height, width)
    offsets = torch.zeros(batch, 2, height, width)
    log_scales = torch.zeros(batch, 2, height, width)
    return logits, offsets, log_scales


def test_prldfc_seed_head_has_requested_outputs_and_bounded_offsets():
    head = PRLDFCSeedHead(channels=8, seed_prior=0.01)
    logits, offsets, log_scales = head(torch.randn(2, 8, 5, 7))

    assert logits.shape == (2, 1, 5, 7)
    assert offsets.shape == (2, 2, 5, 7)
    assert log_scales.shape == (2, 2, 5, 7)
    assert offsets.min().item() >= -0.5
    assert offsets.max().item() <= 0.5
    assert math.isclose(
        head.prediction.bias[0].sigmoid().item(), 0.01,
        rel_tol=0.0, abs_tol=1e-6)


def test_prldfc_seed_targets_assign_same_cell_gt_to_distinct_cells():
    logits, offsets, log_scales = _predictions()
    boxes = [torch.tensor([
        [8.0, 8.0, 16.0, 16.0],
        [7.0, 7.0, 17.0, 17.0],
    ])]
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    targets = build_prldfc_seed_targets(
        logits, offsets, log_scales, boxes, padded_size=(32, 32),
        valid_mask=valid, level_scale_range=(0, 32), matching_radius=2)

    positive_indices = targets['positive_mask'][0, 0].nonzero()
    assert positive_indices.shape == (2, 2)
    assert torch.unique(positive_indices, dim=0).shape[0] == 2
    assert targets['eligible_gt_count'].item() == 2.0
    assert targets['unassigned_gt_count'].item() == 0.0
    assert torch.equal(
        targets['seed_target'].bool(), targets['positive_mask'])


def test_prldfc_seed_targets_filter_scale_and_exclude_padding():
    logits, offsets, log_scales = _predictions()
    boxes = [torch.tensor([
        [4.0, 4.0, 12.0, 12.0],
        [4.0, 4.0, 28.0, 28.0],
    ])]
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    valid[:, :, -1] = False

    targets = build_prldfc_seed_targets(
        logits, offsets, log_scales, boxes, padded_size=(32, 32),
        valid_mask=valid, level_scale_range=(0, 16), matching_radius=2)

    assert targets['eligible_gt_count'].item() == 1.0
    assert targets['positive_mask'].sum().item() == 1
    assert not targets['positive_mask'][~valid].any()
    assert not targets['loss_valid'][~valid].any()
    assert torch.isfinite(targets['offset_target']).all()
    assert torch.isfinite(targets['scale_target']).all()


def test_prldfc_seed_targets_handle_zero_gt_as_finite_negatives():
    logits, offsets, log_scales = _predictions(batch=2)
    boxes = [torch.empty(0, 4), torch.empty(0, 4)]
    valid = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    valid[1, :, -1] = False

    targets = build_prldfc_seed_targets(
        logits, offsets, log_scales, boxes, padded_size=(32, 32),
        valid_mask=valid, level_scale_range=(0, 32))

    assert torch.count_nonzero(targets['seed_target']).item() == 0
    assert torch.count_nonzero(targets['positive_mask']).item() == 0
    assert torch.equal(targets['loss_valid'], valid)
    assert targets['eligible_gt_count'].item() == 0.0
    assert targets['unassigned_gt_count'].item() == 0.0
    for value in targets.values():
        assert torch.isfinite(value.float()).all()


def test_prldfc_cardinality_is_image_wise_for_mixed_empty_batch():
    module = _calibrator()
    logits = torch.full((2, 1, 1, 1), -4.0, requires_grad=True)
    offsets = torch.zeros(2, 2, 1, 1)
    log_scales = torch.zeros(2, 2, 1, 1)
    valid = torch.ones(2, 1, 1, 1, dtype=torch.bool)
    boxes = [torch.tensor([[0.0, 0.0, 8.0, 8.0]]), torch.empty(0, 4)]

    losses = module._seed_losses(
        logits, offsets, log_scales, valid, boxes,
        padded_size=(8, 8), level_scale_range=(0, 32))
    losses['cardinality_loss'].backward()

    # Gradient descent must lower the false seed probability in the empty
    # image even while the non-empty image is under-counted.
    assert logits.grad[0, 0, 0, 0].item() < 0
    assert logits.grad[1, 0, 0, 0].item() > 0


def test_prldfc_seed_targets_reject_inconsistent_shapes():
    logits, offsets, log_scales = _predictions()
    boxes = [torch.empty(0, 4)]
    wrong_valid = torch.ones(1, 1, 3, 4, dtype=torch.bool)

    try:
        build_prldfc_seed_targets(
            logits, offsets, log_scales, boxes, padded_size=(32, 32),
            valid_mask=wrong_valid, level_scale_range=(0, 32))
    except ValueError as error:
        assert 'valid_mask' in str(error)
    else:
        raise AssertionError('PRLDFC accepted an inconsistent valid mask')


def test_prldfc_frequency_bank_partitions_and_reconstructs_odd_even_maps():
    torch.manual_seed(31)
    bank = DynamicFrequencyBank(
        channels=4, frequency_dim=4, num_bands=3,
        temperature=0.02, min_band_width=0.05)

    for height, width in ((5, 7), (6, 8)):
        feature = torch.randn(2, 4, height, width)
        valid = torch.ones(2, 1, height, width, dtype=torch.bool)
        valid[1, :, -1] = False
        bands, diagnostics = bank(feature, valid)

        assert bands.shape == (2, 3, 4, height, width)
        assert diagnostics['masks'].shape == (
            3, height, width // 2 + 1)
        assert torch.allclose(
            diagnostics['masks'].sum(0),
            torch.ones(height, width // 2 + 1), atol=1e-6)
        assert torch.allclose(
            bands.sum(1), diagnostics['projected'], atol=2e-5, rtol=2e-5)
        assert torch.count_nonzero(
            diagnostics['projected'][1, :, -1]).item() == 0


def test_prldfc_frequency_boundaries_are_monotonic_and_receive_gradient():
    torch.manual_seed(37)
    bank = DynamicFrequencyBank(
        channels=4, frequency_dim=3, num_bands=3,
        temperature=0.02, min_band_width=0.05)
    feature = torch.randn(1, 4, 7, 9, requires_grad=True)
    valid = torch.ones(1, 1, 7, 9, dtype=torch.bool)

    bands, diagnostics = bank(feature, valid)
    boundaries = diagnostics['boundaries']
    widths = boundaries[1:] - boundaries[:-1]
    loss = bands[:, 0].square().mean() + 0.3 * bands[:, 2].abs().mean()
    loss.backward()

    assert torch.all(widths >= 0.05 - 1e-7)
    assert boundaries[0].item() == 0.0
    assert torch.allclose(boundaries[-1], torch.tensor(0.5), atol=1e-7)
    assert bank.band_logits.grad.abs().sum().item() > 0
    assert bank.projection.weight.grad.abs().sum().item() > 0
    assert feature.grad.abs().sum().item() > 0


def test_prldfc_frequency_bank_restores_autocast_dtype():
    bank = DynamicFrequencyBank(
        channels=4, frequency_dim=4, num_bands=3,
        temperature=0.02, min_band_width=0.05)
    feature = torch.randn(1, 4, 5, 7)
    valid = torch.ones(1, 1, 5, 7, dtype=torch.bool)

    with torch.autocast('cpu', dtype=torch.bfloat16):
        bands, diagnostics = bank(feature, valid)

    assert bands.dtype == torch.bfloat16
    assert diagnostics['projected'].dtype == torch.bfloat16


def test_prldfc_local_attention_is_finite_for_direct_half_callers():
    module = _calibrator()
    query = torch.randn(1, 4, 5, 7).half().requires_grad_()
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    valid = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    valid[:, :, -1] = False
    valid[:, :, :, -1] = False

    read, attention, entropy = module._local_attention(
        query, key, value, valid, normalize_query_key=True)
    (read.float().square().mean() + entropy.float().mean()).backward()

    assert read.dtype == value.dtype
    assert attention.dtype == value.dtype
    assert entropy.dtype == value.dtype
    assert torch.isfinite(read).all()
    assert torch.isfinite(attention).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key.grad).all()
    assert torch.isfinite(value.grad).all()


def _calibrator(**kwargs):
    defaults = dict(
        channels=8,
        frequency_dim=4,
        prototype_dim=4,
        num_bands=3,
        search_radius=1,
        seed_prior=0.05,
        seed_threshold=0.10,
        seed_temperature=0.5,
        frequency_temperature=0.02,
        min_band_width=0.05,
        residual_epsilon=0.1,
    )
    defaults.update(kwargs)
    return PrototypeRoutedLocalDynamicFrequencyCalibration(**defaults)


def test_prldfc_calibration_is_rgb_only_bounded_and_masks_padding():
    torch.manual_seed(41)
    module = _calibrator()
    rgb = torch.randn(2, 8, 5, 7)
    thermal = torch.randn_like(rgb)
    thermal_before = thermal.clone()
    valid = torch.ones(2, 1, 5, 7, dtype=torch.bool)
    valid[1, :, -1] = False
    valid[1, :, :, -1] = False

    output, aux = module(
        rgb, thermal, valid_mask=valid, return_aux=True)
    delta = output - rgb

    assert output.shape == rgb.shape
    assert torch.equal(thermal, thermal_before)
    assert delta.abs().max().item() <= 0.1 + 1e-6
    assert torch.count_nonzero(delta[1, :, -1]).item() == 0
    assert torch.count_nonzero(delta[1, :, :, -1]).item() == 0
    assert aux['band_weights'].shape == (2, 3, 5, 7)
    assert torch.allclose(
        aux['band_weights'].sum(1), torch.ones(2, 5, 7), atol=1e-6)
    assert torch.allclose(
        aux['reliability'], torch.full((2, 1, 5, 7), 0.5), atol=1e-6)
    assert aux['support_ratio'].ndim == 0
    assert aux['delta_ratio'].ndim == 0
    assert not hasattr(module, 'rgb_objectness_head')


def test_prldfc_seed_predictions_ignore_masked_thermal_values():
    torch.manual_seed(42)
    module = _calibrator().eval()
    with torch.no_grad():
        module.seed_head.prediction.weight.normal_(mean=0.0, std=0.2)
    rgb = torch.randn(1, 8, 5, 7)
    thermal = torch.randn_like(rgb)
    valid = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    valid[:, :, :, -1] = False
    perturbed = thermal.clone()
    perturbed[:, :, :, -1] += 100.0

    with torch.no_grad():
        _, reference = module(
            rgb, thermal, valid_mask=valid, return_aux=True)
        _, changed = module(
            rgb, perturbed, valid_mask=valid, return_aux=True)

    expanded_valid = valid.expand_as(reference['seed_logits'])
    assert torch.equal(
        reference['seed_logits'][expanded_valid],
        changed['seed_logits'][expanded_valid])
    assert torch.equal(
        reference['seed_gate'][expanded_valid],
        changed['seed_gate'][expanded_valid])


def test_prldfc_sparse_inference_is_identity_without_threshold_seeds():
    module = _calibrator(
        seed_prior=0.01, seed_threshold=0.99, sparse_inference=True)
    module.eval()
    rgb = torch.randn(1, 8, 4, 5)
    thermal = torch.randn_like(rgb)

    output, aux = module(rgb, thermal, return_aux=True)

    assert torch.equal(output, rgb)
    assert torch.count_nonzero(aux['seed_gate']).item() == 0
    assert torch.count_nonzero(aux['delta']).item() == 0


def test_prldfc_training_losses_and_detection_path_receive_gradients():
    torch.manual_seed(43)
    module = _calibrator(seed_prior=0.2, seed_temperature=1.0)
    module.train()
    rgb = torch.randn(1, 8, 5, 6, requires_grad=True)
    thermal = torch.randn_like(rgb, requires_grad=True)
    valid = torch.ones(1, 1, 5, 6, dtype=torch.bool)
    boxes = [torch.tensor([
        [8.0, 8.0, 16.0, 16.0],
        [16.0, 8.0, 24.0, 16.0],
    ])]

    output, aux = module(
        rgb, thermal, valid_mask=valid, gt_bboxes=boxes,
        padded_size=(40, 48), level_scale_range=(0, 32),
        return_aux=True)
    loss_names = {
        'seed_loss', 'offset_loss', 'scale_loss', 'cardinality_loss'}
    total = output.square().mean() + sum(aux[name] for name in loss_names)
    total.backward()

    assert loss_names.issubset(aux)
    assert all(torch.isfinite(aux[name]) for name in loss_names)
    assert module.seed_head.prediction.weight.grad.abs().sum().item() > 0
    assert module.frequency_bank.band_logits.grad.abs().sum().item() > 0
    assert module.band_router[-1].weight.grad.abs().sum().item() > 0
    assert module.residual_mlps[0][-1].weight.grad.abs().sum().item() > 0
    assert rgb.grad.abs().sum().item() > 0
    assert thermal.grad.abs().sum().item() > 0
    assert aux['eligible_gt_count'].item() == 2.0
    assert aux['unassigned_gt_count'].item() == 0.0


def test_prldfc_active_seed_initialization_trains_routing_paths():
    """An active seed must not leave the calibration routers starved."""
    torch.manual_seed(45)
    module = _calibrator(seed_prior=0.2, seed_temperature=0.25)
    module.train()
    rgb = torch.randn(2, 8, 9, 11)
    thermal = torch.randn_like(rgb)
    valid = torch.ones(2, 1, 9, 11, dtype=torch.bool)
    valid[:, :, -1] = False
    valid[:, :, :, -1] = False

    output, aux = module(
        rgb, thermal, valid_mask=valid, return_aux=True)
    output.square().mean().backward()

    assert aux['delta_ratio'].item() > 1e-4
    assert module.band_router[-1].weight.grad.norm().item() > 1e-6
    assert module.reliability_router[-1].weight.grad.norm().item() > 1e-6


def test_prldfc_calibration_rejects_mismatched_modalities_and_masks():
    module = _calibrator()
    rgb = torch.randn(1, 8, 4, 5)

    try:
        module(rgb, torch.randn(1, 8, 3, 5))
    except ValueError as error:
        assert 'equal shapes' in str(error)
    else:
        raise AssertionError('PRLDFC accepted unequal modality shapes')

    try:
        module(rgb, torch.randn_like(rgb), valid_mask=torch.ones(1, 1, 3, 5))
    except ValueError as error:
        assert 'valid_mask' in str(error)
    else:
        raise AssertionError('PRLDFC accepted an inconsistent valid mask')


class _CaptureFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.rgb_inputs = []
        self.thermal_inputs = []

    def forward(self, rgb, thermal):
        self.rgb_inputs.append(rgb.detach().clone())
        self.thermal_inputs.append(thermal.detach().clone())
        return rgb + thermal


def _fusion_layer(**overrides):
    arguments = dict(
        in_channels=8,
        reduction=4,
        num_layers=3,
        fs_type='fusionnet-xo',
        use_clfm=[],
        use_trpc=False,
        use_oepc=False,
        use_topc=False,
        use_prldfc=True,
        prldfc_cfg=dict(
            apply_levels=(0, 1),
            frequency_dim=4,
            prototype_dim=4,
            num_bands=3,
            search_radius=(1, 1),
            seed_prior=0.05,
            seed_threshold=0.1,
            seed_temperature=0.5,
            frequency_temperature=0.02,
            min_band_width=0.05,
            level_scale_ranges=((0, 32), (16, 64)),
            residual_epsilon=(0.1, 0.1),
            seed_loss_weight=0.1,
            offset_loss_weight=0.05,
            scale_loss_weight=0.02,
            cardinality_loss_weight=0.01),
        wf_loss=False,
        usepoolup=[])
    arguments.update(overrides)
    return FusionLayer(**arguments)


def test_prldfc_fusion_layer_applies_p3_p4_and_preserves_thermal():
    torch.manual_seed(47)
    layer = _fusion_layer()
    captures = nn.ModuleList(
        [_CaptureFusion(), _CaptureFusion(), _CaptureFusion()])
    layer.hofm_layers = captures
    observed = {}
    handles = [
        layer.prldfc_layers[str(level)].register_forward_hook(
            lambda _module, _inputs, output, level=level:
            observed.update({level: output[1]}))
        for level in (0, 1)]
    layer.train()
    visible = [
        torch.randn(1, 8, 5, 6),
        torch.randn(1, 8, 3, 4),
        torch.randn(1, 8, 2, 3)]
    thermal = [torch.randn_like(feature) for feature in visible]
    thermal_before = [feature.clone() for feature in thermal]
    boxes = [torch.tensor([[8.0, 8.0, 24.0, 24.0]])]
    metas = [dict(
        img_shape=(32, 40, 3), pad_shape=(40, 48, 3),
        batch_input_shape=(40, 48))]

    features, aux = layer(
        visible, thermal, gt_bboxes=boxes, img_metas=metas)
    for handle in handles:
        handle.remove()

    assert len(features) == 3
    assert set(layer.prldfc_layers.keys()) == {'0', '1'}
    assert not hasattr(layer, 'idwt_layers')
    assert not hasattr(layer, 'trpc_layers')
    assert not hasattr(layer, 'oepc_layers')
    assert not hasattr(layer, 'topc_layers')
    for level in range(3):
        assert torch.equal(captures[level].thermal_inputs[0], thermal_before[level])
        assert torch.equal(thermal[level], thermal_before[level])
    assert torch.allclose(
        aux['loss_prldfc_seed'],
        0.1 * sum(item['seed_loss'] for item in observed.values()) / 2)
    assert torch.allclose(
        aux['loss_prldfc_offset'],
        0.05 * sum(item['offset_loss'] for item in observed.values()) / 2)
    assert 'prldfc_band_weight_low' in aux
    assert 'prldfc_delta_ratio' in aux
    assert torch.count_nonzero(
        observed[0]['spatial_support'][:, :, -1]).item() == 0
    assert torch.count_nonzero(
        observed[0]['spatial_support'][:, :, :, -1]).item() == 0


def test_prldfc_fusion_layer_inference_records_detached_diagnostics():
    layer = _fusion_layer()
    layer.hofm_layers = nn.ModuleList(
        [_CaptureFusion(), _CaptureFusion(), _CaptureFusion()])
    layer.eval()
    visible = [
        torch.randn(1, 8, 4, 5),
        torch.randn(1, 8, 2, 3),
        torch.randn(1, 8, 1, 2)]
    thermal = [torch.randn_like(feature) for feature in visible]
    metas = [dict(
        img_shape=(24, 32, 3), pad_shape=(32, 40, 3),
        batch_input_shape=(32, 40))]

    outputs = layer(visible, thermal, img_metas=metas)

    assert len(outputs) == 3
    assert set(layer.last_prldfc_aux) == {0, 1}
    assert not layer.last_prldfc_aux[0]['delta_ratio'].requires_grad


def test_prldfc_forward_train_dispatches_aux_path_when_wf_loss_is_disabled():
    class RecordingHead:

        def __init__(self):
            self.features = None

        def forward_train(self, features, *args):
            self.features = features
            return {'loss_detector': torch.tensor(1.0)}

    class RecordingDetector:
        wf_loss = False
        use_trpc = False
        use_oepc = False
        use_topc = False
        use_prldfc = True

        def __init__(self):
            self.bbox_head = RecordingHead()
            self.extract_args = None
            self.extract_kwargs = None

        def extract_feat(self, img, *args, **kwargs):
            self.extract_args = args
            self.extract_kwargs = kwargs
            return ['p3', 'p4'], {'loss_prldfc_seed': torch.tensor(0.5)}

    detector = RecordingDetector()
    images = (torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16))
    metas = [dict(img_shape=(16, 16, 3), pad_shape=(16, 16, 3))]
    boxes = [torch.tensor([[2.0, 2.0, 8.0, 8.0]])]
    labels = [torch.tensor([0])]

    losses = FusionNetXO.forward_train(
        detector, images, metas, boxes, labels)

    assert detector.extract_args == (boxes, metas)
    assert detector.extract_kwargs == {'gt_bboxes_ignore': None}
    assert detector.bbox_head.features == ['p3', 'p4']
    assert set(losses) == {'loss_detector', 'loss_prldfc_seed'}


def test_prldfc_replacement_is_mutually_exclusive_and_requires_same_shape():
    conflicts = (
        dict(use_clfm=['v3']),
        dict(use_trpc=True),
        dict(use_oepc=True),
        dict(use_topc=True),
    )
    for conflicting in conflicts:
        try:
            _fusion_layer(**conflicting)
        except ValueError as error:
            assert 'mutually exclusive' in str(error) or 'replaces CLFM' in str(error)
        else:
            raise AssertionError('PRLDFC accepted a conflicting fusion path')

    layer = _fusion_layer()
    visible = [
        torch.randn(1, 8, 4, 5),
        torch.randn(1, 8, 2, 3),
        torch.randn(1, 8, 1, 2)]
    thermal = [
        torch.randn(1, 8, 3, 5),
        torch.randn_like(visible[1]),
        torch.randn_like(visible[2])]
    try:
        layer(visible, thermal)
    except ValueError as error:
        assert 'same-stage' in str(error)
    else:
        raise AssertionError('PRLDFC accepted unequal same-stage features')


def test_inactive_prldfc_does_not_constrain_legacy_layer_count():
    layer = FusionLayer(
        in_channels=8, num_layers=1, fs_type='add',
        use_prldfc=False, usepoolup=[])

    assert layer.num_layers == 1
    assert not hasattr(layer, 'prldfc_layers')


def test_prldfc_canonical_and_control_config_contracts():
    canonical = Config.fromfile('configs/coxnet/prldfc/PRLDFC.py').model
    p3_only = Config.fromfile('configs/coxnet/prldfc/PRLDFC_p3.py').model
    control = Config.fromfile(
        'configs/coxnet/prldfc/same_stage_no_calibration.py').model

    assert canonical.neck.start_level == 1
    assert canonical.neck_t.start_level == 1
    assert canonical.use_clfm == []
    assert canonical.use_prldfc is True
    assert canonical.wf_loss is True
    assert canonical.wf_loss_mode == 'kl_v2'
    assert canonical.wf_loss_weight == 0.1
    assert canonical.prldfc_cfg.apply_levels == (0, 1)
    assert canonical.prldfc_cfg.level_scale_ranges == ((0, 32), (16, 64))
    assert canonical.prldfc_cfg.search_radius == (2, 2)
    assert canonical.prldfc_cfg.residual_epsilon == (0.1, 0.1)
    assert p3_only.prldfc_cfg.apply_levels == (0,)
    assert p3_only.prldfc_cfg.level_scale_ranges == ((0, 32),)
    assert control.use_prldfc is False
    assert control.use_clfm == []
    assert control.neck.start_level == control.neck_t.start_level == 1
