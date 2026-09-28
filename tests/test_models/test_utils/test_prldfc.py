import math

import torch

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
