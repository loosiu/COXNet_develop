import torch

from mmdet.models.utils.topc import (
    ThermalAnchoredObjectPrototypeCalibration,
    build_topc_gaussian_targets,
)


def _module(**kwargs):
    defaults = dict(
        channels=8,
        candidate_prior=0.01,
        candidate_threshold=0.5,
        max_candidates=2,
    )
    defaults.update(kwargs)
    return ThermalAnchoredObjectPrototypeCalibration(**defaults)


def test_topc_gaussian_targets_normalize_each_fractional_object_and_mask_padding():
    boxes = [torch.tensor([
        [15.0, 11.0, 24.0, 20.0],
        [24.0, 11.0, 33.0, 20.0],
    ])]
    valid = torch.ones(1, 1, 8, 10, dtype=torch.bool)
    valid[:, :, -1] = False

    target = build_topc_gaussian_targets(
        boxes, padded_size=(64, 80), feat_size=(8, 10),
        device=torch.device('cpu'), valid_mask=valid)

    assert target.shape == (1, 1, 8, 10)
    assert torch.count_nonzero(target == 1.0).item() >= 2
    assert target.min().item() >= 0.0
    assert target.max().item() == 1.0
    assert torch.count_nonzero(target[:, :, -1]).item() == 0


def test_topc_gaussian_targets_merge_by_max_and_handle_empty_ground_truth():
    duplicate_boxes = [torch.tensor([
        [16.0, 16.0, 24.0, 24.0],
        [16.0, 16.0, 24.0, 24.0],
    ])]
    duplicate = build_topc_gaussian_targets(
        duplicate_boxes, (64, 64), (8, 8), torch.device('cpu'))
    single = build_topc_gaussian_targets(
        [duplicate_boxes[0][:1]], (64, 64), (8, 8),
        torch.device('cpu'))
    empty = build_topc_gaussian_targets(
        [torch.empty(0, 4)], (64, 64), (8, 8), torch.device('cpu'))

    assert torch.equal(duplicate, single)
    assert torch.count_nonzero(empty).item() == 0


def test_topc_gaussian_targets_reject_wrong_valid_mask_shape():
    try:
        build_topc_gaussian_targets(
            [torch.empty(0, 4)], (64, 64), (8, 8),
            torch.device('cpu'), valid_mask=torch.ones(
                1, 1, 7, 8, dtype=torch.bool))
    except ValueError as error:
        assert 'valid_mask' in str(error)
    else:
        raise AssertionError('TOPC accepted an invalid target mask shape')


def test_topc_candidate_selection_keeps_adjacent_cells_without_peak_nms():
    module = _module()
    logits = torch.full((1, 1, 5, 5), -20.0)
    logits[0, 0, 2, 2] = 4.0
    logits[0, 0, 2, 3] = 3.0
    valid = torch.ones_like(logits, dtype=torch.bool)

    selected = module._select_candidates(logits, valid)

    expected = {2 * 5 + 2, 2 * 5 + 3}
    actual = set(selected['indices'][0][selected['active'][0]].tolist())
    assert actual == expected
    assert selected['active'].sum().item() == 2
    assert selected['dense_mask'][0, 0, 2, 2]
    assert selected['dense_mask'][0, 0, 2, 3]


def test_topc_candidate_selection_excludes_padding_and_handles_no_candidate():
    module = _module(max_candidates=3)
    logits = torch.full((1, 1, 4, 4), -20.0)
    logits[0, 0, 3, 3] = 20.0
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid[:, :, 3, :] = False

    selected = module._select_candidates(logits, valid)

    assert selected['indices'].shape == (1, 3)
    assert not selected['active'].any()
    assert torch.count_nonzero(selected['scores']).item() == 0
    assert torch.count_nonzero(selected['dense_mask']).item() == 0


def test_topc_objectness_is_thermal_only_and_uses_requested_shape():
    torch.manual_seed(17)
    module = _module()
    thermal = torch.randn(2, 8, 6, 7)

    first = module.objectness_logits(thermal)
    second = module.objectness_logits(thermal.clone())

    assert first.shape == (2, 1, 6, 7)
    assert torch.equal(first, second)
    assert not hasattr(module, 'rgb_candidate_head')


def _selection(indices, active=None):
    indices = torch.as_tensor(indices, dtype=torch.long)
    if indices.ndim == 1:
        indices = indices.unsqueeze(0)
    if active is None:
        active = torch.ones_like(indices, dtype=torch.bool)
    else:
        active = torch.as_tensor(active, dtype=torch.bool)
        if active.ndim == 1:
            active = active.unsqueeze(0)
    return dict(indices=indices, active=active)


def test_topc_reuses_one_projection_module_for_both_modalities():
    module = _module(calibration_dim=4)
    calls = []
    handle = module.shared_projection.register_forward_hook(
        lambda _module, inputs, _output: calls.append(inputs[0].data_ptr()))
    rgb = torch.randn(1, 8, 3, 4)
    thermal = torch.randn(1, 8, 3, 4)

    rgb_shared, thermal_shared = module._shared_features(rgb, thermal)
    handle.remove()

    assert rgb_shared.shape == thermal_shared.shape == (1, 4, 3, 4)
    assert len(calls) == 2
    assert not hasattr(module, 'rgb_projection')
    assert not hasattr(module, 'thermal_projection')
    parameter_names = {name for name, _ in module.named_parameters()}
    assert 'shared_projection.weight' in parameter_names
    assert not any(name.startswith(('rgb_projection', 'thermal_projection'))
                   for name in parameter_names)


def test_topc_thermal_prototype_is_heatmap_weighted_per_candidate():
    module = _module(calibration_dim=1, object_kernel=3)
    thermal_shared = torch.arange(9.0).reshape(1, 1, 3, 3)
    probability = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    valid = torch.ones(1, 1, 3, 3, dtype=torch.bool)

    prototypes, mass, active = module._thermal_prototypes(
        thermal_shared, probability, _selection([4, 0]), valid)

    assert prototypes.shape == (1, 2, 1)
    assert torch.allclose(prototypes[0, 0, 0], torch.tensor(240.0 / 45.0))
    boundary_expected = torch.tensor((0 * 1 + 1 * 2 + 3 * 4 + 4 * 5) / 12.0)
    assert torch.allclose(prototypes[0, 1, 0], boundary_expected)
    assert torch.allclose(mass[0], torch.tensor([45.0, 12.0]))
    assert active.all()


def test_topc_thermal_prototype_respects_padding_and_zero_mass():
    module = _module(calibration_dim=1, object_kernel=3)
    thermal_shared = torch.arange(9.0).reshape(1, 1, 3, 3)
    probability = torch.ones(1, 1, 3, 3)
    valid = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    valid[0, 0, 0, 0] = True

    prototypes, mass, active = module._thermal_prototypes(
        thermal_shared, probability, _selection([0, 8]), valid)

    assert prototypes[0, 0, 0].item() == 0.0
    assert mass[0, 0].item() == 1.0
    assert active[0, 0]
    assert prototypes[0, 1, 0].item() == 0.0
    assert mass[0, 1].item() == 0.0
    assert not active[0, 1]


def test_topc_local_rgb_match_uses_thermal_query_and_masks_invalid_options():
    module = _module(
        calibration_dim=2, search_radius=1, search_temperature=1.0)
    rgb_shared = torch.zeros(1, 2, 3, 3)
    rgb_shared[0, :, 1, 1] = torch.tensor([1.0, 0.0])
    rgb_shared[0, :, 1, 2] = torch.tensor([-1.0, 0.0])
    prototypes = torch.tensor([[[1.0, 0.0]]])
    valid = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    valid[0, 0, 1, 1:3] = True

    match = module._local_rgb_match(
        rgb_shared, prototypes, torch.tensor([[4]]),
        torch.tensor([[True]]), valid)

    expected_attention = torch.softmax(torch.tensor([1.0, -1.0]), dim=0)
    assert match['attention'].shape == (1, 1, 9)
    assert torch.allclose(match['attention'].sum(-1), torch.ones(1, 1))
    assert torch.count_nonzero(match['attention']).item() == 2
    assert torch.allclose(
        match['rgb_prototypes'][0, 0],
        torch.tensor([expected_attention[0] - expected_attention[1], 0.0]))
    assert torch.allclose(match['confidence'][0, 0], expected_attention.max())
    assert 0.0 <= match['confidence'].min() <= match['confidence'].max() <= 1.0


def test_topc_local_rgb_match_handles_inactive_and_all_invalid_windows():
    module = _module(calibration_dim=2, search_radius=1)
    rgb_shared = torch.randn(1, 2, 3, 3)
    prototypes = torch.randn(1, 2, 2)
    valid = torch.zeros(1, 1, 3, 3, dtype=torch.bool)

    match = module._local_rgb_match(
        rgb_shared, prototypes, torch.tensor([[4, 8]]),
        torch.tensor([[True, False]]), valid)

    assert torch.count_nonzero(match['attention']).item() == 0
    assert torch.count_nonzero(match['rgb_prototypes']).item() == 0
    assert torch.count_nonzero(match['confidence']).item() == 0
    assert not match['valid_options'].any()


def test_topc_scatter_keeps_single_candidate_attention_magnitude():
    module = _module(channels=1, calibration_dim=1, search_radius=1)
    residual = torch.tensor([[[2.0]]])
    confidence = torch.tensor([[0.5]])
    attention = torch.zeros(1, 1, 9)
    attention[0, 0, 4] = 0.75
    attention[0, 0, 5] = 0.25

    delta, mass = module._scatter_residual(
        residual, confidence, attention, torch.tensor([[4]]),
        torch.tensor([[True]]), output_size=(3, 3))

    expected = torch.zeros(1, 1, 3, 3)
    expected[0, 0, 1, 1] = 0.75
    expected[0, 0, 1, 2] = 0.25
    assert torch.allclose(delta, expected)
    assert torch.allclose(mass[0, 0, 1, 1:3], torch.tensor([0.75, 0.25]))
    assert torch.count_nonzero(delta).item() == 2


def test_topc_scatter_normalizes_overlap_and_drops_out_of_bounds_offsets():
    module = _module(channels=1, calibration_dim=1, search_radius=1)
    residual = torch.tensor([[[1.0], [3.0], [7.0]]])
    confidence = torch.ones(1, 3)
    attention = torch.zeros(1, 3, 9)
    attention[0, :2, 4] = 1.0
    attention[0, 2, 0] = 1.0  # up-left from cell zero is outside the map

    delta, mass = module._scatter_residual(
        residual, confidence, attention, torch.tensor([[4, 4, 0]]),
        torch.tensor([[True, True, True]]), output_size=(3, 3))

    assert delta[0, 0, 1, 1].item() == 2.0
    assert mass[0, 0, 1, 1].item() == 2.0
    assert torch.count_nonzero(delta).item() == 1


def test_topc_forward_is_identity_without_candidates_and_keeps_thermal_input():
    module = _module(
        candidate_threshold=1.0, calibration_dim=4, search_radius=1)
    rgb = torch.randn(2, 8, 4, 5)
    thermal = torch.randn(2, 8, 4, 5)
    thermal_before = thermal.clone()

    output, aux = module(rgb, thermal, return_aux=True)

    assert torch.equal(output, rgb)
    assert torch.equal(thermal, thermal_before)
    assert aux['candidate_count'].item() == 0.0
    assert torch.count_nonzero(aux['delta']).item() == 0


def test_topc_forward_has_only_objectness_loss_and_no_learned_router_components():
    torch.manual_seed(19)
    module = _module(
        candidate_threshold=0.0, calibration_dim=4, search_radius=1,
        max_candidates=2)
    rgb = torch.randn(1, 8, 4, 5, requires_grad=True)
    thermal = torch.randn(1, 8, 4, 5, requires_grad=True)
    center_target = torch.zeros(1, 1, 4, 5)
    center_target[0, 0, 2, 2] = 1.0

    output, aux = module(
        rgb, thermal, center_target=center_target, return_aux=True)
    total = output.square().mean() + aux['objectness_loss']
    total.backward()

    assert output.shape == rgb.shape
    assert {key for key in aux if key.endswith('_loss')} == {
        'objectness_loss'}
    for forbidden in ('gate', 'router', 'film', 'edl', 'utility'):
        assert not any(forbidden in name.lower()
                       for name, _ in module.named_modules())
    assert module.objectness_head[-1].weight.grad.abs().sum().item() > 0
    assert module.shared_projection.weight.grad.abs().sum().item() > 0
    assert module.residual_projection.weight.grad.abs().sum().item() > 0
    assert rgb.grad.abs().sum().item() > 0
    assert thermal.grad.abs().sum().item() > 0
    assert torch.isfinite(total)


def test_topc_forward_validates_targets_and_masks():
    module = _module(candidate_threshold=0.0)
    rgb = torch.randn(1, 8, 3, 3)
    thermal = torch.randn_like(rgb)

    try:
        module(rgb, thermal, valid_mask=torch.ones(1, 1, 2, 3))
    except ValueError as error:
        assert 'valid' in str(error)
    else:
        raise AssertionError('TOPC accepted a mismatched valid mask')

    try:
        module(
            rgb, thermal, center_target=torch.zeros(1, 1, 2, 3),
            return_aux=True)
    except ValueError as error:
        assert 'center_target' in str(error)
    else:
        raise AssertionError('TOPC accepted a mismatched center target')
