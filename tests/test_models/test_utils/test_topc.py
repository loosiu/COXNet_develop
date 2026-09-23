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
