import math

import torch

from mmdet.models.utils.prldfc import (
    PRLDFCSeedHead,
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
