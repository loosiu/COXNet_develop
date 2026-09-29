import torch

from mmdet.models.utils.tpsc import (
    TPSCDescriptor,
    build_tpsc_gaussian_targets,
    valid_average_pool,
)


def test_tpsc_gaussian_targets_cover_fractional_centers_and_mask_padding():
    boxes = [torch.tensor([
        [15.0, 11.0, 24.0, 20.0],
        [24.0, 11.0, 33.0, 20.0],
    ])]
    valid = torch.ones(1, 1, 8, 10, dtype=torch.bool)
    valid[:, :, -1] = False

    target = build_tpsc_gaussian_targets(
        boxes, padded_size=(64, 80), feat_size=(8, 10),
        device=torch.device('cpu'), valid_mask=valid)

    assert target.shape == (1, 1, 8, 10)
    assert torch.count_nonzero(target == 1.0).item() >= 2
    assert target.min().item() >= 0.0
    assert target.max().item() == 1.0
    assert torch.count_nonzero(target[:, :, -1]).item() == 0


def test_tpsc_gaussian_targets_merge_by_max_and_handle_empty_gt():
    duplicate_boxes = [torch.tensor([
        [16.0, 16.0, 24.0, 24.0],
        [16.0, 16.0, 24.0, 24.0],
    ])]
    duplicate = build_tpsc_gaussian_targets(
        duplicate_boxes, (64, 64), (8, 8), torch.device('cpu'))
    single = build_tpsc_gaussian_targets(
        [duplicate_boxes[0][:1]], (64, 64), (8, 8),
        torch.device('cpu'))
    empty = build_tpsc_gaussian_targets(
        [torch.empty(0, 4)], (64, 64), (8, 8), torch.device('cpu'))

    assert torch.equal(duplicate, single)
    assert torch.count_nonzero(empty).item() == 0
    assert torch.isfinite(empty).all()


def test_tpsc_gaussian_targets_reject_mismatched_valid_mask():
    try:
        build_tpsc_gaussian_targets(
            [torch.empty(0, 4)], (64, 64), (8, 8),
            torch.device('cpu'),
            valid_mask=torch.ones(1, 1, 7, 8, dtype=torch.bool))
    except ValueError as error:
        assert 'valid_mask' in str(error)
    else:
        raise AssertionError('TPSC accepted an invalid target mask shape')


def test_tpsc_valid_average_pool_excludes_padding_and_handles_all_invalid():
    feature = torch.tensor([[[
        [2.0, 100.0, 100.0],
        [100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0],
    ]]])
    valid = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    valid[:, :, 0, 0] = True

    pooled = valid_average_pool(feature, valid, kernel_size=3)
    empty = valid_average_pool(
        feature, torch.zeros_like(valid), kernel_size=3)

    assert pooled.shape == feature.shape
    assert pooled[0, 0, 0, 0].item() == 2.0
    assert pooled[0, 0, 1, 1].item() == 2.0
    assert pooled[0, 0, 2, 2].item() == 0.0
    assert torch.count_nonzero(empty).item() == 0
    assert torch.isfinite(empty).all()


def test_tpsc_descriptor_combines_p3_detail_and_p4_context_without_mutating_inputs():
    torch.manual_seed(5)
    descriptor = TPSCDescriptor(channels=2, prototype_dim=4)
    p3 = torch.randn(1, 2, 4, 4)
    p4 = torch.randn(1, 2, 2, 2)
    p3_before = p3.clone()
    p4_before = p4.clone()
    valid_p3 = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    valid_p4 = torch.ones(1, 1, 2, 2, dtype=torch.bool)

    descriptor_p3, descriptor_p4 = descriptor(
        p3, p4, valid_p3, valid_p4)
    changed_p3, _ = descriptor(
        p3, p4 + 3.0, valid_p3, valid_p4)

    assert descriptor_p3.shape == (1, 4, 4, 4)
    assert descriptor_p4.shape == (1, 4, 2, 2)
    assert not torch.allclose(descriptor_p3, changed_p3)
    assert torch.equal(p3, p3_before)
    assert torch.equal(p4, p4_before)
