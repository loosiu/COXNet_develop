import torch

from mmdet.models.utils.tpsc import (
    SharedSlotPrototypeExtractor,
    TPSCDescriptor,
    build_tpsc_gaussian_targets,
    prototype_coverage_loss,
    prototype_diversity_loss,
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


def test_tpsc_shared_slots_normalize_each_modality_over_valid_space():
    torch.manual_seed(7)
    extractor = SharedSlotPrototypeExtractor(prototype_dim=4, num_slots=3)
    rgb = torch.randn(2, 4, 3, 4)
    thermal = torch.randn(2, 4, 3, 4)
    valid = torch.ones(2, 1, 3, 4, dtype=torch.bool)
    valid[0, :, :, -1] = False

    slots = extractor(rgb, thermal, valid)

    assert slots['rgb_prototypes'].shape == (2, 3, 4)
    assert slots['thermal_prototypes'].shape == (2, 3, 4)
    assert slots['rgb_attention'].shape == (2, 3, 3, 4)
    assert slots['thermal_attention'].shape == (2, 3, 3, 4)
    assert torch.allclose(
        slots['rgb_attention'].flatten(2).sum(-1),
        torch.ones(2, 3), atol=1e-6)
    assert torch.allclose(
        slots['thermal_attention'].flatten(2).sum(-1),
        torch.ones(2, 3), atol=1e-6)
    assert torch.count_nonzero(slots['rgb_attention'][0, :, :, -1]) == 0


def test_tpsc_slots_exclude_padding_and_return_zero_for_all_invalid_sample():
    extractor = SharedSlotPrototypeExtractor(prototype_dim=4, num_slots=2)
    rgb = torch.randn(2, 4, 2, 3)
    thermal = torch.randn(2, 4, 2, 3)
    valid = torch.ones(2, 1, 2, 3, dtype=torch.bool)
    valid[0] = False
    valid[1, :, :, -1] = False

    slots = extractor(rgb, thermal, valid)

    for key in ('rgb_attention', 'thermal_attention'):
        assert torch.count_nonzero(slots[key][0]) == 0
        assert torch.count_nonzero(slots[key][1, :, :, -1]) == 0
        assert torch.isfinite(slots[key]).all()
    for key in ('rgb_prototypes', 'thermal_prototypes'):
        assert torch.count_nonzero(slots[key][0]) == 0
        assert torch.isfinite(slots[key]).all()


def test_tpsc_uses_shared_queries_but_modality_specific_projections():
    extractor = SharedSlotPrototypeExtractor(prototype_dim=4, num_slots=2)

    assert isinstance(extractor.slot_queries, torch.nn.Parameter)
    assert extractor.slot_queries.shape == (2, 4)
    assert extractor.rgb_projection is not extractor.thermal_projection
    assert set(extractor.rgb_projection.parameters()).isdisjoint(
        set(extractor.thermal_projection.parameters()))
    query_parameters = [
        name for name, _ in extractor.named_parameters()
        if 'slot_queries' in name
    ]
    assert query_parameters == ['slot_queries']


def test_tpsc_coverage_prefers_attention_on_gaussian_foreground():
    target = torch.zeros(1, 1, 3, 3)
    target[:, :, 1, 1] = 1.0
    valid = torch.ones_like(target, dtype=torch.bool)
    foreground = torch.full((1, 2, 3, 3), 1e-4)
    foreground[:, :, 1, 1] = 1.0
    background = torch.full((1, 2, 3, 3), 1e-4)
    background[:, :, 0, 0] = 1.0

    foreground_loss = prototype_coverage_loss(foreground, target, valid)
    background_loss = prototype_coverage_loss(background, target, valid)

    assert foreground_loss.ndim == 0
    assert torch.isfinite(foreground_loss)
    assert foreground_loss < background_loss


def test_tpsc_coverage_is_zero_for_empty_target_and_finite_for_all_padding():
    attention = torch.softmax(torch.randn(2, 3, 4, 4), dim=-1)
    empty_target = torch.zeros(2, 1, 4, 4)
    valid = torch.ones_like(empty_target, dtype=torch.bool)
    all_padding = torch.zeros_like(valid)

    empty_loss = prototype_coverage_loss(attention, empty_target, valid)
    padding_loss = prototype_coverage_loss(
        attention, torch.ones_like(empty_target), all_padding)

    assert empty_loss.item() == 0.0
    assert padding_loss.item() == 0.0
    assert torch.isfinite(empty_loss)
    assert torch.isfinite(padding_loss)
    assert empty_loss.requires_grad == attention.requires_grad


def test_tpsc_diversity_penalizes_collapsed_slots_more_than_separated_slots():
    valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    collapsed = torch.zeros(1, 2, 2, 2)
    collapsed[:, :, 0, 0] = 1.0
    separated = torch.zeros_like(collapsed)
    separated[:, 0, 0, 0] = 1.0
    separated[:, 1, 1, 1] = 1.0

    collapsed_loss = prototype_diversity_loss(collapsed, valid)
    separated_loss = prototype_diversity_loss(separated, valid)
    padding_loss = prototype_diversity_loss(
        collapsed, torch.zeros_like(valid))

    assert collapsed_loss.ndim == 0
    assert collapsed_loss > separated_loss
    assert padding_loss.item() == 0.0
    assert torch.isfinite(padding_loss)
