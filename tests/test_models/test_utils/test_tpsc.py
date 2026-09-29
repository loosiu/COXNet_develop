import torch
import torch.nn as nn
from mmcv import Config

from mmdet.models.detectors.fusionnet_xo import FusionNetXO
from mmdet.models.utils.fusion_strategy import FusionLayer
from mmdet.models.utils.tpsc import (
    PrototypeRelationBlock,
    SharedSlotPrototypeExtractor,
    TPSCDescriptor,
    TinyAwarePrototypeSemanticCalibration,
    build_tpsc_gaussian_targets,
    prototype_coverage_loss,
    prototype_diversity_loss,
    valid_average_pool,
)


def _tpsc_inputs(batch=2, channels=4):
    torch.manual_seed(19)
    rgb = (
        torch.randn(batch, channels, 4, 4, requires_grad=True),
        torch.randn(batch, channels, 2, 2, requires_grad=True),
    )
    thermal = (
        torch.randn(batch, channels, 4, 4, requires_grad=True),
        torch.randn(batch, channels, 2, 2, requires_grad=True),
    )
    valid = (
        torch.ones(batch, 1, 4, 4, dtype=torch.bool),
        torch.ones(batch, 1, 2, 2, dtype=torch.bool),
    )
    targets = (
        torch.zeros(batch, 1, 4, 4),
        torch.zeros(batch, 1, 2, 2),
    )
    targets[0][:, :, 1, 1] = 1.0
    targets[1][:, :, 0, 0] = 1.0
    return rgb, thermal, valid, targets


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


def test_tpsc_core_uses_same_slot_thermal_conditioning_without_relation():
    rgb, thermal, valid, targets = _tpsc_inputs()
    model = TinyAwarePrototypeSemanticCalibration(
        channels=4, prototype_dim=4, num_slots=2, num_heads=2,
        relation_depth=1, use_relation=False)

    calibrated, aux = model(
        rgb, thermal, valid_masks=valid, coverage_targets=targets,
        return_aux=True)

    assert len(calibrated) == 2
    assert calibrated[0].shape == rgb[0].shape
    assert calibrated[1].shape == rgb[1].shape
    assert aux['cross_modal_attention_mass'].item() == 0.0
    assert aux['cross_scale_attention_mass'].item() == 0.0


def test_tpsc_relation_connects_modalities_and_scales_with_expected_shapes():
    block = PrototypeRelationBlock(dim=4, num_heads=2)
    nodes = torch.randn(2, 8, 4)
    output, weights = block(nodes)

    assert output.shape == nodes.shape
    assert weights.shape == (2, 2, 8, 8)
    assert not torch.allclose(output, nodes)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 2, 8), atol=1e-6)


def test_tpsc_modifies_only_rgb_and_respects_fixed_ten_percent_bound():
    rgb, thermal, valid, targets = _tpsc_inputs()
    thermal_before = tuple(feature.detach().clone() for feature in thermal)
    model = TinyAwarePrototypeSemanticCalibration(
        channels=4, prototype_dim=4, num_slots=2, num_heads=2,
        modulation_bound=0.1, init_std=1e-2)

    calibrated, aux = model(
        rgb, thermal, valid_masks=valid, coverage_targets=targets,
        return_aux=True)

    for original, output in zip(rgb, calibrated):
        nonzero = original.abs() > 1e-6
        ratio = output[nonzero] / original[nonzero]
        assert ratio.min() >= 0.9 - 1e-6
        assert ratio.max() <= 1.1 + 1e-6
        assert not torch.equal(original, output)
    for before, after in zip(thermal_before, thermal):
        assert torch.equal(before, after.detach())
    assert aux['channel_scale_abs_max'] <= 0.1 + 1e-6
    assert aux['modulation_ratio'] > 0


def test_tpsc_nonzero_initialization_sends_detection_gradient_to_both_modalities():
    rgb, thermal, valid, targets = _tpsc_inputs()
    model = TinyAwarePrototypeSemanticCalibration(
        channels=4, prototype_dim=4, num_slots=2, num_heads=2,
        init_std=1e-2, use_relation=True)
    calibrated = model(rgb, thermal, valid_masks=valid)

    loss = sum(feature.square().mean() for feature in calibrated)
    loss.backward()
    diagnostics = model.gradient_diagnostics()

    assert sum(feature.grad.abs().sum() for feature in rgb).item() > 0
    assert sum(feature.grad.abs().sum() for feature in thermal).item() > 0
    for name in ('rgb_projection', 'thermal_projection', 'relation',
                 'conditioner', 'modulation'):
        assert diagnostics['grad_norm_' + name] > 0


def test_tpsc_disable_modulation_is_exact_rgb_identity():
    rgb, thermal, valid, _ = _tpsc_inputs()
    model = TinyAwarePrototypeSemanticCalibration(
        channels=4, prototype_dim=4, num_slots=2, num_heads=2)

    calibrated = model(
        rgb, thermal, valid_masks=valid, disable_modulation=True)

    assert all(torch.equal(before, after)
               for before, after in zip(rgb, calibrated))


def test_tpsc_disable_relation_matches_core_path():
    rgb, thermal, valid, _ = _tpsc_inputs()
    model = TinyAwarePrototypeSemanticCalibration(
        channels=4, prototype_dim=4, num_slots=2, num_heads=2,
        use_relation=True)

    disabled = model(
        rgb, thermal, valid_masks=valid, disable_relation=True)
    original_flag = model.use_relation
    model.use_relation = False
    core = model(rgb, thermal, valid_masks=valid)
    model.use_relation = original_flag

    assert all(torch.equal(left, right) for left, right in zip(disabled, core))


def test_tpsc_shuffle_thermal_prototypes_changes_batch_two_and_is_noop_for_batch_one():
    rgb, thermal, valid, _ = _tpsc_inputs(batch=2)
    model = TinyAwarePrototypeSemanticCalibration(
        channels=4, prototype_dim=4, num_slots=2, num_heads=2)

    normal = model(rgb, thermal, valid_masks=valid)
    shuffled = model(
        rgb, thermal, valid_masks=valid, shuffle_thermal_prototypes=True)
    assert any(not torch.equal(left, right)
               for left, right in zip(normal, shuffled))

    rgb1 = tuple(feature[:1] for feature in rgb)
    thermal1 = tuple(feature[:1] for feature in thermal)
    valid1 = tuple(mask[:1] for mask in valid)
    normal1 = model(rgb1, thermal1, valid_masks=valid1)
    shuffled1 = model(
        rgb1, thermal1, valid_masks=valid1,
        shuffle_thermal_prototypes=True)
    assert all(torch.equal(left, right)
               for left, right in zip(normal1, shuffled1))


def test_tpsc_forward_handles_empty_gt_partial_padding_and_all_padding_without_nan():
    rgb, thermal, valid, targets = _tpsc_inputs()
    valid[0][0, :, :, -1] = False
    valid[0][1] = False
    valid[1][1] = False
    targets = tuple(torch.zeros_like(target) for target in targets)
    model = TinyAwarePrototypeSemanticCalibration(
        channels=4, prototype_dim=4, num_slots=2, num_heads=2)

    calibrated, aux = model(
        rgb, thermal, valid_masks=valid, coverage_targets=targets,
        return_aux=True)

    assert all(torch.isfinite(feature).all() for feature in calibrated)
    assert all(torch.isfinite(value).all() for value in aux.values()
               if torch.is_tensor(value))
    assert aux['coverage_loss'].item() == 0.0


def test_tpsc_rejects_wrong_level_count_and_modality_shape_mismatch():
    model = TinyAwarePrototypeSemanticCalibration(
        channels=4, prototype_dim=4, num_slots=2, num_heads=2)
    rgb, thermal, valid, _ = _tpsc_inputs()

    for bad_rgb, bad_thermal, bad_valid in (
            (rgb[:1], thermal[:1], valid[:1]),
            (rgb, (thermal[0], thermal[1][:, :, :1]), valid)):
        try:
            model(bad_rgb, bad_thermal, valid_masks=bad_valid)
        except ValueError as error:
            assert 'level' in str(error).lower() or 'shape' in str(error).lower()
        else:
            raise AssertionError('TPSC accepted invalid feature levels')


class _CaptureTPSCFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.rgb_inputs = []
        self.thermal_inputs = []

    def forward(self, rgb, thermal):
        self.rgb_inputs.append(rgb.detach().clone())
        self.thermal_inputs.append(thermal.detach().clone())
        return rgb + thermal


def _tpsc_fusion_layer(**kwargs):
    defaults = dict(
        in_channels=8, reduction=4, num_layers=4,
        fs_type='fusionnet-xo', use_clfm=[], use_trpc=False,
        use_oepc=False, use_topc=False, use_prldfc=False, use_tpsc=True,
        tpsc_cfg=dict(
            apply_levels=(0, 1), prototype_dim=4, num_slots=2,
            num_heads=2, relation_depth=1, modulation_bound=0.1,
            init_std=1e-2, use_relation=True,
            coverage_loss_weight=0.05, diversity_loss_weight=0.01),
        usepoolup=[])
    defaults.update(kwargs)
    return FusionLayer(**defaults)


def test_fusion_layer_rejects_tpsc_with_any_other_clfm_replacement():
    conflicts = (
        dict(use_clfm=['v3']),
        dict(use_trpc=True),
        dict(use_oepc=True),
        dict(use_topc=True),
        dict(use_prldfc=True),
    )
    for conflicting in conflicts:
        try:
            _tpsc_fusion_layer(**conflicting)
        except ValueError as error:
            assert ('mutually exclusive' in str(error) or
                    'replaces CLFM' in str(error))
        else:
            raise AssertionError('TPSC accepted a conflicting fusion path')


def test_fusion_layer_tpsc_calibrates_p3_p4_before_hofm_and_preserves_thermal_values():
    layer = _tpsc_fusion_layer()
    captures = nn.ModuleList([_CaptureTPSCFusion() for _ in range(4)])
    layer.hofm_layers = captures
    layer.eval()
    visible = [
        torch.randn(1, 8, 8, 10), torch.randn(1, 8, 4, 5),
        torch.randn(1, 8, 2, 3), torch.randn(1, 8, 1, 2)]
    thermal = [torch.randn_like(feature) for feature in visible]
    thermal_before = [feature.clone() for feature in thermal]

    outputs = layer(visible, thermal)

    assert len(outputs) == 4
    assert not torch.equal(captures[0].rgb_inputs[0], visible[0])
    assert not torch.equal(captures[1].rgb_inputs[0], visible[1])
    assert torch.equal(captures[2].rgb_inputs[0], visible[2])
    assert torch.equal(captures[3].rgb_inputs[0], visible[3])
    for capture, before, after in zip(captures, thermal_before, thermal):
        assert torch.equal(capture.thermal_inputs[0], before)
        assert torch.equal(after, before)


def test_fusion_layer_tpsc_builds_level_targets_and_weighted_losses():
    layer = _tpsc_fusion_layer()
    layer.hofm_layers = nn.ModuleList(
        [_CaptureTPSCFusion() for _ in range(4)])
    observed = {}
    handle = layer.tpsc.register_forward_hook(
        lambda _module, _inputs, output: observed.update(output[1]))
    layer.train()
    visible = [
        torch.randn(1, 8, 8, 10), torch.randn(1, 8, 4, 5),
        torch.randn(1, 8, 2, 3), torch.randn(1, 8, 1, 2)]
    thermal = [torch.randn_like(feature) for feature in visible]
    boxes = [torch.tensor([[8.0, 8.0, 18.0, 20.0]])]
    metas = [dict(
        img_shape=(48, 64, 3), pad_shape=(64, 80, 3),
        batch_input_shape=(64, 80))]

    _, aux = layer(visible, thermal, gt_bboxes=boxes, img_metas=metas)
    handle.remove()

    assert torch.allclose(
        aux['loss_tpsc_coverage'], 0.05 * observed['coverage_loss'])
    assert torch.allclose(
        aux['loss_tpsc_diversity'], 0.01 * observed['diversity_loss'])
    assert 'tpsc_modulation_ratio' in aux
    assert 'tpsc_grad_norm_modulation' in aux


def test_fusion_layer_tpsc_inference_receives_padding_masks_and_sets_last_aux():
    layer = _tpsc_fusion_layer()
    layer.hofm_layers = nn.ModuleList(
        [_CaptureTPSCFusion() for _ in range(4)])
    masks = []
    handles = [extractor.register_forward_hook(
        lambda _module, inputs, _output: masks.append(inputs[2].detach()))
        for extractor in layer.tpsc.slot_extractors]
    layer.eval()
    visible = [
        torch.randn(1, 8, 8, 10), torch.randn(1, 8, 4, 5),
        torch.randn(1, 8, 2, 3), torch.randn(1, 8, 1, 2)]
    thermal = [torch.randn_like(feature) for feature in visible]
    metas = [dict(
        img_shape=(48, 64, 3), pad_shape=(64, 80, 3),
        batch_input_shape=(64, 80))]

    layer(visible, thermal, img_metas=metas)
    for handle in handles:
        handle.remove()

    assert len(masks) == 2
    assert not masks[0][:, :, 6:].any()
    assert not masks[0][:, :, :, 8:].any()
    assert layer.last_tpsc_aux is not None
    assert not layer.last_tpsc_aux['modulation_ratio'].requires_grad


def test_fusionnet_xo_forwards_tpsc_flags_and_enters_aux_train_path():
    class RecordingHead:

        def forward_train(self, features, *args):
            self.features = features
            return {'loss_detector': torch.tensor(1.0)}

    class RecordingDetector:
        wf_loss = False
        use_trpc = False
        use_oepc = False
        use_topc = False
        use_prldfc = False
        use_tpsc = True

        def __init__(self):
            self.bbox_head = RecordingHead()
            self.extract_args = None

        def extract_feat(self, img, *args, **kwargs):
            self.extract_args = (args, kwargs)
            return ['p3', 'p4'], {'loss_tpsc_coverage': torch.tensor(0.5)}

    detector = RecordingDetector()
    images = (torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16))
    metas = [dict(img_shape=(16, 16, 3), pad_shape=(16, 16, 3))]
    boxes = [torch.tensor([[2.0, 2.0, 8.0, 8.0]])]
    labels = [torch.tensor([0])]

    losses = FusionNetXO.forward_train(
        detector, images, metas, boxes, labels)

    assert detector.extract_args == (
        (boxes, metas), {'gt_bboxes_ignore': None})
    assert detector.bbox_head.features == ['p3', 'p4']
    assert set(losses) == {'loss_detector', 'loss_tpsc_coverage'}


def test_tpsc_configs_define_matched_control_core_and_relation_contracts():
    paths = (
        'configs/coxnet/tpsc/same_stage_control.py',
        'configs/coxnet/tpsc/TPSC_core.py',
        'configs/coxnet/tpsc/TPSC_relation.py')
    control, core, relation = [Config.fromfile(path) for path in paths]

    assert len({config.work_dir for config in (control, core, relation)}) == 3
    for config in (control, core, relation):
        model = config.model
        assert model.neck.start_level == model.neck_t.start_level == 1
        assert model.use_clfm == []
        for legacy in ('use_trpc', 'use_oepc', 'use_topc', 'use_prldfc'):
            assert model[legacy] is False
        assert model.wf_loss is True
        assert model.wf_loss_mode == 'kl_v2'
        assert model.wf_loss_weight == 0.1
    assert control.model.use_tpsc is False
    for config, use_relation in ((core, False), (relation, True)):
        model = config.model
        assert model.use_tpsc is True
        expected = dict(
            apply_levels=(0, 1), prototype_dim=64, num_slots=8,
            num_heads=4, relation_depth=1, modulation_bound=0.1,
            init_std=1e-2, use_relation=use_relation,
            coverage_loss_weight=0.05, diversity_loss_weight=0.01)
        for key, value in expected.items():
            assert model.tpsc_cfg[key] == value
