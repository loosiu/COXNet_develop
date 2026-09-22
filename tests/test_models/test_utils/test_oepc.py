import torch

from mmdet.models.utils.fusion_strategy import FusionLayer
from mmdet.models.utils.oepc import ObjectCentricEvidentialCalibration
from mmdet.models.utils.oepc import build_center_targets


def _assert_nonzero_gradient(parameter, name):
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all(), name
    assert torch.count_nonzero(parameter.grad) > 0, name


def test_oepc_same_stage_rgb_only_calibration_and_losses():
    torch.manual_seed(101)
    module = ObjectCentricEvidentialCalibration(
        channels=16,
        embed_dim=8,
        object_kernel=3,
        context_kernel=5,
        search_radius=1,
        distance_prior_weight=0.2,
        candidate_threshold=0.0,
        max_candidates=2,
        residual_scale=0.2,
        modulation_init_std=1e-2)
    module.train()

    rgb = torch.randn(2, 16, 10, 12, requires_grad=True)
    thermal = torch.randn(2, 16, 10, 12, requires_grad=True)
    thermal_before = thermal.detach().clone()
    valid = torch.ones(2, 1, 10, 12, dtype=torch.bool)
    valid[:, :, -2:] = False
    center_target = torch.zeros(2, 1, 10, 12)
    center_target[0, :, 4, 5] = 1.0
    center_target[1, :, 3, 8] = 1.0
    foreground_target = torch.zeros_like(center_target)
    foreground_target[0, :, 3:6, 4:7] = 1.0
    foreground_target[1, :, 2:5, 7:10] = 1.0

    output, aux = module(
        rgb, thermal, valid_mask=valid, center_target=center_target,
        foreground_target=foreground_target, return_aux=True)
    assert output.shape == rgb.shape
    assert torch.equal(thermal.detach(), thermal_before)
    assert torch.equal(output[:, :, -2:], rgb[:, :, -2:])
    support = aux['spatial_support'].bool()
    outside = ~support.expand_as(output)
    assert torch.equal(output[outside], rgb[outside])
    assert not torch.allclose(output[support.expand_as(output)],
                              rgb[support.expand_as(rgb)])
    assert aux['thermal_candidate_peaks'].sum(dim=(1, 2, 3)).max() <= 2
    assert aux['rgb_candidate_peaks'].sum(dim=(1, 2, 3)).max() <= 2
    assert 0.0 < aux['support_ratio'].item() < 1.0
    assert 0.0 <= aux['matching_confidence_mean'].item() <= 1.0
    assert 0.0 <= aux['transfer_mean'].item() <= 1.0
    assert 0.0 <= aux['uncertainty_rgb_mean'].item() <= 1.0
    assert 0.0 <= aux['uncertainty_thermal_mean'].item() <= 1.0
    assert aux['delta_ratio'].item() > 0
    assert aux['residual_cap_ratio'].item() <= 1.0 + 1e-5
    assert aux['utility_payload'] is None
    assert aux['utility_sampled'].item() == 0
    for key in (
            'candidate_loss', 'foreground_loss', 'contrastive_loss',
            'edl_loss'):
        assert torch.isfinite(aux[key]), key

    loss = output.square().mean()
    loss = loss + aux['candidate_loss'] + aux['foreground_loss']
    loss = loss + aux['contrastive_loss'] + aux['edl_loss']
    loss.backward()
    required = (
        ('RGB input', rgb),
        ('Thermal input', thermal),
        ('RGB embedding', module.rgb_embed.weight),
        ('Thermal embedding', module.thermal_embed.weight),
        ('Thermal candidate head',
         module.thermal_candidate_head[-1].weight),
        ('RGB candidate head', module.rgb_candidate_head[-1].weight),
        ('Thermal foreground head',
         module.thermal_foreground_head[-1].weight),
        ('RGB foreground head', module.rgb_foreground_head[-1].weight),
        ('RGB evidence head', module.rgb_evidence_head.weight),
        ('Thermal evidence head', module.thermal_evidence_head.weight),
        ('router', module.router[-1].weight),
        ('FiLM output', module.film_out.weight),
        ('RGB contrast projection', module.rgb_contrast.weight),
        ('Thermal contrast projection', module.thermal_contrast.weight))
    for name, parameter in required:
        _assert_nonzero_gradient(parameter, name)


def test_oepc_rejects_cross_stage_features():
    module = ObjectCentricEvidentialCalibration(
        channels=8, embed_dim=4, object_kernel=3, context_kernel=5)
    try:
        module(torch.randn(1, 8, 5, 6), torch.randn(1, 8, 10, 12))
    except ValueError as error:
        assert 'equal same-stage RGB/Thermal shapes' in str(error)
    else:
        raise AssertionError('OEPC silently accepted cross-stage features')


def test_oepc_distance_prior_prefers_same_coordinate_for_equal_features():
    module = ObjectCentricEvidentialCalibration(
        channels=8, embed_dim=4, object_kernel=3, context_kernel=5,
        search_radius=1, distance_prior_weight=1.0)
    query = torch.zeros(1, 4, 5, 5)
    key = torch.zeros_like(query)
    valid = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    attention, _, _ = module._local_attention(query, key, valid)
    center_location = 2 * 5 + 2
    assert attention[0, :, center_location].argmax().item() == 4


def test_center_targets_use_thermal_box_centers():
    boxes = [torch.tensor([
        [16.0, 8.0, 32.0, 24.0],
        [48.0, 32.0, 64.0, 48.0]])]
    target = build_center_targets(
        boxes, padded_size=(64, 80), feat_size=(8, 10),
        device=torch.device('cpu'))
    assert target.sum().item() == 2
    assert target[0, 0, 2, 3] == 1
    assert target[0, 0, 5, 7] == 1


def test_oepc_replaces_clfm_and_preserves_thermal_aam_input():
    torch.manual_seed(103)
    layer = FusionLayer(
        in_channels=16,
        reduction=4,
        num_layers=2,
        fs_type='fusionnet-xo',
        use_clfm=[],
        use_trpc=False,
        use_oepc=True,
        oepc_cfg=dict(
            apply_levels=(0,),
            embed_dim=8,
            object_kernel=3,
            context_kernel=5,
            search_radius=1,
            candidate_threshold=0.0,
            max_candidates=4,
            targetness_loss_weight=0.1,
            foreground_loss_weight=0.1,
            contrastive_loss_weight=0.05,
            edl_loss_weight=0.01,
            utility_loss_weight=0.0,
            use_detector_utility=False),
        usepoolup=[])
    layer.train()

    assert not hasattr(layer, 'idwt_layers')
    assert not hasattr(layer, 'trpc_layers')
    assert set(layer.oepc_layers.keys()) == {'0'}
    assert not any(
        'deconv' in name.lower() or 'dwt' in name.lower()
        for name, _ in layer.oepc_layers.named_modules())

    visible = [
        torch.randn(2, 16, 10, 14),
        torch.randn(2, 16, 5, 7)]
    thermal = [
        torch.randn(2, 16, 10, 14),
        torch.randn(2, 16, 5, 7)]
    thermal_before = [feature.clone() for feature in thermal]
    boxes = [
        torch.tensor([[16.0, 12.0, 34.0, 42.0]]),
        torch.tensor([[45.0, 30.0, 68.0, 60.0]])]
    metas = [dict(
        img_shape=(80, 112, 3), pad_shape=(80, 112, 3),
        batch_input_shape=(80, 112)) for _ in range(2)]

    output = layer(
        visible, thermal, gt_bboxes=boxes, img_metas=metas)
    assert isinstance(output, tuple) and len(output) == 2
    features, aux = output
    assert [tuple(feature.shape) for feature in features] == [
        (2, 16, 10, 14), (2, 16, 5, 7)]
    for after, before in zip(thermal, thermal_before):
        assert torch.equal(after, before)
    for key in (
            'loss_oepc_targetness', 'loss_oepc_foreground',
            'loss_oepc_contrastive',
            'loss_oepc_edl',
            'oepc_delta_ratio', 'oepc_support_ratio'):
        assert key in aux
    assert aux['oepc_utility_sampled'].item() == 0


def test_oepc_candidate_selection_precedes_rgb_matching():
    module = ObjectCentricEvidentialCalibration(
        channels=8, embed_dim=4, object_kernel=3, context_kernel=5,
        search_radius=2, candidate_threshold=0.05, max_candidates=4)
    probability = torch.zeros(1, 1, 7, 7)
    probability[0, 0, 3, 3] = 0.7
    valid = torch.ones_like(probability, dtype=torch.bool)
    peak = module._sparse_support(probability, valid)
    # A diffuse 5x5 correspondence would dilute 0.7 to 0.028, but candidate
    # existence has already been decided in Thermal coordinates.
    assert peak[0, 0, 3, 3] == 1


def test_oepc_rgb_candidate_survives_missing_thermal_candidate():
    module = ObjectCentricEvidentialCalibration(
        channels=8, embed_dim=4, object_kernel=3, context_kernel=5,
        search_radius=1, candidate_threshold=0.5, max_candidates=2)
    with torch.no_grad():
        for head, bias in (
                (module.thermal_candidate_head, -20.0),
                (module.rgb_candidate_head, 20.0)):
            head[-1].weight.zero_()
            head[-1].bias.fill_(bias)
    rgb = torch.randn(1, 8, 7, 7)
    thermal = torch.randn_like(rgb)
    _, aux = module(rgb, thermal, return_aux=True)
    assert aux['candidate_count_thermal'].item() == 0
    assert aux['candidate_count_rgb'].item() > 0
    assert aux['candidate_support'].sum().item() > 0


def test_oepc_missing_context_does_not_fall_back_to_object_prototype():
    module = ObjectCentricEvidentialCalibration(
        channels=8, embed_dim=4, object_kernel=3, context_kernel=5)
    feature = torch.randn(1, 8, 7, 7)
    foreground = torch.ones(1, 1, 7, 7)
    uncertainty = torch.zeros_like(foreground)
    valid = torch.ones_like(foreground, dtype=torch.bool)
    _, _, difference, _, _, available = module._object_context(
        feature, foreground, uncertainty, valid)
    assert not available.any()
    assert torch.count_nonzero(difference) == 0


def test_oepc_initialization_follows_user_seed():
    common = dict(
        in_channels=16, reduction=4, num_layers=2,
        fs_type='fusionnet-xo', use_clfm=[], use_trpc=False,
        use_oepc=True,
        oepc_cfg=dict(apply_levels=(0,), embed_dim=8,
                      object_kernel=3, context_kernel=5),
        usepoolup=[])
    torch.manual_seed(109)
    first = FusionLayer(**common)
    torch.manual_seed(110)
    second = FusionLayer(**common)
    assert not torch.equal(
        first.oepc_layers['0'].rgb_embed.weight,
        second.oepc_layers['0'].rgb_embed.weight)


def test_oepc_preserves_control_hofm_initialization():
    common = dict(
        in_channels=16,
        reduction=4,
        num_layers=2,
        fs_type='fusionnet-xo',
        use_clfm=[],
        use_trpc=False,
        usepoolup=[])
    torch.manual_seed(107)
    control = FusionLayer(use_oepc=False, **common)
    torch.manual_seed(107)
    treatment = FusionLayer(
        use_oepc=True,
        oepc_cfg=dict(
            apply_levels=(0,), embed_dim=8,
            object_kernel=3, context_kernel=5),
        **common)
    control_state = control.hofm_layers.state_dict()
    treatment_state = treatment.hofm_layers.state_dict()
    assert control_state.keys() == treatment_state.keys()
    for name in control_state:
        assert torch.equal(control_state[name], treatment_state[name]), name
