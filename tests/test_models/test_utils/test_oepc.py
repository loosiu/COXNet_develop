import torch

from mmdet.models.utils.fusion_strategy import FusionLayer
from mmdet.models.utils.oepc import ObjectCentricEvidentialCalibration


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
        residual_scale=0.2,
        modulation_init_std=1e-2)
    module.train()

    rgb = torch.randn(2, 16, 10, 12, requires_grad=True)
    thermal = torch.randn(2, 16, 10, 12, requires_grad=True)
    thermal_before = thermal.detach().clone()
    valid = torch.ones(2, 1, 10, 12, dtype=torch.bool)
    valid[:, :, -2:] = False
    target = torch.zeros(2, 1, 10, 12)
    target[0, :, 3:6, 4:7] = 1.0
    target[1, :, 2:5, 7:10] = 1.0

    output, aux = module(
        rgb, thermal, valid_mask=valid, target=target, return_aux=True)
    assert output.shape == rgb.shape
    assert torch.equal(thermal.detach(), thermal_before)
    assert torch.equal(output[:, :, -2:], rgb[:, :, -2:])
    assert not torch.allclose(output[:, :, :-2], rgb[:, :, :-2])
    assert 0.0 <= aux['transfer_mean'].item() <= 1.0
    assert 0.0 <= aux['uncertainty_rgb_mean'].item() <= 1.0
    assert 0.0 <= aux['uncertainty_thermal_mean'].item() <= 1.0
    assert aux['delta_ratio'].item() > 0
    for key in ('candidate_loss', 'contrastive_loss', 'edl_loss'):
        assert torch.isfinite(aux[key]), key

    loss = output.square().mean()
    loss = loss + aux['candidate_loss']
    loss = loss + aux['contrastive_loss'] + aux['edl_loss']
    loss.backward()
    required = (
        ('RGB input', rgb),
        ('Thermal input', thermal),
        ('RGB embedding', module.rgb_embed.weight),
        ('Thermal embedding', module.thermal_embed.weight),
        ('candidate head', module.candidate_head[-1].weight),
        ('RGB evidence head', module.rgb_evidence_head.weight),
        ('Thermal evidence head', module.thermal_evidence_head.weight),
        ('router', module.router[-1].weight),
        ('FiLM output', module.film_out.weight))
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
            targetness_loss_weight=0.1,
            contrastive_loss_weight=0.05,
            edl_loss_weight=0.01),
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

    features, aux = layer(
        visible, thermal, gt_bboxes=boxes, img_metas=metas)
    assert [tuple(feature.shape) for feature in features] == [
        (2, 16, 10, 14), (2, 16, 5, 7)]
    for after, before in zip(thermal, thermal_before):
        assert torch.equal(after, before)
    for key in (
            'loss_oepc_targetness', 'loss_oepc_contrastive',
            'loss_oepc_edl', 'oepc_delta_ratio'):
        assert key in aux


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
