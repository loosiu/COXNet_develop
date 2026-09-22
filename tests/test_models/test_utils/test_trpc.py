import torch

from mmdet.models.detectors.fusionnet_xo import FusionNetXO
from mmdet.models.utils.trpc import TRPC
from mmdet.models.utils.trpc import ThermalConditionedTRPC
from mmdet.models.utils.trpc import balanced_binary_focal_loss
from mmdet.models.utils.trpc import make_padding_mask
from mmdet.models.utils.fusion_strategy import FusionLayer


def _assert_nonzero_finite_gradient(parameter, name):
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all(), name
    assert torch.count_nonzero(parameter.grad) > 0, name


def _path_gradient_is_nonzero(module):
    gradients = [
        parameter.grad for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return gradients and any(torch.count_nonzero(grad) > 0 for grad in gradients)


def test_trpc_forward_backward_and_attention_contract():
    torch.manual_seed(7)
    module = TRPC(
        channels=32,
        embed_dim=16,
        num_prototypes=4,
        match_temperature=0.2,
        match_topk=1,
        mutual_matching=True,
        residual_scale=0.2)
    module.train()

    rgb = torch.randn(2, 32, 5, 7, requires_grad=True)
    thermal = torch.randn(2, 32, 10, 14, requires_grad=True)
    thermal_before = thermal.detach().clone()
    valid = torch.ones(2, 10, 14, dtype=torch.bool)
    valid[:, -1, :] = False

    output, aux = module(rgb, thermal, valid_mask=valid, return_aux=True)
    assert output.shape == thermal.shape
    assert torch.isfinite(output).all()
    assert torch.equal(thermal.detach(), thermal_before)  # Thermal is reference-only.
    assert aux['match_weights'].shape == (2, 4, 4)
    assert 0.0 <= aux['match_rate'].item() <= 1.0
    assert 0.0 <= aux['match_confidence'].item() <= 1.0

    # The reconstruction map is the RGB assignment over prototypes, gated by
    # RGB objectness, and invalid padded positions receive no assignment.
    where = aux['rgb_where_attention']
    obj = aux['rgb_objectness'].flatten(2)
    expected = obj.squeeze(1) * valid.flatten(1).to(obj.dtype)
    assert torch.allclose(where.sum(dim=1), expected, atol=1e-5, rtol=1e-4)
    assert torch.count_nonzero(where[:, :, -14:]) == 0
    assert torch.isfinite(aux['proto_rgb_cos_before'])
    assert torch.isfinite(aux['proto_rgb_cos_after'])
    assert 0.0 <= aux['attention_entropy_rgb'].item() <= 1.0
    assert 0.0 <= aux['attention_entropy_thermal'].item() <= 1.0
    assert 1.0 <= aux['prototype_usage'].item() <= 4.0

    # Zero-initialized residual actuator makes iteration-zero TRPC exactly the
    # retained CLFM DeConv baseline.
    with torch.no_grad():
        deconv_only = module.visible_upsample(rgb.detach())
    assert torch.allclose(output.detach(), deconv_only, atol=1e-6, rtol=1e-6)

    loss = output.square().mean() + aux['diversity_loss']
    loss.backward()
    required = (
        'visible_upsample.Deconv.weight',
        'rgb_extractor.embed.weight',
        'thermal_extractor.embed.weight',
        'delta_to_rgb.weight',
    )
    grads = dict(module.named_parameters())
    for name in required:
        assert grads[name].grad is not None, name
        assert torch.isfinite(grads[name].grad).all(), name
    _assert_nonzero_finite_gradient(
        grads['delta_to_rgb.weight'], 'delta_to_rgb.weight')


def test_thermal_reference_is_detached_with_active_residual_projection():
    """Stop-gradient must hold even when the residual actuator is active."""
    torch.manual_seed(13)
    module = TRPC(channels=16, embed_dim=8, num_prototypes=4)
    with torch.no_grad():
        module.delta_to_rgb.weight.fill_(0.01)

    rgb = torch.randn(2, 16, 4, 5, requires_grad=True)
    thermal = torch.randn(2, 16, 8, 10, requires_grad=True)
    output = module(rgb, thermal)
    deconv_only = module.visible_upsample(rgb)
    assert not torch.allclose(output, deconv_only)

    rgb_grad, thermal_grad = torch.autograd.grad(
        output.square().mean(), (rgb, thermal), allow_unused=True)
    assert rgb_grad is not None and torch.count_nonzero(rgb_grad) > 0
    assert thermal_grad is None or torch.count_nonzero(thermal_grad) == 0


def test_gate_and_delta_paths_learn_after_zero_init_optimizer_step():
    """The zero-init actuator must open learning paths after its first step."""
    torch.manual_seed(17)
    module = TRPC(channels=16, embed_dim=8, num_prototypes=4)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    rgb = torch.randn(2, 16, 4, 5)
    thermal = torch.randn(2, 16, 8, 10)

    first_output = module(rgb, thermal)
    first_output.square().mean().backward()
    _assert_nonzero_finite_gradient(
        module.delta_to_rgb.weight, 'delta_to_rgb.weight at first step')
    optimizer.step()
    assert torch.count_nonzero(module.delta_to_rgb.weight) > 0

    optimizer.zero_grad(set_to_none=True)
    second_output = module(rgb, thermal)
    second_output.square().mean().backward()
    assert _path_gradient_is_nonzero(module.gate_mlp)
    assert _path_gradient_is_nonzero(module.delta_mlp)


def test_thermal_auxiliary_losses_train_thermal_extractor():
    """Targetness and diversity retain learning signals for Thermal features."""
    torch.manual_seed(19)
    module = TRPC(channels=16, embed_dim=8, num_prototypes=4)
    rgb = torch.randn(2, 16, 4, 5)
    thermal = torch.randn(2, 16, 8, 10)
    _, aux = module(rgb, thermal, return_aux=True)

    logits = aux['thermal_objectness_logits']
    target = torch.zeros_like(logits)
    target[:, :, 2:6, 3:8] = 1.0
    valid = torch.ones_like(logits, dtype=torch.bool)
    targetness = balanced_binary_focal_loss(logits, target, valid)
    target_grads = torch.autograd.grad(
        targetness,
        (module.thermal_extractor.embed.weight,
         module.thermal_extractor.objectness_out.weight),
        retain_graph=True)
    for name, gradient in zip(
            ('thermal embed from targetness', 'thermal objectness head'),
            target_grads):
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name

    diversity_grad = torch.autograd.grad(
        aux['diversity_loss'], module.thermal_extractor.embed.weight)[0]
    assert torch.isfinite(diversity_grad).all()
    assert torch.count_nonzero(diversity_grad) > 0


def test_simple_test_passes_padding_geometry_to_trpc_path():
    """Padded feature cells must remain masked during simple inference."""
    class RecordingDetector:
        class EmptyHead:
            num_classes = 1

            @staticmethod
            def simple_test(feats, img_metas, rescale=False):
                batch = len(img_metas)
                return [
                    (torch.empty(0, 5), torch.empty(0, dtype=torch.long))
                    for _ in range(batch)
                ]

        bbox_head = EmptyHead()

        def extract_feat(self, img, img_metas=None):
            assert img_metas is not None
            shapes = [meta['img_shape'][:2] for meta in img_metas]
            padded = img_metas[0]['batch_input_shape']
            self.valid_mask = make_padding_mask(
                shapes, padded, (48, 80), img[1].device)
            return [img[1]]

    detector = RecordingDetector()
    padded_rgb = torch.randn(1, 3, 384, 640)
    padded_thermal = torch.randn(1, 3, 384, 640)
    img_metas = [dict(
        img_shape=(360, 640, 3),
        pad_shape=(384, 640, 3),
        batch_input_shape=(384, 640))]

    FusionNetXO.simple_test(
        detector, (padded_rgb, padded_thermal), img_metas)
    assert detector.valid_mask.shape == (1, 1, 48, 80)
    assert detector.valid_mask[:, :, :45].all()
    assert not detector.valid_mask[:, :, 45:].any()


def test_trpc_odd_resolution_fallback():
    module = TRPC(channels=16, embed_dim=8, num_prototypes=2)
    output = module(torch.randn(1, 16, 4, 5), torch.randn(1, 16, 9, 11))
    assert output.shape == (1, 16, 9, 11)


def test_same_stage_trpc_has_no_deconv_and_rejects_shape_mismatch():
    module = TRPC(
        channels=16, embed_dim=8, num_prototypes=2, same_stage=True)
    assert isinstance(module.visible_upsample, torch.nn.Identity)
    assert not any(
        name.startswith('visible_upsample.')
        for name, _ in module.named_parameters())

    rgb = torch.randn(1, 16, 9, 11)
    thermal = torch.randn(1, 16, 9, 11)
    output = module(rgb, thermal)
    assert output.shape == rgb.shape
    assert torch.allclose(output, rgb, atol=1e-6, rtol=1e-6)

    try:
        module(torch.randn(1, 16, 5, 6), thermal)
    except ValueError as error:
        assert 'requires equal RGB/Thermal feature shapes' in str(error)
    else:
        raise AssertionError('same-stage TRPC silently accepted mismatched shapes')


def test_thermal_conditioned_trpc_modulates_rgb_and_learns_end_to_end():
    torch.manual_seed(29)
    module = ThermalConditionedTRPC(
        channels=16, embed_dim=8, num_prototypes=4,
        epsilon=0.1, modulation_init_std=1e-3)
    rgb = torch.randn(2, 16, 6, 8, requires_grad=True)
    thermal = torch.randn(2, 16, 6, 8, requires_grad=True)
    valid = torch.ones(2, 6, 8, dtype=torch.bool)
    valid[:, -1] = False

    output, aux = module(rgb, thermal, valid_mask=valid, return_aux=True)
    assert output.shape == rgb.shape
    assert not torch.allclose(output[:, :, :-1], rgb[:, :, :-1])
    assert torch.equal(output[:, :, -1], rgb[:, :, -1])
    assert torch.allclose(
        aux['conditioning_attention'].sum(dim=-1), valid.flatten(1).float(),
        atol=1e-6, rtol=1e-6)
    assert 0.0 <= aux['conditioning_attention_entropy'].item() <= 1.0
    assert 1.0 <= aux['conditioning_prototype_usage'].item() <= 4.0
    assert aux['delta_ratio'].item() > 0
    assert aux['modulation_ratio'].item() > aux['delta_ratio'].item()
    assert torch.count_nonzero(module.film_out.weight) > 0
    assert 'epsilon' not in dict(module.named_parameters())
    assert not hasattr(module, 'rgb_extractor')
    assert not hasattr(module, 'gate_mlp')

    output.square().mean().backward()
    required = (
        ('rgb input', rgb),
        ('thermal input', thermal),
        ('RGB query', module.rgb_query.weight),
        ('Thermal extractor', module.thermal_extractor.embed.weight),
        ('Thermal assignment', module.thermal_extractor.assignment.weight),
        ('Thermal key', module.thermal_key.weight),
        ('Thermal value', module.thermal_value.weight),
        ('FiLM output', module.film_out.weight),
    )
    for name, tensor in required:
        _assert_nonzero_finite_gradient(tensor, name)

    try:
        module(rgb, torch.randn(2, 16, 3, 4))
    except ValueError as error:
        assert 'requires equal RGB/Thermal shapes' in str(error)
    else:
        raise AssertionError('conditioning TRPC accepted mismatched shapes')


def test_same_stage_trpc_in_fusion_slot_has_no_clfm_modules():
    layer = FusionLayer(
        in_channels=32,
        reduction=8,
        num_layers=2,
        fs_type='fusionnet-xo',
        use_clfm=[],
        use_trpc=True,
        trpc_cfg=dict(
            variant='thermal_conditioning',
            num_prototypes=4,
            embed_dim=16,
            targetness_loss_weight=0.1,
            diversity_loss_weight=0.0),
        usepoolup=[])
    assert layer.trpc_same_stage
    assert layer.trpc_variant == 'thermal_conditioning'
    assert not hasattr(layer, 'idwt_layers')
    assert all(
        isinstance(module, ThermalConditionedTRPC)
        for module in layer.trpc_layers)
    assert all(
        not hasattr(module, 'visible_upsample')
        for module in layer.trpc_layers)

    visible = [torch.randn(2, 32, 10, 14), torch.randn(2, 32, 6, 8)]
    thermal = [torch.randn(2, 32, 10, 14), torch.randn(2, 32, 6, 8)]
    features, aux = layer(
        visible, thermal,
        gt_bboxes=[torch.empty(0, 4), torch.empty(0, 4)],
        img_metas=[
            dict(
                img_shape=(80, 112, 3), pad_shape=(80, 112, 3),
                batch_input_shape=(80, 112))
            for _ in range(2)])
    assert [tuple(feature.shape) for feature in features] == [
        (2, 32, 10, 14), (2, 32, 6, 8)]
    assert 'loss_trpc_targetness' in aux
    assert 'trpc_delta_ratio' in aux
    assert 'trpc_conditioning_attention_entropy' in aux


def test_same_stage_trpc_preserves_control_hofm_initialization():
    """Adding TRPC must not change the same-seed AAM/HOFM initialization."""
    common = dict(
        in_channels=32,
        reduction=8,
        num_layers=2,
        fs_type='fusionnet-xo',
        use_clfm=[],
        usepoolup=[])
    torch.manual_seed(23)
    control = FusionLayer(use_trpc=False, **common)
    torch.manual_seed(23)
    treatment = FusionLayer(
        use_trpc=True,
        trpc_cfg=dict(
            variant='thermal_conditioning',
            num_prototypes=4,
            embed_dim=16,
            targetness_loss_weight=0.1,
            diversity_loss_weight=0.0),
        **common)

    control_state = control.hofm_layers.state_dict()
    treatment_state = treatment.hofm_layers.state_dict()
    assert control_state.keys() == treatment_state.keys()
    for name in control_state:
        assert torch.equal(control_state[name], treatment_state[name]), name


def test_trpc_in_coxnet_fusion_slot():
    torch.manual_seed(11)
    layer = FusionLayer(
        in_channels=32,
        reduction=8,
        num_layers=2,
        fs_type='fusionnet-xo',
        use_clfm=[],
        use_trpc=True,
        trpc_cfg=dict(
            num_prototypes=4,
            embed_dim=16,
            targetness_loss_weight=0.1,
            diversity_loss_weight=0.01),
        usepoolup=[])
    layer.train()
    visible = [torch.randn(2, 32, 5, 7), torch.randn(2, 32, 3, 4)]
    thermal = [torch.randn(2, 32, 10, 14), torch.randn(2, 32, 6, 8)]
    boxes = [torch.tensor([[16.0, 12.0, 34.0, 42.0]]),
             torch.tensor([[45.0, 30.0, 68.0, 60.0]])]
    metas = [dict(img_shape=(80, 112, 3), pad_shape=(80, 112, 3),
                  batch_input_shape=(80, 112)) for _ in range(2)]

    features, aux = layer(visible, thermal, gt_bboxes=boxes, img_metas=metas)
    assert [tuple(x.shape) for x in features] == [(2, 32, 10, 14), (2, 32, 6, 8)]
    assert 'loss_trpc_targetness' in aux
    assert 'loss_trpc_diversity' in aux
    assert 'trpc_match_rate' in aux
    total = sum(x.mean() for x in features)
    total = total + sum(v for k, v in aux.items() if 'loss' in k)
    total.backward()
    assert layer.trpc_layers[0].delta_to_rgb.weight.grad is not None
    assert layer.hofm_layers[0].conv.weight.grad is not None
