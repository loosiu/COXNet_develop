import torch

from mmdet.models.utils.trpc import TRPC
from mmdet.models.utils.fusion_strategy import FusionLayer


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
    # The calibration path cannot optimize the thermal reference.  Thermal is
    # still trained by COXNet's downstream HOFM and its own auxiliary losses.
    calibration_grad = torch.autograd.grad(
        output.square().mean(), thermal, retain_graph=True, allow_unused=True)[0]
    assert calibration_grad is None or torch.count_nonzero(calibration_grad) == 0
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
        'gate_mlp.0.weight',
        'delta_mlp.0.weight',
        'delta_to_rgb.weight',
    )
    grads = dict(module.named_parameters())
    for name in required:
        assert grads[name].grad is not None, name
        assert torch.isfinite(grads[name].grad).all(), name


def test_trpc_odd_resolution_fallback():
    module = TRPC(channels=16, embed_dim=8, num_prototypes=2)
    output = module(torch.randn(1, 16, 4, 5), torch.randn(1, 16, 9, 11))
    assert output.shape == (1, 16, 9, 11)


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
