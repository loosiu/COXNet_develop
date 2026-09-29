#!/usr/bin/env python
"""Run one real RGBTDronePerson batch through ICBFC forward/backward."""

import argparse
import os

import torch
from mmcv import Config
from mmcv.parallel import scatter

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import update_data_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='configs/coxnet/icbfc/ICBFC.py')
    return parser.parse_args()


def _nonzero_finite_gradient(parameters, name):
    gradients = [parameter.grad for parameter in parameters
                 if parameter.requires_grad]
    if not gradients or any(gradient is None for gradient in gradients):
        raise RuntimeError(f'missing gradient: {name}')
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError(f'non-finite gradient: {name}')
    if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
        raise RuntimeError(f'zero gradient: {name}')


def mean_loss_value(value):
    """Match MMDetection's tensor-or-list loss reduction contract."""
    if isinstance(value, torch.Tensor):
        return value.mean()
    if isinstance(value, list):
        return sum(item.mean() for item in value)
    raise TypeError(f'unsupported loss value: {type(value)}')


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the ICBFC smoke test')

    cfg = Config.fromfile(args.config)
    update_data_root(cfg)
    cfg.model.backbone.init_cfg = None
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1,
        dist=False, shuffle=False, seed=0)

    batch = None
    for index, candidate in enumerate(loader):
        boxes = candidate['gt_bboxes'].data[0][0]
        if boxes.shape[0] >= 2:
            batch = candidate
            break
        if index >= 100:
            break
    if batch is None:
        raise RuntimeError(
            'no multi-instance training sample found in 100 batches')

    model = build_detector(cfg.model).cuda().train()
    batch = scatter(batch, [torch.cuda.current_device()])[0]
    losses = model(return_loss=True, **batch)
    optimized = [mean_loss_value(value) for key, value in losses.items()
                 if 'loss' in key]
    if not optimized:
        raise RuntimeError('model returned no optimized loss')
    total_loss = sum(optimized)
    if not torch.isfinite(total_loss):
        raise RuntimeError('non-finite total loss')
    total_loss.backward()

    prior = model.fuse_layer.icbfc_prior
    for name in ('center_head', 'offset_head', 'scale_head'):
        _nonzero_finite_gradient(getattr(prior, name).parameters(), name)
    for level_index, level in enumerate(model.fuse_layer.icbfc_layers):
        checks = dict(
            q_proj=level.q_proj,
            k_proj=level.k_proj,
            v_proj=level.v_proj,
            router=level.router,
            out_proj=level.out_proj,
            deconv=level.deconv)
        for name, module in checks.items():
            _nonzero_finite_gradient(
                module.parameters(), f'level{level_index}.{name}')

    count = losses['icbfc_candidate_count'].detach().item()
    recall = losses['icbfc_center_recall'].detach().item()
    router_variance = losses['icbfc_router_variance'].detach().item()
    delta_ratio = losses['icbfc_delta_ratio'].detach().item()
    if count <= 0:
        raise RuntimeError('no training instances reached ICBFC')
    if router_variance <= 0:
        raise RuntimeError('instance router variance is zero')
    if not 0 < delta_ratio < 0.2:
        raise RuntimeError(f'unsafe delta_ratio: {delta_ratio}')
    print(
        f'ICBFC_SMOKE_OK center_count={count:.1f} '
        f'center_recall={recall:.6f} '
        f'router_variance={router_variance:.8f} '
        f'delta_ratio={delta_ratio:.8f} '
        f'data_root={os.environ.get("MMDET_DATASETS", cfg.data_root)}')


if __name__ == '__main__':
    main()
