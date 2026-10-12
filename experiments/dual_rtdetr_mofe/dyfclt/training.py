"""Optional RT-DETR optimizer groups and EMA, with explicit legacy defaults.

Reference: lyuwenyu/RT-DETR rtdetr_pytorch, commit
29320b6fd828f8e0987a71426cf2d961b09dfed7, include/optimizer.yml and
src/optim/ema.py. Names match the reference regex, including its treatment
of anonymously named normalization layers. EMA averages floating state only.
"""
from copy import deepcopy
import math
import re

import torch


def build_optimizer(model, settings):
    lr = float(settings['lr'])
    wd = float(settings['weight_decay'])
    mode = settings.get('optimizer_groups', 'legacy')
    if mode == 'legacy':
        return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=lr, weight_decay=wd)
    if mode != 'rtdetr':
        raise ValueError(f'Unknown optimizer_groups: {mode}')
    multiplier = float(settings.get('backbone_lr_multiplier', .1))
    if multiplier <= 0:
        raise ValueError('backbone_lr_multiplier must be positive')
    buckets = {k: [] for k in ('backbone', 'encoder_no_decay', 'decoder_no_decay', 'default')}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if 'backbone' in name:
            bucket = 'backbone'
        elif name.startswith('encoder.') and re.search(r'bias|norm.*weight', name):
            bucket = 'encoder_no_decay'
        elif name.startswith('decoder.') and re.search(r'bias|norm.*weight', name):
            bucket = 'decoder_no_decay'
        else:
            bucket = 'default'
        buckets[bucket].append(parameter)
    groups = []
    for name, parameters in buckets.items():
        if parameters:
            scale = multiplier if name == 'backbone' else 1.
            groups.append(dict(params=parameters, group_name=name, lr_scale=scale,
                               lr=lr * scale, weight_decay=0. if name.endswith('no_decay') else wd))
    return torch.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(.9, .999))


def set_learning_rate(optimizer, base_lr, factor):
    """Retain parameter-group ratios through warmup and checkpoint resume."""
    for group in optimizer.param_groups:
        group['lr'] = float(base_lr) * group.get('lr_scale', 1.) * factor


class ModelEMA:
    """Single-device EMA with the official RT-DETR exponential warmup schedule."""
    def __init__(self, model, decay=.9999, warmups=2000):
        if not 0 <= decay < 1 or warmups <= 0:
            raise ValueError('EMA requires 0 <= decay < 1 and positive warmups')
        self.module = deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay = float(decay)
        self.warmups = int(warmups)
        self.updates = 0
        self.constant_keys = {('.'.join((prefix, name)) if prefix else name)
                              for prefix, module in model.named_modules()
                              for name in getattr(module, 'ema_constant_buffers', ())}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.warmups))
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            if name in self.constant_keys:
                value.copy_(source[name].detach())
            elif value.dtype.is_floating_point:
                value.mul_(decay).add_(source[name].detach(), alpha=1 - decay)

    def state_dict(self):
        return dict(module=self.module.state_dict(), updates=self.updates,
                    decay=self.decay, warmups=self.warmups)

    def load_state_dict(self, state):
        self.module.load_state_dict(state['module'], strict=True)
        self.updates = int(state['updates'])
        self.decay = float(state.get('decay', self.decay))
        self.warmups = int(state.get('warmups', self.warmups))


def build_ema(model, settings):
    options = dict(settings.get('ema', {}))
    if not options.pop('enabled', False):
        return None
    return ModelEMA(model, **options)


def optimizer_summary(optimizer):
    return [dict(name=g.get('group_name', 'legacy'), lr=g['lr'],
                 lr_scale=g.get('lr_scale', 1.), weight_decay=g['weight_decay'],
                 parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups]
