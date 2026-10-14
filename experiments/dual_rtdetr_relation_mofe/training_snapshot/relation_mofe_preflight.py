"""Three real detection updates for the direct-feature relation MoFE variant.

This is validation only. It never starts production training or saves updated
weights. A future train.py run initializes independently from public backbone
pretrained weights, using the same A recipe.
"""
from datetime import datetime, timezone
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from dyfclt.model import DyFCLT
from dyfclt.losses import build_criterion
from dyfclt.runtime import (load_config, seed_everything, make_dataset, make_loader,
    move_targets, resize_training_batch, warmup_factor, apply_optimizer_step)
from dyfclt.training import build_optimizer, build_ema, set_learning_rate
from dyfclt.frequency_training import sha256, source_manifest, gate_statistics


def main():
    cfg = load_config(ROOT/'configs/rgbtdroneperson_relation_direct_seed0.yaml')
    reference = json.loads(Path('/data/siwoo/DyFCLT/outputs/rgbtdroneperson_dual_rtdetr_seed0/config.json').read_text())
    current, baseline = copy.deepcopy(cfg), copy.deepcopy(reference)
    current['model'].pop('frequency_utility')
    for c in [current, baseline]: c['train'].pop('output_dir', None)
    assert current == baseline
    frozen = json.loads((ROOT/'artifacts/original_spatial_source_frozen.json').read_text())
    assert all(sha256(path) == digest for path, digest in frozen.items())
    seed_everything(0); a = DyFCLT(**reference['model']); rng = torch.get_rng_state()
    seed_everything(0); model = DyFCLT(**cfg['model'])
    assert torch.equal(rng, torch.get_rng_state())
    target = model.state_dict()
    for name, value in a.state_dict().items(): assert torch.equal(value, target[name]), name
    common_count = len(a.state_dict()); del a, target; gc.collect()
    assert torch.cuda.device_count() == 1 and os.environ['RESIDUAL_GPU_UUID'] == 'GPU-c1faf7b9-2a34-010b-9739-4712a6f089a2'
    model.cuda().train(); module = model.encoder.frequency_module
    assert module.output_mode == 'direct'
    params = dict(module.named_parameters()); assert not any('gamma' in name for name in params)
    before = {name: p.detach().clone() for name, p in params.items()}
    optimizer = build_optimizer(model, cfg['train']); ema = build_ema(model, cfg['train'])
    ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in ids for p in params.values())
    train, val = make_dataset(cfg, 'train', training=True), make_dataset(cfg, 'val')
    assert len(train) == 4900 and len(val) == 1225 and train.category_ids == val.category_ids == [0, 1, 2]
    criterion = build_criterion(3, **cfg['loss']).cuda()
    scaler = torch.cuda.amp.GradScaler(enabled=False); batches = iter(make_loader(train, 2, 0, True))
    steps = []
    for index, size in enumerate([(448, 576), (576, 704), (512, 640)]):
        rgb, thermal, targets = resize_training_batch(*next(batches), [size])
        rgb, thermal, targets = rgb.cuda(), thermal.cuda(), move_targets(targets, 'cuda:0')
        set_learning_rate(optimizer, cfg['train']['lr'], warmup_factor(index, 2000, .001))
        optimizer.zero_grad(set_to_none=True); started = time.monotonic()
        out = model(rgb, thermal, targets, return_frequency_aux=True)
        assert out['frequency_aux']['routing_weights'].shape == (2, 2, 4, size[0]//8, size[1]//8)
        losses = criterion(out, targets); loss = sum(losses.values()); assert torch.isfinite(loss)
        assert not any('utility' in name or 'kl' in name or 'mask' in name for name in losses)
        loss.backward()
        grads = {name: dict(finite=p.grad is not None and bool(torch.isfinite(p.grad).all()),
                    abs_sum=float(p.grad.abs().sum()) if p.grad is not None else None) for name, p in params.items()}
        assert all(v['finite'] and v['abs_sum'] > 0 for v in grads.values()), grads
        norm, applied = apply_optimizer_step(model, optimizer, scaler, .1, ema)
        assert applied and torch.isfinite(norm); torch.cuda.synchronize()
        row = dict(step=index+1, image_size=list(size), loss=float(loss.detach()), grad_norm=float(norm),
                   seconds=time.monotonic()-started, gradients=grads, gate_statistics=gate_statistics(out['frequency_aux']))
        steps.append(row)
        print(json.dumps({k: row[k] for k in ['step', 'image_size', 'loss', 'grad_norm', 'seconds']}), flush=True)
        del out, losses, loss
    changed = {name: not torch.equal(p, before[name]) for name, p in params.items()}
    assert all(changed.values()), changed
    with torch.no_grad():
        r, t, _ = next(iter(make_loader(val, 2, 0)))
        left, right = ema.module(r.cuda(), t.cuda()), ema.module(r.cuda(), t.cuda())
        assert all(torch.equal(left[k], right[k]) and torch.isfinite(left[k]).all() for k in ['pred_logits', 'pred_boxes'])
    proof = dict(verified_utc=datetime.now(timezone.utc).isoformat(), config=cfg,
        all_A_common_initial_tensors_equal=True, common_tensor_count=common_count, CPU_rng_equal=True,
        original_spatial_source_preserved=True, source_sha256=source_manifest(ROOT),
        all_module_gradients_finite_nonzero=True, all_module_parameters_updated=True,
        optimizer_includes_all_module_parameters=True, EMA_eval_finite_repeatable=True,
        mode='direct_feature_replacement', module_parameter_tensors=len(params), changed=changed, steps=steps,
        parameters=sum(p.numel() for p in model.parameters()), module_parameters=sum(p.numel() for p in module.parameters()),
        physical_gpu=1, gpu_uuid=os.environ['RESIDUAL_GPU_UUID'], pretrained_sha256=sha256(ROOT/cfg['model']['pretrained']),
        annotation_sha256={s: sha256(cfg['data'][s]['ann_file']) for s in ['train', 'val']},
        trained_A_B_checkpoint_loaded=False, production_training_launched=False, AP_measured=False)
    (ROOT/'artifacts/verification_relation_direct_gpu1.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(dict(verified=True, updated_tensors=sum(changed.values()), parameters=proof['parameters'],
                          module_parameters=proof['module_parameters'], production_training_launched=False)), flush=True)


if __name__ == '__main__': main()
