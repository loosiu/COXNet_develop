"""Real detection updates, initial B equality and spatial routing verification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dyfclt.model import DyFCLT
from dyfclt.losses import build_criterion
from dyfclt.runtime import (load_config, seed_everything, make_dataset, make_loader,
    move_targets, resize_training_batch, warmup_factor, apply_optimizer_step)
from dyfclt.training import build_optimizer, build_ema, set_learning_rate
from dyfclt.frequency_training import sha256, gate_statistics
from tools.spatial_mofe_protocol import (CONFIG, ORIGINAL, EXPECTED_UUID, check_recipe,
                                        manifest, verify_tests, verify_original)


def fingerprints(model):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def main():
    cfg = load_config(CONFIG); check_recipe(cfg); verify_tests()
    preserved = verify_original()
    reference_script = '''import hashlib,json,torch
from pathlib import Path
from dyfclt.model import DyFCLT
from dyfclt.runtime import seed_everything
c=json.loads(Path("outputs/rgbtdroneperson_B_pre_seed0/config.json").read_text())
seed_everything(c["seed"]);m=DyFCLT(**c["model"])
print(json.dumps({"hashes":{n:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,v in m.state_dict().items()},"rng":torch.get_rng_state().tolist()}))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    reference = json.loads(subprocess.check_output([sys.executable, '-c', reference_script],
                         cwd=ORIGINAL, env=env, text=True).splitlines()[-1])
    seed_everything(0); model = DyFCLT(**cfg['model'])
    initial = fingerprints(model)
    assert initial == reference['hashes'], 'Expert/router/common parameter initialization changed'
    assert torch.get_rng_state().tolist() == reference['rng'], 'Initialization RNG changed'
    assert torch.cuda.device_count() == 1 and os.environ['RESIDUAL_GPU_UUID'] == EXPECTED_UUID
    model.cuda().train(); module = model.encoder.frequency_module
    params = dict(module.named_parameters()); assert len(params) == 36
    before = {n: p.detach().clone() for n, p in params.items()}
    optimizer = build_optimizer(model, cfg['train']); ema = build_ema(model, cfg['train'])
    ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in ids for p in params.values())
    train, val = make_dataset(cfg, 'train', training=True), make_dataset(cfg, 'val')
    assert len(train) == 4900 and len(val) == 1225 and train.category_ids == val.category_ids == [0, 1, 2]
    criterion = build_criterion(3, **cfg['loss']).cuda()
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    batches = iter(make_loader(train, 2, 0, True)); steps = []
    for index, size in enumerate([(448, 576), (576, 704), (512, 640)]):
        rgb, thermal, targets = resize_training_batch(*next(batches), [size])
        rgb, thermal, targets = rgb.cuda(), thermal.cuda(), move_targets(targets, 'cuda:0')
        set_learning_rate(optimizer, cfg['train']['lr'], warmup_factor(index, 2000, .001))
        optimizer.zero_grad(set_to_none=True); started = time.monotonic()
        out = model(rgb, thermal, targets, return_frequency_aux=True)
        weights = out['frequency_aux']['routing_weights']
        assert weights.shape == (2, 2, 4, size[0]//8, size[1]//8)
        assert torch.allclose(weights.sum(2), torch.ones_like(weights[:, :, 0]), atol=1e-6)
        losses = criterion(out, targets); loss = sum(losses.values()); assert torch.isfinite(loss)
        assert not any('utility' in n or 'gain' in n or 'mask' in n or 'kl' in n for n in losses)
        loss.backward()
        grads = {n: {'finite': p.grad is not None and bool(torch.isfinite(p.grad).all()),
                     'abs_sum': float(p.grad.abs().sum()) if p.grad is not None else None} for n, p in params.items()}
        assert all(x['finite'] and x['abs_sum'] > 0 for x in grads.values()), grads
        norm, applied = apply_optimizer_step(model, optimizer, scaler, .1, ema)
        assert applied and torch.isfinite(norm); torch.cuda.synchronize()
        stats = gate_statistics(out['frequency_aux'])
        assert all(sum(s['weight_spatial_std_mean']) > 0 for s in stats.values())
        row = dict(step=index+1, image_size=list(size), loss=float(loss.detach()), grad_norm=float(norm),
                   seconds=time.monotonic()-started, gradients=grads, gate_statistics=stats)
        steps.append(row)
        print(json.dumps({k: row[k] for k in ['step', 'image_size', 'loss', 'grad_norm', 'seconds']}), flush=True)
        del out, losses, loss
    changed = {n: not torch.equal(p, before[n]) for n, p in params.items()}
    assert all(changed.values()), changed
    with torch.no_grad():
        r, t, _ = next(iter(make_loader(val, 2, 0)))
        left, right = ema.module(r.cuda(), t.cuda()), ema.module(r.cuda(), t.cuda())
        assert all(torch.equal(left[k], right[k]) and torch.isfinite(left[k]).all() for k in ['pred_logits', 'pred_boxes'])
    proof = dict(resolved_config=cfg, source_sha256=manifest(), all_B_pre_initial_tensors_equal=True,
        initial_tensor_count=len(initial), CPU_rng_equal=True, original_source_files_preserved=preserved,
        cpu_tests_passed=True, all_module_gradients_finite_nonzero=True, all_module_parameters_updated=True,
        optimizer_includes_all_module_parameters=True, EMA_eval_finite_repeatable=True, spatial_weights_verified=True,
        changed=changed, steps=steps, parameters=sum(p.numel() for p in model.parameters()),
        module_parameters=sum(p.numel() for p in module.parameters()), physical_gpu=1, gpu_uuid=EXPECTED_UUID,
        pretrained_sha256=sha256(ROOT/cfg['model']['pretrained']),
        input_sha256={s: sha256(cfg['data'][s]['ann_file']) for s in ['train', 'val']},
        trained_A_B_E_checkpoint_loaded=False, preflight_updated_weights_used_for_production=False, AP_measured=False)
    (ROOT/'artifacts/verification_spatial_gpu1.json').write_text(json.dumps(proof, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(verified=True, changed_tensors=sum(changed.values()), parameters=proof['parameters'])), flush=True)


if __name__ == '__main__': main()
