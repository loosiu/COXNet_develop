"""Disposable real-data GPU0 detection updates; never save smoke-trained weights."""
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/data/siwoo/DualRTDETR-MoFE-Staged')
sys.path.insert(0, str(ROOT))
import torch
from dyfclt.model import DyFCLT
from dyfclt.losses import build_criterion
from dyfclt.training import build_optimizer, build_ema, set_learning_rate
from dyfclt.runtime import (load_config, seed_everything, make_dataset, make_loader,
                           move_targets, resize_training_batch, warmup_factor, apply_optimizer_step)
from dyfclt.frequency_training import sha256, source_manifest, gate_statistics
from tools.staged_queue import GpuLocks, probe_cuda_mapping, atomic_json
from tools.train import validate_detection_config

GPU0_UUID = 'GPU-13eb8897-9466-51a2-84eb-e2d28ce8deba'
CONFIG = 'configs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0.yaml'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('Preflight requires physical GPU0 numeric mapping')
    locks = GpuLocks(sorted(set(Path('/tmp').glob('*_gpu0.lock')) | {Path('/tmp/self_band_mofe_gpu0.lock')}))
    if not locks.acquire():raise RuntimeError('GPU0 resource lock is held')
    try:
        jobs = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
        if GPU0_UUID in jobs:raise RuntimeError('Preserve existing GPU0 occupants')
        mapping = probe_cuda_mapping(os.environ, GPU0_UUID)
        torch.set_num_threads(4)
        cfg = load_config(ROOT/CONFIG);validate_detection_config(cfg)
        source_before = source_manifest(ROOT)
        plain_cfg = copy.deepcopy(cfg);plain_cfg['model'].pop('frequency_utility')
        seed_everything(0)
        plain = DyFCLT(**plain_cfg['model'])
        shared = {n:v.clone() for n,v in plain.state_dict().items()}
        plain_rng = torch.get_rng_state().clone()
        del plain;gc.collect()
        seed_everything(0)
        model = DyFCLT(**cfg['model'])
        assert torch.equal(torch.get_rng_state(),plain_rng)
        assert all(torch.equal(model.state_dict()[n],v) for n,v in shared.items())
        shared_count = len(shared);del shared;gc.collect()
        model.cuda();module=model.encoder.frequency_module
        criterion=build_criterion(cfg['model']['num_classes'],**cfg.get('loss',{})).cuda()
        settings=cfg['train'];optimizer=build_optimizer(model,settings)
        optimizer_ids={id(p) for group in optimizer.param_groups for p in group['params']}
        assert all(id(p) in optimizer_ids for p in module.parameters())
        ema=build_ema(model,settings);scaler=torch.cuda.amp.GradScaler(enabled=False)
        train=make_dataset(cfg,'train',training=True);val=make_dataset(cfg,'val')
        assert len(train)==4900 and len(val)==1225 and train.category_ids==val.category_ids==[0,1,2]
        rgb,thermal,targets=next(iter(make_loader(train,2,0,True)))
        rgb,thermal=rgb.cuda(),thermal.cuda()
        model.eval()
        with torch.no_grad():
            initial=model(rgb,thermal,return_frequency_aux=True)
            model.encoder.frequency_module=None
            baseline=model(rgb,thermal)
            model.encoder.frequency_module=module
            assert all(torch.equal(initial[k],baseline[k]) for k in ('pred_logits','pred_boxes'))
        del initial,baseline
        initial_module={n:p.detach().clone() for n,p in module.named_parameters()}
        proof=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),physical_gpu=0,
                   gpu_uuid=GPU0_UUID,pid=os.getpid(),mapping_probe=mapping,
                   shared_A_initialization_exact=True,shared_A_tensors=shared_count,CPU_rng_equal=True,
                   initial_prediction_equals_bypass=True,optimizer_covers_all_module_parameters=True,
                   parameters=sum(p.numel() for p in model.parameters()),
                   module_parameters=sum(p.numel() for p in module.parameters()),
                   module_parameter_tensors=len(initial_module),config=cfg,source_sha256=source_before,
                   categories=train.category_ids,train_images=len(train),val_images=len(val),
                   trained_checkpoint_loaded=False,updated_weights_saved=False,updates=[])
        observed=set();model.train()
        for step in range(6):
            size=settings['resize_sizes'][step%len(settings['resize_sizes'])]
            r,t,y=resize_training_batch(rgb,thermal,copy.deepcopy(targets),[size]);y=move_targets(y,'cuda:0')
            factor=warmup_factor(step,settings['warmup_steps'],settings['warmup_factor'])
            set_learning_rate(optimizer,settings['lr'],factor)
            optimizer.zero_grad(set_to_none=True)
            outputs=model(r,t,y,return_frequency_aux=True)
            losses=criterion(outputs,y);loss=sum(losses.values());assert bool(torch.isfinite(loss))
            scaler.scale(loss).backward();nonzero=set()
            for n,p in module.named_parameters():
                assert p.grad is not None and bool(torch.isfinite(p.grad).all()),n
                if p.grad.abs().max()>0:nonzero.add(n)
            if step==0:assert nonzero=={'gamma_rgb','gamma_thermal'},nonzero
            observed|=nonzero
            norm,applied=apply_optimizer_step(model,optimizer,scaler,settings['clip_grad_norm'],ema=ema)
            assert applied and bool(torch.isfinite(norm))
            record=dict(step=step+1,loss=float(loss.detach()),grad_norm=float(norm),image_size=size,
                        nonzero_module_gradient_tensors=len(nonzero),gamma=[float(module.gamma_rgb),float(module.gamma_thermal)],
                        statistics=gate_statistics(outputs['frequency_aux']))
            proof['updates'].append(record);print(json.dumps(record),flush=True)
            del outputs,loss,losses
        assert observed==set(initial_module),sorted(set(initial_module)-observed)
        changed=[n for n,p in module.named_parameters() if not torch.equal(initial_module[n],p.detach())]
        assert 'gamma_rgb' in changed and 'gamma_thermal' in changed
        assert any('.experts.' in n for n in changed)
        assert ema.updates==6
        ema.module.eval()
        with torch.no_grad():
            first=ema.module(rgb,thermal);second=ema.module(rgb,thermal)
            assert all(torch.equal(first[k],second[k]) and bool(torch.isfinite(first[k]).all()) for k in ('pred_logits','pred_boxes'))
        old_manifest=json.loads((ROOT/'artifacts/preserved_staged_source_sha256.json').read_text())
        assert source_manifest(OLD)==old_manifest
        assert source_manifest(ROOT)==source_before
        proof.update(all_module_gradients_finite_and_eventually_nonzero=True,parameter_tensors_updated=changed,
                     EMA_eval_finite_repeatable=True,ema_updates=ema.updates,original_stage2_sources_unchanged=True,
                     note='Production reinitializes seed0; gamma0 opens before inner gates, so early warmup can leave some FP32 weights unchanged despite nonzero gradients.',ready=True)
        inputs={str((ROOT/cfg['model']['pretrained']).resolve()):sha256(ROOT/cfg['model']['pretrained'])}
        for split in ('train','val'):
            path=cfg['data'][split]['ann_file'];inputs[path]=sha256(path)
        proof['input_sha256']=inputs
        atomic_json(ROOT/'artifacts/gpu_preflight.json',proof)
        print(json.dumps(dict(preflight_passed=True,all_module_gradient_tensors=len(observed),updated_tensors=len(changed))),flush=True)
    finally:
        locks.release()


if __name__=='__main__':main()
