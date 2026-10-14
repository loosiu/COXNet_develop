#!/usr/bin/env python3
"""Single-device training. Run from the repository root; see README.md."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dyfclt.model import DyFCLT
from dyfclt.losses import build_criterion
from dyfclt.training import build_optimizer, build_ema, set_learning_rate, optimizer_summary
from dyfclt.frequency_training import validate_output_destination
from dyfclt.runtime import (load_config, seed_everything, warmup_factor, make_dataset,
                            make_loader, move_targets, save_checkpoint, load_checkpoint, evaluate, resize_training_batch, select_training_config, apply_optimizer_step, checkpoint_metric)


def validate_detection_config(config):
    """The ordinary A trainer can optimize differentiable detection gates directly."""
    if config.get('initialization'):
        raise ValueError('Fresh detection training does not load a trained A checkpoint')
    utility = config.get('utility', {})
    if utility.get('enabled', False) or utility.get('loss_weight', 0.) != 0:
        raise ValueError('This trainer supports detection loss only')
    frequency = config['model'].get('frequency_utility')
    if frequency is not None and frequency.get('gate_mode') not in ('sigmoid_detection', 's3am_mofe', 'residual_mofe', 'residual_mamba_mofe', 'spatial_mofe', 'relation_mofe'):
        raise ValueError('Detached Beta gates require the legacy utility trainer')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Defaults to saved config on resume, otherwise configs/rgbtdroneperson.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume')
    parser.add_argument('--output-dir')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-steps', type=int, help='Debug run only; saves debug.pth, not resumable mid-epoch')
    parser.add_argument('--no-pretrained', action='store_true', help='Explicitly use random initialization')
    parser.add_argument('--no-validation', action='store_true')
    args = parser.parse_args()
    resume_state = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    config = select_training_config(args.config, resume_state['config'] if resume_state else None)
    validate_detection_config(config)
    settings = config['train']
    for key in ('output_dir','epochs','workers','batch_size'):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    output = Path(settings['output_dir'])
    validate_output_destination(output, args.resume, resume_state['config'] if resume_state else None)
    if (output / 'last.pth').exists() and not args.resume:
        raise FileExistsError(f'{output}/last.pth already exists; use --resume or a new --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config.get('seed', 0))
    device = torch.device(args.device)
    train_data = make_dataset(config, 'train', training=True)
    val_data = make_dataset(config, 'val') if not args.no_validation else None
    if len(train_data.category_ids) != config['model']['num_classes']:
        raise ValueError('model.num_classes must match the selected dataset categories')
    if val_data and train_data.category_ids != val_data.category_ids:
        raise ValueError('Training and validation category mappings differ')
    train_loader = make_loader(train_data, settings['batch_size'], settings['workers'], True)
    val_loader = make_loader(val_data, settings['batch_size'], settings['workers']) if val_data else None
    model_kwargs = dict(config['model'])
    if args.no_pretrained or args.resume:
        model_kwargs['pretrained'] = False
    if args.no_pretrained:
        config['model']['pretrained'] = False
    model = DyFCLT(**model_kwargs).to(device)
    criterion = build_criterion(config['model']['num_classes'], **config.get('loss', {})).to(device)
    optimizer = build_optimizer(model, settings)
    ema = build_ema(model, settings)
    amp = bool(settings.get('amp', False) and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    epoch_start, global_step, best_ap = 0, 0, -math.inf
    if args.resume:
        state = load_checkpoint(resume_state, model, optimizer, scaler, restore_rng=True, ema=ema)
        if state.get('partial_epoch'):
            raise ValueError('Debug checkpoint is incomplete; use an epoch-complete last.pth for resume')
        if state['category_ids'] != train_data.category_ids:
            raise ValueError('Resume checkpoint category mapping differs')
        epoch_start, global_step, best_ap = state['epoch'] + 1, state['global_step'], state['best_ap']
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(dict(device=str(device), classes=train_data.category_names,
                          train_images=len(train_data), parameters=sum(p.numel() for p in model.parameters()),
                          output_dir=str(output), amp=amp, model_config=model_kwargs, architecture=model.architecture,
                          modules=dict(dfca=model.use_dfca, sse=model.use_sse, fref=model.use_fref),
                          optimizer_groups=optimizer_summary(optimizer), ema_enabled=ema is not None,
                          evaluation_weights='ema' if ema else 'model')), flush=True)
    run_steps = 0
    consecutive_overflows = 0
    for epoch in range(epoch_start, settings['epochs']):
        model.train()
        started = time.time()
        for batch_index, (rgb, thermal, targets) in enumerate(train_loader):
            rgb, thermal, targets = resize_training_batch(rgb, thermal, targets, settings.get('resize_sizes'))
            factor = warmup_factor(global_step, settings['warmup_steps'], settings['warmup_factor'])
            set_learning_rate(optimizer, settings['lr'], factor)
            optimizer.zero_grad(set_to_none=True)
            targets = move_targets(targets, device)
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(rgb.to(device), thermal.to(device), targets,
                                return_frequency_aux=config['model'].get('frequency_utility') is not None)
                losses = criterion(outputs, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {global_step}: {losses}')
            scaler.scale(loss).backward()
            norm, applied = apply_optimizer_step(model, optimizer, scaler, settings['clip_grad_norm'], ema=ema)
            if not applied:
                consecutive_overflows += 1
                print(json.dumps(dict(amp_overflow=True, skipped_step=global_step, loss_scale=scaler.get_scale())), flush=True)
                if consecutive_overflows >= 16:
                    raise FloatingPointError('AMP gradients remained nonfinite after 16 loss-scale reductions')
                continue
            consecutive_overflows = 0
            global_step += 1
            run_steps += 1
            if batch_index % 20 == 0:
                record = dict(epoch=epoch+1, step=global_step, loss=float(loss.detach()),
                              grad_norm=float(norm), lr=settings['lr'] * factor,
                              group_lrs={g.get('group_name', 'legacy'): g['lr'] for g in optimizer.param_groups},
                              ema_updates=ema.updates if ema else None,
                              losses={k: float(v.detach()) for k,v in losses.items()})
                if 'frequency_aux' in outputs:
                    from dyfclt.frequency_training import gate_statistics
                    record['gate_statistics'] = gate_statistics(outputs['frequency_aux'])
                print(json.dumps(record), flush=True)
                with (output / 'train.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
            if args.max_steps is not None and run_steps >= args.max_steps:
                save_checkpoint(output / 'debug.pth', model, optimizer, scaler, config,
                                epoch-1, global_step, best_ap, train_data.category_ids, partial_epoch=True, ema=ema)
                print('Debug step limit reached. Saved debug.pth (partial epoch; not a completed training result).', flush=True)
                return
        metrics = evaluate(ema.module if ema else model, val_loader, device, config['data']['val']['ann_file'],
                           train_data.category_ids, output / 'validation',
                           protocol=config.get('evaluation', {}).get('protocol', 'coco')) if val_loader else None
        is_best = metrics is not None and checkpoint_metric(metrics, config) > best_ap
        if is_best:
            best_ap = checkpoint_metric(metrics, config)
        save_checkpoint(output / 'last.pth', model, optimizer, scaler, config,
                        epoch, global_step, best_ap, train_data.category_ids, ema=ema)
        if is_best:
            save_checkpoint(output / 'best.pth', model, optimizer, scaler, config,
                            epoch, global_step, best_ap, train_data.category_ids, ema=ema)
        print(json.dumps(dict(epoch=epoch+1, seconds=time.time()-started, validation=metrics,
                              evaluation_weights="ema" if ema else "model", ema_updates=ema.updates if ema else None)), flush=True)


if __name__ == '__main__':
    main()
