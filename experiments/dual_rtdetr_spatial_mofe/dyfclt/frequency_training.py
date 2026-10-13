"""Experiment provenance, strict A initialization and continuation recipe checks."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tarfile
from datetime import datetime, timezone

import torch


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def source_manifest(root):
    root = Path(root)
    return {str(path.relative_to(root)): sha256(path)
            for folder in ['dyfclt', 'tools', 'configs']
            for path in sorted((root/folder).rglob('*'))
            if path.is_file() and path.suffix in ('.py', '.yaml', '.sh')}


def validate_output_destination(output, resume_path=None, resume_config=None):
    """Nonempty destinations belong to their original checkpoint/config only."""
    output = Path(output).resolve()
    if not output.exists(): return
    if not output.is_dir(): raise FileExistsError('Output is not a directory: '+str(output))
    if not any(output.iterdir()): return
    if resume_path is None or output != Path(resume_path).resolve().parent:
        raise FileExistsError('Refusing to reuse unrelated nonempty output directory: '+str(output))
    config_path = output/'config.json'
    if not config_path.is_file() or resume_config is None:
        raise ValueError('Existing resume output must contain its original config.json')
    original, checkpoint = json.loads(config_path.read_text()), copy.deepcopy(resume_config)
    for cfg in (original, checkpoint):
        cfg['train'].pop('output_dir', None); cfg['train'].pop('workers', None)
    if original != checkpoint:
        raise ValueError('Resume checkpoint does not belong to the existing output configuration')


def write_launch_provenance(output, root, manifest, config, launch):
    """Keep initial provenance immutable; record each continuation separately."""
    output, root = Path(output), Path(root)
    config_path = output/'config.json'
    if not config_path.exists():
        with config_path.open('x') as handle: handle.write(json.dumps(config, indent=2)+'\n')
    history = (output/'launch.json').exists()
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch_name = 'resume_launch_'+suffix+'.json' if history else 'launch.json'
    snapshot_name = 'resume_source_snapshot_'+suffix+'.tar.gz' if history else 'source_snapshot.tar.gz'
    with (output/launch_name).open('x') as handle: handle.write(json.dumps(launch, indent=2)+'\n')
    with tarfile.open(output/snapshot_name, 'x:gz') as archive:
        for relative in manifest: archive.add(root/relative, arcname=relative)


def initialize_from_a(model, checkpoint, weights='ema'):
    if weights not in ('ema', 'model'):
        raise ValueError('A initialization requires explicit ema or model weights')
    path = Path(checkpoint).resolve()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    options = saved['config']['model']
    if saved.get('partial_epoch') or saved.get('epoch', -1) < 0:
        raise ValueError('Initialization requires an epoch-complete trained A checkpoint')
    if options.get('architecture') != 'dual_rtdetr' or options.get('use_dfca') or options.get('use_sse') or any(
            options.get(key) is not None for key in ('fixed_experts', 'spatial_experts', 'frequency_utility')):
        raise ValueError('Initialization checkpoint must be the plain experiment A')
    if weights == 'ema':
        if saved.get('ema') is None: raise ValueError('A checkpoint has no EMA state')
        source = saved['ema']['module']
    else:
        source = saved['model']
    target = model.state_dict()
    missing, extra = sorted(set(target)-set(source)), sorted(set(source)-set(target))
    if extra or any(not key.startswith('encoder.frequency_module.') for key in missing):
        raise ValueError('A checkpoint/common model keys differ: missing={} extra={}'.format(missing, extra))
    for key, value in source.items():
        if value.shape != target[key].shape:
            raise ValueError('A checkpoint shape mismatch: '+key)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('A checkpoint contains nonfinite weights: '+key)
    model.load_state_dict(source, strict=False)
    loaded = model.state_dict()
    exact = all(torch.equal(loaded[k], v.to(loaded[k].device, loaded[k].dtype)) for k, v in source.items())
    if not exact: raise RuntimeError('Shared A tensors did not load exactly')
    return dict(source_checkpoint=str(path), checkpoint_sha256=sha256(path), selected_weights=weights,
                source_completed_epoch=int(saved['epoch'])+1, source_global_step=int(saved['global_step']),
                source_best_AP50=float(saved['best_ap']), category_ids=saved['category_ids'],
                loaded_state_tensors=len(source), new_state_keys=missing, shared_states_exact=exact,
                optimizer_restored=False, old_epoch_counter_restored=False)


def kl_coefficient(progress_epochs, maximum=.01, ramp_epochs=2):
    if not all(math.isfinite(x) for x in (progress_epochs, maximum, ramp_epochs)) or maximum < 0 or ramp_epochs <= 0:
        raise ValueError('KL schedule requires finite nonnegative maximum and positive ramp')
    return maximum*min(max(progress_epochs/ramp_epochs, 0), 1)


def validate_paired_recipes(control, variant):
    a, d = copy.deepcopy(control), copy.deepcopy(variant)
    if a['model'].get('frequency_utility') is not None or d['model'].get('frequency_utility') is None:
        raise ValueError('Control must omit frequency_utility and variant must enable it')
    for cfg in (a, d):
        cfg['model'].pop('frequency_utility', None)
        cfg['train'].pop('output_dir', None)
        cfg['utility'].pop('enabled', None)
    if a != d:
        raise ValueError('A continuation and D differ beyond module/utility enabled/output directory')
    return True


def validate_config(config):
    settings, utility = config['train'], config.get('utility', {})
    frequency = config['model'].get('frequency_utility')
    frequency_enabled = frequency is not None
    mode = frequency.get('gate_mode', 'beta_utility') if frequency_enabled else None
    enabled = bool(utility.get('enabled', False))
    if mode == 'sigmoid_detection':
        if enabled or utility.get('loss_weight', 0.) != 0:
            raise ValueError('D2 sigmoid mode trains detection only; auxiliary utility is a later experiment')
    elif enabled != frequency_enabled:
        raise ValueError('Utility enabled must match model frequency_utility')
    for key in ('epochs', 'batch_size', 'workers', 'warmup_steps'):
        v = settings[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < (0 if key in ('workers', 'warmup_steps') else 1):
            raise ValueError('Invalid integer train.'+key)
    if enabled:
        if isinstance(utility['probe_every'], bool) or not isinstance(utility['probe_every'], int) or utility['probe_every'] < 1:
            raise ValueError('probe_every must be a positive integer')
        if not 0 <= utility['teacher_decay'] < 1 or not 0 <= utility['gt_probability'] <= 1:
            raise ValueError('Invalid teacher decay or GT sampling probability')
        if utility['loss_weight'] < 0 or not math.isfinite(utility['loss_weight']):
            raise ValueError('Utility loss weight must be finite and nonnegative')
        kl_coefficient(0, utility['kl_max'], utility['kl_ramp_epochs'])
    return enabled


def validate_training_pair(config, root):
    from .runtime import load_config
    has_frequency = config['model'].get('frequency_utility') is not None
    paired = load_config(Path(root)/'configs'/('rgbtdroneperson_A_continuation.yaml' if has_frequency
                                             else 'rgbtdroneperson_frequency_utility.yaml'))
    return validate_paired_recipes(paired, config) if has_frequency else validate_paired_recipes(config, paired)


def gate_statistics(aux):
    with torch.no_grad():
        if 'diagnostics' in aux:
            return {direction: {key+'_mean': value.detach().float().mean(0)[index].cpu().tolist()
                                for key, value in aux['diagnostics'].items()}
                    for index, direction in enumerate(['rgb_from_thermal', 'thermal_from_rgb'])}
        if 'routing_weights' in aux:
            weights = aux['routing_weights'].detach().float()
            return dict(routing_weight_mean=weights.mean(0).cpu().tolist(),
                        routing_weight_image_std=weights.std(0, unbiased=False).cpu().tolist(),
                        routing_entropy=float(-(weights * weights.clamp_min(1e-12).log()).sum(1).mean()),
                        candidate_rms=aux['candidate_rms'].mean(0).cpu().tolist(),
                        rgb_relative_update=float(aux['relative_update'].mean()))
        return {key+'_mean': aux[key].detach().float().mean((0, 3, 4)).cpu().tolist()
                for key in ('w', 'p', 'u', 's') if key in aux} | {
            'w_spatial_std': aux['w'].detach().float().flatten(3).std(-1, unbiased=False).mean(0).cpu().tolist()}


def serialize_probe(probe):
    if probe is None: return None
    def detached(value):
        if torch.is_tensor(value): return value.detach().cpu().tolist()
        if isinstance(value, dict): return {key: detached(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)): return [detached(item) for item in value]
        return value
    result = {key: detached(value)
              for key, value in probe.items() if key != 'query_indices'}
    indices = probe['query_indices'].detach().cpu().contiguous()
    result['query_index_sha256'] = hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    return result


def training_step(model, teacher, ema, sampler, criterion, optimizer, scaler,
                  batch, utility, progress_epochs, do_probe=False, check_noop=False,
                  max_grad_norm=.1, probe_region_sizes=(1,)):
    """One actual detection/utility step, shared by the CLI and bounded preflight."""
    from .runtime import apply_optimizer_step
    from .utility_probe import make_utility_labels, sampled_utility_loss
    rgb, thermal, targets = batch
    enabled = bool(utility.get('enabled', False))
    frequency_module = getattr(getattr(model, 'encoder', None), 'frequency_module', None)
    frequency_enabled = frequency_module is not None
    if enabled and getattr(frequency_module, 'gate_mode', None) == 'sigmoid_detection':
        raise ValueError('D2 detection gates do not use binary/Beta auxiliary utility')
    probe = None
    if enabled and do_probe:
        if teacher is None or sampler is None:
            raise ValueError('Utility training requires a teacher and independent sampler')
        # Function returns only small detached costs/sites, releasing the large teacher cache.
        probe = make_utility_labels(teacher.module, rgb, thermal, targets, criterion,
                                    sampler, check_noop=check_noop, diagnostic_region_sizes=probe_region_sizes)
    valid_count = int((~probe['neutral']).sum().item()) if probe is not None else 0
    positive_valid = int(((~probe['neutral']) & (probe['labels'] == 1)).sum().item()) if probe is not None else 0
    negative_valid = valid_count-positive_valid
    probe_count = len(probe['labels']) if probe is not None else 0
    positive_all = int((probe['labels'] == 1).sum().item()) if probe is not None else 0
    utility_active = bool(enabled and probe is not None and valid_count > 0 and utility.get('loss_weight', .1) > 0)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    kl = kl_coefficient(progress_epochs, utility.get('kl_max', .01), utility.get('kl_ramp_epochs', 2))
    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        outputs = model(rgb, thermal, targets, return_frequency_aux=frequency_enabled)
        losses = criterion(outputs, targets)
        detection_loss = sum(losses.values())
        utility_loss = (sampled_utility_loss(outputs['frequency_aux'], probe, kl)
                        if utility_active else detection_loss.new_zeros(()))
        loss = detection_loss + utility.get('loss_weight', .1)*utility_loss
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite detection/utility training loss')
    scaler.scale(loss).backward()
    norm, applied = apply_optimizer_step(model, optimizer, scaler, max_grad_norm, ema=ema)
    if teacher is not None: teacher.update(model, applied=applied)
    return dict(applied=applied, loss=float(loss.detach()), detection_loss=float(detection_loss.detach()),
                utility_loss=float(utility_loss.detach()), kl_coefficient=kl, grad_norm=float(norm),
                valid_count=valid_count, positive_valid_count=positive_valid, negative_valid_count=negative_valid,
                utility_active=utility_active, positive_fraction_all=positive_all/probe_count if probe_count else None,
                positive_fraction_valid=positive_valid/valid_count if valid_count else None,
                losses={k: float(v.detach()) for k, v in losses.items()},
                gate_statistics=gate_statistics(outputs['frequency_aux']) if frequency_enabled else None,
                probe=serialize_probe(probe))


def utility_preflight_ready(probe_images, nonneutral_fraction, steps, finite, constants_exact):
    """Regional signal and synthetic tests cannot establish single-cell readiness."""
    active_backward = any(row.get('applied') and row.get('utility_active') and row.get('valid_count', 0) > 0
                          and row.get('evaluator_nonzero_grad') and row.get('evaluator_gradients_finite')
                          for row in steps)
    return bool(probe_images >= 16 and nonneutral_fraction is not None and nonneutral_fraction >= .1
                and finite and constants_exact and sum(bool(row.get('applied')) for row in steps) >= 2
                and active_backward)


def verify_training_certificate(path, config, root):
    import json
    certificate = json.loads(Path(path).read_text())
    if not certificate.get('ready_for_long_training'):
        raise ValueError('Single-cell utility preflight did not establish adequate signal; inspect its report before long training')
    signal = certificate['single_cell_signal']
    if not utility_preflight_ready(signal['images'], signal['nonneutral_fraction'], certificate['training_steps'],
                                   certificate['finite'], certificate['constants_exact']):
        raise ValueError('Long D1 training requires actual nonneutral single-cell utility backward')
    if certificate['resolved_config'] != config or certificate['source_sha256'] != source_manifest(root):
        raise ValueError('Preflight certificate config/source differ from this experiment; repeat preflight')
    if certificate.get('tested_amp') != bool(config['train']['amp']):
        raise ValueError('Preflight must verify the configured AMP/FP32 mode')
    for key, path_key in [('train_annotations', 'train'), ('val_annotations', 'val')]:
        if certificate['input_sha256'][key] != sha256(config['data'][path_key]['ann_file']):
            raise ValueError('Preflight annotation changed: '+key)
    return certificate
