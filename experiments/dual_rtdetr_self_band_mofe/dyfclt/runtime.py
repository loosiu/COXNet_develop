"""Training utilities and standard COCO evaluation (not a custom tiny evaluator)."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def select_training_config(config_path=None, saved_config=None):
    """An explicit config wins; otherwise resume keeps its saved architecture/data."""
    if config_path is not None:
        return load_config(config_path)
    if saved_config is not None:
        return copy.deepcopy(saved_config)
    return load_config('configs/rgbtdroneperson.yaml')


def checkpoint_metric(metrics, config):
    key = config.get('evaluation', {}).get('selection_metric', 'AP')
    if key not in metrics:
        raise ValueError(f'Configured checkpoint metric {key} is absent from evaluation')
    return float(metrics[key])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def warmup_factor(step, steps=2000, start_factor=.001):
    if steps <= 0:
        return 1.
    return start_factor + (1. - start_factor) * min(max(step / steps, 0.), 1.)


def apply_optimizer_step(model, optimizer, scaler, max_norm, ema=None):
    """Unscale, clip, and step; allow GradScaler to skip/back off AMP overflow."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm, error_if_nonfinite=not scaler.is_enabled())
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    applied = not scaler.is_enabled() or scaler.get_scale() >= previous_scale
    if applied and ema is not None:
        ema.update(model)
    return norm, applied


def move_targets(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def resize_training_batch(rgb, thermal, targets, sizes):
    """Synchronized RT-DETR-style batch resizing; normalized boxes stay invariant."""
    if not sizes:
        return rgb, thermal, targets
    height, width = random.choice(sizes)
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError('Training resize dimensions must be positive multiples of 32')
    rgb = torch.nn.functional.interpolate(rgb, size=(height,width), mode='bilinear', align_corners=False)
    thermal = torch.nn.functional.interpolate(thermal, size=(height,width), mode='bilinear', align_corners=False)
    resized_targets = []
    for target in targets:
        updated = dict(target)
        old_height, old_width = target['size']
        updated['size'] = target['size'].new_tensor([height,width])
        if 'transform' in target:
            scale = target['transform'].new_tensor([[width / old_width,0,0],[0,height / old_height,0],[0,0,1]])
            updated['transform'] = scale @ target['transform']
            updated['inverse_transform'] = torch.linalg.inv(updated['transform'])
        resized_targets.append(updated)
    return rgb, thermal, resized_targets


def postprocess(outputs, targets, category_ids, topk=300, score_threshold=0.):
    """RT-DETR sigmoid top-k, normalized cxcywh -> original COCO xywh."""
    logits = outputs['pred_logits'].detach().float()
    boxes = outputs['pred_boxes'].detach().float()
    if logits.shape[-1] != len(category_ids):
        raise ValueError('Checkpoint class count does not match dataset category mapping')
    scores, indices = logits.sigmoid().flatten(1).topk(min(topk, logits.shape[1] * logits.shape[2]), dim=1)
    labels = indices % logits.shape[2]
    query_indices = indices // logits.shape[2]
    selected = boxes.gather(1, query_indices.unsqueeze(-1).expand(-1, -1, 4))
    xywh = torch.cat((selected[..., :2] - selected[..., 2:] / 2, selected[..., 2:]), -1)
    records = []
    for b, target in enumerate(targets):
        height, width = [int(v) for v in target['orig_size']]
        scaled = xywh[b] * xywh.new_tensor([width, height, width, height])
        image_id = int(target['image_id'])
        for score, label, box in zip(scores[b].cpu().tolist(), labels[b].cpu().tolist(), scaled.cpu().tolist()):
            if score >= score_threshold:
                records.append(dict(image_id=image_id, category_id=int(category_ids[label]), bbox=box, score=score))
    return records


def save_checkpoint(path, model, optimizer, scaler, config, epoch, global_step, best_ap, category_ids, partial_epoch=False, ema=None, extra_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(model=model.state_dict(), ema=ema.state_dict() if ema else None,
                 evaluation_weights='ema' if ema else 'model', optimizer=optimizer.state_dict() if optimizer else None,
                 scaler=scaler.state_dict() if scaler else None, config=config,
                 epoch=epoch, global_step=global_step, best_ap=best_ap, category_ids=list(category_ids), partial_epoch=partial_epoch,
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    if extra_state is not None:
        if set(extra_state).intersection(state):
            raise ValueError('Extra checkpoint state cannot replace standard state fields')
        state.update(extra_state)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer=None, scaler=None, restore_rng=False, ema=None, weights="model"):
    state = path if isinstance(path, dict) else torch.load(path, map_location='cpu', weights_only=False)
    if weights not in ('auto', 'model', 'ema'):
        raise ValueError(f'Unknown checkpoint weights: {weights}')
    if ema is not None and state.get('ema') is None:
        raise ValueError('Resume requested EMA but checkpoint has no EMA state; start a separate run')
    selected = 'ema' if weights == 'auto' and state.get('ema') is not None else weights
    if selected == 'ema':
        if state.get('ema') is None:
            raise ValueError('Checkpoint contains no EMA weights')
        if optimizer is not None or ema is not None:
            raise ValueError('Resume uses raw model weights with separate EMA state')
        model.load_state_dict(state['ema']['module'], strict=True)
    else:
        model.load_state_dict(state['model'], strict=True)
    if ema is not None:
        ema.load_state_dict(state['ema'])
    if optimizer is not None and state.get('optimizer') is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scaler is not None and state.get('scaler') is not None:
        scaler.load_state_dict(state['scaler'])
    if restore_rng and 'rng' in state:
        rng = state['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if torch.cuda.is_available() and rng['cuda'] is not None:
            torch.cuda.set_rng_state_all(rng['cuda'])
    return state


def make_dataset(config, split, training=False):
    from .data import PairedCocoDataset
    data = config['data']
    kwargs = dict(data[split])
    kwargs.update(image_size=tuple(data['image_size']), training=training)
    if training:
        kwargs.update(data.get('augmentation', {}))
    return PairedCocoDataset(**kwargs)


def make_loader(dataset, batch_size=2, workers=0, training=False):
    from .data import paired_collate_fn
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=training,
                                       num_workers=workers, collate_fn=paired_collate_fn,
                                       pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                                       drop_last=False)


@torch.no_grad()
def evaluate(model, loader, device, ann_file, category_ids, output_dir, max_batches=None, protocol='coco'):
    """Evaluate only image IDs actually visited, including negative images."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    was_training = model.training
    model.eval()
    records, image_ids = [], []
    for index, (rgb, thermal, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        outputs = model(rgb.to(device), thermal.to(device))
        records.extend(postprocess(outputs, targets, category_ids))
        image_ids.extend(int(t['image_id']) for t in targets)
    model.train(was_training)
    if not image_ids:
        raise ValueError('Evaluation received no images')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'predictions.json').open('w') as handle:
        json.dump(records, handle)
    if protocol == 'rgbtdroneperson':
        from .evaluation import evaluate_records
        result = evaluate_records(ann_file, records, sorted(set(image_ids)), category_ids)
        result.update(images=len(set(image_ids)), partial=max_batches is not None)
        with (output_dir / 'metrics.json').open('w') as handle:
            json.dump(result, handle, indent=2)
        return result
    if protocol != 'coco':
        raise ValueError(f'Unknown evaluation protocol: {protocol}')
    coco = COCO(str(ann_file))
    # Preserve explicit ignore annotations as crowd regions for stock COCOeval.
    # Original JSON remains untouched. Custom tiny benchmark rules are separate.
    for annotation in coco.dataset['annotations']:
        if annotation.get('ignore', False):
            annotation['iscrowd'] = 1
        if 'area' not in annotation:
            annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    if records:
        # Older pycocotools assumes an info entry in loadRes.
        coco.dataset.setdefault('info', {})
        predictions = coco.loadRes(records)
    else:
        predictions = COCO()
        predictions.dataset = dict(images=coco.dataset['images'], categories=coco.dataset['categories'], annotations=[])
        predictions.createIndex()
    metric = COCOeval(coco, predictions, 'bbox')
    metric.params.imgIds = sorted(set(image_ids))
    metric.params.catIds = list(category_ids)
    metric.evaluate()
    metric.accumulate()
    metric.summarize()
    names = ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
    result = {key: float(value) for key, value in zip(names, metric.stats)}
    result.update(images=len(set(image_ids)), evaluation='standard COCO; maxDets=100', partial=max_batches is not None)
    with (output_dir / 'metrics.json').open('w') as handle:
        json.dump(result, handle, indent=2)
    return result
