"""Fixed-object weight sweeps; diagnostics only, with no auxiliary training targets."""
import hashlib
import math

import torch
from torch.nn import functional as F

from .frequency_utility import BANDS
from .utility_probe import FrozenDetectionCost
from .utility_diagnostics import cache_tensors, per_image_max_difference
from .vendor.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class FixedObjectCost:
    """One frozen reference GT/query per image, normalized by one object, not image GT count."""
    def __init__(self, frozen, sites):
        batch, queries, _ = frozen.shape
        if sites.shape != (batch, 6) or sites.dtype != torch.long or sites[:, 0].tolist() != list(range(batch)):
            raise ValueError('Object sites must be [B,6]: image,direction,query,gt,y,x')
        self.frozen, self.sites = frozen, sites.detach().clone()
        self.truth = []
        for image, _, query, gt, _, _ in sites.tolist():
            if not 0 <= query < queries: raise ValueError('Reference object query out of range')
            matched_q, matched_gt = frozen.indices[image]
            mapping = {q: (g, index) for index, (q, g) in enumerate(zip(matched_q.tolist(), matched_gt.tolist()))}
            if gt >= 0:
                if query not in mapping or mapping[query][0] != gt:
                    raise ValueError('Selected GT/query must preserve reference matching')
                self.truth.append(frozen.target_boxes[image][mapping[query][1]])
            else:
                if gt != -1 or query in mapping:
                    raise ValueError('Background cost requires a reference-unmatched query')
                self.truth.append(None)

    def components(self, outputs):
        logits, boxes = outputs['pred_logits'].float(), outputs['pred_boxes'].float()
        if logits.shape != self.frozen.shape or boxes.shape != (*self.frozen.shape[:2], 4):
            raise ValueError('Object replay changed query prediction shape')
        image, query = self.sites[:, 0], self.sites[:, 2]
        vfl = F.binary_cross_entropy_with_logits(logits[image, query], self.frozen.quality_targets[image, query],
            weight=self.frozen.classification_weights[image, query], reduction='none').sum(-1)*self.frozen.coefficients['loss_vfl']
        bbox, giou = [], []
        for index, (_, _, q, _, _, _) in enumerate(self.sites.tolist()):
            prediction = boxes[index, q:q+1]
            truth = self.truth[index]
            if truth is None:
                bbox.append(boxes.new_zeros(())); giou.append(boxes.new_zeros(()))
            else:
                truth = truth.unsqueeze(0)
                bbox.append(F.l1_loss(prediction, truth, reduction='sum')*self.frozen.coefficients['loss_bbox'])
                giou.append((1-generalized_box_iou(box_cxcywh_to_xyxy(prediction),
                            box_cxcywh_to_xyxy(truth))[0, 0])*self.frozen.coefficients['loss_giou'])
        bbox, giou = torch.stack(bbox), torch.stack(giou)
        return dict(vfl=vfl, bbox=bbox, giou=giou, total=vfl+bbox+giou)


class ObjectProbeSampler:
    def __init__(self, seed=0, gt_probability=1.):
        if not 0 <= gt_probability <= 1: raise ValueError('Invalid object GT probability')
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.gt_probability = float(gt_probability)

    def state_dict(self):
        return dict(generator=self.generator.get_state(), gt_probability=self.gt_probability)

    def load_state_dict(self, state):
        self.generator.set_state(state['generator'].cpu())
        self.gt_probability = float(state['gt_probability'])

    def integer(self, upper):
        return int(torch.randint(upper, (), generator=self.generator))

    def sample(self, reference, targets, matching, cache):
        masks = cache['valid_masks']; height, width = masks.shape[-2:]
        anchors = reference['query_metadata']['anchor_centers']
        scores = reference['pred_logits'].sigmoid().max(-1).values
        rows, kinds = [], []
        for image, (target, indices) in enumerate(zip(targets, matching)):
            direction = self.integer(2); mask = masks[image, direction]
            if not mask.any(): raise ValueError('No valid receiver cells for object sweep')
            gt_choices = []
            for query, gt in zip(indices[0].tolist(), indices[1].tolist()):
                center = target['boxes'][gt, :2]
                yy, xx = min(height-1, max(0, int(center[1]*height))), min(width-1, max(0, int(center[0]*width)))
                if mask[yy, xx]: gt_choices.append((query, gt, yy, xx))
            prefer_gt = bool(torch.rand((), generator=self.generator) < self.gt_probability)
            selected, kind = None, None
            if prefer_gt and gt_choices:
                selected, kind = gt_choices[self.integer(len(gt_choices))], 'matched_gt'
            else:
                matched = set(indices[0].tolist())
                boxes = target['boxes'].detach()
                for query in scores[image].argsort(descending=True).tolist():
                    if query in matched: continue
                    center = anchors[image, query]
                    if not ((center >= 0) & (center < 1)).all(): continue
                    yy, xx = int(center[1]*height), int(center[0]*width)
                    if not mask[yy, xx]: continue
                    if len(boxes):
                        lower, upper = boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2
                        if (((center >= lower) & (center <= upper)).all(-1)).any(): continue
                    selected, kind = (query, -1, yy, xx), 'unmatched_background'
                    break
            if selected is None and gt_choices:
                selected, kind = gt_choices[self.integer(len(gt_choices))], 'matched_gt_fallback'
            if selected is None:
                raise ValueError('No reference matched GT or valid unmatched background query for sweep')
            rows.append([image, direction, *selected]); kinds.append(kind)
        return torch.tensor(rows, dtype=torch.long, device=masks.device), kinds


def object_sweep_gates(reference, sites, band, value, valid_masks, region_size=1):
    if reference.ndim != 5 or reference.shape[1:3] != (2, 4): raise ValueError('Expected gates [B,2,4,H,W]')
    batch, _, _, height, width = reference.shape
    if sites.shape != (batch, 6) or sites.dtype != torch.long or sites[:, 0].tolist() != list(range(batch)):
        raise ValueError('One ordered object site per image required')
    if isinstance(band, bool) or not isinstance(band, int) or band not in range(4): raise ValueError('Invalid sweep band')
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1: raise ValueError('Sweep weight must be in[0,1]')
    if isinstance(region_size, bool) or not isinstance(region_size, int) or region_size not in (1, 3, 5): raise ValueError('Invalid sweep region size')
    if valid_masks.shape != (batch, 2, height, width) or valid_masks.dtype != torch.bool or valid_masks.device != reference.device:
        raise ValueError('Invalid receiver validity mask')
    gates = reference.detach().clone(); radius = region_size//2
    for image, direction, _, _, yy, xx in sites.tolist():
        if not (0 <= direction < 2 and 0 <= yy < height and 0 <= xx < width): raise ValueError('Invalid object sweep site')
        ys, xs = slice(max(0, yy-radius), min(height, yy+radius+1)), slice(max(0, xx-radius), min(width, xx+radius+1))
        gates[image, direction, band, ys, xs][valid_masks[image, direction, ys, xs]] = value
    return gates


@torch.no_grad()
def fixed_object_sweep(teacher, rgb, thermal, targets, criterion, sampler,
                       weights=(0., .25, .5, .75, 1.), region_size=1):
    if teacher.training: raise ValueError('Object sweep requires a frozen eval-mode teacher')
    values = tuple(weights)
    if not values or values[0] != 0 or len(set(values)) != len(values) or any(
            isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Distinct finite sweep weights in[0,1], starting with0, required')
    cache = teacher.prepare_frequency(rgb, thermal)
    versions = [(t, t._version) for t in cache_tensors(cache)]
    reference = teacher.replay_frequency(cache, return_probe=True)
    forced = reference['query_metadata']['topk_indices']
    frozen = FrozenDetectionCost.from_reference(reference, targets, criterion)
    sites, kinds = sampler.sample(reference, targets, frozen.indices, cache['frequency'])
    local = FixedObjectCost(frozen, sites)
    object_ref, image_ref = local.components(reference), frozen.components(reference)
    object_eps, image_eps = [1e-4*cost['total'].abs().clamp_min(1) for cost in (object_ref, image_ref)]
    noop = teacher.replay_frequency(cache, forced_indices=forced)
    object_error = (local.components(noop)['total']-object_ref['total']).abs()
    image_error = (frozen.cost(noop)-image_ref['total']).abs()
    if (object_error >= .1*object_eps).any() or (image_error >= .1*image_eps).any():
        raise FloatingPointError('Object/image no-op error exceeds the measurement floor')
    result = dict(sites=sites.cpu().tolist(), site_kinds=kinds, region_size=region_size,
        reference_object_components={k:v.cpu().tolist() for k,v in object_ref.items()},
        reference_image_components={k:v.cpu().tolist() for k,v in image_ref.items()},
        object_epsilon=object_eps.cpu().tolist(), image_epsilon=image_eps.cpu().tolist(),
        object_noop_error=object_error.cpu().tolist(), image_noop_error=image_error.cpu().tolist(),
        image_gt_normalizers=frozen.normalizers.cpu().tolist(),
        query_index_sha256=hashlib.sha256(forced.cpu().contiguous().numpy().tobytes()).hexdigest(),
        bands={}, diagnostic_only=True, query_indices_fixed=True, matching_unchanged=True)
    for band_index, band_name in enumerate(BANDS):
        object_parts, image_parts, metrics = [], [], []
        zero_output = None
        for value in values:
            gates = object_sweep_gates(cache['frequency']['w'], sites, band_index, value, cache['frequency']['valid_masks'], region_size)
            output = teacher.replay_frequency(cache, gates, forced_indices=forced, return_probe=True, return_diagnostics=True)
            if not torch.equal(output['query_metadata']['topk_indices'], forced): raise AssertionError('Sweep changed reference query order')
            object_parts.append(local.components(output)); image_parts.append(frozen.components(output))
            if zero_output is None: zero_output = output
            diffs = {key:per_image_max_difference(output['diagnostic_features'][key], zero_output['diagnostic_features'][key])
                     for key in ('rgb_p3', 'thermal_p3', 'fused_p3')}
            diffs['receiver_p3'] = [max(r,t) for r,t in zip(diffs['rgb_p3'],diffs['thermal_p3'])]
            diffs['logits'] = per_image_max_difference(output['pred_logits'], zero_output['pred_logits'])
            diffs['boxes'] = per_image_max_difference(output['pred_boxes'], zero_output['pred_boxes'])
            metrics.append(diffs)
        obj_cost, img_cost = [torch.stack([part['total'] for part in parts]) for parts in (object_parts, image_parts)]
        obj_utility, img_utility = obj_cost[0]-obj_cost, img_cost[0]-img_cost
        candidate_rms = []; ref_weights = []
        for image, direction, _, _, yy, xx in sites.tolist():
            candidate_rms.append(float(cache['frequency']['candidates'][image,direction,band_index,:,yy,xx].float().square().mean().sqrt()))
            ref_weights.append(float(cache['frequency']['w'][image,direction,band_index,yy,xx]))
        result['bands'][band_name] = dict(weights=list(values), object_cost=obj_cost.cpu().tolist(), image_cost=img_cost.cpu().tolist(),
            object_components={k:torch.stack([p[k] for p in object_parts]).cpu().tolist() for k in object_ref},
            image_components={k:torch.stack([p[k] for p in image_parts]).cpu().tolist() for k in image_ref},
            object_utility_from_zero=obj_utility.cpu().tolist(), image_utility_from_zero=img_utility.cpu().tolist(),
            measurable_object_range=((obj_cost.max(0).values-obj_cost.min(0).values)>object_eps).cpu().tolist(),
            measurable_image_range=((img_cost.max(0).values-img_cost.min(0).values)>image_eps).cpu().tolist(),
            candidate_center_rms=candidate_rms, reference_center_weight=ref_weights,
            **{key+'_max_abs_diff':[m[key] for m in metrics] for key in metrics[0]})
        del zero_output, output
    if any(t._version != before for t,before in versions): raise AssertionError('Object sweep changed candidate/feature cache')
    result['cache_unchanged'] = True
    return result
