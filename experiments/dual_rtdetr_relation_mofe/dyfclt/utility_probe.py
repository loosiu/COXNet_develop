"""Controlled single-cell teacher interventions and sampled evidence training."""
from copy import deepcopy

import torch
from torch.nn import functional as F

from .vendor.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


class FrozenDetectionCost:
    """Final decoder-only VFL/L1/GIoU; matching and reference VFL terms frozen."""
    @classmethod
    @torch.no_grad()
    def from_reference(cls, reference, targets, criterion):
        base = criterion.detector if hasattr(criterion, 'detector') else criterion
        result = cls()
        logits, boxes = reference['pred_logits'].float(), reference['pred_boxes'].float()
        result.shape = logits.shape
        result.indices = [(q.to(logits.device), t.to(logits.device))
                          for q, t in base.matcher(reference, targets)]
        result.quality_targets = torch.zeros_like(logits)
        onehot = torch.zeros_like(logits)
        result.target_boxes = []
        for i, ((queries, matches), target) in enumerate(zip(result.indices, targets)):
            matched_boxes = target['boxes'][matches].float().detach().clone()
            result.target_boxes.append(matched_boxes)
            if len(queries):
                quality = box_iou(box_cxcywh_to_xyxy(boxes[i, queries]),
                                  box_cxcywh_to_xyxy(matched_boxes))[0].diag()
                classes = target['labels'][matches]
                result.quality_targets[i, queries, classes] = quality
                onehot[i, queries, classes] = 1
        result.classification_weights = (
            base.alpha*logits.sigmoid().pow(base.gamma)*(1-onehot) + result.quality_targets)
        result.normalizers = logits.new_tensor([max(len(t['labels']), 1) for t in targets])
        result.coefficients = {k: float(base.weight_dict[k]) for k in ['loss_vfl', 'loss_bbox', 'loss_giou']}
        return result

    def components(self, outputs):
        logits, boxes = outputs['pred_logits'].float(), outputs['pred_boxes'].float()
        if logits.shape != self.shape or boxes.shape != (*self.shape[:2], 4):
            raise ValueError('Replay prediction shapes differ from the reference queries')
        classification = F.binary_cross_entropy_with_logits(
            logits, self.quality_targets, weight=self.classification_weights, reduction='none').sum((1, 2))
        bbox_costs, giou_costs = [], []
        for i, (queries, _) in enumerate(self.indices):
            selected, truth = boxes[i, queries], self.target_boxes[i]
            bbox = F.l1_loss(selected, truth, reduction='sum')
            if len(queries):
                giou = (1-generalized_box_iou(box_cxcywh_to_xyxy(selected), box_cxcywh_to_xyxy(truth)).diag()).sum()
            else:
                giou = boxes[i].sum()*0
            bbox_costs.append(self.coefficients['loss_bbox']*bbox/self.normalizers[i])
            giou_costs.append(self.coefficients['loss_giou']*giou/self.normalizers[i])
        vfl = self.coefficients['loss_vfl']*classification/self.normalizers
        bbox, giou = torch.stack(bbox_costs), torch.stack(giou_costs)
        return dict(vfl=vfl, bbox=bbox, giou=giou, total=vfl+bbox+giou)

    def cost(self, outputs):
        return self.components(outputs)['total']


def beta_utility_loss(a, b, labels, kl_weight=0.):
    if a.shape != b.shape or labels.shape != a.shape or a.numel() == 0:
        raise ValueError('Utility loss expects equal nonempty sampled a/b/label shapes')
    if kl_weight < 0 or not torch.isfinite(a).all() or not torch.isfinite(b).all() or (a < 1).any() or (b < 1).any():
        raise ValueError('Evidence shapes must be finite and >=1; KL weight nonnegative')
    if ((labels != 0) & (labels != 1)).any():
        raise ValueError('Utility labels must be binary')
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, b, labels = a.float(), b.float(), labels.float()
        total = a+b
        fit = (labels-a/total).square() + a*b/(total.square()*(total+1))
        ta, tb = labels+(1-labels)*a, (1-labels)+labels*b
        ts = ta+tb
        kl = (torch.lgamma(ts)-torch.lgamma(ta)-torch.lgamma(tb)
              +(ta-1)*(torch.digamma(ta)-torch.digamma(ts))
              +(tb-1)*(torch.digamma(tb)-torch.digamma(ts)))
        return (fit+kl_weight*kl).mean()


def sampled_utility_loss(aux, probe, kl_weight=0.):
    site = probe['selections'].to(device=aux['a'].device)
    if site.ndim != 2 or site.shape[1] != 5 or site.dtype != torch.long:
        raise ValueError('Sampled sites must be [N,5]: image,direction,band,y,x')
    labels = probe['labels'].to(device=aux['a'].device)
    neutral = probe['neutral'].to(device=aux['a'].device)
    if labels.shape != (len(site),):
        raise ValueError('Labels must match sampled sites')
    if neutral.shape != labels.shape or neutral.dtype != torch.bool:
        raise ValueError('Neutral mask must be boolean and match labels')
    valid = ~neutral
    if not valid.any():
        # No graph connection: AdamW must skip evaluators, including old momentum.
        return aux['a'].new_zeros(())
    site, labels = site[valid], labels[valid]
    index = tuple(site[:, i] for i in range(5))
    return beta_utility_loss(aux['a'][index], aux['b'][index], labels, kl_weight)


def constant_buffer_keys(model):
    return {('.'.join((prefix, name)) if prefix else name)
            for prefix, module in model.named_modules()
            for name in getattr(module, 'ema_constant_buffers', ())}


class UtilityTeacher:
    """A separate frozen .99 EMA, without evaluation-EMA warmup."""
    def __init__(self, model, decay=.99):
        if not 0 <= decay < 1:
            raise ValueError('Teacher decay must be in [0,1)')
        self.module = deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = float(decay), 0
        self.constant_keys = constant_buffer_keys(model)

    @torch.no_grad()
    def update(self, model, applied=True):
        if not applied:
            return
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.is_floating_point() and name not in self.constant_keys:
                value.mul_(self.decay).add_(source[name].detach(), alpha=1-self.decay)
            else:
                value.copy_(source[name].detach())
        self.updates += 1

    def state_dict(self):
        return dict(module=self.module.state_dict(), decay=self.decay, updates=self.updates)

    def load_state_dict(self, state):
        self.module.load_state_dict(state['module'], strict=True)
        if not 0 <= float(state['decay']) < 1 or int(state['updates']) < 0:
            raise ValueError('Invalid utility teacher checkpoint state')
        self.decay, self.updates = float(state['decay']), int(state['updates'])
        self.module.eval().requires_grad_(False)


def utility_targets(reference_cost, on_cost, off_cost):
    if not (reference_cost.shape == on_cost.shape == off_cost.shape) or reference_cost.ndim != 1:
        raise ValueError('Probe costs must have one scalar per image')
    if not all(torch.isfinite(x).all() for x in [reference_cost, on_cost, off_cost]):
        raise FloatingPointError('Nonfinite teacher intervention cost')
    epsilon = 1e-4*reference_cost.abs().clamp_min(1)
    return ((off_cost-on_cost) > epsilon).float(), epsilon


def coalition_gates(reference, selections, generator):
    if reference.ndim != 5 or reference.shape[1:3] != (2, 4):
        raise ValueError('Reference gates must have shape [B,2,4,H,W]')
    if selections.shape != (reference.shape[0], 5) or selections[:, 0].tolist() != list(range(reference.shape[0])):
        raise ValueError('Intervene at exactly one cell per image in batch order')
    on, off = reference.detach().clone(), reference.detach().clone()
    coalition = (torch.rand((reference.shape[0], 4), generator=generator) < .5).to(reference.device, reference.dtype)
    for row, values in zip(selections.tolist(), coalition):
        image, direction, band, yy, xx = row
        if not (0 <= direction < 2 and 0 <= band < 4 and 0 <= yy < reference.shape[-2] and 0 <= xx < reference.shape[-1]):
            raise ValueError('Intervention cell/direction/band out of range')
        on[image, direction, :, yy, xx] = values
        off[image, direction, :, yy, xx] = values
        on[image, direction, band, yy, xx], off[image, direction, band, yy, xx] = 1, 0
    return on, off, coalition


def region_coalition_gates(reference, selections, coalitions, valid_masks, region_size):
    """Diagnostic-only joint intervention; never a set of cell utility labels."""
    if reference.ndim != 5 or reference.shape[1:3] != (2, 4):
        raise ValueError('Reference gates must have shape [B,2,4,H,W]')
    batch, _, _, height, width = reference.shape
    if selections.shape != (batch, 5) or selections.dtype != torch.long or selections[:, 0].tolist() != list(range(batch)):
        raise ValueError('Select one diagnostic site per image in batch order')
    if isinstance(region_size, bool) or not isinstance(region_size, int) or region_size not in (1, 3, 5):
        raise ValueError('Diagnostic region_size must be 1, 3 or 5')
    if (valid_masks.shape != (batch, 2, height, width) or valid_masks.dtype != torch.bool
            or valid_masks.device != reference.device):
        raise ValueError('Receiver valid_masks must be boolean [B,2,H,W] on gate device')
    if coalitions.shape != (batch, 4) or not ((coalitions == 0) | (coalitions == 1)).all():
        raise ValueError('Reuse binary coalition draws [B,4]')
    draws = coalitions.to(device=reference.device, dtype=reference.dtype)
    on, off = reference.detach().clone(), reference.detach().clone()
    expected_difference = torch.zeros_like(reference, dtype=torch.bool)
    radius = region_size//2
    for image, direction, band, yy, xx in selections.tolist():
        if not (0 <= direction < 2 and 0 <= band < 4 and 0 <= yy < height and 0 <= xx < width):
            raise ValueError('Diagnostic cell/direction/band out of range')
        ys, xs = slice(max(0, yy-radius), min(height, yy+radius+1)), slice(max(0, xx-radius), min(width, xx+radius+1))
        mask = valid_masks[image, direction, ys, xs]
        for gates in (on, off):
            view = gates[image, direction, :, ys, xs]
            view.copy_(torch.where(mask.unsqueeze(0), draws[image, :, None, None], view))
        on[image, direction, band, ys, xs][mask] = 1
        off[image, direction, band, ys, xs][mask] = 0
        expected_difference[image, direction, band, ys, xs] = mask
    if not torch.equal(on-off, expected_difference.to(reference.dtype)):
        raise AssertionError('On/off difference must equal exactly the selected band/direction/valid region')
    return on, off


class UtilityProbeSampler:
    def __init__(self, seed=0, gt_probability=.5):
        if not 0 <= gt_probability <= 1:
            raise ValueError('GT probability must lie in [0,1]')
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.gt_probability = float(gt_probability)

    def state_dict(self):
        return dict(generator=self.generator.get_state(), gt_probability=self.gt_probability)

    def load_state_dict(self, state):
        self.generator.set_state(state['generator'].cpu())
        self.gt_probability = float(state['gt_probability'])

    def integer(self, upper):
        return int(torch.randint(upper, (), generator=self.generator))

    def sample(self, reference, targets, matching, frequency_cache):
        masks = frequency_cache['valid_masks']
        height, width = masks.shape[-2:]
        anchors = reference['query_metadata']['anchor_centers']
        scores = reference['pred_logits'].sigmoid().max(-1).values
        rows, kinds = [], []
        for image, (target, indices) in enumerate(zip(targets, matching)):
            direction, band = self.integer(2), self.integer(4)
            mask = masks[image, direction]
            available = mask.nonzero(as_tuple=False)
            if len(available) == 0:
                raise ValueError('Cannot probe an image with no valid receiver cells')
            boxes = target['boxes'].detach()
            gt_sites = []
            for box in boxes:
                yy, xx = min(height-1, max(0, int(box[1]*height))), min(width-1, max(0, int(box[0]*width)))
                if mask[yy, xx]: gt_sites.append((yy, xx))
            prefer_gt = bool(torch.rand((), generator=self.generator) < self.gt_probability)
            selected, kind = None, None
            if prefer_gt and gt_sites:
                selected, kind = gt_sites[self.integer(len(gt_sites))], 'gt_center'
            else:
                matched = set(indices[0].tolist())
                for query in scores[image].argsort(descending=True).tolist():
                    if query in matched: continue
                    center = anchors[image, query]
                    if not ((center >= 0) & (center < 1)).all(): continue
                    yy, xx = int(center[1]*height), int(center[0]*width)
                    if not mask[yy, xx]: continue
                    if len(boxes):
                        lower, upper = boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2
                        if (((center >= lower) & (center <= upper)).all(-1)).any(): continue
                    selected, kind = (yy, xx), 'unmatched_background'
                    break
            if selected is None and gt_sites:
                selected, kind = gt_sites[self.integer(len(gt_sites))], 'gt_fallback'
            if selected is None:
                selected, kind = tuple(available[self.integer(len(available))].tolist()), 'valid_cell_fallback'
            rows.append([image, direction, band, *selected]); kinds.append(kind)
        return torch.tensor(rows, dtype=torch.long, device=masks.device), kinds


@torch.no_grad()
def make_utility_labels(teacher, rgb, thermal, targets, criterion, sampler,
                        valid_masks=None, check_noop=False, diagnostic_region_sizes=(1,)):
    if teacher.training:
        raise ValueError('Utility teacher must be in eval mode')
    cache = teacher.prepare_frequency(rgb, thermal, valid_masks)
    reference = teacher.replay_frequency(cache, return_probe=True)
    forced = reference['query_metadata']['topk_indices']
    frozen = FrozenDetectionCost.from_reference(reference, targets, criterion)
    reference_cost = frozen.cost(reference)
    selections, kinds = sampler.sample(reference, targets, frozen.indices, cache['frequency'])
    on_gates, off_gates, coalitions = coalition_gates(cache['frequency']['w'], selections, sampler.generator)
    from .utility_diagnostics import diagnose_regions
    diagnostics = diagnose_regions(teacher, cache, reference, frozen, selections, coalitions,
                                   diagnostic_region_sizes, single_gates=(on_gates, off_gates))
    on_cost = diagnostics['single_on_components']['total']
    off_cost = diagnostics['single_off_components']['total']
    labels, epsilon = utility_targets(reference_cost, on_cost, off_cost)
    noop_delta = None
    if check_noop:
        noop_delta = (frozen.cost(teacher.replay_frequency(cache, forced_indices=forced))-reference_cost).abs()
        if (noop_delta >= .1*epsilon).any():
            raise FloatingPointError('No-op replay error is not below the utility tolerance')
    # No candidate/encoder cache or reference predictions escape this function.
    return dict(selections=selections, site_kinds=kinds, labels=labels,
                delta_cost=off_cost-on_cost, reference_cost=reference_cost,
                on_cost=on_cost, off_cost=off_cost, epsilon=epsilon,
                neutral=(off_cost-on_cost).abs() <= epsilon,
                diagnostic_regions=diagnostics['regions'], cache_unchanged=diagnostics['cache_unchanged'],
                coalitions=coalitions, noop_delta=noop_delta,
                query_indices=forced.detach().clone())
