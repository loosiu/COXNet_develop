"""Paired region-size diagnostics, without producing any regional training labels."""
import torch

from .utility_probe import region_coalition_gates


def cache_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values(): yield from cache_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value: yield from cache_tensors(item)


def per_image_max_difference(a, b):
    return (a-b).abs().flatten(1).amax(1).float().cpu().tolist()


@torch.no_grad()
def diagnose_regions(teacher, cache, reference, frozen, selections, coalitions,
                     region_sizes=(1, 3, 5), single_gates=None):
    """Use one teacher/cache/site/draw/query/cost context for every intervention size."""
    if teacher.training:
        raise ValueError('Region diagnostics require an eval-mode teacher')
    sizes = tuple(region_sizes)
    if not sizes or 1 not in sizes or len(set(sizes)) != len(sizes) or any(
            isinstance(n, bool) or not isinstance(n, int) or n not in (1, 3, 5) for n in sizes):
        raise ValueError('Diagnostic region sizes must be distinct 1/3/5 and include1')
    versions = [(tensor, tensor._version) for tensor in cache_tensors(cache)]
    forced = reference['query_metadata']['topk_indices']
    ref_gates = cache['frequency']['w']
    masks = cache['frequency']['valid_masks']
    epsilon = 1e-4*frozen.cost(reference).abs().clamp_min(1)
    result = dict(regions={})
    for size in sizes:
        on_gates, off_gates = region_coalition_gates(ref_gates, selections, coalitions, masks, size)
        if size == 1 and single_gates is not None:
            if not all(torch.equal(actual, original) for actual, original in zip((on_gates, off_gates), single_gates)):
                raise AssertionError('Masked 1x1 diagnostic must reproduce the training single-cell coalition')
        on = teacher.replay_frequency(cache, on_gates, forced_indices=forced, return_probe=True, return_diagnostics=True)
        off = teacher.replay_frequency(cache, off_gates, forced_indices=forced, return_probe=True, return_diagnostics=True)
        if not all(torch.equal(output['query_metadata']['topk_indices'], forced) for output in (on, off)):
            raise AssertionError('Regional replay changed reference query indices/order')
        on_parts, off_parts = frozen.components(on), frozen.components(off)
        delta = {key: off_parts[key]-on_parts[key] for key in on_parts}
        info = {key+'_max_abs_diff': per_image_max_difference(on['diagnostic_features'][key], off['diagnostic_features'][key])
                for key in ('rgb_p3', 'thermal_p3', 'fused_p3')}
        info.update(logits_max_abs_diff=per_image_max_difference(on['pred_logits'], off['pred_logits']),
                    boxes_max_abs_diff=per_image_max_difference(on['pred_boxes'], off['pred_boxes']),
                    **{'delta_'+key: value.cpu().tolist() for key, value in delta.items()})
        info.update(on_components={key: value.cpu().tolist() for key, value in on_parts.items()},
                    off_components={key: value.cpu().tolist() for key, value in off_parts.items()},
                    epsilon=epsilon.cpu().tolist(), neutral=(delta['total'].abs() <= epsilon).cpu().tolist(),
                    positive_valid=(delta['total'] > epsilon).cpu().tolist(),
                    negative_valid=(delta['total'] < -epsilon).cpu().tolist(),
                    intervened_cells=(on_gates-off_gates).flatten(1).count_nonzero(1).cpu().tolist(),
                    query_indices_fixed=True, gt_normalizers=frozen.normalizers.cpu().tolist())
        rms, center_rms = [], []
        radius = size//2
        height, width = masks.shape[-2:]
        for image, direction, band, yy, xx in selections.tolist():
            ys, xs = slice(max(0, yy-radius), min(height, yy+radius+1)), slice(max(0, xx-radius), min(width, xx+radius+1))
            valid = masks[image, direction, ys, xs]
            candidate = cache['frequency']['candidates'][image, direction, band]
            values = candidate[:, ys, xs][:, valid].float()
            rms.append(float(values.square().mean().sqrt()) if values.numel() else 0.)
            center_rms.append(float(candidate[:, yy, xx].float().square().mean().sqrt()))
        info.update(selected_candidate_rms=rms, center_candidate_rms=center_rms)
        result['regions'][str(size)] = info
        if size == 1:
            result.update(single_on_components=on_parts, single_off_components=off_parts)
        del on, off
    if any(tensor._version != version for tensor, version in versions):
        raise AssertionError('Region diagnostics modified the shared candidate/feature/gate cache')
    result['cache_unchanged'] = True
    return result
