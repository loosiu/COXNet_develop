"""Offline object-crop analysis following supplementary section 6.3.

This module does not participate in the detector FFT or training path. Choices
not specified by the supplement are explicit: original RGB pixels are used
without resizing or grayscale conversion; per-channel PSDs are averaged. Both
padding stages use NumPy reflection excluding the edge pixel, centered with
an odd extra pixel on the bottom/right. The next power of two means the
smallest power >= the square side, and a symmetric (not periodic) Hann window
is used. Radial frequencies are cycles/pixel, with intervals [0,1/8),
[1/8,1/4), [1/4,1/2]. Corner frequencies beyond radius 1/2 are excluded from
the per-instance denominator. Channel means are removed after padding and
before windowing; no additional spectral DC-bin removal is performed.

Size bins are caller-supplied original annotation bbox width*height intervals
in pixels squared, lower inclusive and upper exclusive. They are not claimed
to reproduce RGBT-Tiny's official scale definitions. Clipping the crop to the
decoded image does not change that annotation-area classification.
"""

from collections import Counter
import math
from numbers import Real

import numpy as np
from PIL import Image


BAND_NAMES = ('low', 'mid', 'high')
ASSUMPTIONS = {
    'source': 'DyFCLT supplementary section 6.3; implementation choices below are not author code',
    'pixels': 'PIL RGB conversion of original files; no resizing, interpolation, or augmentation',
    'channel_policy': 'subtract each channel mean; average channel PSDs (no grayscale averaging)',
    'pair_boxes': 'thermal-reference annotation bbox applied to both modalities; assumes spatial alignment',
    'dimensions': 'decoded paired image dimensions are authoritative; do not rescale stale annotation coordinates',
    'crop_edges': 'clip continuous xywh bbox to decoded image; floor left/top and ceil right/bottom',
    'padding': 'reflect excluding edge pixel, first square then smallest enclosing power of two; odd extra bottom/right',
    'minimum_size': 'input height and width >= 2; final padded side >= 4 for a useful Hann window',
    'dc_window': 'per-channel padded-patch mean subtraction then symmetric 2D Hann; no second DC removal',
    'psd': 'abs(FFT)^2 / sum(window^2), averaged across channels; integrate with frequency cell area 1/N^2',
    'frequency': 'FFT-shifted isotropic radial cycles/pixel; [0,.125), [.125,.25), [.25,.5]',
    'normalization': 'each instance normalized over radius <= .5; corners outside disk excluded',
    'aggregation': 'unweighted mean of valid per-instance ratios, independently per modality',
    'selection': 'configured categories; ignore and iscrowd annotations excluded as in PairedCocoDataset',
    'size_bins': 'manual original annotation bbox width*height in pixels squared; [lower,upper), null upper is unbounded; not official RGBT-Tiny bins',
    'skips': 'invalid/empty boxes, too-small crops, nonfinite pixels, zero/nonfinite energy; counts retained',
}


class SkippedPatch(ValueError):
    """An instance cannot produce a meaningful normalized energy distribution."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def reflection_pad_square_power2(patch):
    """Reflect-pad HxW or HxWxC pixels in two centered stages without resizing."""
    patch = np.asarray(patch, dtype=np.float64)
    if patch.ndim not in (2, 3) or (patch.ndim == 3 and patch.shape[2] == 0):
        raise ValueError('Expected HxW or HxWxC image pixels')
    if not np.isfinite(patch).all():
        raise SkippedPatch('nonfinite_pixels')
    height, width = patch.shape[:2]
    if min(height, width) < 2:
        raise SkippedPatch('too_small_for_reflection')
    side = max(height, width)
    fft_side = 1 << (side - 1).bit_length()
    if fft_side < 4:
        raise SkippedPatch('too_small_for_hann')
    for target_side in (side, fft_side):
        extra_y, extra_x = target_side - patch.shape[0], target_side - patch.shape[1]
        padding = [(extra_y // 2, extra_y - extra_y // 2),
                   (extra_x // 2, extra_x - extra_x // 2)]
        if patch.ndim == 3:
            padding.append((0, 0))
        patch = np.pad(patch, padding, mode='reflect')
    return patch


def analyze_patch(patch):
    """Return compensated band energies and a unit-sum instance distribution.

    The unnormalized band energies integrate the window-compensated PSD. Their
    sum plus outside_disk_energy equals the compensated mean-square signal.
    Empty energy is skipped, never converted into an invented distribution.
    """
    original_shape = list(np.asarray(patch).shape[:2])
    padded = reflection_pad_square_power2(patch)
    if padded.ndim == 2:
        padded = padded[..., None]
    size = padded.shape[0]
    # Detect channelwise constants before averaging, whose rounding error can
    # otherwise invent low-frequency energy for values such as float64(0.1).
    if np.all(padded == padded[0, 0, :]):
        raise SkippedPatch('zero_energy')
    centered = padded - padded.mean(axis=(0, 1), keepdims=True)
    window = np.outer(np.hanning(size), np.hanning(size))
    window_power = float(np.square(window).sum())
    spectrum = np.fft.fftshift(np.fft.fft2(centered * window[..., None], axes=(0, 1)), axes=(0, 1))
    psd = np.square(np.abs(spectrum)).mean(axis=2) / window_power
    frequency = np.fft.fftshift(np.fft.fftfreq(size, d=1.))
    radial = np.hypot(frequency[:, None], frequency[None, :])
    masks = (radial < .125, (radial >= .125) & (radial < .25),
             (radial >= .25) & (radial <= .5))
    energies = np.array([psd[mask].sum() / (size * size) for mask in masks])
    energy = float(energies.sum())
    if not np.isfinite(energy):
        raise SkippedPatch('nonfinite_energy')
    if energy <= 0.:
        raise SkippedPatch('zero_energy')
    return {
        'ratios': (energies / energy).tolist(),
        'band_energies': energies.tolist(),
        'outside_disk_energy': float(psd[radial > .5].sum() / (size * size)),
        'crop_shape': original_shape,
        'fft_size': size,
        'window_mean_square': window_power / (size * size),
    }


class EnergyAccumulator:
    """Streaming equal-instance average, retaining explicit skip denominators."""

    def __init__(self):
        self.valid = 0
        self.ratio_sum = np.zeros(3, dtype=np.float64)
        self.skipped = Counter()

    def add(self, result):
        self.valid += 1
        self.ratio_sum += np.asarray(result['ratios'], dtype=np.float64)

    def skip(self, reason):
        self.skipped[reason] += 1

    def summary(self):
        skipped = sum(self.skipped.values())
        return {'attempted': self.valid + skipped, 'valid': self.valid, 'skipped': skipped,
                'skipped_by_reason': dict(sorted(self.skipped.items())),
                'mean_ratios': (self.ratio_sum / self.valid).tolist() if self.valid else None}


def validate_scale_bins(scale_bins):
    """Validate user-defined disjoint area intervals, preserving display order."""
    if scale_bins is None:
        return {}
    if not isinstance(scale_bins, dict):
        raise ValueError('scale bins must be a JSON object mapping names to [lower, upper]')
    validated = {}
    for name, bounds in scale_bins.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError('scale bin names must be nonempty strings')
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f'Scale bin {name!r} must have [lower, upper] bounds')
        lower, upper = bounds
        if not isinstance(lower, Real) or isinstance(lower, bool) or not math.isfinite(lower) or lower < 0:
            raise ValueError(f'Invalid lower area bound for {name!r}')
        if upper is not None and (not isinstance(upper, Real) or isinstance(upper, bool)
                                  or not math.isfinite(upper) or upper <= lower):
            raise ValueError(f'Invalid upper area bound for {name!r}; use null for no upper limit')
        validated[name] = [float(lower), None if upper is None else float(upper)]
    ordered = sorted(validated.items(), key=lambda entry: entry[1][0])
    for (left, (_, upper)), (right, (lower, _)) in zip(ordered, ordered[1:]):
        if upper is None or upper > lower:
            raise ValueError(f'Overlapping scale bins: {left!r} and {right!r}')
    return validated


def scale_bin_for_area(area, scale_bins):
    """Return one manual bin name, or None when the area is outside all bins."""
    for name, (lower, upper) in scale_bins.items():
        if lower <= area and (upper is None or area < upper):
            return name
    return None


def _crop_bounds(bbox, width, height):
    try:
        x, y, box_width, box_height = map(float, bbox)
    except (TypeError, ValueError):
        raise SkippedPatch('invalid_bbox')
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)) or min(box_width, box_height) <= 0:
        raise SkippedPatch('invalid_bbox')
    area = box_width * box_height
    if not math.isfinite(area):
        raise SkippedPatch('invalid_bbox')
    left, top = max(0., x), max(0., y)
    right, bottom = min(float(width), x + box_width), min(float(height), y + box_height)
    if right <= left or bottom <= top:
        raise SkippedPatch('empty_bbox_after_clipping')
    return (math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)), area


def analyze_dataset(dataset, max_instances=None, scale_bins=None):
    """Read original paired files using PairedCocoDataset metadata only.

    The optional bound counts eligible annotations attempted, including skipped
    geometry or zero-energy instances. Iteration follows source JSON image and
    annotation order, so a bounded run is a smoke check, not a random sample.
    """
    if max_instances is not None and (isinstance(max_instances, bool)
                                      or not isinstance(max_instances, int) or max_instances < 1):
        raise ValueError('max_instances must be a positive integer')
    bins = validate_scale_bins(scale_bins)
    modes = ('visible', 'thermal')
    overall = {mode: EnergyAccumulator() for mode in modes}
    groups = {mode: {name: EnergyAccumulator() for name in bins} for mode in modes}
    available = sum(len(dataset.annotations_by_image[record['id']]) for record in dataset.images)
    counts = {'selected_annotations_available': available, 'attempted_instances': 0,
              'images_read': 0, 'stale_metadata_images': 0, 'valid_in_both_modalities': 0,
              'instances_outside_manual_scale_bins': 0}
    for record in dataset.images:
        if max_instances is not None and counts['attempted_instances'] >= max_instances:
            break
        annotations = dataset.annotations_by_image[record['id']]
        if not annotations:
            continue
        paths = dataset._pair_paths(record['file_name'])
        pixels = {}
        for mode, path in zip(modes, paths):
            with Image.open(path) as image:
                pixels[mode] = np.array(image.convert('RGB'), copy=True)
        if pixels['visible'].shape != pixels['thermal'].shape:
            raise ValueError(f"Paired image size mismatch for image {record['id']}")
        height, width = pixels['thermal'].shape[:2]
        counts['images_read'] += 1
        if record.get('width') != width or record.get('height') != height:
            counts['stale_metadata_images'] += 1
        for annotation in annotations:
            if max_instances is not None and counts['attempted_instances'] >= max_instances:
                break
            counts['attempted_instances'] += 1
            try:
                (left, top, right, bottom), area = _crop_bounds(annotation.get('bbox'), width, height)
            except SkippedPatch as error:
                for accumulator in overall.values():
                    accumulator.skip(error.reason)
                continue
            group = scale_bin_for_area(area, bins)
            if bins and group is None:
                counts['instances_outside_manual_scale_bins'] += 1
            valid_pair = True
            for mode in modes:
                targets = [overall[mode]] + ([groups[mode][group]] if group is not None else [])
                try:
                    result = analyze_patch(pixels[mode][top:bottom, left:right])
                except SkippedPatch as error:
                    valid_pair = False
                    for accumulator in targets:
                        accumulator.skip(error.reason)
                else:
                    for accumulator in targets:
                        accumulator.add(result)
            counts['valid_in_both_modalities'] += int(valid_pair)
    return {
        'schema_version': 1,
        'assumptions': dict(ASSUMPTIONS),
        'band_names': list(BAND_NAMES),
        'band_intervals_cycles_per_pixel': ['[0,0.125)', '[0.125,0.25)', '[0.25,0.5]'],
        'category_ids': dataset.category_ids,
        'category_names': dataset.category_names,
        'ann_file': str(dataset.ann_file),
        'image_root': str(dataset.image_root),
        'max_instances': max_instances,
        'complete_dataset': counts['attempted_instances'] == available,
        'scale_bins': bins,
        'counts': counts,
        'modalities': {mode: {'all': overall[mode].summary(),
                              'by_scale': {name: accumulator.summary() for name, accumulator in groups[mode].items()}}
                       for mode in modes},
    }


def save_band_plot(report, path):
    """Save a paired bar chart for all instances and any explicit manual bins."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    labels = ['all'] + list(report['scale_bins'])
    figure, axes = plt.subplots(len(labels), 1, figsize=(7, 3.2 * len(labels)), squeeze=False)
    positions = np.arange(3)
    for index, label in enumerate(labels):
        axis = axes[index, 0]
        for mode, offset, color in (('visible', -.18, '#D55E00'), ('thermal', .18, '#0072B2')):
            summary = report['modalities'][mode]['all'] if index == 0 else report['modalities'][mode]['by_scale'][label]
            ratios = summary['mean_ratios']
            values = np.full(3, np.nan) if ratios is None else 100 * np.asarray(ratios)
            axis.bar(positions + offset, values, width=.36, color=color,
                     label=f"{mode} (valid n={summary['valid']})")
        axis.set_xticks(positions, report['band_names'])
        axis.set_ylim(0, 100)
        axis.set_ylabel('Mean energy (%)')
        axis.set_title('All eligible instances' if index == 0 else f'Manual area bin: {label}')
        axis.legend()
    figure.suptitle('Per-instance radial energy ratios' + ('' if report['complete_dataset'] else ' (bounded subset)'))
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
