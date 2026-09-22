"""Build a three-seed TRPC report in the COXNet Table 1 column order.

Example:
    python tools/misc/build_trpc_table1.py \
        --baseline-root /path/to/ptcc \
        --out docs/trpc_table1_ko.md

``baseline-root`` is optional. When supplied, it must contain
``A_clfm_seed0``, ``A_clfm_seed1``, and ``A_clfm_seed2`` directories.
"""
import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRPC_ROOT = ROOT / 'work_dir/coxmamba/rgbtdroneperson/trpc'
METRICS = (
    ('mAP25', 'bbox_mAP_25'),
    ('mAP50', 'bbox_mAP_50'),
    ('tiny', 'bbox_mAP_50_tiny'),
    ('tiny1', 'bbox_mAP_50_tiny1'),
    ('tiny2', 'bbox_mAP_50_tiny2'),
    ('tiny3', 'bbox_mAP_50_tiny3'),
    ('small', 'bbox_mAP_50_small'),
)
HEADER = (
    '| Method | Seed | Best epoch | mAP25 | mAP50 (all) | '
    'mAP50 (tiny) | tiny1 | tiny2 | tiny3 | small |')
SEPARATOR = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'


def read_run(run_dir):
    """Read all validation rows, keyed by epoch, from an MMDet work dir."""
    rows = {}
    for log_path in sorted(run_dir.glob('*.log.json')):
        for line in log_path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            if record.get('mode') != 'val':
                continue
            if not all(source in record for _, source in METRICS):
                continue
            epoch = int(record['epoch'])
            rows[epoch] = {
                name: 100.0 * float(record[source])
                for name, source in METRICS
            }
    if not rows:
        raise RuntimeError(f'No validation metrics found in {run_dir}')
    return rows


def load_group(root, prefix):
    runs = {}
    for seed in range(3):
        values = read_run(root / f'{prefix}{seed}')
        if 12 not in values:
            raise RuntimeError(f'Seed {seed} did not finish 12 epochs')
        best_epoch = max(
            values, key=lambda epoch: (values[epoch]['mAP50'], -epoch))
        runs[seed] = dict(
            best_epoch=best_epoch,
            best=values[best_epoch],
            final=values[max(values)],
        )
    return runs


def summary(runs, key='best'):
    result = {}
    for metric, _ in METRICS:
        values = [runs[seed][key][metric] for seed in range(3)]
        result[metric] = (
            statistics.mean(values), statistics.stdev(values))
    return result


def fmt(value):
    return f'{value:.2f}'


def fmt_summary(value):
    mean, std = value
    return f'{mean:.2f} ± {std:.2f}'


def seed_row(method, seed, run):
    values = run['best']
    cells = [fmt(values[name]) for name, _ in METRICS]
    return (
        f'| {method} | {seed} | {run["best_epoch"]} | '
        + ' | '.join(cells) + ' |')


def summary_row(method, values, epoch='10/12/12'):
    cells = [fmt_summary(values[name]) for name, _ in METRICS]
    return f'| **{method}** | 0/1/2 | {epoch} | ' + ' | '.join(cells) + ' |'


def delta_row(trpc, baseline):
    cells = []
    for metric, _ in METRICS:
        differences = [
            trpc[seed]['best'][metric] - baseline[seed]['best'][metric]
            for seed in range(3)
        ]
        cells.append(
            f'{statistics.mean(differences):+.2f} ± '
            f'{statistics.stdev(differences):.2f}')
    return '| **Paired Δ (TRPC−COXNet)** | 0/1/2 | — | ' + ' | '.join(cells) + ' |'


def render(trpc, baseline=None):
    trpc_summary = summary(trpc)
    final_summary = summary(trpc, key='final')
    lines = [
        '# TRPC 결과 — COXNet Table 1 형식',
        '',
        '- Dataset: RGBTDronePerson validation 1,225 images',
        '- Training: 12 epochs, deterministic seeds 0/1/2',
        '- Selection: each seed checkpoint with the highest `bbox_mAP_50`; '
        'ties select the earlier epoch',
        '- `±` is the sample standard deviation over three seeds',
        '',
        '## Table 1 — best checkpoint per seed',
        '',
        HEADER,
        SEPARATOR,
    ]
    for seed in range(3):
        lines.append(seed_row('TRPC', seed, trpc[seed]))
    lines.append(summary_row('TRPC mean ± SD', trpc_summary))

    if baseline is not None:
        baseline_summary = summary(baseline)
        lines.extend([
            '',
            '## Controlled comparison with COXNet',
            '',
            HEADER,
            SEPARATOR,
            summary_row('COXNet mean ± SD', baseline_summary, '10/9/12'),
            summary_row('TRPC mean ± SD', trpc_summary),
            delta_row(trpc, baseline),
        ])

    lines.extend([
        '',
        '## Final epoch reference',
        '',
        f'- TRPC epoch-12 mAP50 (all): '
        f'**{fmt_summary(final_summary["mAP50"])}**',
        '',
        '## Interpretation',
        '',
    ])
    if baseline is None:
        lines.append(
            '- A COXNet baseline root was not supplied, so no paired comparison '
            'was generated.')
    else:
        trpc_map50 = trpc_summary['mAP50']
        baseline_map50 = summary(baseline)['mAP50']
        differences = [
            trpc[seed]['best']['mAP50'] - baseline[seed]['best']['mAP50']
            for seed in range(3)
        ]
        lines.extend([
            f'- COXNet mAP50: **{fmt_summary(baseline_map50)}**; '
            f'TRPC mAP50: **{fmt_summary(trpc_map50)}**.',
            f'- Paired mAP50 difference: '
            f'**{statistics.mean(differences):+.2f} ± '
            f'{statistics.stdev(differences):.2f}**.',
            '- The mAP50 difference is smaller than the seed variation. This '
            'table does not support a claim that TRPC improves overall AP.',
            '- `tiny1` has high variance and should not be interpreted from its '
            'mean alone.',
        ])
    return '\n'.join(lines) + '\n'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trpc-root', type=Path, default=DEFAULT_TRPC_ROOT)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--out', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    trpc = load_group(args.trpc_root, 'TRPC_seed')
    baseline = (
        load_group(args.baseline_root, 'A_clfm_seed')
        if args.baseline_root else None)
    report = render(trpc, baseline)
    print(report, end='')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
