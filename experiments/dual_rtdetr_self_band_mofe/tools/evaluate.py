#!/usr/bin/env python3
"""Standard COCO evaluation for a saved DyFCLT checkpoint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dyfclt.model import DyFCLT
from dyfclt.runtime import load_config, load_checkpoint, make_dataset, make_loader, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', choices=['auto', 'model', 'ema'], default='auto',
                        help='auto uses EMA when present; model selects raw training weights')
    parser.add_argument('--config', help='Optional data path overrides; model always comes from checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='outputs/evaluation')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-batches', type=int, help='Partial debug evaluation, explicitly marked in metrics')
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = load_config(args.config) if args.config else saved['config']
    kwargs = dict(saved['config']['model'], pretrained=False)
    model = DyFCLT(**kwargs).to(args.device)
    load_checkpoint(saved, model, weights=args.weights)
    dataset = make_dataset(config, 'val')
    if saved['category_ids'] != dataset.category_ids:
        raise ValueError('Checkpoint and validation category mappings differ')
    loader = make_loader(dataset, args.batch_size, args.workers)
    result = evaluate(model, loader, args.device, config['data']['val']['ann_file'],
                      dataset.category_ids, args.output_dir, args.max_batches,
                      protocol=config.get('evaluation', {}).get('protocol', 'coco'))
    result['checkpoint_weights'] = 'ema' if args.weights == 'auto' and saved.get('ema') is not None else ('model' if args.weights == 'auto' else args.weights)
    (Path(args.output_dir) / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
