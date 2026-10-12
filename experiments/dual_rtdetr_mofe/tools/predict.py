#!/usr/bin/env python3
"""Infer one RGB/thermal pair using the saved class mapping and input size."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from PIL import Image, ImageDraw
from dyfclt.model import DyFCLT
from dyfclt.runtime import load_checkpoint, postprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', choices=['auto', 'model', 'ema'], default='auto',
                        help='auto uses EMA when present; model selects raw training weights')
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--thermal', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score', type=float, default=.3)
    parser.add_argument('--output', default='outputs/prediction.json')
    parser.add_argument('--visualization', help='Optional annotated RGB output path')
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = state['config']
    rgb, thermal = Image.open(args.rgb).convert('RGB'), Image.open(args.thermal).convert('RGB')
    if rgb.size != thermal.size:
        raise ValueError('RGB and thermal image dimensions must match')
    height, width = config['data']['image_size']
    def tensor(image):
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.array(resized).copy()).permute(2,0,1).float().div(255).unsqueeze(0).to(args.device)
    model = DyFCLT(**dict(config['model'], pretrained=False)).to(args.device).eval()
    load_checkpoint(state, model, weights=args.weights)
    with torch.no_grad():
        outputs = model(tensor(rgb), tensor(thermal))
    target = dict(orig_size=torch.tensor([rgb.height,rgb.width]), image_id=torch.tensor(0))
    records = postprocess(outputs, [target], state['category_ids'], score_threshold=args.score)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2))
    if args.visualization:
        draw = ImageDraw.Draw(rgb)
        for record in records:
            x,y,w,h = record['bbox']
            draw.rectangle((x,y,x+w,y+h), outline='red', width=2)
            draw.text((x,y), f"{record['category_id']} {record['score']:.2f}", fill='yellow')
        destination = Path(args.visualization)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(destination)
    print(f'Saved {len(records)} detections to {output}')

if __name__ == '__main__':
    main()
