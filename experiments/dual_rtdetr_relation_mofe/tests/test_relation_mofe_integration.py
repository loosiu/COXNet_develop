import copy
import json
from pathlib import Path
import unittest

import torch

from dyfclt.model import DyFCLT
from dyfclt.losses import build_criterion
from dyfclt.training import build_optimizer, ModelEMA
from dyfclt.runtime import load_config
from dyfclt.frequency_training import gate_statistics
from tools.train import validate_detection_config

ROOT = Path(__file__).resolve().parents[1]


class RelationIntegrationTests(unittest.TestCase):
    def opts(self):
        return dict(num_classes=2, hidden_dim=32, num_queries=8, num_decoder_layers=1,
                    num_denoising=4, architecture='dual_rtdetr', pretrained=False)

    def create(self):
        return DyFCLT(**self.opts(), frequency_utility=dict(gate_mode='relation_mofe',
            dim=8, expert_hidden=8, router_hidden=8, output_mode='direct',
            insertion_point='projected_p3'))

    def data(self):
        return torch.randn(2, 3, 64, 96), torch.randn(2, 3, 64, 96), [
            dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5, .5, .2, .2]])) for _ in range(2)]

    def test_A_shared_initialization_and_RNG_unchanged(self):
        torch.manual_seed(0); a = DyFCLT(**self.opts()); rng = torch.get_rng_state()
        torch.manual_seed(0); model = self.create()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_detection_loss_updates_all_relation_experts_and_router(self):
        torch.manual_seed(0); model = self.create()
        optimizer = build_optimizer(model, dict(lr=.00025, weight_decay=.0001, optimizer_groups='rtdetr'))
        params = dict(model.encoder.frequency_module.named_parameters())
        before = {name: p.detach().clone() for name, p in params.items()}
        ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in ids for p in params.values()))
        self.assertFalse(any('gamma' in name for name in params))
        r, t, targets = self.data(); out = model(r, t, targets, return_frequency_aux=True)
        losses = build_criterion(2, mask_weight=0)(out, targets)
        self.assertFalse(any('utility' in key or 'kl' in key or 'mask' in key for key in losses))
        sum(losses.values()).backward(); optimizer.step()
        for name, value in params.items():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
            self.assertGreater(float(value.grad.abs().sum()), 0, name)
            self.assertFalse(torch.equal(value, before[name]), name)
        for stats in gate_statistics(out['frequency_aux']).values():
            self.assertGreater(sum(stats['weight_spatial_std_mean']), 0)
            self.assertEqual(len(stats['relation_attention_mean']), 4)
            self.assertGreater(stats['output_rms_mean'], 0)

    def test_replay_and_regular_forward_exact_and_all_pyramid_levels_change(self):
        model = self.create().eval(); r, t, _ = self.data()
        with torch.no_grad():
            cache = model.prepare_frequency(r, t)
            out, replay = model(r, t), model.replay_frequency(cache)
            for key in ['pred_logits', 'pred_boxes']:
                self.assertTrue(torch.equal(out[key], replay[key]), key)
            original = model.encoder.fuse_frequency_streams(cache['rgb_features'], cache['thermal_features'])
            rf, tf = list(cache['rgb_features']), list(cache['thermal_features'])
            rf[0], tf[0] = model.encoder.frequency_module.mix(cache['frequency'])
            enhanced = model.encoder.fuse_frequency_streams(rf, tf)
            for level in range(3):
                self.assertFalse(torch.equal(original[level], enhanced[level]), str(level))

    def test_EMA_finite_repeatable_without_unused_scale_parameters(self):
        model = self.create(); ema = ModelEMA(model)
        ema.update(model); r, t, _ = self.data()
        with torch.no_grad(): left, right = ema.module(r, t), ema.module(r, t)
        for key in ['pred_logits', 'pred_boxes']:
            self.assertTrue(torch.isfinite(left[key]).all())
            self.assertTrue(torch.equal(left[key], right[key]))

    def test_direct_and_residual_recipes_preserve_A_training(self):
        a = json.loads((ROOT/'results/A_original_config.json').read_text())
        for filename, mode in [('rgbtdroneperson_relation_direct_seed0.yaml', 'direct'),
                               ('rgbtdroneperson_relation_residual_seed0.yaml', 'residual')]:
            cfg = load_config(ROOT/'configs'/filename)
            self.assertTrue(validate_detection_config(cfg))
            self.assertEqual(cfg['model']['frequency_utility']['output_mode'], mode)
            current, baseline = copy.deepcopy(cfg), copy.deepcopy(a)
            current['model'].pop('frequency_utility')
            for split in ('train', 'val'):
                for field in ('ann_file', 'image_root'):
                    current['data'][split][field] = baseline['data'][split][field]
            for value in [current, baseline]: value['train'].pop('output_dir', None)
            self.assertEqual(current, baseline)


if __name__ == '__main__': unittest.main()
