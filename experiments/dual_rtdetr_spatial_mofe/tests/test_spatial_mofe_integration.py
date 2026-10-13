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


class SpatialIntegrationTests(unittest.TestCase):
    def opts(self):
        return dict(num_classes=2, hidden_dim=32, num_queries=8, num_decoder_layers=1,
                    num_denoising=4, architecture='dual_rtdetr', pretrained=False)

    def create(self):
        return DyFCLT(**self.opts(), frequency_utility=dict(gate_mode='spatial_mofe',
            dim=8, expert_hidden=8, router_hidden=8, residual_scale=.1,
            residual_multiplier=.5, insertion_point='projected_p3'))

    def data(self):
        return torch.randn(2, 3, 64, 96), torch.randn(2, 3, 64, 96), [
            dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5, .5, .2, .2]])) for _ in range(2)]

    def test_A_initial_tensors_and_random_state_are_unchanged(self):
        torch.manual_seed(0); baseline = DyFCLT(**self.opts()); rng = torch.get_rng_state()
        torch.manual_seed(0); model = self.create()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_detection_loss_updates_all_parameters_and_logs_spatial_variation(self):
        torch.manual_seed(0); model = self.create()
        optimizer = build_optimizer(model, dict(lr=.00025, weight_decay=.0001, optimizer_groups='rtdetr'))
        params = dict(model.encoder.frequency_module.named_parameters())
        self.assertEqual(len(params), 36)
        before = {name: p.detach().clone() for name, p in params.items()}
        ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in ids for p in params.values()))
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
            self.assertEqual(len(stats['routing_weight_mean']), 4)
            self.assertGreater(sum(stats['weight_spatial_std_mean']), 0)
            self.assertEqual(stats['residual_multiplier_mean'], .5)

    def test_replay_and_zero_gamma_bypass_are_exact(self):
        model = self.create().eval(); r, t, _ = self.data()
        with torch.no_grad():
            cache = model.prepare_frequency(r, t)
            out, replay = model(r, t), model.replay_frequency(cache)
            for key in ['pred_logits', 'pred_boxes']:
                self.assertTrue(torch.equal(out[key], replay[key]), key)
            model.encoder.frequency_module.gamma_rgb.zero_()
            model.encoder.frequency_module.gamma_thermal.zero_()
            out = model(r, t)
            model.encoder.frequency_module = None
            bypass = model(r, t)
        for key in ['pred_logits', 'pred_boxes']:
            self.assertTrue(torch.equal(out[key], bypass[key]), key)

    def test_enhancement_is_before_FPN_and_changes_P4_P5(self):
        model = self.create().eval(); r, t, _ = self.data()
        with torch.no_grad():
            cache = model.prepare_frequency(r, t)
            self.assertEqual(cache['frequency']['w'].shape[-2:], cache['rgb_features'][0].shape[-2:])
            original = model.encoder.fuse_frequency_streams(cache['rgb_features'], cache['thermal_features'])
            rf, tf = list(cache['rgb_features']), list(cache['thermal_features'])
            rf[0], tf[0] = model.encoder.frequency_module.mix(cache['frequency'])
            enhanced = model.encoder.fuse_frequency_streams(rf, tf)
            for level in range(3):
                self.assertFalse(torch.equal(original[level], enhanced[level]), str(level))

    def test_EMA_is_finite_and_repeatable(self):
        model = self.create(); ema = ModelEMA(model)
        with torch.no_grad(): model.encoder.frequency_module.gamma_rgb.add_(.01)
        ema.update(model); r, t, _ = self.data()
        with torch.no_grad(): left, right = ema.module(r, t), ema.module(r, t)
        for key in ['pred_logits', 'pred_boxes']:
            self.assertTrue(torch.isfinite(left[key]).all())
            self.assertTrue(torch.equal(left[key], right[key]))

    def test_fresh_seed0_recipe_matches_A_and_B_pre(self):
        cfg = load_config(ROOT/'configs/rgbtdroneperson_spatial_mofe_seed0.yaml')
        self.assertTrue(validate_detection_config(cfg))
        a = json.loads((ROOT/'results/A_original_config.json').read_text())
        current = copy.deepcopy(cfg)
        current['model'].pop('frequency_utility')
        for split in ('train', 'val'):
            for field in ('ann_file', 'image_root'):
                current['data'][split][field] = a['data'][split][field]
        for x in (current, a): x['train'].pop('output_dir', None)
        self.assertEqual(current, a)
        old = json.loads((ROOT/'results/B_pre_original_config.json').read_text())
        current = copy.deepcopy(cfg)
        current['model']['frequency_utility']['gate_mode'] = 'residual_mofe'
        current['model']['frequency_utility'].pop('residual_multiplier')
        for split in ('train', 'val'):
            for field in ('ann_file', 'image_root'):
                current['data'][split][field] = old['data'][split][field]
        for x in (current, old): x['train'].pop('output_dir', None)
        self.assertEqual(current, old)


if __name__ == '__main__':
    unittest.main()
