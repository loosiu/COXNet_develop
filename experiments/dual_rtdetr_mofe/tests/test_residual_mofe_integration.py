import json
from pathlib import Path
import unittest

import torch
from dyfclt.model import DyFCLT
from dyfclt.losses import build_criterion
from dyfclt.training import build_optimizer, ModelEMA
from dyfclt.frequency_training import gate_statistics
from dyfclt.runtime import load_config
from tools.train import validate_detection_config

ROOT = Path(__file__).resolve().parents[1]


class ResidualMoFEIntegrationTests(unittest.TestCase):
    def opts(self):
        return dict(num_classes=2, hidden_dim=32, num_queries=8, num_decoder_layers=1,
                    num_denoising=4, architecture='dual_rtdetr', pretrained=False)

    def create(self):
        return DyFCLT(**self.opts(), frequency_utility=dict(gate_mode='residual_mofe',
            dim=8, expert_hidden=8, router_hidden=8, residual_scale=.1))

    def data(self):
        return torch.randn(2, 3, 64, 96), torch.randn(2, 3, 64, 96), [
            dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5, .5, .2, .2]])) for _ in range(2)]

    def test_A_initial_tensors_and_cpu_rng_match(self):
        torch.manual_seed(0)
        a = DyFCLT(**self.opts())
        rng = torch.get_rng_state()
        torch.manual_seed(0)
        model = self.create()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, tensor in a.state_dict().items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[name]), name)

    def test_all_original_RGB_bank_initial_values_match_including_router(self):
        torch.manual_seed(0)
        old = DyFCLT(**self.opts(), frequency_utility=dict(gate_mode='s3am_mofe', dim=8,
                                                        expert_hidden=8, router_hidden=8))
        torch.manual_seed(0)
        new = self.create()
        for name, tensor in old.encoder.frequency_module.state_dict().items():
            self.assertTrue(torch.equal(tensor, new.encoder.frequency_module.rgb_from_thermal.state_dict()[name]), name)

    def test_detection_only_gradients_and_updates_all_36_module_parameters(self):
        torch.manual_seed(0)
        model = self.create()
        optimizer = build_optimizer(model, dict(lr=.00025, weight_decay=.0001, optimizer_groups='rtdetr'))
        params = dict(model.encoder.frequency_module.named_parameters())
        self.assertEqual(len(params), 36)
        before = {name: p.detach().clone() for name, p in params.items()}
        ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in ids for p in params.values()))
        r, t, targets = self.data()
        out = model(r, t, targets, return_frequency_aux=True)
        losses = build_criterion(2, mask_weight=0)(out, targets)
        self.assertFalse(any('kl' in key or 'utility' in key or 'mask' in key for key in losses))
        sum(losses.values()).backward()
        optimizer.step()
        for name, value in params.items():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
            self.assertGreater(float(value.grad.abs().sum()), 0, name)
            self.assertFalse(torch.equal(value, before[name]), name)
        stats = gate_statistics(out['frequency_aux'])
        for direction in ['rgb_from_thermal', 'thermal_from_rgb']:
            self.assertGreater(stats[direction]['relative_update_mean'], 0)
            self.assertEqual(len(stats[direction]['routing_weight_mean']), 4)

    def test_zero_scales_replay_and_bypass_predictions_are_identical(self):
        model = self.create().eval()
        model.encoder.frequency_module.gamma_rgb.data.zero_()
        model.encoder.frequency_module.gamma_thermal.data.zero_()
        r, t, _ = self.data()
        with torch.no_grad():
            cache = model.prepare_frequency(r, t)
            replay, out = model.replay_frequency(cache), model(r, t)
            model.encoder.frequency_module = None
            bypass = model(r, t)
        for key in ['pred_logits', 'pred_boxes']:
            self.assertTrue(torch.equal(replay[key], out[key]))
            self.assertTrue(torch.equal(out[key], bypass[key]))

    def test_only_P3_changes_before_original_multiscale_fusion(self):
        model = self.create().eval()
        with torch.no_grad():
            r, t, _ = self.data()
            cache = model.prepare_frequency(r, t)
            rf, tf = list(cache['rgb_features']), list(cache['thermal_features'])
            rf[0], tf[0] = model.encoder.frequency_module.mix(cache['frequency'])
            self.assertFalse(torch.equal(rf[0], cache['rgb_features'][0]))
            self.assertFalse(torch.equal(tf[0], cache['thermal_features'][0]))
            original = model.encoder.fuse_streams(cache['rgb_features'], cache['thermal_features'])
            fused = model.encoder.fuse_streams(rf, tf)
            for level in [1, 2]: self.assertTrue(torch.equal(original[level], fused[level]))

    def test_ema_preserves_banks_and_scales_and_is_repeatable(self):
        model = self.create()
        ema = ModelEMA(model)
        with torch.no_grad(): model.encoder.frequency_module.gamma_rgb.add_(.01)
        ema.update(model)
        self.assertGreater(float(ema.module.encoder.frequency_module.gamma_rgb), .1)
        self.assertLessEqual(float(ema.module.encoder.frequency_module.gamma_rgb), .11)
        r, t, _ = self.data()
        with torch.no_grad(): left, right = ema.module(r, t), ema.module(r, t)
        for key in ['pred_logits', 'pred_boxes']: self.assertTrue(torch.equal(left[key], right[key]))

    def test_recipe_matches_A_except_module_and_output(self):
        path = ROOT/'configs/rgbtdroneperson_residual_mofe.yaml'
        self.assertTrue(path.exists(), 'Matched fresh configuration is missing')
        cfg = load_config(path)
        self.assertTrue(validate_detection_config(cfg))
        self.assertNotIn('routing_regularization', cfg)
        self.assertNotIn('initialization', cfg)
        a = load_config(ROOT/'configs/rgbtdroneperson_dual_rtdetr.yaml')
        cfg['model'].pop('frequency_utility')
        for value in [a, cfg]: value['train'].pop('output_dir')
        self.assertEqual(a, cfg)


if __name__ == '__main__': unittest.main()
