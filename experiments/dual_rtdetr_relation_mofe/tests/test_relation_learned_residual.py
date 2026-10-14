import copy
import json
from pathlib import Path
import unittest

import torch

from dyfclt.frequency_training import gate_statistics
from dyfclt.losses import build_criterion
from dyfclt.model import DyFCLT
from dyfclt.relation_mofe import BidirectionalRelationMoFE
from dyfclt.runtime import load_config
from dyfclt.training import build_optimizer, ModelEMA
from tools.train import validate_detection_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/rgbtdroneperson_relation_learned_residual_seed0.yaml'


class LearnedResidualRelationTests(unittest.TestCase):
    def test_primary_recipe_uses_original_feature_with_independent_learned_strengths(self):
        self.assertTrue(CONFIG.is_file(), 'The learned-residual primary config must exist')
        cfg = load_config(CONFIG)
        self.assertTrue(validate_detection_config(cfg))
        options = cfg['model']['frequency_utility']
        self.assertEqual(options['output_mode'], 'residual')
        self.assertEqual(options['residual_scale'], .1)
        self.assertEqual(options['residual_multiplier'], 1.)
        baseline = json.loads((ROOT/'results/A_original_config.json').read_text())
        current = copy.deepcopy(cfg); current['model'].pop('frequency_utility')
        for split in ('train', 'val'):
            for field in ('ann_file', 'image_root'):
                current['data'][split][field] = baseline['data'][split][field]
        for value in (current, baseline): value['train'].pop('output_dir', None)
        self.assertEqual(current, baseline)
        self.assertNotEqual(cfg['train']['output_dir'], load_config(ROOT/'configs/rgbtdroneperson_relation_residual_seed0.yaml')['train']['output_dir'])

    def test_logged_strengths_follow_current_parameters_without_half_multiplier(self):
        module = BidirectionalRelationMoFE(channels=8, dim=4, expert_hidden=4,
            router_hidden=4, output_mode='residual', residual_scale=.1, residual_multiplier=1.)
        with torch.no_grad():
            module.gamma_rgb.fill_(.37); module.gamma_thermal.fill_(-.12)
        r, t = torch.randn(2, 8, 5, 7), torch.randn(2, 8, 5, 7)
        _, _, aux = module(r, t)
        self.assertIn('residual_scale', aux['diagnostics'])
        self.assertIn('effective_residual_scale', aux['diagnostics'])
        values = gate_statistics(aux)
        for name, expected in [('rgb_from_thermal', .37), ('thermal_from_rgb', -.12)]:
            self.assertAlmostEqual(values[name]['residual_scale_mean'], expected)
            self.assertAlmostEqual(values[name]['effective_residual_scale_mean'], expected)
            self.assertEqual(values[name]['residual_multiplier_mean'], 1.)
        self.assertFalse(aux['diagnostics']['residual_scale'].requires_grad)

    def test_detection_loss_updates_both_strengths_and_all_experts(self):
        torch.manual_seed(0)
        cfg = load_config(CONFIG)
        options = dict(cfg['model']['frequency_utility'], dim=8, expert_hidden=8, router_hidden=8)
        model = DyFCLT(num_classes=2, hidden_dim=32, num_queries=8, num_decoder_layers=1,
            num_denoising=4, architecture='dual_rtdetr', pretrained=False, frequency_utility=options)
        module = model.encoder.frequency_module
        params = dict(module.named_parameters())
        self.assertEqual(len(params), 100)
        before = {name: p.detach().clone() for name, p in params.items()}
        optimizer = build_optimizer(model, cfg['train']); ema = ModelEMA(model)
        ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in ids for p in params.values()))
        self.assertIsNot(module.gamma_rgb, module.gamma_thermal)
        r, t = torch.randn(2, 3, 64, 96), torch.randn(2, 3, 64, 96)
        targets = [dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5, .5, .2, .2]])) for _ in range(2)]
        out = model(r, t, targets, return_frequency_aux=True)
        losses = build_criterion(2, mask_weight=0)(out, targets)
        self.assertFalse(any('utility' in key or 'kl' in key or 'mask' in key for key in losses))
        sum(losses.values()).backward(); optimizer.step(); ema.update(model)
        for name, p in params.items():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(float(p.grad.abs().sum()), 0, name)
            self.assertFalse(torch.equal(p, before[name]), name)
        self.assertFalse(torch.equal(ema.module.encoder.frequency_module.gamma_rgb, before['gamma_rgb']))
        self.assertFalse(torch.equal(ema.module.encoder.frequency_module.gamma_thermal, before['gamma_thermal']))

    def test_zero_strength_preserves_baseline_predictions(self):
        torch.manual_seed(0)
        cfg = load_config(CONFIG)
        options = dict(cfg['model']['frequency_utility'], dim=8, expert_hidden=8, router_hidden=8)
        model = DyFCLT(num_classes=2, hidden_dim=32, num_queries=8, num_decoder_layers=1,
            num_denoising=4, architecture='dual_rtdetr', pretrained=False, frequency_utility=options).eval()
        r, t = torch.randn(2, 3, 64, 96), torch.randn(2, 3, 64, 96)
        with torch.no_grad():
            model.encoder.frequency_module.gamma_rgb.zero_(); model.encoder.frequency_module.gamma_thermal.zero_()
            zero = model(r, t)
            model.encoder.frequency_module = None
            baseline = model(r, t)
        for name in ['pred_logits', 'pred_boxes']:
            self.assertTrue(torch.equal(zero[name], baseline[name]), name)


if __name__ == '__main__': unittest.main()
