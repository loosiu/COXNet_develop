import copy
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from dyfclt.frequency_training import gate_statistics
from dyfclt.losses import build_criterion
from dyfclt.model import DyFCLT
from dyfclt.runtime import load_config
from dyfclt.s3am_mofe import build_frequency_pairs
from dyfclt.spatial_mofe import SpatialResidualMoFE
from dyfclt.training import build_optimizer, ModelEMA
from tools.train import validate_detection_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/rgbtdroneperson_cnn_mofe_zero_seed0.yaml'


class CNNZeroMoFETests(unittest.TestCase):
    def options(self):
        return dict(num_classes=2, hidden_dim=32, num_queries=8, num_decoder_layers=1,
            num_denoising=4, architecture='dual_rtdetr', pretrained=False)

    def create(self):
        cfg = load_config(CONFIG)
        module = dict(cfg['model']['frequency_utility'],dim=8,expert_hidden=8,router_hidden=8)
        return DyFCLT(**self.options(),frequency_utility=module)

    def inputs(self):
        r,t=torch.randn(2,3,64,96),torch.randn(2,3,64,96)
        targets=[dict(labels=torch.tensor([0]),boxes=torch.tensor([[.5,.5,.2,.2]])) for _ in range(2)]
        return r,t,targets

    def test_recipe_is_zero_init_CNN_spatial_residual_with_the_same_A_training(self):
        self.assertTrue(CONFIG.is_file(), 'The CNN-only zero-init config must exist')
        cfg=load_config(CONFIG); self.assertTrue(validate_detection_config(cfg))
        module=cfg['model']['frequency_utility']
        self.assertEqual(module['gate_mode'],'spatial_mofe')
        self.assertEqual(module['residual_scale'],0.)
        self.assertEqual(module['residual_multiplier'],1.)
        self.assertEqual(module['insertion_point'],'projected_p3')
        self.assertNotIn('correlation_eps',module); self.assertNotIn('output_mode',module)
        a=json.loads((ROOT/'results/A_original_config.json').read_text())
        current=copy.deepcopy(cfg); current['model'].pop('frequency_utility')
        for split in ('train','val'):
            for field in ('ann_file','image_root'): current['data'][split][field]=a['data'][split][field]
        for item in (current,a): item['train'].pop('output_dir',None)
        self.assertEqual(current,a)

    def test_frequency_module_has_CNN_experts_and_spatial_weights_without_QKV(self):
        model=self.create(); module=model.encoder.frequency_module
        self.assertIsInstance(module,SpatialResidualMoFE)
        self.assertEqual(len(list(module.parameters())),36)
        self.assertFalse(any(any(token in name for token in ['query','key','value','modulator']) for name,_ in module.named_parameters()))
        r,t=torch.randn(2,32,5,7),torch.randn(2,32,5,7)
        cache=module.build_cache(r,t)
        self.assertEqual(cache['w'].shape,(2,2,4,5,7))
        torch.testing.assert_close(cache['w'].sum(2),torch.ones(2,2,5,7))
        for name in ['rgb_from_thermal','thermal_from_rgb']:
            bank=getattr(module,name); cached=cache[name]
            self.assertEqual(len(bank.experts),4)
            self.assertIsInstance(bank.router[2],nn.Identity)
            mask=torch.ones(2,1,5,7,dtype=torch.bool)
            pairs=build_frequency_pairs(*bank.prepare_bands(cached['rgb'],cached['thermal'],mask,mask))
            self.assertEqual([p.shape[1] for p in pairs],[16,32,32,48])
            for index,expert in enumerate(bank.experts):
                self.assertIsInstance(expert[0],nn.Conv2d)
                self.assertIsInstance(expert[2],nn.Conv2d)
                self.assertEqual(expert[2].groups,8)
                torch.testing.assert_close(cached['candidates'][:,index],expert(pairs[index]))
        self.assertIs(cache['rgb_from_thermal']['thermal'],t)
        self.assertIs(cache['thermal_from_rgb']['thermal'],r)

    def test_zero_init_preserves_A_tensors_RNG_and_initial_predictions(self):
        torch.manual_seed(0); a=DyFCLT(**self.options()).eval(); rng=torch.get_rng_state()
        torch.manual_seed(0); model=self.create().eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,value in a.state_dict().items(): self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
        module=model.encoder.frequency_module
        self.assertEqual(float(module.gamma_rgb),0.); self.assertEqual(float(module.gamma_thermal),0.)
        r,t,_=self.inputs()
        with torch.no_grad(): before,after=a(r,t),model(r,t)
        for name in ['pred_logits','pred_boxes']: self.assertTrue(torch.equal(before[name],after[name]),name)

    def test_detection_gradients_unlock_experts_after_gamma_update_and_EMA_tracks_them(self):
        torch.manual_seed(0); model=self.create(); cfg=load_config(CONFIG)
        module=model.encoder.frequency_module; params=dict(module.named_parameters())
        optimizer=build_optimizer(model,cfg['train']); ema=ModelEMA(model)
        ids={id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in ids for p in params.values()))
        r,t,targets=self.inputs(); criterion=build_criterion(2,mask_weight=0)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            out=model(r,t,targets,return_frequency_aux=True); losses=criterion(out,targets)
            self.assertFalse(any('utility' in name or 'kl' in name or 'mask' in name for name in losses))
            sum(losses.values()).backward()
            for name,p in params.items():
                self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)
                if step==0 and name not in ('gamma_rgb','gamma_thermal'): self.assertEqual(float(p.grad.abs().sum()),0.,name)
                else: self.assertGreater(float(p.grad.abs().sum()),0.,name)
            optimizer.step(); ema.update(model)
        self.assertNotEqual(float(module.gamma_rgb),0.); self.assertNotEqual(float(module.gamma_thermal),0.)
        stats=gate_statistics(out['frequency_aux'])
        self.assertTrue(all(sum(s['weight_spatial_std_mean'])>0 for s in stats.values()))
        with torch.no_grad(): left,right=ema.module(r,t),ema.module(r,t)
        for name in ['pred_logits','pred_boxes']:
            self.assertTrue(torch.equal(left[name],right[name])); self.assertTrue(torch.isfinite(left[name]).all())


if __name__=='__main__': unittest.main()
