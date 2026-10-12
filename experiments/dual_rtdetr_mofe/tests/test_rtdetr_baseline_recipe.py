import copy
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from dyfclt.model import DyFCLT
from dyfclt.losses import build_criterion
from dyfclt.runtime import save_checkpoint, load_checkpoint


class BaselineTests(unittest.TestCase):
    def test_full_dual_encoders_receive_gradients_without_proposed_modules(self):
        # Catches replacing one encoder with the top-down proposal or sharing branches.
        from dyfclt.vendor.hybrid_encoder import HybridEncoder
        model = DyFCLT(3, hidden_dim=32, num_queries=8, num_decoder_layers=2,
                       num_denoising=4, architecture='dual_rtdetr')
        encoders = [m for m in model.encoder.modules() if isinstance(m, HybridEncoder)]
        self.assertEqual(len(encoders), 2)
        self.assertFalse(set(map(id, encoders[0].parameters())) & set(map(id, encoders[1].parameters())))
        from dyfclt.frequency import DFCA
        from dyfclt.sse import SSE
        self.assertFalse(any(isinstance(m, (DFCA, SSE)) for m in model.modules()))
        targets = [{'labels': torch.tensor([0]), 'boxes': torch.tensor([[.5,.5,.1,.1]])},
                   {'labels': torch.empty(0,dtype=torch.long), 'boxes': torch.empty(0,4)}]
        outputs = model(torch.randn(2,3,64,96), torch.randn(2,3,64,96), targets)
        self.assertEqual(outputs['pred_logits'].shape, (2,8,3))
        losses = build_criterion(3)(outputs, targets)
        self.assertNotIn('loss_mask', losses)
        self.assertIn('dn_aux_outputs', outputs)
        self.assertTrue(all(torch.isfinite(x) for x in losses.values()))
        sum(losses.values()).backward()
        for enc in encoders:
            for group in (enc.input_proj, enc.encoder, enc.fpn_blocks, enc.pan_blocks):
                grads=[p.grad for p in group.parameters() if p.requires_grad]
                self.assertTrue(any(g is not None and g.abs().sum()>0 for g in grads))
        for group in (model.rgb_backbone, model.thermal_backbone, model.encoder.modal_fusion, model.decoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in group.parameters()))


class ToyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone=nn.Linear(2,2)
        self.thermal_backbone=nn.Linear(2,2)
        self.encoder=nn.ModuleDict({'norm': nn.LayerNorm(2), 'proj': nn.Linear(2,2),
                                   'input_proj': nn.Sequential(nn.Linear(2,2), nn.LayerNorm(2))})
        self.decoder=nn.Linear(2,2)
        self.register_buffer('counter',torch.tensor(0))
        self.thermal_backbone.bias.requires_grad_(False)


class RecipeTests(unittest.TestCase):
    def recipe(self):
        import dyfclt.training as recipe
        return recipe

    def test_groups_cover_trainable_parameters_once_and_preserve_warmup_ratio(self):
        recipe=self.recipe();model=ToyDetector()
        settings={'lr': .00025, 'weight_decay': .0001, 'optimizer_groups': 'rtdetr'}
        opt=recipe.build_optimizer(model,settings)
        all_params=[p for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(all_params),len(set(map(id,all_params))))
        self.assertEqual(set(map(id,all_params)),{id(p) for p in model.parameters() if p.requires_grad})
        recipe.set_learning_rate(opt,settings['lr'],.001)
        groups={id(p):g for g in opt.param_groups for p in g['params']}
        for name,p in model.named_parameters():
            if not p.requires_grad:continue
            expected_lr=2.5e-8 if 'backbone' in name else 2.5e-7
            self.assertAlmostEqual(groups[id(p)]['lr'],expected_lr)
        self.assertEqual(groups[id(model.encoder['norm'].weight)]['weight_decay'],0)
        self.assertEqual(groups[id(model.encoder['proj'].bias)]['weight_decay'],0)
        # Official regex is name-based: anonymous norm .1.weight is not matched.
        self.assertEqual(groups[id(model.encoder['input_proj'][1].weight)]['weight_decay'],.0001)
        self.assertEqual(groups[id(model.rgb_backbone.bias)]['weight_decay'],.0001)
        recipe.set_learning_rate(opt,settings['lr'],1)
        self.assertAlmostEqual(groups[id(model.rgb_backbone.weight)]['lr'],.000025)
        self.assertAlmostEqual(groups[id(model.decoder.weight)]['lr'],.00025)

    def test_ema_ramp_and_checkpoint_continue_same_trajectory(self):
        recipe=self.recipe();model=ToyDetector()
        settings={'lr': .00025,'weight_decay': .0001,'optimizer_groups':'rtdetr'}
        opt=recipe.build_optimizer(model,settings)
        ema=recipe.ModelEMA(model,decay=.9,warmups=2)
        before=model.decoder.weight.detach().clone()
        with torch.no_grad():
            model.decoder.weight.add_(2)
            model.counter.add_(1)
        ema.update(model)
        decay=.9*(1-math.exp(-.5))
        torch.testing.assert_close(ema.module.decoder.weight,before+2*(1-decay))
        self.assertFalse(ema.module.training)
        self.assertFalse(any(p.requires_grad for p in ema.module.parameters()))
        self.assertEqual(int(ema.module.counter),0)  # Matches official non-float buffer policy.
        recipe.set_learning_rate(opt,settings['lr'],.25)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pth'
            save_checkpoint(path,model,opt,None,{},0,1,.1,[0],ema=ema)
            clone=ToyDetector();clone_opt=recipe.build_optimizer(clone,settings)
            clone_ema=recipe.ModelEMA(clone,decay=.8,warmups=3)
            load_checkpoint(path,clone,clone_opt,ema=clone_ema)
            self.assertEqual(clone_ema.updates,1)
            self.assertEqual(clone_ema.decay,.9)
            self.assertEqual(clone_ema.warmups,2)
            recipe.set_learning_rate(clone_opt,settings['lr'],1)
            ratios=[g['lr'] for g in clone_opt.param_groups]
            self.assertIn(.000025,ratios);self.assertIn(.00025,ratios)
            with torch.no_grad():
                model.decoder.weight.add_(3);clone.decoder.weight.add_(3)
            ema.update(model);clone_ema.update(clone)
            for k,v in ema.module.state_dict().items():torch.testing.assert_close(v,clone_ema.module.state_dict()[k])

    def test_inference_selects_ema_and_legacy_remains_loadable(self):
        recipe=self.recipe();model=ToyDetector();ema=recipe.ModelEMA(model)
        with torch.no_grad():
            model.decoder.weight.fill_(2);ema.module.decoder.weight.fill_(7)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'ema.pth';save_checkpoint(p,model,None,None,{},0,1,0,[0],ema=ema)
            target=ToyDetector()
            load_checkpoint(p,target,weights='auto')
            self.assertTrue(torch.all(target.decoder.weight==7))
            load_checkpoint(p,target,weights='model')
            self.assertTrue(torch.all(target.decoder.weight==2))
            old={'model':model.state_dict()}
            load_checkpoint(old,target,weights='auto')
            self.assertTrue(torch.all(target.decoder.weight==2))
            with self.assertRaisesRegex(ValueError,'EMA'):
                load_checkpoint(old,target,weights='ema')
            with self.assertRaisesRegex(ValueError,'EMA'):
                load_checkpoint(old,target,ema=ema)


if __name__=='__main__':unittest.main()
