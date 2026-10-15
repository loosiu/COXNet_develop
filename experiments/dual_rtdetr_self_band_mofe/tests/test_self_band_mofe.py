import copy
from pathlib import Path
import unittest

import torch

from dyfclt.runtime import load_config
from dyfclt.spatial_mofe import SpatialResidualMoFE

try:
    from dyfclt.self_band_mofe import SelfBandResidualMoFE
except ImportError:
    SelfBandResidualMoFE = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0.yaml'
OPTIONS = dict(channels=8, dim=4, expert_hidden=5, router_hidden=6, residual_scale=0., residual_multiplier=1.)


class SelfBandMoFETests(unittest.TestCase):
    def make(self, **overrides):
        self.assertIsNotNone(SelfBandResidualMoFE, 'receiver-value relation expert not implemented')
        return SelfBandResidualMoFE(**dict(OPTIONS, **overrides))

    def test_receiver_zero_forbids_direct_donor_values(self):
        module = self.make(residual_scale=.2)
        receiver, donor = torch.zeros(2,8,5,7), torch.randn(2,8,5,7)
        cache = module.build_cache(receiver, donor)
        self.assertTrue(torch.equal(cache['rgb_from_thermal']['candidates'], torch.zeros(2,4,8,5,7)))
        out = module(receiver, donor)
        self.assertTrue(torch.equal(out[0], receiver))
        self.assertGreater(float((out[1]-donor).abs().sum()), 0)

    def test_initial_gate_is_one_but_expert_is_nonzero(self):
        module = self.make()
        r,t = torch.randn(2,8,5,7), torch.randn(2,8,5,7)
        cache = module.build_cache(r,t)
        for name in ('rgb_from_thermal','thermal_from_rgb'):
            b=cache[name]
            self.assertEqual(b['relation_gates'].shape, (2,4,5,5,7))
            self.assertTrue(torch.equal(b['relation_gates'], torch.ones_like(b['relation_gates'])))
            self.assertGreater(float(b['candidates'].abs().sum()), 0)
        ro,to,aux=module(r,t)
        self.assertTrue(torch.equal(ro,r) and torch.equal(to,t))
        self.assertTrue(torch.allclose(aux['routing_weights'].sum(2), torch.ones(2,2,5,7)))

    def test_two_values_shared_across_four_experts(self):
        module = self.make()
        bank=module.rgb_from_thermal
        self.assertEqual(len(bank.values),2)
        seen=[[],[]]
        hooks=[value.register_forward_hook(lambda m,i,o,index=i: seen[index].append(o)) for i,value in enumerate(bank.values)]
        cache=module.build_cache(torch.randn(2,8,5,7), torch.randn(2,8,5,7))
        for h in hooks:h.remove()
        self.assertEqual([len(v) for v in seen],[1,1])
        self.assertEqual(len(bank.experts),4)

    def test_learned_relation_gate_depends_on_donor_but_value_does_not(self):
        torch.manual_seed(11);module=self.make();bank=module.rgb_from_thermal
        with torch.no_grad():
            for expert in bank.experts: expert.gate[-1].weight.normal_(0,.2)
        r,t=torch.randn(2,8,5,7,requires_grad=True),torch.randn(2,8,5,7,requires_grad=True)
        cache=bank.build_cache(r,t)
        cache['candidates'].square().mean().backward()
        self.assertGreater(float(r.grad.abs().sum()),0)
        self.assertGreater(float(t.grad.abs().sum()),0)
        changed=bank.build_cache(r.detach(),torch.flip(t.detach(),[-1]))
        self.assertGreater(float((cache['relation_gates']-changed['relation_gates']).abs().sum()),0)
        self.assertTrue(((cache['relation_gates']>=0)&(cache['relation_gates']<=2)).all())

    def test_gradients_open_after_gamma_and_final_gate_layers(self):
        torch.manual_seed(31);module=self.make();optimizer=torch.optim.SGD(module.parameters(),lr=.1)
        r,t=torch.randn(2,8,5,7),torch.randn(2,8,5,7)
        yr,yt=torch.randn_like(r),torch.randn_like(t);observed=set()
        for step in range(5):
            optimizer.zero_grad();ro,to,_=module(r,t)
            ((ro-yr).square().mean()+(to-yt).square().mean()).backward()
            nonzero=set()
            for name,p in module.named_parameters():
                self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
                if p.grad.abs().sum()>0:nonzero.add(name)
            if step==0:self.assertEqual(nonzero,{'gamma_rgb','gamma_thermal'})
            observed |= nonzero;optimizer.step()
        self.assertEqual(observed,{n for n,_ in module.named_parameters()})

    def test_banks_read_original_pair_and_masks_block_injection(self):
        module=self.make(residual_scale=.2);r,t=torch.randn(2,8,5,7),torch.randn(2,8,5,7)
        cache=module.build_cache(r,t)
        self.assertIs(cache['rgb_from_thermal']['rgb'],r)
        self.assertIs(cache['thermal_from_rgb']['rgb'],t)
        mask=torch.zeros(2,1,5,7,dtype=torch.bool)
        ro,to,_=module(r,t,(mask,mask));self.assertTrue(torch.equal(ro,r) and torch.equal(to,t))

    def test_projection_router_and_rng_match_existing_bank(self):
        torch.manual_seed(0);old=SpatialResidualMoFE(**OPTIONS);rng=torch.get_rng_state()
        torch.manual_seed(0);new=self.make()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for n,v in old.state_dict().items():
            if '.experts.' not in n:self.assertTrue(torch.equal(v,new.state_dict()[n]),n)

    def test_config_only_changes_expert_mode_and_output(self):
        self.assertTrue(CONFIG.exists(), 'self-band config is missing')
        old=load_config(ROOT/'configs/rgbtdroneperson_cnn_mofe_zero_post_encoder_seed0.yaml')
        new=load_config(CONFIG);expected=copy.deepcopy(old)
        expected['model']['frequency_utility']['gate_mode']='self_band_mofe'
        expected['train']['output_dir']='outputs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0'
        self.assertEqual(new,expected)


if __name__=='__main__':unittest.main()
