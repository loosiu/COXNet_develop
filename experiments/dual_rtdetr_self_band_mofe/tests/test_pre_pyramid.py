import copy
import unittest
import torch
from dyfclt.model import DyFCLT
from dyfclt.dual_encoder import DualRTDETREncoder
from dyfclt.vendor.hybrid_encoder import HybridEncoder


class PrePyramidTests(unittest.TestCase):
    def options(self, point='projected_p3', mamba=False):
        out=dict(gate_mode='residual_mamba_mofe' if mamba else 'residual_mofe',
                 dim=8, expert_hidden=8, router_hidden=8, residual_scale=.1,
                 insertion_point=point)
        if mamba:out.update(state_dim=2,eta=.05)
        return out

    def model(self, frequency=None):
        torch.manual_seed(0)
        return DyFCLT(3,hidden_dim=32,num_queries=4,num_decoder_layers=1,
                      num_denoising=0,architecture='dual_rtdetr',frequency_utility=frequency)

    def inputs(self):
        torch.manual_seed(4)
        return torch.randn(2,3,64,96),torch.randn(2,3,64,96)

    def test_split_encoder_exact_legacy_forward(self):
        torch.manual_seed(0)
        enc=HybridEncoder(in_channels=[8,16,32],hidden_dim=32).eval()
        feats=[torch.randn(2,c,8//(2**i),12//(2**i)) for i,c in enumerate([8,16,32])]
        with torch.no_grad():
            a=enc(feats);b=enc.forward_projected(enc.project_features(feats))
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))

    def test_gamma_zero_reproduces_A_all_outputs_and_RNG(self):
        a=self.model().eval();rng=torch.get_rng_state().clone()
        b=self.model(self.options()).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,x in a.state_dict().items():self.assertTrue(torch.equal(x,b.state_dict()[name]),name)
        with torch.no_grad():
            b.encoder.frequency_module.gamma_rgb.zero_();b.encoder.frequency_module.gamma_thermal.zero_()
            r,t=self.inputs();aa,bb=a(r,t),b(r,t)
        for key in ['pred_logits','pred_boxes']:self.assertTrue(torch.equal(aa[key],bb[key]),key)

    def test_only_P3_input_changes_but_all_pyramid_levels_respond(self):
        b=self.model(self.options()).eval();r,t=self.inputs();seen=[]
        handles=[layer.register_forward_hook(lambda *args:seen.append(1)) for enc in [b.encoder.rgb_encoder,b.encoder.thermal_encoder] for layer in enc.input_proj]
        with torch.no_grad():
            cache=b.prepare_frequency(r,t)
            baseline=list(cache['rgb_features']);enhanced=list(baseline)
            enhanced[0]=enhanced[0]+.05*torch.randn_like(enhanced[0])
            a=b.encoder.rgb_encoder.forward_projected(baseline)
            out=b.encoder.rgb_encoder.forward_projected(enhanced)
        for h in handles:h.remove()
        self.assertEqual(len(seen),6)
        for x,y in zip(a,out):self.assertFalse(torch.equal(x,y))

    def test_replay_matches_normal_after_pyramid(self):
        b=self.model(self.options()).eval();r,t=self.inputs()
        with torch.no_grad():
            cache=b.prepare_frequency(r,t);a=b(r,t);z=b.replay_frequency(cache)
        for key in ['pred_logits','pred_boxes']:self.assertTrue(torch.equal(a[key],z[key]),key)

    def test_E_initial_output_exact_B(self):
        b=self.model(self.options()).eval();e=self.model(self.options(mamba=True)).eval()
        r,t=self.inputs()
        with torch.no_grad():a,z=b(r,t),e(r,t)
        for key in ['pred_logits','pred_boxes']:self.assertTrue(torch.equal(a[key],z[key]),key)

    def test_reject_invalid_location(self):
        with self.assertRaises(ValueError):self.model(self.options('wrong'))

    def test_valid_mask_applied_at_projected_P3_once(self):
        b=self.model(self.options()).eval();r,t=self.inputs();mask=torch.ones(2,1,64,96,dtype=torch.bool);mask[...,-16:]=False
        with torch.no_grad():
            cache=b.prepare_frequency(r,t,(mask,mask));rr,tt=b.encoder.frequency_module.mix(cache['frequency'])
        for x,y in [(cache['rgb_features'][0],rr),(cache['thermal_features'][0],tt)]:
            invalid=~cache['frequency']['rgb_from_thermal']['valid_mask'].expand_as(x)
            self.assertTrue(torch.equal(x[invalid],y[invalid]))

if __name__=='__main__':unittest.main()
