import importlib,unittest
import torch
class S3AMMoFETests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('dyfclt.s3am_mofe'),'S3AM pair-expert module is missing')
  self.api=importlib.import_module('dyfclt.s3am_mofe')
  torch.manual_seed(7)
 def module(self):return self.api.S3AMFrequencyMixture(channels=8,dim=4,expert_hidden=8,router_hidden=8)
 def features(self):return torch.randn(2,8,9,11),torch.randn(2,8,9,11)
 def test_paper_frequency_pair_order_and_channel_counts(self):
  r=tuple(torch.full((2,2,3,5),float(i)) for i in [1,2,3,4]);t=tuple(torch.full((2,2,3,5),float(i)) for i in [11,12,13,14])
  pairs=self.api.build_frequency_pairs(r,t)
  expected=[(r[0],t[0]),(r[0],*t[1:]),(*r[1:],t[0]),(*r[1:],*t[1:])]
  for actual,parts,c in zip(pairs,expected,[4,8,8,12]):
   self.assertEqual(actual.shape,(2,c,3,5));self.assertTrue(torch.equal(actual,torch.cat(parts,1)))
 def test_softmax_is_image_level_simplex_with_adaptive_initialization(self):
  m=self.module();r,t=self.features();cache=m.build_cache(r,t)
  self.assertEqual(cache['w'].shape,(2,4));self.assertEqual(cache['s'].shape,(2,4))
  self.assertTrue(torch.allclose(cache['w'].sum(1),torch.ones(2),atol=1e-6));self.assertTrue((cache['w']>0).all())
  self.assertLess(float((cache['w']-.25).abs().max()),.03)
  self.assertTrue(cache['w'].requires_grad)
 def test_weighted_sum_is_added_directly_only_to_rgb(self):
  m=self.module();r,t=self.features()
  c=torch.stack([torch.full_like(r,float(i)) for i in [1,2,3,4]],1)
  w=torch.tensor([[.1,.2,.3,.4],[.1,.2,.3,.4]])
  cache={'rgb':r,'thermal':t,'candidates':c,'w':w,'valid_mask':torch.ones(2,1,9,11,dtype=torch.bool)}
  ro,to=m.mix(cache)
  self.assertTrue(torch.allclose(ro,r+3,atol=1e-6));self.assertIs(to,t)
 def test_forward_matches_cached_mixture_without_recomputation(self):
  m=self.module();r,t=self.features();cache=m.build_cache(r,t);expected=m.mix(cache);ro,to,aux=m(r,t)
  self.assertTrue(torch.equal(ro,expected[0]));self.assertIs(to,t)
  self.assertEqual(aux['routing_weights'].shape,(2,4));self.assertEqual(aux['candidate_rms'].shape,(2,4));self.assertEqual(aux['relative_update'].shape,(2,))
 def test_all_experts_and_router_learn_on_first_backward(self):
  m=self.module();r,t=self.features();ro,_,_=m(r,t);ro.square().mean().backward()
  for n,p in m.named_parameters():
   self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n);self.assertGreater(float(p.grad.abs().sum()),0,n)
 def test_zero_experts_return_exact_identity(self):
  m=self.module();r,t=self.features()
  for e in m.experts:torch.nn.init.zeros_(e[-1].weight)
  ro,to,_=m(r,t);self.assertTrue(torch.equal(ro,r));self.assertIs(to,t)
 def test_validity_intersection_controls_injection_and_pooling(self):
  m=self.module();r,t=self.features();a=torch.ones(2,1,9,11,dtype=torch.bool);b=a.clone();b[:,:,:,:3]=False
  ro,_,_=m(r,t,valid_masks=(a,b));self.assertTrue(torch.equal(ro[:,:,:,:3],r[:,:,:,:3]))
 def test_all_invalid_is_finite_identity(self):
  m=self.module();r,t=self.features();mask=torch.zeros(2,1,9,11,dtype=torch.bool)
  ro,to,aux=m(r,t,valid_masks=(mask,mask));self.assertTrue(torch.equal(ro,r));self.assertIs(to,t);self.assertTrue(torch.isfinite(aux['routing_weights']).all())
 def test_single_pixel_and_odd_size_inputs_preserve_shape(self):
  m=self.module()
  for h,w in [(1,1),(3,5)]:
   r=torch.randn(1,8,h,w);t=torch.randn_like(r);ro,to,_=m(r,t);self.assertEqual(ro.shape,r.shape);self.assertIs(to,t)
 def test_bad_input_and_non_simplex_overrides_are_rejected(self):
  m=self.module();r,t=self.features()
  with self.assertRaises(ValueError):m(r,t[:,:,:,:-1])
  cache=m.build_cache(r,t)
  for weights in [torch.zeros(2,4),torch.full((2,4),float('nan')),torch.ones(2,3)]:
   with self.assertRaises(ValueError):m.mix(cache,weights)
 def test_invalid_constructor_values_are_rejected(self):
  for value in [0,True,-1]:
   with self.assertRaises(ValueError):self.api.S3AMFrequencyMixture(channels=value)
