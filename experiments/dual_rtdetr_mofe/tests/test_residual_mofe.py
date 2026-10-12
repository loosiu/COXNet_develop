import importlib
import importlib.util
import unittest

import torch

from dyfclt.s3am_mofe import S3AMFrequencyMixture


class ResidualMoFETests(unittest.TestCase):
    def create(self, **options):
        self.assertIsNotNone(importlib.util.find_spec('dyfclt.residual_mofe'), 'Residual MoFE missing')
        cls = importlib.import_module('dyfclt.residual_mofe').BidirectionalResidualMoFE
        return cls(channels=8, dim=4, expert_hidden=4, router_hidden=4, **options)

    def data(self):
        torch.manual_seed(7)
        return torch.randn(2, 8, 7, 9), torch.randn(2, 8, 7, 9)

    def test_banks_are_independent_and_keep_original_all_parameters(self):
        torch.manual_seed(0)
        old = S3AMFrequencyMixture(channels=8, dim=4, expert_hidden=4, router_hidden=4)
        torch.manual_seed(0)
        new = self.create()
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.rgb_from_thermal.state_dict()[name]), name)
        rgb_ids = {id(p) for p in new.rgb_from_thermal.parameters()}
        thermal_ids = {id(p) for p in new.thermal_from_rgb.parameters()}
        self.assertTrue(rgb_ids.isdisjoint(thermal_ids))
        self.assertEqual(set(new.rgb_from_thermal.state_dict()), set(new.thermal_from_rgb.state_dict()))
        self.assertEqual(len(list(new.parameters())), 36)
        self.assertAlmostEqual(float(new.gamma_rgb), .1, places=7)
        self.assertAlmostEqual(float(new.gamma_thermal), .1, places=7)

    def test_bidirectional_output_uses_original_inputs_and_small_residual(self):
        module = self.create()
        r, t = self.data()
        rc = module.rgb_from_thermal.build_cache(r, t)
        tc = module.thermal_from_rgb.build_cache(t, r)
        def expected(cache, gamma):
            return cache['rgb'] + gamma * (cache['w'][:, :, None, None, None] * cache['candidates']).sum(1)
        ro, to, _ = module(r, t)
        self.assertEqual(ro.shape, r.shape)
        self.assertEqual(to.shape, t.shape)
        self.assertTrue(torch.equal(ro, expected(rc, module.gamma_rgb)))
        self.assertTrue(torch.equal(to, expected(tc, module.gamma_thermal)))
        self.assertFalse(torch.equal(ro, r))
        self.assertFalse(torch.equal(to, t))

    def test_zero_scales_are_exact_identity(self):
        module = self.create(residual_scale=0.)
        r, t = self.data()
        ro, to, _ = module(r, t)
        self.assertTrue(torch.equal(ro, r))
        self.assertTrue(torch.equal(to, t))

    def test_global_simplex_weights_and_diagnostics(self):
        module = self.create()
        r, t = self.data()
        _, _, aux = module(r, t)
        self.assertEqual(aux['routing_weights'].shape, (2, 2, 4))
        self.assertTrue(torch.allclose(aux['routing_weights'].sum(-1), torch.ones(2, 2)))
        self.assertTrue((aux['routing_weights'] >= 0).all())
        self.assertEqual(aux['diagnostics']['candidate_rms'].shape, (2, 2, 4))
        self.assertEqual(aux['diagnostics']['relative_update'].shape, (2, 2))
        for value in aux['diagnostics'].values():
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value).all())

    def test_deterministic_in_train_and_eval_without_sampling_or_rng_use(self):
        module = self.create()
        r, t = self.data()
        rng = torch.get_rng_state()
        left = module(r, t)
        right = module(r, t)
        module.eval()
        evaluation = module(r, t)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for index in (0, 1):
            self.assertTrue(torch.equal(left[index], right[index]))
            self.assertTrue(torch.equal(left[index], evaluation[index]))

    def test_masked_inputs_cannot_leak_and_invalid_updates_are_identity(self):
        module = self.create()
        r, t = self.data()
        rm = torch.ones(2, 1, 7, 9, dtype=torch.bool)
        tm = rm.clone()
        rm[:, :, -2:] = False
        tm[:, :, :, -2:] = False
        ro, to, _ = module(r, t, valid_masks=(rm, tm))
        r2, t2 = r.masked_fill(~rm, 999.), t.masked_fill(~tm, -999.)
        ro2, to2, _ = module(r2, t2, valid_masks=(rm, tm))
        intersection = (rm & tm).expand_as(r)
        self.assertTrue(torch.equal(ro[intersection], ro2[intersection]))
        self.assertTrue(torch.equal(to[intersection], to2[intersection]))
        self.assertTrue(torch.equal(ro[~intersection], r[~intersection]))
        self.assertTrue(torch.equal(to[~intersection], t[~intersection]))

    def test_all_invalid_masks_preserve_original_features(self):
        module = self.create()
        r, t = self.data()
        mask = torch.zeros(2, 1, 7, 9, dtype=torch.bool)
        ro, to, _ = module(r, t, valid_masks=(mask, mask))
        self.assertTrue(torch.equal(ro, r))
        self.assertTrue(torch.equal(to, t))

    def test_override_selects_expert_in_both_directions_and_validates_simplex(self):
        module = self.create()
        r, t = self.data()
        cache = module.build_cache(r, t)
        weights = torch.zeros(2, 2, 4)
        weights[:, 0, 1] = 1
        weights[:, 1, 3] = 1
        ro, to = module.mix(cache, weights)
        self.assertTrue(torch.equal(ro, r + module.gamma_rgb * cache['rgb_from_thermal']['candidates'][:, 1]))
        self.assertTrue(torch.equal(to, t + module.gamma_thermal * cache['thermal_from_rgb']['candidates'][:, 3]))
        for invalid in [torch.zeros_like(weights), weights[:, 0], weights.clone().fill_(float('nan'))]:
            with self.assertRaises(ValueError): module.mix(cache, invalid)

    def test_every_bank_parameter_and_scale_receives_gradient_and_updates(self):
        torch.manual_seed(0)
        module = self.create()
        r, t = self.data()
        optimizer = torch.optim.AdamW(module.parameters(), lr=.001)
        before = {name: p.detach().clone() for name, p in module.named_parameters()}
        ro, to, _ = module(r, t)
        (ro.square().mean() + to.square().mean()).backward()
        optimizer.step()
        for name, p in module.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(float(p.grad.abs().sum()), 0, name)
            self.assertFalse(torch.equal(p, before[name]), name)

    def test_checkpoint_roundtrip(self):
        module = self.create()
        other = self.create()
        other.load_state_dict(module.state_dict(), strict=True)
        r, t = self.data()
        left, right = module(r, t), other(r, t)
        self.assertTrue(torch.equal(left[0], right[0]))
        self.assertTrue(torch.equal(left[1], right[1]))

    def test_reject_nonfinite_scale_and_mismatched_inputs(self):
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError): self.create(residual_scale=value)
        module = self.create()
        r, t = self.data()
        with self.assertRaises(ValueError): module(r, t[..., :-1])


if __name__ == '__main__':
    unittest.main()
