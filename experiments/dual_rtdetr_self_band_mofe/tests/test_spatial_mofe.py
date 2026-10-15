import unittest

import torch

from dyfclt.residual_mofe import BidirectionalResidualMoFE
from dyfclt.spatial_mofe import SpatialResidualMoFE


class SpatialMoFETests(unittest.TestCase):
    def create(self):
        return SpatialResidualMoFE(channels=8, dim=4, expert_hidden=4,
                                   router_hidden=4, residual_multiplier=.5)

    def test_bank_parameters_and_rng_are_identical_to_B(self):
        torch.manual_seed(0)
        baseline = BidirectionalResidualMoFE(8, 4, 4, 4)
        rng = torch.get_rng_state()
        torch.manual_seed(0)
        spatial = self.create()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(baseline.state_dict()), set(spatial.state_dict()))
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, spatial.state_dict()[name]), name)

    def test_routing_is_local_and_probabilities_sum_at_each_position(self):
        module = self.create()
        rgb, thermal = torch.randn(2, 8, 5, 7), torch.randn(2, 8, 5, 7)
        cache = module.build_cache(rgb, thermal)
        weights = cache['w']
        self.assertEqual(weights.shape, (2, 2, 4, 5, 7))
        torch.testing.assert_close(weights.sum(2), torch.ones(2, 2, 5, 7))
        self.assertGreater(float(weights.std(dim=(-2, -1)).sum()), 0)
        # Candidates are injected under their local weights, not a global average.
        custom = torch.zeros_like(weights)
        custom[:, :, 0, :, :3] = 1
        custom[:, :, 3, :, 3:] = 1
        outputs = module.mix(cache, custom)
        for direction, (name, source, gamma) in enumerate([
            ('rgb_from_thermal', rgb, module.gamma_rgb),
            ('thermal_from_rgb', thermal, module.gamma_thermal),
        ]):
            selected = cache[name]['candidates'][:, 0].clone()
            selected[:, :, :, 3:] = cache[name]['candidates'][:, 3, :, :, 3:]
            torch.testing.assert_close(outputs[direction], source + .5 * gamma * selected)

    def test_both_directions_use_original_features_and_half_residual(self):
        module = self.create()
        rgb, thermal = torch.randn(2, 8, 5, 7), torch.randn(2, 8, 5, 7)
        cache = module.build_cache(rgb, thermal)
        self.assertIs(cache['rgb_from_thermal']['rgb'], rgb)
        self.assertIs(cache['thermal_from_rgb']['rgb'], thermal)
        self.assertIs(cache['thermal_from_rgb']['thermal'], rgb)
        out = module.mix(cache)
        for direction, (name, source, gamma) in enumerate([
            ('rgb_from_thermal', rgb, module.gamma_rgb),
            ('thermal_from_rgb', thermal, module.gamma_thermal),
        ]):
            update = (cache['w'][:, direction].unsqueeze(2) * cache[name]['candidates']).sum(1)
            torch.testing.assert_close(out[direction], source + .5 * gamma * update)

    def test_invalid_regions_preserve_original_and_diagnostics_ignore_them(self):
        module = self.create()
        rgb, thermal = torch.randn(2, 8, 5, 7), torch.randn(2, 8, 5, 7)
        rm = torch.ones(2, 1, 5, 7, dtype=torch.bool)
        tm = rm.clone(); tm[0, :, :, 3:] = False; tm[1] = False
        outputs = module(rgb, thermal, (rm, tm))
        for out, source in zip(outputs[:2], (rgb, thermal)):
            self.assertTrue(torch.equal(out.masked_select(~tm), source.masked_select(~tm)))
        for value in outputs[2]['diagnostics'].values():
            self.assertTrue(torch.isfinite(value).all())
            self.assertFalse(value.requires_grad)
        self.assertEqual(float(outputs[2]['diagnostics']['relative_update'][1].sum()), 0)
        self.assertEqual(float(outputs[2]['diagnostics']['valid_fraction'][1].sum()), 0)
        self.assertTrue(outputs[2]['routing_weights'].requires_grad)

    def test_detected_signal_trains_router_and_every_expert(self):
        torch.manual_seed(0)
        module = self.create()
        optimizer = torch.optim.AdamW(module.parameters(), lr=.001)
        before = {n: p.detach().clone() for n, p in module.named_parameters()}
        rgb, thermal = torch.randn(2, 8, 5, 7), torch.randn(2, 8, 5, 7)
        r, t, _ = module(rgb, thermal)
        (r.square().mean() + t.square().mean()).backward()
        optimizer.step()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
            self.assertFalse(torch.equal(before[name], parameter), name)

    def test_bad_multiplier_or_wrong_expert_axis_is_rejected(self):
        for value in [True, -1, 0, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                SpatialResidualMoFE(residual_multiplier=value)
        module = self.create()
        cache = module.build_cache(torch.randn(1, 8, 3, 5), torch.randn(1, 8, 3, 5))
        for value in [torch.ones_like(cache['w']), cache['w'].mean((-2, -1)),
                      torch.full_like(cache['w'], float('nan'))]:
            with self.assertRaises(ValueError):
                module.mix(cache, value)


if __name__ == '__main__':
    unittest.main()
