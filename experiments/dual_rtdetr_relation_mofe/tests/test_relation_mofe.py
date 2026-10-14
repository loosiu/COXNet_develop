import unittest

import torch

from dyfclt.relation_mofe import BidirectionalRelationMoFE, RelationExpert, normalized_frequency_relation
from dyfclt.spatial_mofe import SpatialResidualMoFE


class RelationMoFETests(unittest.TestCase):
    def create(self, mode='direct'):
        return BidirectionalRelationMoFE(channels=8, dim=4, expert_hidden=4,
            router_hidden=4, output_mode=mode)

    def inputs(self):
        return torch.randn(2, 8, 5, 7), torch.randn(2, 8, 5, 7)

    def test_frequency_relation_matches_circular_cross_correlation_oracle(self):
        torch.manual_seed(0)
        q, k = torch.randn(1, 2, 3, 5), torch.randn(1, 2, 3, 5)
        expected = torch.empty_like(q)
        for y in range(3):
            for x in range(5):
                expected[..., y, x] = (q * torch.roll(k, shifts=(y, x), dims=(-2, -1))).sum((-2, -1))
        expected = expected / (expected.square().mean((-2, -1), keepdim=True) + 1e-8).sqrt()
        torch.testing.assert_close(normalized_frequency_relation(q, k, eps=1e-4), expected, rtol=2e-5, atol=2e-6)

    def test_zero_and_large_finite_relations_have_finite_gradients(self):
        for scale in [0., 1., 1e8]:
            q = (torch.randn(2, 4, 5, 7) * scale).requires_grad_()
            k = torch.randn_like(q, requires_grad=True)
            value = normalized_frequency_relation(q, k)
            self.assertTrue(torch.isfinite(value).all())
            value.square().mean().backward()
            self.assertTrue(torch.isfinite(q.grad).all())
            self.assertTrue(torch.isfinite(k.grad).all())

    def test_relation_depends_on_donor_and_receiver(self):
        expert = RelationExpert(receiver_channels=4, donor_channels=12, hidden_dim=4, channels=8)
        r, t = torch.randn(2, 4, 5, 7), torch.randn(2, 12, 5, 7)
        first, diagnostics = expert.forward_with_diagnostics(r, t)
        changed_r, _ = expert.forward_with_diagnostics(r + torch.randn_like(r), t)
        changed_t, _ = expert.forward_with_diagnostics(r, t + torch.randn_like(t))
        self.assertEqual(first.shape, (2, 8, 5, 7))
        self.assertFalse(torch.equal(first, changed_r))
        self.assertFalse(torch.equal(first, changed_t))
        self.assertEqual(diagnostics['attention'].shape, (2, 4, 5, 7))
        self.assertTrue(((diagnostics['attention'] >= 0) & (diagnostics['attention'] <= 1)).all())

    def test_original_projection_router_initialization_and_CPU_rng_preserved(self):
        torch.manual_seed(0); old = SpatialResidualMoFE(8, 4, 4, 4); rng = torch.get_rng_state()
        torch.manual_seed(0); new = self.create()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for direction in ['rgb_from_thermal', 'thermal_from_rgb']:
            before, after = getattr(old, direction), getattr(new, direction)
            for attribute in ['project', 'router']:
                for name, value in getattr(before, attribute).state_dict().items():
                    self.assertTrue(torch.equal(value, getattr(after, attribute).state_dict()[name]), (direction, attribute, name))

    def test_direct_output_is_candidate_mixture_without_original_addition(self):
        module = self.create(); rgb, thermal = self.inputs(); cache = module.build_cache(rgb, thermal)
        weights = torch.zeros_like(cache['w']); weights[:, :, 2] = 1
        r, t = module.mix(cache, weights)
        torch.testing.assert_close(r, cache['rgb_from_thermal']['candidates'][:, 2])
        torch.testing.assert_close(t, cache['thermal_from_rgb']['candidates'][:, 2])
        for bank in ['rgb_from_thermal', 'thermal_from_rgb']: cache[bank]['candidates'].zero_()
        r, t = module.mix(cache)
        self.assertEqual(float(r.abs().sum()), 0)
        self.assertEqual(float(t.abs().sum()), 0)
        self.assertFalse(any('gamma' in n for n, _ in module.named_parameters()))

    def test_residual_control_uses_same_candidate_definition_and_half_scale(self):
        module = self.create('residual'); rgb, thermal = self.inputs(); cache = module.build_cache(rgb, thermal)
        with torch.no_grad():
            module.gamma_rgb.fill_(.1); module.gamma_thermal.fill_(.1)
        outputs = module.mix(cache)
        for d, (name, source, gamma) in enumerate([
            ('rgb_from_thermal', rgb, module.gamma_rgb), ('thermal_from_rgb', thermal, module.gamma_thermal),
        ]):
            mixed = (cache['w'][:, d].unsqueeze(2) * cache[name]['candidates']).sum(1)
            torch.testing.assert_close(outputs[d], source + .5 * gamma * mixed)

    def test_both_banks_read_original_pair_and_invalid_regions_preserve_source(self):
        module = self.create(); rgb, thermal = self.inputs()
        rm = torch.ones(2, 1, 5, 7, dtype=torch.bool); tm = rm.clone(); tm[0, :, :, 3:] = False; tm[1] = False
        cache = module.build_cache(rgb, thermal, (rm, tm))
        self.assertIs(cache['thermal_from_rgb']['thermal'], rgb)
        self.assertIs(cache['rgb_from_thermal']['thermal'], thermal)
        r, t, aux = module(rgb, thermal, (rm, tm))
        for output, source in [(r, rgb), (t, thermal)]:
            self.assertTrue(torch.equal(output.masked_select(~tm), source.masked_select(~tm)))
        self.assertTrue(aux['routing_weights'].requires_grad)
        torch.testing.assert_close(aux['routing_weights'].sum(2), torch.ones(2, 2, 5, 7))
        for value in aux['diagnostics'].values():
            self.assertTrue(torch.isfinite(value).all())
            self.assertFalse(value.requires_grad)

    def test_invalid_mode_and_unstable_eps_rejected(self):
        with self.assertRaises(ValueError): self.create('unknown')
        for eps in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): normalized_frequency_relation(torch.ones(1, 1, 3, 3), torch.ones(1, 1, 3, 3), eps)


if __name__ == '__main__': unittest.main()
