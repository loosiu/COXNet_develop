import unittest

import torch

from mmdet.models.utils.icbfc import (
    ICBFCLevel, ThermalInstancePrior, build_instance_targets,
    extract_instance_candidates, haar_dwt, haar_idwt, sparsemax)


class TestICBFCPrimitives(unittest.TestCase):

    def test_haar_round_trip_preserves_values_dtype_and_device(self):
        source = torch.arange(64, dtype=torch.float64).reshape(1, 1, 8, 8)

        components = haar_dwt(source)
        reconstructed = haar_idwt(*components)

        torch.testing.assert_close(reconstructed, source, atol=1e-12, rtol=0)
        self.assertEqual(reconstructed.dtype, source.dtype)
        self.assertEqual(reconstructed.device, source.device)

    def test_sparsemax_is_normalized_sparse_and_differentiable(self):
        logits = torch.tensor([[1.0, 0.8, -2.0]], requires_grad=True)

        probabilities = sparsemax(logits, dim=-1)

        torch.testing.assert_close(
            probabilities,
            torch.tensor([[0.6, 0.4, 0.0]]),
            atol=1e-6,
            rtol=0)
        torch.testing.assert_close(
            probabilities.sum(dim=-1), torch.ones(1), atol=1e-6, rtol=0)
        self.assertEqual((probabilities == 0).sum().item(), 1)

        (probabilities * torch.tensor([[1.0, 2.0, 3.0]])).sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_stride4_targets_keep_centers_that_stride8_would_merge(self):
        boxes = [torch.tensor([
            [8.0, 8.0, 12.0, 12.0],
            [12.0, 8.0, 16.0, 12.0],
        ])]
        metas = [dict(
            img_shape=(64, 64, 3),
            pad_shape=(64, 64, 3),
            batch_input_shape=(64, 64))]

        targets_s4 = build_instance_targets(
            boxes, metas, (16, 16), torch.device('cpu'))
        targets_s8 = build_instance_targets(
            boxes, metas, (8, 8), torch.device('cpu'))

        self.assertEqual(targets_s4['regression_mask'].sum().item(), 2)
        self.assertEqual(targets_s8['regression_mask'].sum().item(), 1)
        self.assertTrue(targets_s4['regression_mask'][0, 0, 2, 2])
        self.assertTrue(targets_s4['regression_mask'][0, 0, 2, 3])

    def test_candidate_extraction_keeps_more_than_100_valid_peaks(self):
        logits = torch.full((1, 1, 32, 32), -12.0)
        logits[..., 0::2, 0::2] = 12.0
        offsets = torch.zeros(1, 2, 32, 32)
        log_scales = torch.zeros(1, 2, 32, 32)
        valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)

        candidates = extract_instance_candidates(
            logits, offsets, log_scales, valid,
            score_threshold=0.5, chunk_size=37)

        self.assertEqual(candidates[0]['centers'].shape, (256, 2))
        self.assertEqual(candidates[0]['scores'].numel(), 256)
        self.assertEqual(candidates[0]['chunks'].shape, (7, 2))
        self.assertEqual(candidates[0]['chunks'][-1].tolist(), [222, 256])

    def test_candidate_extraction_removes_local_duplicates_and_padding(self):
        logits = torch.full((1, 1, 6, 6), -12.0)
        logits[0, 0, 2, 2] = 7.0
        logits[0, 0, 2, 3] = 6.0
        logits[0, 0, 5, 5] = 9.0
        offsets = torch.zeros(1, 2, 6, 6)
        offsets[0, :, 2, 2] = torch.tensor([0.25, 0.5])
        log_scales = torch.zeros(1, 2, 6, 6)
        valid = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
        valid[..., :5, :5] = True

        candidates = extract_instance_candidates(
            logits, offsets, log_scales, valid, score_threshold=0.5)

        self.assertEqual(candidates[0]['scores'].numel(), 1)
        torch.testing.assert_close(
            candidates[0]['centers'][0], torch.tensor([2.25, 2.5]))
        torch.testing.assert_close(
            candidates[0]['scales'][0], torch.tensor([1.0, 1.0]))

    def test_candidate_extraction_allows_zero_candidates(self):
        logits = torch.full((2, 1, 4, 5), -12.0)
        offsets = torch.zeros(2, 2, 4, 5)
        log_scales = torch.zeros(2, 2, 4, 5)
        valid = torch.ones(2, 1, 4, 5, dtype=torch.bool)

        candidates = extract_instance_candidates(
            logits, offsets, log_scales, valid, score_threshold=0.5)

        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertEqual(candidate['centers'].shape, (0, 2))
            self.assertEqual(candidate['scales'].shape, (0, 2))
            self.assertEqual(candidate['scores'].shape, (0,))
            self.assertEqual(candidate['chunks'].shape, (0, 2))


class TestThermalInstancePrior(unittest.TestCase):

    @staticmethod
    def _metas():
        return [dict(
            img_shape=(64, 64, 3),
            pad_shape=(64, 64, 3),
            batch_input_shape=(64, 64))]

    def test_training_prior_uses_every_gt_instance_and_returns_three_losses(self):
        torch.manual_seed(1)
        prior = ThermalInstancePrior(in_channels=8, hidden_channels=8)
        prior.train()
        thermal = torch.randn(1, 8, 16, 16)
        boxes = [torch.tensor([
            [8.0, 8.0, 12.0, 12.0],
            [12.0, 8.0, 16.0, 12.0],
        ])]

        instances, aux = prior(
            thermal, boxes, self._metas(), return_loss=True)

        self.assertEqual(instances[0]['centers'].shape, (2, 2))
        torch.testing.assert_close(
            instances[0]['centers'],
            torch.tensor([[2.5, 2.5], [3.5, 2.5]]))
        self.assertEqual(
            {key for key in aux if key.startswith('loss_icbfc_')},
            {'loss_icbfc_center', 'loss_icbfc_offset',
             'loss_icbfc_scale'})
        self.assertTrue(all(torch.isfinite(aux[key]) for key in (
            'loss_icbfc_center', 'loss_icbfc_offset',
            'loss_icbfc_scale')))

    def test_empty_gt_prior_losses_are_finite(self):
        prior = ThermalInstancePrior(in_channels=8, hidden_channels=8)
        prior.train()

        instances, aux = prior(
            torch.randn(1, 8, 16, 16),
            [torch.empty(0, 4)], self._metas(), return_loss=True)

        self.assertEqual(instances[0]['centers'].shape, (0, 2))
        for key in ('loss_icbfc_center', 'loss_icbfc_offset',
                    'loss_icbfc_scale'):
            self.assertTrue(torch.isfinite(aux[key]))

    def test_prior_predictions_exclude_padding(self):
        prior = ThermalInstancePrior(
            in_channels=8, hidden_channels=8, score_threshold=0.5)
        prior.eval()
        torch.nn.init.zeros_(prior.center_head.weight)
        torch.nn.init.constant_(prior.center_head.bias, 10.0)
        metas = [dict(
            img_shape=(32, 48, 3),
            pad_shape=(64, 64, 3),
            batch_input_shape=(64, 64))]

        instances, _ = prior(torch.zeros(1, 8, 16, 16), img_metas=metas)

        self.assertGreater(instances[0]['centers'].shape[0], 0)
        self.assertTrue((instances[0]['centers'][:, 0] < 12).all())
        self.assertTrue((instances[0]['centers'][:, 1] < 8).all())

    def test_center_offset_and_scale_parameters_receive_nonzero_gradients(self):
        torch.manual_seed(2)
        prior = ThermalInstancePrior(in_channels=8, hidden_channels=8)
        prior.train()
        boxes = [torch.tensor([[8.0, 8.0, 20.0, 24.0]])]

        _, aux = prior(
            torch.randn(1, 8, 16, 16), boxes, self._metas(),
            return_loss=True)
        loss = sum(aux[key] for key in (
            'loss_icbfc_center', 'loss_icbfc_offset',
            'loss_icbfc_scale'))
        loss.backward()

        for head_name in ('center_head', 'offset_head', 'scale_head'):
            gradients = [
                parameter.grad
                for parameter in getattr(prior, head_name).parameters()
            ]
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertGreater(
                sum(gradient.abs().sum().item() for gradient in gradients),
                0.0)

    def test_inference_prior_returns_variable_candidate_counts(self):
        prior = ThermalInstancePrior(
            in_channels=8, hidden_channels=8, score_threshold=0.5)
        prior.eval()
        torch.nn.init.zeros_(prior.center_head.weight)
        torch.nn.init.constant_(prior.center_head.bias, 10.0)
        metas = [
            dict(
                img_shape=(64, 64, 3), pad_shape=(64, 64, 3),
                batch_input_shape=(64, 64)),
            dict(
                img_shape=(32, 32, 3), pad_shape=(64, 64, 3),
                batch_input_shape=(64, 64)),
        ]

        instances, _ = prior(torch.zeros(2, 8, 16, 16), img_metas=metas)

        self.assertEqual(instances[0]['centers'].shape[0], 256)
        self.assertEqual(instances[1]['centers'].shape[0], 64)
        self.assertTrue((instances[0]['batch_index'] == 0).all())
        self.assertTrue((instances[1]['batch_index'] == 1).all())


class TestICBFCLevel(unittest.TestCase):

    @staticmethod
    def _instance(centers=None, scales=None):
        if centers is None:
            centers = torch.tensor([[4.0, 5.0], [11.0, 10.0]])
        if scales is None:
            scales = torch.tensor([[2.0, 3.0], [4.0, 2.0]])
        count = centers.shape[0]
        return [dict(
            centers=centers,
            scales=scales,
            scores=torch.ones(count),
            batch_index=torch.zeros(count, dtype=torch.long),
            chunks=torch.tensor([[0, count]], dtype=torch.long).reshape(-1, 2),
            prior_size=torch.tensor([16, 20], dtype=torch.long))]

    @staticmethod
    def _features(requires_grad=False):
        torch.manual_seed(11)
        thermal = torch.randn(
            1, 8, 16, 20, requires_grad=requires_grad)
        visible = torch.randn(
            1, 8, 8, 10, requires_grad=requires_grad)
        return thermal, visible

    def test_level_extracts_eight_tokens_per_instance(self):
        module = ICBFCLevel(channels=8, token_dim=8)
        thermal, visible = self._features()

        _, aux = module(
            thermal, visible, self._instance(), return_aux=True)

        self.assertEqual(aux['rgb_tokens'].shape, (2, 4, 8))
        self.assertEqual(aux['thermal_tokens'].shape, (2, 4, 8))

    def test_relation_is_four_by_four_and_rows_sum_to_one(self):
        module = ICBFCLevel(channels=8, token_dim=8)
        thermal, visible = self._features()

        _, aux = module(
            thermal, visible, self._instance(), return_aux=True)

        self.assertEqual(aux['relation'].shape, (2, 4, 4))
        torch.testing.assert_close(
            aux['relation'].sum(dim=-1), torch.ones(2, 4),
            atol=1e-6, rtol=0)

    def test_off_diagonal_relation_changes_rgb_output(self):
        torch.manual_seed(12)
        module = ICBFCLevel(channels=8, token_dim=8)
        thermal, visible = self._features()
        diagonal = torch.full((4, 4), -20.0)
        diagonal.fill_diagonal_(20.0)
        off_diagonal = torch.full((4, 4), 20.0)
        off_diagonal.fill_diagonal_(-20.0)

        with torch.no_grad():
            module.band_pair_bias.copy_(diagonal)
        diagonal_output = module(thermal, visible, self._instance())
        with torch.no_grad():
            module.band_pair_bias.copy_(off_diagonal)
        off_diagonal_output = module(thermal, visible, self._instance())

        self.assertFalse(torch.allclose(
            diagonal_output, off_diagonal_output, atol=1e-7, rtol=0))

    def test_sparse_router_can_choose_different_bands_per_instance(self):
        torch.manual_seed(13)
        module = ICBFCLevel(channels=8, token_dim=8)
        thermal, visible = self._features()

        _, aux = module(
            thermal, visible, self._instance(), return_aux=True)

        self.assertEqual(aux['router_weights'].shape, (2, 4))
        torch.testing.assert_close(
            aux['router_weights'].sum(dim=-1), torch.ones(2),
            atol=1e-6, rtol=0)
        self.assertGreater(
            (aux['router_weights'][0] -
             aux['router_weights'][1]).abs().sum().item(), 0.0)

    def test_overlapping_supports_are_normalized_not_summed(self):
        torch.manual_seed(14)
        module = ICBFCLevel(channels=8, token_dim=8)
        thermal, visible = self._features()
        one = self._instance(
            centers=torch.tensor([[6.0, 6.0]]),
            scales=torch.tensor([[3.0, 3.0]]))
        duplicate = self._instance(
            centers=torch.tensor([[6.0, 6.0], [6.0, 6.0]]),
            scales=torch.tensor([[3.0, 3.0], [3.0, 3.0]]))

        one_output = module(thermal, visible, one)
        duplicate_output = module(thermal, visible, duplicate)

        torch.testing.assert_close(
            duplicate_output, one_output, atol=1e-5, rtol=1e-5)

    def test_no_instances_return_exact_deconv_identity(self):
        module = ICBFCLevel(channels=8, token_dim=8)
        module.eval()
        thermal, visible = self._features()
        empty = self._instance(
            centers=torch.empty(0, 2), scales=torch.empty(0, 2))

        output, aux = module(thermal, visible, empty, return_aux=True)
        expected = module.deconv(visible)

        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        self.assertEqual(aux['candidate_count'].item(), 0)

    def test_level_preserves_thermal_and_matches_thermal_shape(self):
        module = ICBFCLevel(channels=8, token_dim=8)
        thermal, visible = self._features()
        thermal_before = thermal.clone()

        output = module(thermal, visible, self._instance())

        self.assertEqual(output.shape, thermal.shape)
        torch.testing.assert_close(thermal, thermal_before)

    def test_first_backward_reaches_qkv_router_output_and_deconv(self):
        torch.manual_seed(15)
        module = ICBFCLevel(channels=8, token_dim=8)
        thermal, visible = self._features(requires_grad=True)

        output = module(thermal, visible, self._instance())
        output.square().mean().backward()

        for prefix in ('q_proj', 'k_proj', 'v_proj', 'router',
                       'correction_mlp', 'out_proj', 'deconv'):
            gradients = [
                parameter.grad for name, parameter in module.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(all(gradient is not None for gradient in gradients),
                            prefix)
            self.assertGreater(
                sum(gradient.abs().sum().item() for gradient in gradients),
                0.0, prefix)
        self.assertGreater(thermal.grad.abs().sum().item(), 0.0)
        self.assertGreater(visible.grad.abs().sum().item(), 0.0)

    def test_initial_delta_ratio_is_finite_nonzero_and_below_point_two(self):
        torch.manual_seed(16)
        module = ICBFCLevel(channels=8, token_dim=8)
        thermal, visible = self._features()

        _, aux = module(
            thermal, visible, self._instance(), return_aux=True)

        ratio = aux['delta_ratio']
        self.assertTrue(torch.isfinite(ratio))
        self.assertGreater(ratio.item(), 0.0)
        self.assertLess(ratio.item(), 0.2)


if __name__ == '__main__':
    unittest.main()
