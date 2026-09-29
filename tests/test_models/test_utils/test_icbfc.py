import unittest

import torch

from mmdet.models.utils.icbfc import (
    build_instance_targets, extract_instance_candidates, haar_dwt,
    haar_idwt, sparsemax)


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


if __name__ == '__main__':
    unittest.main()
