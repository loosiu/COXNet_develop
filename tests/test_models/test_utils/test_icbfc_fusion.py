import unittest

import torch

from mmdet.models.utils.fusion_strategy import FusionLayer


def _layer(**kwargs):
    defaults = dict(
        in_channels=16,
        reduction=4,
        num_layers=4,
        fs_type='fusionnet-xo',
        use_om=False,
        use_msf=False,
        use_clfm=[],
        use_icbfc=True,
        icbfc_cfg=dict(
            prior_in_channels=16,
            prior_hidden_channels=16,
            token_dim=8,
            score_threshold=0.5,
            candidate_chunk_size=37,
            aux_loss_weight=0.1),
        usepoolup=[])
    defaults.update(kwargs)
    return FusionLayer(**defaults)


def _features(batch=1):
    visible = [
        torch.randn(batch, 16, 8, 8),
        torch.randn(batch, 16, 4, 4),
        torch.randn(batch, 16, 2, 2),
        torch.randn(batch, 16, 1, 1),
    ]
    thermal = [
        torch.randn(batch, 16, 16, 16),
        torch.randn(batch, 16, 8, 8),
        torch.randn(batch, 16, 4, 4),
        torch.randn(batch, 16, 2, 2),
    ]
    return visible, thermal


class TestICBFCFusion(unittest.TestCase):

    @staticmethod
    def _metas():
        return [dict(
            img_shape=(64, 64, 3),
            pad_shape=(64, 64, 3),
            batch_input_shape=(64, 64))]

    def test_fusion_layer_builds_one_prior_and_four_levels(self):
        layer = _layer()

        self.assertEqual(layer.icbfc_prior.__class__.__name__,
                         'ThermalInstancePrior')
        self.assertEqual(len(layer.icbfc_layers), 4)
        self.assertEqual(
            [module.__class__.__name__ for module in layer.icbfc_layers],
            ['ICBFCLevel'] * 4)

    def test_fusion_layer_returns_four_thermal_resolution_features(self):
        layer = _layer().eval()
        visible, thermal = _features()
        thermal_s4 = torch.randn(1, 16, 16, 16)

        output = layer(
            visible, thermal, img_metas=self._metas(),
            thermal_s4=thermal_s4)

        self.assertEqual(len(output), 4)
        self.assertEqual(
            [tuple(feature.shape) for feature in output],
            [tuple(feature.shape) for feature in thermal])

    def test_training_returns_center_losses_and_nonoptimized_diagnostics(self):
        layer = _layer().train()
        visible, thermal = _features()
        boxes = [torch.tensor([[8.0, 8.0, 20.0, 24.0]])]

        output, aux = layer(
            visible, thermal, gt_bboxes=boxes, img_metas=self._metas(),
            thermal_s4=torch.randn(1, 16, 16, 16))

        self.assertEqual(len(output), 4)
        self.assertEqual(
            {key for key in aux if key.startswith('loss_icbfc_')},
            {'loss_icbfc_center', 'loss_icbfc_offset',
             'loss_icbfc_scale'})
        self.assertIn('icbfc_candidate_count', aux)
        self.assertIn('icbfc_delta_ratio', aux)
        for key, value in aux.items():
            self.assertTrue(torch.isfinite(value).all(), key)
            if not key.startswith('loss_icbfc_'):
                self.assertFalse(value.requires_grad, key)

    def test_fusion_rejects_icbfc_with_other_clfm_replacements(self):
        conflicts = (
            dict(use_clfm=['v3']),
            dict(use_trpc=True),
            dict(use_oepc=True),
            dict(use_topc=True),
            dict(use_prldfc=True),
            dict(use_tpsc=True),
        )
        for conflict in conflicts:
            with self.assertRaises(ValueError):
                _layer(**conflict)

    def test_more_than_100_instances_complete_in_chunks_without_truncation(self):
        layer = _layer().eval()
        torch.nn.init.zeros_(layer.icbfc_prior.center_head.weight)
        torch.nn.init.constant_(layer.icbfc_prior.center_head.bias, 10.0)
        visible, thermal = _features()

        output = layer(
            visible, thermal,
            img_metas=[dict(
                img_shape=(48, 48, 3), pad_shape=(48, 48, 3),
                batch_input_shape=(48, 48))],
            thermal_s4=torch.zeros(1, 16, 12, 12))

        self.assertEqual(len(output), 4)
        self.assertEqual(
            layer.last_icbfc_aux['prior']['candidate_count'].item(), 144)
        for level in range(4):
            self.assertEqual(
                layer.last_icbfc_aux['levels'][level][
                    'candidate_count'].item(), 144)


if __name__ == '__main__':
    unittest.main()
