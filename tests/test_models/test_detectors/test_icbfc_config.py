import unittest

import torch
from mmcv import Config

from mmdet.models import build_detector
from mmdet.models.detectors.fusionnet_xo import FusionNetXO


class TestICBFCDetectorConfig(unittest.TestCase):

    def test_icbfc_config_preserves_cross_stage_baseline_contract(self):
        model = Config.fromfile('configs/coxnet/icbfc/ICBFC.py').model

        self.assertEqual(model.neck.start_level, 2)
        self.assertEqual(model.neck_t.start_level, 1)
        self.assertEqual(
            model.bbox_head.anchor_generator.strides, [8, 16, 32, 64])
        self.assertTrue(model.wf_loss)
        self.assertEqual(model.wf_loss_mode, 'kl_v2')
        self.assertEqual(model.train_cfg.assigner.type, 'QLSAssigner')
        self.assertEqual(model.test_cfg.nms.iou_threshold, 0.3)
        self.assertEqual(model.use_clfm, [])
        self.assertTrue(model.use_icbfc)

    def test_icbfc_config_builds_detector(self):
        model_cfg = Config.fromfile(
            'configs/coxnet/icbfc/ICBFC.py').model.copy()
        model_cfg.backbone.init_cfg = None

        detector = build_detector(model_cfg)

        self.assertIsInstance(detector, FusionNetXO)
        self.assertTrue(detector.use_icbfc)
        self.assertEqual(len(detector.fuse_layer.icbfc_layers), 4)

    def test_detector_passes_raw_stride4_thermal_and_img_metas(self):
        class Backbone:
            def __call__(self, visible, thermal):
                visible_raw = tuple(
                    torch.randn(1, 4, size, size)
                    for size in (16, 8, 4, 2))
                thermal_raw = tuple(
                    torch.randn(1, 4, size, size)
                    for size in (16, 8, 4, 2))
                return visible_raw, thermal_raw

        class Neck:
            def __call__(self, features):
                return features[1:]

        class RecordingFusion:
            def __call__(self, visible, thermal, *args, **kwargs):
                self.visible = visible
                self.thermal = thermal
                self.args = args
                self.kwargs = kwargs
                return visible

        class Detector:
            backbone = Backbone()
            neck = Neck()
            neck_t = Neck()
            with_neck = True
            use_icbfc = True
            fuse_layer = RecordingFusion()

        detector = Detector()
        images = (torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
        metas = [dict(
            img_shape=(64, 64, 3), pad_shape=(64, 64, 3),
            batch_input_shape=(64, 64))]

        FusionNetXO.extract_feat(detector, images, img_metas=metas)

        self.assertEqual(
            detector.fuse_layer.kwargs['thermal_s4'].shape[-2:], (16, 16))
        self.assertIs(detector.fuse_layer.args[1], metas)

    def test_simple_test_excludes_padding_candidates(self):
        class RecordingDetector:
            class EmptyHead:
                num_classes = 1

                @staticmethod
                def simple_test(feats, img_metas, rescale=False):
                    return [(torch.empty(0, 5),
                             torch.empty(0, dtype=torch.long))]

            bbox_head = EmptyHead()

            def extract_feat(self, img, img_metas=None):
                self.metas = img_metas
                return [torch.randn(1, 4, 2, 2)]

        detector = RecordingDetector()
        metas = [dict(
            img_shape=(48, 64, 3), pad_shape=(64, 64, 3),
            batch_input_shape=(64, 64))]

        FusionNetXO.simple_test(
            detector,
            (torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)),
            metas)

        self.assertIs(detector.metas, metas)

    def test_parse_losses_does_not_optimize_icbfc_diagnostics(self):
        losses = dict(
            loss_detector=torch.tensor(2.0),
            loss_icbfc_center=torch.tensor(0.5),
            icbfc_delta_ratio=torch.tensor(100.0),
            icbfc_candidate_count=torch.tensor(50.0))

        total, log_vars = FusionNetXO._parse_losses(None, losses)

        self.assertEqual(total.item(), 2.5)
        self.assertEqual(log_vars['loss'], 2.5)

    def test_baseline_and_legacy_configs_still_build(self):
        for path in (
                'configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py',
                'configs/coxnet/tpsc/TPSC_relation.py'):
            model_cfg = Config.fromfile(path).model.copy()
            model_cfg.backbone.init_cfg = None
            detector = build_detector(model_cfg)
            self.assertIsInstance(detector, FusionNetXO)


if __name__ == '__main__':
    unittest.main()
