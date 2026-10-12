"""Observable contracts for the paired thermal-reference COCO loader."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from dyfclt.data import PairedCocoDataset, paired_collate_fn


class PairedCocoDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for mode in ("thermal", "visible"):
            (self.root / mode).mkdir()
        pixels = np.zeros((8, 8, 3), dtype=np.uint8)
        pixels[1:3, 2:4] = (255, 255, 255)
        Image.fromarray(pixels).save(self.root / "thermal" / "a.png")
        Image.fromarray(pixels).save(self.root / "visible" / "a.png")
        self.annotation = {
            "images": [{"id": 42, "file_name": "a.png", "height": 8, "width": 8}],
            "categories": [{"id": 3, "name": "person"}, {"id": 8, "name": "rider"}],
            "annotations": [
                {"id": 1, "image_id": 42, "category_id": 3, "bbox": [2, 1, 2, 2], "ignore": 0, "iscrowd": 0},
                {"id": 2, "image_id": 42, "category_id": 8, "bbox": [6, 6, 4, 4], "ignore": 0, "iscrowd": 0},
                {"id": 3, "image_id": 42, "category_id": 8, "bbox": [0, 0, 2, 2], "ignore": 1, "iscrowd": 0},
                {"id": 4, "image_id": 42, "category_id": 8, "bbox": [0, 0, 2, 2], "ignore": 0, "iscrowd": 1},
                {"id": 5, "image_id": 42, "category_id": 3, "bbox": [4, 4, -1, 2], "ignore": 0, "iscrowd": 0},
                {"id": 6, "image_id": 42, "category_id": 3, "bbox": [20, 20, 1, 1], "ignore": 0, "iscrowd": 0},
            ],
        }
        self.write_annotation()

    def write_annotation(self):
        self.ann_file = self.root / "annotations.json"
        self.ann_file.write_text(json.dumps(self.annotation))

    def dataset(self, **kwargs):
        return PairedCocoDataset(self.ann_file, self.root, image_size=(8, 8), **kwargs)

    def test_contiguous_labels_clipped_boxes_and_metadata(self):
        ds = self.dataset()
        rgb, thermal, target = ds[0]
        self.assertEqual(ds.category_ids, [3, 8])
        self.assertEqual(ds.category_names, ["person", "rider"])
        self.assertEqual(rgb.shape, (3, 8, 8))
        self.assertTrue(torch.equal(rgb, thermal))
        self.assertEqual(rgb.dtype, torch.float32)
        self.assertGreaterEqual(float(rgb.min()), 0)
        self.assertLessEqual(float(rgb.max()), 1)
        self.assertEqual(target["labels"].tolist(), [0, 1])
        self.assertEqual(target["labels"].dtype, torch.int64)
        torch.testing.assert_close(target["boxes"], torch.tensor([[.375, .25, .25, .25], [.875, .875, .25, .25]]))
        self.assertEqual(target["orig_size"].tolist(), [8, 8])
        self.assertEqual(target["size"].tolist(), [8, 8])
        self.assertEqual(target["image_id"].item(), 42)

    def test_explicit_category_filter_and_empty_shape(self):
        ds = self.dataset(categories=["rider"])
        self.assertEqual(ds.category_ids, [8])
        self.assertEqual(ds[0][2]["labels"].tolist(), [0])
        self.annotation["annotations"] = []
        self.write_annotation()
        empty = self.dataset()[0][2]
        self.assertEqual(empty["boxes"].shape, (0, 4))
        self.assertEqual(empty["labels"].shape, (0,))

    def test_tiny_box_is_preserved_after_resize(self):
        self.annotation["annotations"] = [{"id": 1, "image_id": 42, "category_id": 3, "bbox": [3, 3, .25, .5]}]
        self.write_annotation()
        target = PairedCocoDataset(self.ann_file, self.root, image_size=(16, 16))[0][2]
        torch.testing.assert_close(target["boxes"], torch.tensor([[.390625, .40625, .03125, .0625]]))

    def test_synchronized_crop_and_flip_move_images_and_boxes(self):
        self.annotation["annotations"] = self.annotation["annotations"][:1]
        self.write_annotation()
        with patch("dyfclt.data.random.random", side_effect=[0.0, 0.0]), \
             patch("dyfclt.data.random.uniform", return_value=.5), \
             patch("dyfclt.data.random.randint", side_effect=[2, 1]):
            rgb, thermal, target = self.dataset(training=True, crop_prob=1, flip_prob=1, min_crop_scale=.5)[0]
        self.assertTrue(torch.equal(rgb, thermal))
        torch.testing.assert_close(target["boxes"], torch.tensor([[.75, .25, .5, .5]]))
        self.assertEqual(target["orig_size"].tolist(), [8, 8])
        self.assertEqual(target["size"].tolist(), [8, 8])
        original_edges = torch.tensor([2., 1., 1.])
        output_edges = target["transform"] @ original_edges
        torch.testing.assert_close(output_edges, torch.tensor([8., 0., 1.]))
        torch.testing.assert_close(target["inverse_transform"] @ output_edges, original_edges)

    def test_eval_resize_records_invertible_pixel_mapping(self):
        target = PairedCocoDataset(self.ann_file, self.root, image_size=(16, 12))[0][2]
        torch.testing.assert_close(target["transform"] @ torch.tensor([2., 1., 1.]),
                                   torch.tensor([3., 2., 1.]))
        torch.testing.assert_close(target["inverse_transform"] @ torch.tensor([3., 2., 1.]),
                                   torch.tensor([2., 1., 1.]))

    def test_stale_annotation_dimensions_use_decoded_pixels_and_keep_source(self):
        # Reproduces RGBTDronePerson metadata smaller than both paired images.
        self.annotation["images"][0].update(width=7, height=6)
        self.write_annotation()
        before = self.ann_file.read_bytes()
        ds = PairedCocoDataset(self.ann_file, self.root, image_size=(16, 12))
        rgb, thermal, target = ds[0]
        self.assertEqual(target["orig_size"].tolist(), [8, 8])
        self.assertEqual(target["size"].tolist(), [16, 12])
        self.assertEqual(rgb.shape, (3, 16, 12))
        self.assertTrue(torch.equal(rgb, thermal))
        torch.testing.assert_close(target["boxes"], torch.tensor([
            [.375, .25, .25, .25], [.875, .875, .25, .25]]))
        torch.testing.assert_close(target["transform"] @ torch.tensor([2., 1., 1.]),
                                   torch.tensor([3., 2., 1.]))
        # Normalized predictions recover original absolute box coordinates.
        from dyfclt.runtime import postprocess
        predictions = postprocess({"pred_logits": torch.tensor([[[10., -10.]]]),
                                   "pred_boxes": target["boxes"][:1].unsqueeze(0)},
                                  [target], ds.category_ids, topk=1)
        torch.testing.assert_close(torch.tensor(predictions[0]["bbox"]),
                                   torch.tensor([2., 1., 2., 2.]))
        self.assertEqual(self.ann_file.read_bytes(), before)
        self.assertEqual(ds.images[0]["width"], 7)

    def test_stale_metadata_and_crop_keep_original_decoded_size(self):
        self.annotation["images"][0].update(width=7, height=6)
        self.annotation["annotations"] = self.annotation["annotations"][:1]
        self.write_annotation()
        with patch("dyfclt.data.random.random", side_effect=[0.0, 0.0]), \
             patch("dyfclt.data.random.uniform", return_value=.5), \
             patch("dyfclt.data.random.randint", side_effect=[2, 1]):
            _, _, target = self.dataset(training=True, crop_prob=1, flip_prob=1, min_crop_scale=.5)[0]
        self.assertEqual(target["orig_size"].tolist(), [8, 8])
        torch.testing.assert_close(target["boxes"], torch.tensor([[.75, .25, .5, .5]]))
        torch.testing.assert_close(target["inverse_transform"] @ torch.tensor([8., 0., 1.]),
                                   torch.tensor([2., 1., 1.]))

    def test_missing_pair_and_dimension_mismatch_raise(self):
        (self.root / "visible" / "a.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "visible"):
            self.dataset()[0]
        Image.new("RGB", (7, 8)).save(self.root / "visible" / "a.png")
        with self.assertRaisesRegex(ValueError, "size"):
            self.dataset()[0]

    def test_collate_stacks_images_and_keeps_variable_targets(self):
        ds = self.dataset()
        rgb, thermal, targets = paired_collate_fn([ds[0], ds[0]])
        self.assertEqual(rgb.shape, (2, 3, 8, 8))
        self.assertEqual(thermal.shape, (2, 3, 8, 8))
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0]["image_id"].item(), 42)


if __name__ == "__main__":
    unittest.main()
