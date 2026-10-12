"""Behavioral checks for the released RGBTDronePerson scoring rules."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from dyfclt.evaluation import evaluate_records


class TinyEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.annotations = Path(self.temp.name) / 'annotations.json'
        self.dataset = {
            'images': [{'id': 1, 'width': 128, 'height': 128},
                       {'id': 2, 'width': 128, 'height': 128}],
            'categories': [{'id': 0, 'name': 'person'}, {'id': 1, 'name': 'rider'},
                           {'id': 2, 'name': 'crowd'}, {'id': 3, 'name': 'uncertain'}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 0, 'bbox': [4, 4, 10, 10], 'area': 100, 'iscrowd': 0},
                {'id': 2, 'image_id': 1, 'category_id': 0, 'bbox': [40, 40, 24, 24], 'area': 576, 'iscrowd': 0},
                {'id': 3, 'image_id': 1, 'category_id': 0, 'bbox': [80, 4, 10, 10], 'area': 100,
                 'iscrowd': 0, 'uncertain': 1},
                {'id': 4, 'image_id': 1, 'category_id': 3, 'bbox': [80, 80, 10, 10], 'area': 100, 'iscrowd': 0},
            ],
        }
        self.annotations.write_text(json.dumps(self.dataset))

    def score(self, records, images=(1,)):
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluate_records(self.annotations, records, images, [0, 1, 2])

    def test_tiny_and_small_are_distinct_and_uncertain_is_ignored(self):
        prediction = {'image_id': 1, 'category_id': 0, 'bbox': [4, 4, 10, 10], 'score': 0.9}
        before = copy.deepcopy(prediction)
        score = self.score([prediction])
        self.assertAlmostEqual(score['AP50_tiny'], 1.0)
        self.assertAlmostEqual(score['AP50_small'], 0.0)
        self.assertAlmostEqual(score['AP50'], 0.504950495, places=6)
        self.assertEqual(prediction, before)
        self.assertEqual(json.loads(self.annotations.read_text()), self.dataset)

    def test_perfect_small_detection_and_ignored_person_region(self):
        records = [{'image_id': 1, 'category_id': 0, 'bbox': [80, 4, 10, 10], 'score': .99},
                   {'image_id': 1, 'category_id': 0, 'bbox': [40, 40, 24, 24], 'score': .9},
                   {'image_id': 1, 'category_id': 0, 'bbox': [4, 4, 10, 10], 'score': .8}]
        score = self.score(records)
        self.assertAlmostEqual(score['AP50'], 1.)
        self.assertAlmostEqual(score['AP50_tiny'], 1.)
        self.assertAlmostEqual(score['AP50_small'], 1.)
        self.assertAlmostEqual(score['AP25'], 1.)
        self.assertAlmostEqual(score['AP75'], 1.)

    def test_empty_results_and_selected_negative_image(self):
        empty = self.score([], images=(1, 2))
        self.assertEqual(empty['images'], 2)
        self.assertEqual(empty['AP50'], 0.)
        self.assertEqual(empty['AP50_tiny1'], -1.)
        positive = [{'image_id': 1, 'category_id': 0, 'bbox': [4, 4, 10, 10], 'score': .8},
                    {'image_id': 1, 'category_id': 0, 'bbox': [40, 40, 24, 24], 'score': .7}]
        fp = {'image_id': 2, 'category_id': 0, 'bbox': [4, 4, 10, 10], 'score': .95}
        with_negative = self.score([fp] + positive, images=(1, 2))
        without_negative = self.score(positive, images=(1,))
        self.assertLess(with_negative['AP50'], without_negative['AP50'])
        with self.assertRaises(ValueError):
            self.score([fp], images=(1,))

    def test_four_class_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'person, rider, crowd'):
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_records(self.annotations, [], [1], [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
