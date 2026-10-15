"""Official RGBTDronePerson three-class tiny-object evaluation protocol."""
import copy
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

from .vendor.tiny_cocoeval import COCOeval


_AREA_NAMES = ('all', 'tiny', 'tiny1', 'tiny2', 'tiny3', 'small', 'reasonable')


def evaluate_records(ann_file, records, image_ids, category_ids):
    """Evaluate COCO xywh predictions on selected images using QFDet's protocol.

    AP is a fraction in [0, 1]. A size bin without ground truth returns -1,
    following the official evaluator. Only person, rider, and crowd are scored.
    """
    image_ids = sorted(set(int(value) for value in image_ids))
    category_ids = [int(value) for value in category_ids]
    if not image_ids:
        raise ValueError('Evaluation received no images')
    coco_gt = COCO()
    with Path(ann_file).open(encoding='utf-8') as stream:
        coco_gt.dataset = json.load(stream)
    coco_gt.createIndex()
    cats = {int(cat['id']): cat['name'] for cat in coco_gt.dataset['categories']}
    if len(category_ids) != 3 or {cats.get(value) for value in category_ids} != {'person', 'rider', 'crowd'}:
        raise ValueError('RGBTDronePerson benchmark categories must be person, rider, crowd')
    if not set(image_ids).issubset(coco_gt.imgs):
        raise ValueError('Evaluation image ID absent from annotations')
    records = copy.deepcopy(list(records))
    if any(int(record['image_id']) not in image_ids or int(record['category_id']) not in category_ids
           for record in records):
        raise ValueError('Prediction image or category is outside the evaluated set')

    # COCO loads fresh objects from disk. Normalize only the in-memory copy;
    # neither the source JSON nor caller-owned prediction records are changed.
    for annotation in coco_gt.dataset['annotations']:
        annotation.setdefault('iscrowd', 0)
        annotation.setdefault('area', annotation['bbox'][2] * annotation['bbox'][3])
    # Build the bbox result COCO object directly. Older pycocotools releases
    # have a Python 2-only `unicode` reference in loadRes, so this mirrors its
    # bbox branch without patching process-wide module globals.
    coco_dt = COCO()
    for index, record in enumerate(records, start=1):
        box = record['bbox']
        record['id'] = index
        record['area'] = box[2] * box[3]
        record['iscrowd'] = 0
    coco_dt.dataset = dict(info={}, images=copy.deepcopy(coco_gt.dataset['images']),
                           categories=copy.deepcopy(coco_gt.dataset['categories']),
                           annotations=records)
    coco_dt.createIndex()

    evaluator = COCOeval(coco_gt, coco_dt, 'bbox', ignore_uncertain=True,
                         use_iod_for_ignore=True)
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = category_ids
    evaluator.params.maxDets = [100, 300, 1000]
    evaluator.params.iouThrs = np.array([0.25, 0.5, 0.75])
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval['precision']  # [IoU, recall, class, size, maxDet]
    result = {}
    for iou_index, iou in enumerate((25, 50, 75)):
        for area_index, area_name in enumerate(_AREA_NAMES):
            values = precision[iou_index, :, :, area_index, -1]
            valid = values[values > -1]
            key = f'AP{iou}' if area_name == 'all' else f'AP{iou}_{area_name}'
            result[key] = float(valid.mean()) if valid.size else -1.0
    result.update(images=len(image_ids), evaluation='RGBTDronePerson tiny; maxDets=1000')
    return result
