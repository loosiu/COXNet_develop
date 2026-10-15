"""Paired COCO images with thermal-reference detection targets.

``image_root`` is a split directory containing ``visible/`` and ``thermal/``.
The annotation file is left unchanged for COCO evaluation. Evaluation applies
only an exact resize, so ``orig_size`` reverses the normalized model boxes.
Decoded image dimensions are authoritative: some source JSON width/height
fields are stale. COCO bbox values remain absolute pixels and are not rescaled
to reconcile metadata. RGB and thermal must still have matching dimensions.
``transform`` and ``inverse_transform`` record the full pixel-coordinate
mapping for callers that also need training augmentation provenance.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class PairedCocoDataset(Dataset):
    """Return ``(visible, thermal, target)`` for paired COCO images.

    Images are float RGB tensors in [0, 1], without ImageNet normalization.
    ``image_size`` uses (height, width). Boxes are normalized ``cxcywh``.
    A crop, when enabled, takes a square fraction of each original dimension;
    both modalities and thermal-reference boxes receive the same geometry.
    """

    def __init__(
        self,
        ann_file: Union[str, Path],
        image_root: Union[str, Path],
        image_size: Tuple[int, int] = (512, 640),
        training: bool = False,
        categories: Optional[Sequence[Union[str, int]]] = None,
        crop_prob: float = 0.5,
        min_crop_scale: float = 0.7,
        flip_prob: float = 0.5,
    ):
        self.ann_file = Path(ann_file)
        self.image_root = Path(image_root)
        if not self.ann_file.is_file():
            raise FileNotFoundError(f"COCO annotation file missing: {self.ann_file}")
        with self.ann_file.open() as stream:
            coco = json.load(stream)

        self.image_size = tuple(int(v) for v in image_size)
        if len(self.image_size) != 2 or any(v <= 0 for v in self.image_size):
            raise ValueError("image_size must be positive (height, width)")
        if not 0 <= crop_prob <= 1 or not 0 <= flip_prob <= 1:
            raise ValueError("crop_prob and flip_prob must be in [0, 1]")
        if not 0 < min_crop_scale <= 1:
            raise ValueError("min_crop_scale must be in (0, 1]")
        self.training = bool(training)
        self.crop_prob = float(crop_prob)
        self.min_crop_scale = float(min_crop_scale)
        self.flip_prob = float(flip_prob)

        all_categories = coco["categories"]
        ids = [int(cat["id"]) for cat in all_categories]
        names = [str(cat["name"]) for cat in all_categories]
        if len(ids) != len(set(ids)) or len(names) != len(set(names)):
            raise ValueError("COCO category ids and names must be unique")
        if categories is None:
            selected = all_categories
        else:
            requested = set(categories)
            selected = [cat for cat in all_categories if cat["id"] in requested or cat["name"] in requested]
            found = {cat["id"] for cat in selected} | {cat["name"] for cat in selected}
            unknown = requested - found
            if unknown:
                raise ValueError(f"Unknown COCO categories: {sorted(map(str, unknown))}")
        if not selected:
            raise ValueError("No categories selected")
        self.category_ids = [int(cat["id"]) for cat in selected]
        self.category_names = [str(cat["name"]) for cat in selected]
        self.category_id_to_label = {category_id: label for label, category_id in enumerate(self.category_ids)}

        self.images = coco["images"]
        self.annotations_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            if ann.get("ignore", 0) or ann.get("iscrowd", 0):
                continue
            if ann["category_id"] in self.category_id_to_label:
                self.annotations_by_image[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def _pair_paths(self, file_name: str):
        relative = Path(file_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid COCO image file_name: {file_name}")
        if relative.parts and relative.parts[0] in ("thermal", "visible"):
            relative = Path(*relative.parts[1:])
        return self.image_root / "visible" / relative, self.image_root / "thermal" / relative

    @staticmethod
    def _tensor(image: Image.Image):
        return torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div_(255.0)

    def __getitem__(self, index):
        record = self.images[index]
        visible_path, thermal_path = self._pair_paths(record["file_name"])
        for path in (visible_path, thermal_path):
            if not path.is_file():
                raise FileNotFoundError(f"Paired image missing: {path}")
        with Image.open(visible_path) as image:
            visible = image.convert("RGB")
        with Image.open(thermal_path) as image:
            thermal = image.convert("RGB")
        width, height = thermal.size
        original_width, original_height = width, height
        if visible.size != thermal.size:
            raise ValueError(
                f"Paired image size mismatch for image {record['id']}: "
                f"visible={visible.size}, thermal={thermal.size}, "
                f"annotation={(record['width'], record['height'])}"
            )

        boxes, labels = [], []
        for ann in self.annotations_by_image[record["id"]]:
            x, y, box_width, box_height = map(float, ann["bbox"])
            if not all(math.isfinite(v) for v in (x, y, box_width, box_height)):
                continue
            if box_width <= 0 or box_height <= 0:
                continue
            x1, y1 = max(0., x), max(0., y)
            x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.category_id_to_label[ann["category_id"]])

        # Homogeneous matrix mapping original continuous pixel edges to output.
        transform = torch.eye(3, dtype=torch.float32)
        if self.training and random.random() < self.crop_prob:
            scale = random.uniform(self.min_crop_scale, 1.0)
            crop_width = max(1, round(width * scale))
            crop_height = max(1, round(height * scale))
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            bounds = (left, top, left + crop_width, top + crop_height)
            visible = visible.crop(bounds)
            thermal = thermal.crop(bounds)
            cropped_boxes, cropped_labels = [], []
            for (x1, y1, x2, y2), label in zip(boxes, labels):
                x1 = max(0., min(float(crop_width), x1 - left))
                y1 = max(0., min(float(crop_height), y1 - top))
                x2 = max(0., min(float(crop_width), x2 - left))
                y2 = max(0., min(float(crop_height), y2 - top))
                if x2 > x1 and y2 > y1:
                    cropped_boxes.append([x1, y1, x2, y2])
                    cropped_labels.append(label)
            boxes, labels = cropped_boxes, cropped_labels
            transform = torch.tensor([[1., 0., -left], [0., 1., -top], [0., 0., 1.]]) @ transform
            width, height = crop_width, crop_height

        if self.training and random.random() < self.flip_prob:
            visible = ImageOps.mirror(visible)
            thermal = ImageOps.mirror(thermal)
            boxes = [[width - x2, y1, width - x1, y2] for x1, y1, x2, y2 in boxes]
            transform = torch.tensor([[-1., 0., float(width)], [0., 1., 0.], [0., 0., 1.]]) @ transform

        output_height, output_width = self.image_size
        visible = visible.resize((output_width, output_height), Image.Resampling.BILINEAR)
        thermal = thermal.resize((output_width, output_height), Image.Resampling.BILINEAR)
        transform = torch.tensor([[output_width / width, 0., 0.], [0., output_height / height, 0.], [0., 0., 1.]]) @ transform

        normalized = []
        for x1, y1, x2, y2 in boxes:
            normalized.append([(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                               (x2 - x1) / width, (y2 - y1) / height])
        target = {
            "boxes": torch.tensor(normalized, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "orig_size": torch.tensor([original_height, original_width], dtype=torch.int64),
            "size": torch.tensor([output_height, output_width], dtype=torch.int64),
            "image_id": torch.tensor(record["id"], dtype=torch.int64),
            "transform": transform,
            "inverse_transform": torch.linalg.inv(transform),
        }
        return self._tensor(visible), self._tensor(thermal), target


def paired_collate_fn(samples):
    """Stack equal-sized image pairs and retain per-image target dictionaries."""
    visible, thermal, targets = zip(*samples)
    return torch.stack(visible), torch.stack(thermal), list(targets)
