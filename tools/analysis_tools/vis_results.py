import argparse
import json
import os
import pickle

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COXNet detections')
    parser.add_argument('annotation', help='COCO-format validation annotation')
    parser.add_argument('prediction', help='Detection result pickle file')
    parser.add_argument('image_root', help='Directory containing RGB images')
    parser.add_argument('output_dir', help='Directory for rendered images')
    parser.add_argument('--iou-thr', type=float, default=0.1)
    return parser.parse_args()


def calculate_iou(box1, box2):
    """Calculate IoU for two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union else 0


def visualize_results(image_path, gt_boxes, pred_boxes, output_path,
                      iou_threshold=0.1):
    """Render matched, missed, and false-positive boxes."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Unable to read image: {image_path}')

    for gt in gt_boxes:
        matched = False
        for pred in pred_boxes:
            if calculate_iou(gt, pred[:4]) >= iou_threshold:
                cv2.rectangle(
                    image, (int(pred[0]), int(pred[1])),
                    (int(pred[2]), int(pred[3])), (0, 255, 0), 2)
                matched = True
        if not matched:
            cv2.rectangle(
                image, (int(gt[0]), int(gt[1])),
                (int(gt[2]), int(gt[3])), (0, 165, 255), 2)

    for pred in pred_boxes:
        if not any(
                calculate_iou(gt, pred[:4]) >= iou_threshold
                for gt in gt_boxes):
            cv2.rectangle(
                image, (int(pred[0]), int(pred[1])),
                (int(pred[2]), int(pred[3])), (0, 0, 255), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def main():
    args = parse_args()
    with open(args.annotation, 'r', encoding='utf-8') as file:
        gt_data = json.load(file)
    with open(args.prediction, 'rb') as file:
        pred_data = pickle.load(file)

    gt_boxes = {}
    for annotation in gt_data['annotations']:
        x1, y1, width, height = annotation['bbox']
        gt_boxes.setdefault(annotation['image_id'], []).append(
            [x1, y1, x1 + width, y1 + height])

    for image_info in gt_data['images']:
        image_id = image_info['id']
        image_name = image_info['file_name']
        predictions = pred_data.get(image_name, {}).get('pred_boxes', [])
        visualize_results(
            os.path.join(args.image_root, image_name),
            gt_boxes.get(image_id, []), predictions,
            os.path.join(args.output_dir, image_name), args.iou_thr)


if __name__ == '__main__':
    main()
