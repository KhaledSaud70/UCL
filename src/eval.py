import torch
from torchvision.ops import box_iou
import os
import json
from PIL import Image
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_arguments():
    parser = argparse.ArgumentParser(description="SMS Evaluate",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', type=str, help='path of data dir', required=True, default='/data')
    parser.add_argument('--iou_threshold', type=int, help='IoU threshold between refernce box and detected box', default=0.5)
    parser.add_argument('--camera_names', type=str, help='camera directory names to be evaluated', default='all')


    opts = parser.parse_args()

    return opts


def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def calculate_metrics(ground_truth_boxes, detected_boxes, iou_threshold=0.15):
    iou_matrix = box_iou(detected_boxes, ground_truth_boxes)

    tp = (iou_matrix.max(dim=0).values > iou_threshold).sum().item()
    fp = (iou_matrix == 0.0).all(dim=1).sum().item()
    fn = len(ground_truth_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    return metrics


def evaluate(cfg):
    if cfg.camera_names == 'all':
        cfg.camera_names = [camera for camera in os.listdir(cfg.data_dir) if camera.startswith("camera")]
    elif not isinstance(cfg.camera_names, list):
        cfg.camera_names = [cfg.camera_names]
    
    results = OrderedDict({camera_name: {'statusPerImage': [], 'statusPerCamera': {}} for camera_name in cfg.camera_names})
    inference_metadata = load_json(os.path.join(cfg.data_dir, 'inference.json'))

    for camera_name in cfg.camera_names[:1]:
        camera_dir = os.path.join(cfg.data_dir, camera_name)
        inference_data = inference_metadata[camera_name]
        labels_dir = os.path.join(camera_dir, 'labels')

        metrics_sum = {'precision': 0, 'recall': 0, 'f1': 0}
        processed_images = 0

        for label_file in os.listdir(labels_dir):
            labels_data = load_json(os.path.join(labels_dir, label_file))
            label_file_name = os.path.splitext(label_file)[0]
            infer_data = next((d for d in inference_data if os.path.splitext(d['image_file'])[0] == label_file_name), None)
            if infer_data is None:
                print(f"No inference data found for label file '{label_file_name}'. Skipping...")
                continue
            
            processed_images += 1

            detected_boxes = [list(d['detectedObject']['boundingBox'].values()) for d in infer_data['misplacedProducts']]
            detected_boxes = torch.tensor(detected_boxes)
            label_boxes = torch.stack([torch.tensor(list(entry["box"].values())) for entry in labels_data])

            metrics_status = calculate_metrics(label_boxes, detected_boxes)
            metrics_sum['precision'] += metrics_status['precision']
            metrics_sum['recall'] += metrics_status['recall']
            metrics_sum['f1'] += metrics_status['f1']

            metrics_status['label_file'] = label_file
            results[camera_name]['statusPerImage'].append(metrics_status)
        
        if processed_images > 0:
            avg_metrics = {k: v / processed_images for k, v in metrics_sum.items()}
        else:
            avg_metrics = {'precision': 0, 'recall': 0, 'f1': 0}

        results[camera_name]['statusPerCamera'] = avg_metrics

    json_path = os.path.join(cfg.data_dir, 'evaluation.json')
    with open(json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)


def main():
    cfg = parse_arguments()
    evaluate(cfg)


if __name__ == '__main__':
    main()