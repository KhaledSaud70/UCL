from checkpoint import load_checkpoint
from ultralytics import YOLO
import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import box_iou
import os
import json
from PIL import Image
import faiss
import argparse
from collections import OrderedDict


def parse_arguments():
    parser = argparse.ArgumentParser(description="SMS Inference",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', type=str, help='path of data dir', required=True, default='/data')
    parser.add_argument('--camera_names', type=str, help='camera directory names', default='all')
    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--backbone', type=str, help='model architecture')
    parser.add_argument('--features_dim', type=int, help='size of projection head', default=128)
    parser.add_argument('--checkpoint', type=str, help='path to model weights')
    parser.add_argument('--yolo_weights', type=str, help='path to YOLO weights')
    parser.add_argument('--iou',
                        type=int,
                        help='Intersection Over Union (IoU) threshold between the reference and detected boxes. Objects detected with IoU below this threshold will be considered as a gap', default=0.15)
    parser.add_argument('--k_max', type=int, help='Maximum number of nearest neighbors to consider during product feature matching', default=3)

    opts = parser.parse_args()

    return opts


def build_transforms():
    # Mean and std should be computed offline
    mean = (0.5525, 0.4640, 0.4124)
    std = (0.2703, 0.2622, 0.2679)
    resize_size = (224, 224)

    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def build_model(cfg):
    model = models.SupConModel(cfg.backbone, num_classes=0)
    cfg.features_dim = model.features_dim
    
    if cfg.checkpoint:
        model.load_checkpoint(cfg.checkpoint)


    if cfg.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)
    model = model.to(cfg.device)
    
    return model.encoder


def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def find_misplaced_products(cfg, image, metadata, indices, detected_boxes, rf_boxes, feat_index, model, transform):
    misplaced_products_position = {"misplacedProducts": []}
    with torch.no_grad():
        for q_idx, rf_idx in indices:
            q_box = detected_boxes[q_idx]
            rf_id = metadata[rf_idx]["id"]

             # Determine the value of k based on the number of samples in the current class
            num_samples_in_class = sum(1 for entery in metadata if entery["id"] == rf_id)
            k = min(num_samples_in_class, cfg.k_max) if num_samples_in_class > 0 else 1

            q_img = image.crop(tuple(q_box.tolist()))
            q_img = transform(q_img).unsqueeze(0).to(cfg.device)
            q_feat = model(q_img).detach().cpu()
            q_feat = F.normalize(q_feat, dim=1).squeeze().unsqueeze(0)
            topk_score, topk_idxs = feat_index.search(q_feat.numpy(), k)

            topk_products = []
            topk_boxes = [rf_boxes[idx] for idx in topk_idxs][0]
            for top_box in topk_boxes:
                top_box = top_box.view(1, 4)
                for box_idx, box in enumerate(rf_boxes):
                    box = box.view(1, 4)
                    iou = box_iou(top_box.cpu(), box.cpu()).item()
                    if iou >= cfg.iou:
                        topk_products.append({
                            "boundingBox": box,
                            "id": metadata[box_idx]["id"],
                        })

            topk_ids = [product["id"] for product in topk_products]
            if rf_id not in topk_ids:
                q_box = q_box.tolist()
                detected_product = {
                    "positionProductId": rf_id,
                    "detectedObject": {
                        "id": topk_ids[topk_score.argmax()],
                        "boundingBox": {
                            "x1": q_box[0],
                            "y1": q_box[1],
                            "x2": q_box[2],
                            "y2": q_box[3]
                        },
                        "classifications": [
                            {
                                "confidence": float(topk_score.max()),
                                
                            }
                        ]
                    }
                }

                misplaced_products_position['misplacedProducts'].append(detected_product)

        return misplaced_products_position


def find_gaps(cfg, iou_mat, rf_boxes, metadata):
    mask = torch.any(iou_mat > cfg.iou, dim=0)
    indices = torch.nonzero(~mask)

    gaps = {"gaps": []}
    for idx in indices:
        box = rf_boxes[idx.item()].tolist()
        detected_gaps = {
            "id": metadata[idx]["id"],
            "boundingBox": {
                "x1": box[0],
                "x2": box[1],
                "y1": box[2],
                "y2": box[3]
            }
        }
        gaps['gaps'].append(detected_gaps)

    return gaps


def process_image(cfg, image, model, yolo_model, feat_index, rf_boxes, metadata, transform):
    image_res = yolo_model.predict(image, iou=0.5)
    detected_boxes = image_res[0].boxes.xyxy

    iou_mat = box_iou(detected_boxes, rf_boxes)
    indices = torch.nonzero(iou_mat >= cfg.iou)

    # Find unique indices with maximum IoU for each detected box
    unique_indices = {}
    for idx in indices:
        q_idx, rf_idx = idx.tolist()
        iou = iou_mat[q_idx, rf_idx].item()
        if q_idx not in unique_indices or iou > unique_indices[q_idx][0]:
            unique_indices[q_idx] = (iou, rf_idx)

    unique_indices = torch.tensor([[q_idx, rf_idx] for q_idx, (_, rf_idx) in unique_indices.items()])

    gaps_position = find_gaps(cfg, iou_mat, rf_boxes, metadata)

    misplaced_products_position = find_misplaced_products(cfg,
                                                          image, metadata,
                                                          unique_indices,
                                                          detected_boxes, rf_boxes,
                                                          feat_index,
                                                          model, transform)
    
    results = OrderedDict()
    results.update(misplaced_products_position)
    results.update(gaps_position)
    return results


def inference(cfg, model, yolo_model, transform):
    if cfg.camera_names == 'all':
        cfg.camera_names = [camera for camera in os.listdir(cfg.data_dir) if camera.startswith("camera")]
    elif not isinstance(cfg.camera_names, list):
        cfg.camera_names = [cfg.camera_names]
    
    results_per_camera = OrderedDict({camera_name: [] for camera_name in cfg.camera_names})


    for camera_name in cfg.camera_names:
        camera_dir = os.path.join(cfg.data_dir, camera_name)
        rf_image = Image.open(os.path.join(camera_dir, 'reference.jpg')) # Ensure consistent refernce image names for all cameras
        metadata = load_json(os.path.join(camera_dir, 'metadata.json')) # Ensure consistent metadata file name for all cameras

        boxes = [[entery['box']['x1'], entery['box']['y1'], entery['box']['x2'], entery['box']['y2']] for entery in metadata]
        rf_boxes = torch.tensor(boxes, device=cfg.device)

        feat_index = faiss.IndexFlatIP(cfg.features_dim)

        features = torch.empty(len(rf_boxes), cfg.features_dim, dtype=torch.float32)
        with torch.no_grad():
            for i, box in enumerate(rf_boxes):
                img = rf_image.crop(tuple(box.tolist()))
                feat = model(transform(img).unsqueeze(0)).squeeze().detach().cpu()
                features[i] = feat

        features = F.normalize(features)
        feat_index.add(features.numpy())

        images_dir = os.path.join(camera_dir, 'images')
        for image_file in os.listdir(images_dir):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(images_dir, image_file)
                image = Image.open(image_path)

                results = process_image(cfg, image, model, yolo_model, feat_index, rf_boxes, metadata, transform)
                results['image_file'] = image_file
                results_per_camera[camera_name].append(results)
        
    
    # json_path = os.path.join(cfg.data_dir, 'inference.json')
    # with open(json_path, 'w') as json_file:
    #         json.dump(results_per_camera, json_file, indent=2)
    
    return results_per_camera


def main():
    cfg = parse_arguments()

    model = build_model(cfg)
    model.eval()
    yolo_model = YOLO(cfg.yolo_weights)

    result = inference(cfg, model, yolo_model, transform=build_transforms())


if __name__ == '__main__':
    main()