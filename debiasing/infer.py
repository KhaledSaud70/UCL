import checkpoint as checkpoint
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="SMS Inference",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', type=str, help='path of data dir', required=True, default='/data')
    parser.add_argument('--dataset', type=str, help='dataset (format name_attr e.g. biased-mnist_0.999)', required=True)
    parser.add_argument('--camera_names', type=str, help='camera directory names', default='all')

    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--model', type=str, help='model architecture')
    parser.add_argument('--feat_dim', type=int, help='size of projection head', default=128)
    parser.add_argument('--model_weights', type=str, help='path to model weights')
    parser.add_argument('--yolo_weights', type=str, help='path to YOLO weights')
    parser.add_argument('--iou_threshold', type=int, help='IoU threshold to between refernce box and query box. IoU less than this value will be make the query box as a gap', default=0.15)
    parser.add_argument('--k_max', type=int, help='Maximum number of nearest neighbors to consider during product feature matching', default=3)

    opts = parser.parse_args()

    return opts

def build_transforms():
    # Mean and std should be computed offline
    mean = (0.5732, 0.4699, 0.4269)
    std = (0.2731, 0.2662, 0.2697)
    resize_size = 224

    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def setup_model(cfg):
    if 'resnet' in cfg.model:
        model = models.SupConResNet(cfg.model)

    elif cfg.model == 'simpleconvnet':
        model = models.SupConSimpleConvNet()
    
    else:
        ValueError(f'Unsupported model name {cfg.model}')
    
    if cfg.model_weights:
        checkpoint.load_checkpoint(cfg.model_weights, model, remove_fc=True)


    if cfg.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)
    model = model.to(cfg.device)
    
    layers = torch.nn.Sequential(*list(model.children()))

    try:
        potential_last_layer = layers[-1]
        while not isinstance(potential_last_layer, nn.Linear):
            potential_last_layer = potential_last_layer[-1]
    except TypeError:
        raise TypeError('Can\'t find the linear layer of the model')
    
    cfg.feat_dim = potential_last_layer.in_features
    model = torch.nn.Sequential(*list(model.children())[:-1])
    
    return model

def load_image(image_path):
    return Image.open(image_path)

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def evaluate():
    pass

def find_misplaced_products(cfg, image, metadata, indices, detected_boxes, rf_boxes, feat_index, model, transform):
    misplaced_products_position = []
    with torch.no_grad():
        for q_idx, rf_idx in indices:
            q_box = detected_boxes[q_idx]
            rf_class = metadata[rf_idx]['class']

             # Determine the value of k based on the number of samples in the current class
            num_samples_in_class = sum(1 for entery in metadata if entery['class'] == rf_class)
            k = min(num_samples_in_class, cfg.k_max) if num_samples_in_class > 0 else 1

            q_img = image.crop(tuple(q_box.tolist()))
            q_feat = model(transform(q_img).unsqueeze(0)).detach().cpu()
            q_feat = F.normalize(q_feat, dim=1).squeeze().unsqueeze(0)
            topk_score, topk_idxs = feat_index.search(q_feat.numpy(), k)

            topk_classes = set()
            topk_rf_boxes = [rf_boxes[idx] for idx in topk_idxs][0]
            for box in topk_rf_boxes:
                box = box.view(1, 4)
                for box_idx, box2 in enumerate(rf_boxes):
                    rf_box = box2.view(1, 4)
                    iou = box_iou(box.cpu(), rf_box.cpu()).item()
                    if iou >= cfg.iou_threshold:
                        topk_classes.add(metadata[box_idx]["class"])

            if rf_class not in topk_classes:
                print(rf_class, topk_classes)
                misplaced_products_position.append(q_box)

        return torch.stack(misplaced_products_position)

def find_gaps(cfg, iou_mat, rf_boxes):
    mask = torch.any(iou_mat > cfg.iou_threshold, dim=1)
    indices = torch.nonzero(~mask)
    gaps = rf_boxes[indices]
    return gaps

def process_image(cfg, image, model, yolo_model, feat_index, rf_boxes, metadata, transform):
    image_res = yolo_model.predict(image, iou=0.5)
    detected_boxes = image_res[0].boxes.xyxy

    iou_mat = box_iou(detected_boxes, rf_boxes)
    indices = torch.nonzero(iou_mat >= cfg.iou_threshold)

    # Find unique indices with maximum IoU for each detected box
    unique_indices = {}
    for idx in indices:
        q_idx, rf_idx = idx.tolist()
        iou = iou_mat[q_idx, rf_idx].item()
        if q_idx not in unique_indices or iou > unique_indices[q_idx][0]:
            unique_indices[q_idx] = (iou, rf_idx)

    unique_indices = torch.tensor([[q_idx, rf_idx] for q_idx, (_, rf_idx) in unique_indices.items()])

    gaps_position = find_gaps(cfg, iou_mat, rf_boxes)

    misplaced_products_position = find_misplaced_products(cfg,
                                                          image, metadata,
                                                          unique_indices,
                                                          detected_boxes, rf_boxes,
                                                          feat_index,
                                                          model, transform)
    
    print('Gaps', gaps_position)
    print('mis', misplaced_products_position)
    return None

def inference(cfg, model, yolo_model, transform):
    if cfg.camera_names == 'all':
        cfg.camera_names = [camera for camera in os.listdir(cfg.data_dir) if camera.startswith("camera")]
    elif not isinstance(cfg.camera_names, list):
        cfg.camera_names = [cfg.camera_names]

    for camera_name in cfg.camera_names[:1]:
        camera_dir = os.path.join(cfg.data_dir, camera_name)
        rf_image = load_image(os.path.join(camera_dir, 'reference.jpg')) # Ensure consistent refernce image names for all cameras
        metadata = load_json(os.path.join(camera_dir, 'metadata.json')) # Ensure consistent metadata file name for all cameras

        boxes = [[entery['box']['x1'], entery['box']['y1'], entery['box']['x2'], entery['box']['y2']] for entery in metadata]
        rf_boxes = torch.tensor(boxes)

        feat_index_path = os.path.join(camera_dir, 'feat_index.index')
        if not os.path.exists(feat_index_path):
            feat_index = faiss.IndexFlatIP(cfg.feat_dim)

            features = []
            with torch.no_grad():
                for box in rf_boxes:
                    img = rf_image.crop(tuple(box.tolist()))
                    feat = model(transform(img).unsqueeze(0)).squeeze().detach().cpu().numpy()
                    features.append(feat)

            features = F.normalize(torch.tensor(features, dtype=torch.float32)) 

            feat_index.add(features.numpy())

            faiss.write_index(feat_index, feat_index_path)
        else:
            feat_index = faiss.read_index(feat_index_path)

        images_dir = os.path.join(camera_dir, 'images')
        for image_file in os.listdir(images_dir)[:1]:
            if image_file.endswith('.jpg'):
                image_path = os.path.join(images_dir, image_file)
                image = load_image(image_path)

                results = process_image(cfg, image, model, yolo_model, feat_index, rf_boxes, metadata, transform)


    

def main():
    cfg = parse_arguments()

    model = setup_model(cfg)
    model.eval()
    yolo_model = YOLO(cfg.yolo_weights)

    result = inference(cfg, model, yolo_model, transform=build_transforms())

    # if cfg.eval:
    #     result_metrics = evaluate(cfg)





if __name__ == '__main__':
    main()