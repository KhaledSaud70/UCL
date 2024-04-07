import os
import json
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, Subset
import torch
from map import MapDataset


class DanubeDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = set() 
        self._load_data()

    def _load_data(self):
        self.image = Image.open(os.path.join(self.root, 'reference.jpg')).rotate(180).convert('RGB')
        with open(os.path.join(self.root, 'detection_data.json'), 'r') as f:
            self._detection_data = json.load(f)
        
        for item_info in self._detection_data:
            self.classes.add(item_info['class'])

    def _apply_transform(self, image):
        if self.transform:
            return self.transform(image)
        return image
    
    def _compute_mean_std(self):
        pixel_means = []
        pixel_stds = []

        for item_info in self._detection_data:
            x1, x2, y1, y2 = item_info['box']['x1'], item_info['box']['x2'], item_info['box']['y1'], item_info['box']['y2']
            item_image = self.image.crop((x1, y1, x2, y2))
            item_image = self._apply_transform(item_image)

            if not isinstance(item_image, torch.Tensor):
                item_image = transforms.ToTensor()(item_image)

            pixel_means.append(torch.mean(item_image, dim=(1, 2)))
            pixel_stds.append(torch.std(item_image, dim=(1, 2)))

        mean = torch.stack(pixel_means).mean(0)
        std = torch.stack(pixel_stds).mean(0)

        return mean, std

    def __getitem__(self, idx):
        item_info = self._detection_data[idx]
        x1, x2, y1, y2 = item_info['box']['x1'], item_info['box']['x2'], item_info['box']['y1'], item_info['box']['y2']
        item_image = self.image.crop((x1, y1, x2, y2))
        item_image = self._apply_transform(item_image)
        item_class = item_info['class']
        return item_image, item_class

    def __len__(self):
        return len(self._detection_data)

    def num_classes(self):
        return len(self.classes)


if __name__ == '__main__':
    root = f'/home/{os.environ.get("USER")}/workspace/projects/ucl/debiasing/data/camera_ip'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = DanubeDataset(root, transform=transform)
    train_dataset = MapDataset(train_dataset, lambda x, y: (x, y, 0))


    test_dataset = DanubeDataset(root, transform=transform)
    test_dataset = MapDataset(test_dataset, lambda x, y: (x, y, 0))


    test_size = 0.1
    data_size = len(train_dataset)
    indices = list(range(data_size))
    np.random.shuffle(indices)
    split = int(np.floor(test_size * data_size))
    train_idx, test_idx = indices[split:], indices[:split]

    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(test_dataset, test_idx)

    print(f"Train size: {len(train_dataset)}, test size: {len(val_dataset)}")