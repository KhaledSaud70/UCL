import os
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


class DanubeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = set() 
        self.metadata = []
        self._load_data()

    def _load_data(self):
        cameras = [camera for camera in os.listdir(self.data_dir) if camera.startswith("camera")]
        for camera in cameras:
            camera_path = os.path.join(self.data_dir, camera)
            if os.path.isdir(camera_path):
                reference_image_path = os.path.join(camera_path, 'reference.jpg')
                metadata_path = os.path.join(camera_path, 'metadata.json')

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                for entry in metadata:
                    entry['camera'] = camera
                    self.classes.add(entry['class'])
                    entry['image_path'] = reference_image_path
                    self.metadata.append(entry)

    def _apply_transform(self, image):
        if self.transform:
            return self.transform(image)
        return image
    
    def __getitem__(self, idx):
        entry = self.metadata[idx]
        # print(entry)
        image_path = entry['image_path']
        x1, x2, y1, y2 = entry['box']['x1'], entry['box']['x2'], entry['box']['y1'], entry['box']['y2']
        
        image = Image.open(image_path).convert('RGB').crop((x1, y1, x2, y2))
        image = self._apply_transform(image)
        
        item_class = entry['class']
        return image, item_class

    def __len__(self):
        return len(self.metadata)

    @property
    def num_classes(self):
        return len(self.classes)

def compute_mean_std(dataloader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for images, _ in tqdm(dataloader):  # (B,H,W,C)
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == '__main__':
    data_dir = f'/home/{os.environ.get("USER")}/workspace/projects/shlef-monitoring/src/data'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = DanubeDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)

    # Mean = tensor([0.5732, 0.4699, 0.4269]), Std = tensor([0.2731, 0.2662, 0.2697])
    mean, std = compute_mean_std(dataloader)
    print(f"Mean = {mean}, Std = {std}")



    # products_dir = os.path.join(data_dir, 'products')
    # os.makedirs(products_dir, exist_ok=True)

    # to_pil = transforms.ToPILImage()
    # for image, label in dataset:
    #     image_pil = to_pil(image)
    #     label_dir = os.path.join(products_dir, str(label))
    #     os.makedirs(label_dir, exist_ok=True)
    #     filename = f"{label}_{len(os.listdir(label_dir))}.jpg"
    #     save_path = os.path.join(label_dir, filename)
    #     image_pil.save(save_path)

    # print("Cropped images saved successfully.")


    # the class of entery 1164 in camera_2/metadata changed from 68 to 32