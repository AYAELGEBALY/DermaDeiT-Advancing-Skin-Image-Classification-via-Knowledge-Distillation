import os
import cv2
import torch
from torch.utils.data import Dataset

class ParallelDermoscopicDataset(Dataset):
    def __init__(self, directory, transform=None, resize=(224, 224)):
        self.directory = directory
        self.transform = transform
        self.resize = resize
        self.data = []
        self.labels = []
        class_labels = {'nevus': 0, 'others': 1}

        for class_name in os.listdir(directory):
            class_dir = os.path.join(directory, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    self.data.append((img_path, class_labels[class_name]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image at {img_path}")
        image = cv2.resize(image, self.resize)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
