import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from dataset import ParallelDermoscopicDataset
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from monai.transforms import (
    Compose, Rand2DElasticd, RandRotate90d, RandFlipd, RandAffined,
    RandCoarseShuffled, EnsureTyped, LoadImaged, Resized, ToTensord, NormalizeIntensityd
)
from monai.data import PersistentDataset
from PIL import Image
import torch
from torch import nn, tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score
from tqdm import tqdm
from torchvision import models

if torch.backends.mps.is_available():
    device = torch.device("mps")
    #print("Using MPS (Metal Performance Shaders) on Apple Silicon GPU.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA on device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    device = torch.device("cpu")
    print("Using CPU for computations.")
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss
base_transforms = [
    LoadImaged('image',ensure_channel_first=True),
    Resized('image',(224, 224), mode='bilinear'),
    ToTensord(['image','label']),
    NormalizeIntensityd('image')
]

augmentations = [
    #Rand2DElastic('image',prob=0.5, spacing=100, magnitude_range=(3, 12), padding_mode="zeros"),
    RandRotate90d('image',prob=0.5, spatial_axes=[0, 1]),
    RandFlipd('image',prob=0.5, spatial_axis=0),
    RandFlipd('image',prob=0.5, spatial_axis=1),
    #RandAffine('image',prob=0.5,translate_range=(50, 50),rotate_range=(0.75, 0.75),scale_range=(0.25, 0.25),shear_range=(0.25, 0.25),padding_mode="zeros",),
    #RandCoarseShuffle('image',holes=1, spatial_size=50, max_holes=5, max_spatial_size=150, prob=0.5),
]

train_transform = Compose(base_transforms + augmentations)
val_transform = Compose(base_transforms)

def train_and_evaluate_model(model, train_loader, val_loader, device, epochs=30, results_file="results.txt"):
    model = model.to(device)
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20*len(train_loader), eta_min=1e-6)
    best_f1 = 0
    with open(results_file, "a") as f:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                scheduler.step()
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader):.4f}")
            f.write(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader):.4f}\n")
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_preds)
            cm = confusion_matrix(all_labels, all_preds)
            results = (
                f"Validation Accuracy: {accuracy:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
            )
            print(results)
            print(cm)
            f.write(results + "\n")
            f.write(str(cm) + "\n")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f"{model.__class__.__name__}_best.pth")
        classification_results = classification_report(all_labels, all_preds, target_names=["nevus", "others"])
        print(classification_results)
        f.write(classification_results + "\n")
    return accuracy, precision, recall, f1

def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    train_dir = '../train2C/train'
    val_dir = '../val2C/val'
    train_dataset = ParallelDermoscopicDataset(train_dir, transform=train_transform)
    val_dataset = ParallelDermoscopicDataset(val_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    pretrained_models = [models.mobilenet_v2]
    results = {}
    results_file = "results.txt"
    if os.path.exists(results_file):
        os.remove(results_file)  
    for model_fn in pretrained_models:
        print(f"\nTraining {model_fn.__name__}...")
        with open(results_file, "a") as f:
            f.write(f"\nTraining {model_fn.__name__}...\n")
        model = model_fn(weights="IMAGENET1K_V1")
        if hasattr(model, 'fc'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                num_ftrs = model.classifier[-1].in_features
            else:
                num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 2)
        acc, prec, rec, f1 = train_and_evaluate_model(model, train_loader, val_loader, device, results_file=results_file)
        results[model_fn.__name__] = (acc, prec, rec, f1)
    print("\nModel Comparison Results:")
    with open(results_file, "a") as f:
        f.write("\nModel Comparison Results:\n")
    for model_name, metrics in results.items():
        result_line = (
            f"{model_name}: Accuracy={metrics[0]:.4f}, Precision={metrics[1]:.4f}, "
            f"Recall={metrics[2]:.4f}, F1 Score={metrics[3]:.4f}"
        )
        print(result_line)
        with open(results_file, "a") as f:
            f.write(result_line + "\n")
