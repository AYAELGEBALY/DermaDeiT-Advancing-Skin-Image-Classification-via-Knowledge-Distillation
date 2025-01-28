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
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

def train_and_evaluate_model(model, train_loader, val_loader, device, epochs=30, results_file="results.txt"):
    model = model.to(device)
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)  
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
