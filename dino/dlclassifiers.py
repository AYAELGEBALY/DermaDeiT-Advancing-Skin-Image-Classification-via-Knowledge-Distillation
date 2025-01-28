import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class
class MelanomaDataset(Dataset):
    def __init__(self, embeddings_path, labels_path):
        self.embeddings = np.load(embeddings_path)
        self.labels = np.load(labels_path)
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# DataLoader
train_dataset = MelanomaDataset("fine_tuned_train.npy", "fine_tuned_train_labels.npy")
val_dataset = MelanomaDataset("fine_tuned_val.npy", "fine_tuned_val_labels.npy")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# Improved Fully Connected Classifier
class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FullyConnectedClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),  # Increased neurons
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

input_size = train_dataset.embeddings.shape[1] 
num_classes = 2
classifiers = {
    "FullyConnectedClassifier": FullyConnectedClassifier(input_size, num_classes),
}

# Training function with early stopping
def train_model(model, train_loader, val_loader, device, epochs=200, patience=15):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    log_file = "FT_fullyCL_classifiers_log.txt"
    with open(log_file, "a") as f:
        f.write(f"\nModel: {model.__class__.__name__}\n")

    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for embeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]"):
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for embeddings, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]"):
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        # Metrics
        precision = precision_score(all_labels, all_preds, average="binary")
        recall = recall_score(all_labels, all_preds, average="binary")
        f1 = f1_score(all_labels, all_preds, average="binary")
        kappa = cohen_kappa_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        # Early stopping check
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_fullyCL_model.pth")
            print("Saved the best model!")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best epoch was {best_epoch} with val_accuracy {best_val_accuracy:.4f}.")
            break

        # Logging
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}: \n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write(f"Accuracy: {val_accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Cohen's Kappa: {kappa:.4f}\n")
            f.write(f"Training Loss: {avg_train_loss:.4f}\n")
            f.write(f"Validation Loss: {avg_val_loss:.4f}\n")
            f.write("=" * 50 + "\n")

        print(f"Epoch {epoch + 1} Metrics:")
        print(f"Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    print(f"Training completed. Best epoch: {best_epoch} with val_accuracy: {best_val_accuracy:.4f}.")

# Main
if __name__ == "__main__":
    for name, model in classifiers.items():
        print(f"\nTraining {name}...")
        train_model(model, train_loader, val_loader, device, epochs=200, patience=15)
