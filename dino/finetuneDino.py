import torch 
import torch.nn as nn
from torchvision import transforms, datasets
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    print("Loaded pretrained DINOv2 ViT-S/14")
    for idx, block in enumerate(dino_model.blocks):
        if idx < 8: 
            for param in block.parameters():
                param.requires_grad = False
    print("Frozen first 8 layers. Fine-tuning the last few layers.")
    dino_model.head = nn.Linear(dino_model.embed_dim, 2)  
    dino_model.to(device)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset_dir = "../train2C/train"
    val_dataset_dir = "../val2C/val"
    train_dataset = datasets.ImageFolder(root=train_dataset_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dataset_dir, transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, dino_model.parameters()), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    num_epochs = 400
    best_val_loss = float("inf")
    patience = 15  
    patience_counter = 0  
    for epoch in range(num_epochs):
        dino_model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = dino_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        dino_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]"):
                images, labels = images.to(device), labels.to(device)
                outputs = dino_model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  
            torch.save(dino_model.state_dict(), "finetuned_dinov2_vits14.pth")
            print("Saved the best fine-tuned model!")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    print("Fine-tuning complete!")
if __name__ == "__main__":
    main()
