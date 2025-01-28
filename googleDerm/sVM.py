import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

train_embeddings = np.load("derm_train_embeddings.npy")
train_labels = np.load("derm_train_labels.npy")
val_embeddings = np.load("derm_val_embeddings.npy")
val_labels = np.load("derm_val_labels.npy")

scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
val_embeddings = scaler.transform(val_embeddings)


models = [
    {"C": 1.0, "gamma": "scale"},
    {"C": 0.1, "gamma": "scale"},
    {"C": 10.0, "gamma": "scale"},
    {"C": 100.0, "gamma": "scale"}
]
log_file = "svm_rbf_results.txt"
with open(log_file, "w") as f:
    f.write("SVM RBF Kernel Results:\n")
    f.write("=" * 60 + "\n")
for idx, params in enumerate(models):
    print(f"\nTraining Model {idx + 1} with params: {params}")
    model = SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], class_weight="balanced", random_state=42)
    model.fit(train_embeddings, train_labels) 
    val_preds = model.predict(val_embeddings)
    accuracy = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds, average="binary")
    recall = recall_score(val_labels, val_preds, average="binary")
    f1 = f1_score(val_labels, val_preds, average="binary")
    kappa = cohen_kappa_score(val_labels, val_preds)
    cm = confusion_matrix(val_labels, val_preds)
    report = classification_report(val_labels, val_preds)

    with open(log_file, "a") as f:
        f.write(f"Model {idx + 1} Parameters: {params}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write("=" * 60 + "\n")
    print(f"Model {idx + 1} Metrics:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
    print("Confusion Matrix:")
    print(cm)
print(f"\nResults logged to {log_file}")
