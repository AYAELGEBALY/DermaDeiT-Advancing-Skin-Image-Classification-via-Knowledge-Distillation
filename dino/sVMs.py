from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,)
import numpy as np
X_train = np.load("fine_tuned_train.npy")
y_train = np.load("fine_tuned_train_labels.npy")
X_val = np.load("fine_tuned_val.npy")
y_val = np.load("fine_tuned_val_labels.npy")
svm_params = [
    {"C": 1.0, "gamma": "scale"},  
    {"C": 0.1, "gamma": "scale"},  
    {"C": 10.0, "gamma": "scale"},  
    {"C": 1.0, "gamma": "auto"},  
    {"C": 1.0, "gamma": 0.01},  
    {"C": 1.0, "gamma": 0.1},  
    {"C": 1.0, "gamma": "scale", "class_weight": "balanced"},  
    {"C": 10.0, "gamma": "scale", "class_weight": "balanced"},  
    {"C": 0.1, "gamma": 0.01},  
    {"C": 10.0, "gamma": 0.1},  
]
log_file = "rbf_svm_comparison_log_FT.txt"
with open(log_file, "w") as f:
    f.write("RBF SVM Model Comparison:\n")
    f.write("=" * 50 + "\n")
for i, params in enumerate(svm_params, 1):
    print(f"\nTraining RBF SVM Model {i} with parameters: {params}")
    svm = SVC(kernel="rbf", probability=True, random_state=42, **params)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="binary")
    recall = recall_score(y_val, y_pred, average="binary")
    f1 = f1_score(y_val, y_pred, average="binary")
    kappa = cohen_kappa_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    with open(log_file, "a") as f:
        f.write(f"RBF SVM Model {i}\n")
        f.write(f"Parameters: {params}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(f"{classification_report(y_val, y_pred)}\n")
        f.write("=" * 50 + "\n")
    print(f"Model {i} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
    print("Confusion Matrix:")
    print(cm)
print(f"\nAll results have been logged to {log_file}.")
