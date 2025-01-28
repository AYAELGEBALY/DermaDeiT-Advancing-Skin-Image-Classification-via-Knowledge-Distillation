from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np

data_files=[
    "Dino_3C/train_3Cls_embeddings_0.npy",
    "Dino_3C/train_3Cls_embeddings_1.npy",
    "Dino_3C/train_3Cls_embeddings_2.npy",
    "Dino_3C/train_3Cls_embeddings_3.npy",
    "Dino_3C/train_3Cls_embeddings_4.npy",
    "Dino_3C/train_3Cls_embeddings_5.npy",
    "Dino_3C/train_3Cls_embeddings_6.npy",
    "Dino_3C/train_3Cls_embeddings_7.npy",
    "Dino_3C/train_3Cls_embeddings_8.npy",
    "Dino_3C/train_3Cls_embeddings_9.npy"
]

label_files=[
    "Dino_3C/train_3Cls_labels_0.npy",
    "Dino_3C/train_3Cls_labels_1.npy",
    "Dino_3C/train_3Cls_labels_2.npy",
    "Dino_3C/train_3Cls_labels_3.npy",
    "Dino_3C/train_3Cls_labels_4.npy",
    "Dino_3C/train_3Cls_labels_5.npy",
    "Dino_3C/train_3Cls_labels_6.npy",
    "Dino_3C/train_3Cls_labels_7.npy",
    "Dino_3C/train_3Cls_labels_8.npy",
    "Dino_3C/train_3Cls_labels_9.npy"
]

X_list = [np.load(f) for f in data_files]
y_list = [np.load(f) for f in label_files]

X_train = np.concatenate(X_list, axis=0)
y_train = np.concatenate(y_list, axis=0)

print(f"Combined data shape: {X_train.shape}")
print(f"Combined labels shape: {y_train.shape}")

X_val = np.load("Dino_3C/val_3Cls_embeddings.npy")
y_val = np.load("Dino_3C/val_3Cls_labels.npy")

print(f"Validation data shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")

classifiers = {
    "SVM (rbf Kernel)": SVC(kernel="rbf", probability=True, random_state=42, C=10.0, gamma="scale", class_weight="balanced"),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
}

log_file = "Dino_3C/classification_ml_3c.txt"
with open(log_file, "w") as f:
    f.write("Classifier Metrics:\n")
    f.write("=" * 50 + "\n")

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="weighted")
    recall = recall_score(y_val, y_pred, average="weighted")
    f1 = f1_score(y_val, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    with open(log_file, "a") as f:
        f.write(f"\n{name}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(f"{classification_report(y_val, y_pred)}\n")
        f.write("=" * 50 + "\n")

    print(f"{name}:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
    print("Confusion Matrix:")
    print(cm)

print(f"\nAll results have been logged to {log_file}.")
