import optuna
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Load the embeddings and labels
X_train = np.load("fine_tuned_train.npy")
y_train = np.load("fine_tuned_train_labels.npy")
X_val = np.load("fine_tuned_val.npy")
y_val = np.load("fine_tuned_val_labels.npy")

# Combine train and validation data for splitting during Optuna search
X = np.concatenate((X_train, X_val), axis=0)
y = np.concatenate((y_train, y_val), axis=0)

# Objective function for Optuna
def objective(trial):
    # Define the hyperparameter search space
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    gamma = trial.suggest_loguniform("gamma", 1e-3, 1e3)
    kernel = trial.suggest_categorical("kernel", ["poly", "rbf", "sigmoid"])
    degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3

    # Train-test split within the objective function
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train the SVM
    svc = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=42)
    svc.fit(X_train_split, y_train_split)

    # Evaluate on the validation set
    y_pred = svc.predict(X_val_split)
    accuracy = accuracy_score(y_val_split, y_pred)

    return accuracy  # Objective is to maximize accuracy

# Running Optuna study
study_name = "svm_hyperparameter_optimization"
storage = f"sqlite:///{study_name}.db"

study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage,
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)

# Best hyperparameters
print("\nBest Hyperparameters:")
print(study.best_params)
print(f"Best Validation Accuracy: {study.best_value:.4f}")

# Train and evaluate with the best hyperparameters
best_params = study.best_params
best_svc = SVC(
    kernel=best_params["kernel"],
    C=best_params["C"],
    gamma=best_params["gamma"],
    degree=best_params["degree"] if best_params["kernel"] == "poly" else 3,
    random_state=42,
)

best_svc.fit(X_train, y_train)

y_val_pred = best_svc.predict(X_val)

# Metrics
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average="binary")
recall = recall_score(y_val, y_val_pred, average="binary")
f1 = f1_score(y_val, y_val_pred, average="binary")

print("\nPerformance on Validation Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# Save the best SVM model
import joblib
joblib.dump(best_svc, "best_svm_model.pkl")
print("\nBest SVM model saved as 'best_svm_model.pkl'.")
