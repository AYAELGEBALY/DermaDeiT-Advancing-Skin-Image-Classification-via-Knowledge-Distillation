import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import logging

log_file = "NN_training_logs.txt"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(message)s")
logging.info("Starting Training...")

logging.info("Loading embeddings and labels...")
train_embeddings = np.load("derm_train_embeddings.npy")
train_labels = np.load("derm_train_labels.npy")
val_embeddings = np.load("derm_val_embeddings.npy")
val_labels = np.load("derm_val_labels.npy")
logging.info(f"Train embeddings shape: {train_embeddings.shape}")
logging.info(f"Validation embeddings shape: {val_embeddings.shape}")

logging.info("Normalizing embeddings...")
scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
val_embeddings = scaler.transform(val_embeddings)

logging.info("Defining the classifier...")
model = Sequential([
    Dense(256, activation='relu', input_shape=(train_embeddings.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  
])

logging.info("Compiling the model...")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

logging.info("Setting up callbacks...")
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

logging.info("Starting training...")
history = model.fit(
    train_embeddings, train_labels,
    validation_data=(val_embeddings, val_labels),
    epochs=150,
    batch_size=64,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

logging.info("Training completed. Saving training history...")
for key, values in history.history.items():
    logging.info(f"{key}: {values}")

logging.info("Evaluating the model on validation data...")
val_predictions = (model.predict(val_embeddings) > 0.5).astype(int).flatten()

logging.info("Calculating metrics...")
accuracy = accuracy_score(val_labels, val_predictions)
precision = precision_score(val_labels, val_predictions)
recall = recall_score(val_labels, val_predictions)
f1 = f1_score(val_labels, val_predictions)
kappa = cohen_kappa_score(val_labels, val_predictions)
cm = confusion_matrix(val_labels, val_predictions)
report = classification_report(val_labels, val_predictions)

logging.info("Validation Metrics:")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall: {recall:.4f}")
logging.info(f"F1 Score: {f1:.4f}")
logging.info(f"Cohen's Kappa: {kappa:.4f}")
logging.info("Confusion Matrix:")
logging.info(cm)
logging.info("Classification Report:")
logging.info(report)

print("Validation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)

logging.info("Plotting training and validation loss...")
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("loss_plot.png")
plt.show()

logging.info("Training complete. Results saved.")
