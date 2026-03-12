from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import joblib
import json

IMG_SIZE = 224
BATCH_SIZE = 32

# Image preprocessing
datagen = ImageDataGenerator(rescale=1./255)

data_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Classes:", data_generator.class_indices)

# VGG16 feature extractor
base_model = VGG16(weights="imagenet", include_top=False, pooling="avg")

features = base_model.predict(data_generator)
labels = data_generator.classes

print("Feature shape:", features.shape)

# NMF dimensionality reduction
nmf = NMF(n_components=50, random_state=42)
reduced_features = nmf.fit_transform(features)

# Quantum encoding
quantum_features = np.concatenate([
    np.sin(reduced_features),
    np.cos(reduced_features)
], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    quantum_features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Metrics calculation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("\nModel Performance")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)

# Save models
joblib.dump(model, "quantum_gnb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(nmf, "nmf_model.pkl")

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist()
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("\nModel training completed")
print("Files saved:")
print("quantum_gnb_model.pkl")
print("scaler.pkl")
print("nmf_model.pkl")
print("metrics.json")