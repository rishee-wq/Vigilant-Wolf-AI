import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2

# Load dataset
print("Loading dataset...")
df = pd.read_csv('assets/dataset/dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nClasses in dataset:")
print(df['category'].value_counts())

# Encode labels
label_encoder = LabelEncoder()
df['class_encoded'] = label_encoder.fit_transform(df['category'])

# Create features from available data
print("\nPreparing features...")
features_list = []
labels_list = []

for idx, row in df.iterrows():
    try:
        # Extract features from the data
        feature_vector = [
            row['severity_level'],
            row['bounding_box_x'],
            row['bounding_box_y'],
            row['bounding_box_w'],
            row['bounding_box_h'],
            hash(row['filename']) % 1000,  # Filename hash as feature
            hash(row['behavior_description'] if pd.notna(row['behavior_description']) else '') % 1000,
        ]
        
        features_list.append(feature_vector)
        labels_list.append(row['class_encoded'])
    except:
        continue

X = np.array(features_list)
y = np.array(labels_list)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and encoder
print("\nSaving model...")
os.makedirs('models', exist_ok=True)

with open('models/driver_alertness_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✓ Model saved to models/driver_alertness_model.pkl")
print("✓ Label encoder saved to models/label_encoder.pkl")
print("\nModel training complete!")
print(f"Classes: {list(label_encoder.classes_)}")
