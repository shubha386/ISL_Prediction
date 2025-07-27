import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Parameters
IMG_SIZE = 64
EPOCHS = 10
DATASET_DIR = "ISL_dataset"

# Load and preprocess images
data, labels = [], []
for label in os.listdir(DATASET_DIR):
    folder = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(folder): continue
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(label)

X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
le = LabelEncoder()
y = to_categorical(le.fit_transform(labels))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build CNN model
model = Sequential([
    input_shape=(IMG_SIZE, IMG_SIZE, 1)
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS)

# Save model and labels
model.save("gesture_cnn_model.h5")
np.save("gesture_labels.npy", le.classes_)

# Evaluate and print results
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

# Classification report
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred_labels, target_names=le.classes_))
