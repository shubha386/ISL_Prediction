import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 64
MODEL_PATH = "gesture_cnn_model.h5"
LABELS_PATH = "gesture_labels.npy"

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

cap = cv2.VideoCapture(0)

x1, y1 = 100, 100
x2, y2 = x1 + 200, y1 + 200

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if roi.shape[:2] == (200, 200):
        roi_input = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).reshape(1, IMG_SIZE, IMG_SIZE, 3) / 255.0
        pred = model.predict(roi_input)
        label = labels[np.argmax(pred)]

        cv2.putText(frame, f"Prediction: {label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
