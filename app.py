from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import base64
from flask_cors import CORS

app = Flask(__name__)

# Load model and labels
model = load_model("gesture_cnn_model.h5")
labels = np.load("gesture_labels.npy")
IMG_SIZE = 64

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3) / 255.0

        prediction = model.predict(img)
        label = labels[np.argmax(prediction)]
        return jsonify({"prediction": str(label)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
