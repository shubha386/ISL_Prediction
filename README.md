# Indian Sign Language Gesture Recognition using CNN

This project uses a Convolutional Neural Network (CNN) to recognize Indian Sign Language gestures from images. The model is trained on a dataset of labeled hand gesture images and can make predictions in real-time via webcam or through an API.

## Features

- CNN-based classification of ISL gestures
- Real-time gesture prediction using webcam (local only)
- Flask-based API for deployment on Render
- Dataset with labeled color images of hand gestures

---

## Folder Structure

gesture-recognition/
├── ISL_dataset/ # Folder with subfolders A-Z containing training images
├── app.py # Flask API for inference
├── real_time_prediction.py # Webcam-based prediction (local only)
├── train_cnn.py # Model training script
├── gesture_cnn_model.h5 # Saved CNN model
├── gesture_labels.npy # Encoded gesture labels
├── requirements.txt # Python dependencies
└── README.md

yaml
Copy
Edit

---

## Requirements

- Python 3.8+
- TensorFlow
- OpenCV
- Flask
- NumPy
- Scikit-learn

Install all dependencies:

```bash
pip install -r requirements.txt
Training the Model
bash
Copy
Edit
python train_cnn.py
Ensure ISL_dataset/ is structured like:

css
Copy
Edit
ISL_dataset/
├── A/
│   ├── A1.jpg
│   ├── A2.jpg
├── B/
│   ├── B1.jpg
...
Real-Time Prediction (Local Only)
bash
Copy
Edit
python real_time_prediction.py
Opens webcam

Draws a box on screen

Place your hand in the box to get predictions

Flask API Deployment (Render Compatible)
Run locally:
bash
Copy
Edit
python app.py
