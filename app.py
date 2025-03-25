from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
import os

app = Flask(__name__)

# Google Drive file ID (Extract from your Drive link)
FILE_ID = "1dtxCw1PCIQcbfkb2rgd6vr4Zpv-Y4Qv5"
MODEL_PATH = "model.h5"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):  # Download only if the file is not present
        print("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# Download model
download_model()

# Load the trained model
print("Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded Successfully!")

# Route to serve the frontend UI
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint to handle image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected!"})

    # Read image and preprocess
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict using model
    predictions = model.predict(image)
    skin_tone_labels = ["light", "mid-light", "mid-dark", "dark"]  # Modify based on your dataset
    predicted_skin_tone = skin_tone_labels[np.argmax(predictions)]

    return jsonify({"skin_tone": predicted_skin_tone})

if __name__ == "__main__":
    # Ensure it uses the PORT environment variable when deployed on Render
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
