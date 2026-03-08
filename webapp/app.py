from flask import Flask, render_template, request
import librosa
import numpy as np
import joblib
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("MODEL PATH:", MODEL_PATH)

# Load trained model
model = joblib.load(MODEL_PATH)
print("Model loaded successfully")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    try:
        file = request.files["file"]

        file_path = os.path.join(UPLOAD_FOLDER, "uploaded_song.wav")
        file.save(file_path)

    except Exception as e:
        return f"Error saving file: {e}"

    # Load audio
    audio, sample_rate = librosa.load(file_path, duration=5)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=10)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    # Reshape for model
    mfcc_scaled = mfcc_scaled.reshape(1, -1)

    # Predict genre
    prediction = model.predict(mfcc_scaled)

    return render_template("index.html", genre=prediction[0])




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
 
    app.run(host="0.0.0.0", port=port)