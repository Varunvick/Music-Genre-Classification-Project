from flask import Flask, render_template, request
import librosa
import numpy as np
import joblib
import os

app = Flask(__name__)

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# Temporary upload folder (required for Render)
UPLOAD_FOLDER = "/tmp"

print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)

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
    try:
        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]

        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, "uploaded_song.wav")
        file.save(file_path)

        # Load audio
        audio, sample_rate = librosa.load(file_path, duration=5)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_scaled = mfcc_scaled.reshape(1, -1)

        # Predict
        prediction = model.predict(mfcc_scaled)

        return render_template("index.html", genre=prediction[0])

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)