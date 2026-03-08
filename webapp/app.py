from os import name

from flask import Flask, render_template, request
import librosa
import numpy as np
import joblib
import os

app = Flask(__name__)

# load trained model

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
model = joblib.load(model_path)

# home page

@app.route("/")
def home():
    return render_template("index.html")

# prediction route

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"
    try:
        file = request.files["file"]
        file_path = os.path.join(os.path.dirname(__file__), "..", "uploads", "uploaded_song.wav")
        file.save(file_path)
    except Exception as e:
        return f"Error saving file: {e}"

    # load audio
    audio, sample_rate = librosa.load(file_path, duration=5)
    # extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=10)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    # reshape for model
    mfcc_scaled = mfcc_scaled.reshape(1, -1)

    # predict genre
    prediction = model.predict(mfcc_scaled)

    return render_template("index.html", genre=prediction[0])

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
