from os import name

from flask import Flask, render_template, request
import librosa
import numpy as np
import joblib

app = Flask(__name__)

# load trained model

model = joblib.load("models/music_genre_model.pkl")

# home page

@app.route("/")
def home():
    return render_template("index.html")

# prediction route

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    file_path = "uploaded_song.wav"
    file.save(file_path)

    # load audio
    audio, sample_rate = librosa.load(file_path)

    # extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    # reshape for model
    mfcc_scaled = mfcc_scaled.reshape(1, -1)

    # predict genre
    prediction = model.predict(mfcc_scaled)

    return render_template("index.html", genre=prediction[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
