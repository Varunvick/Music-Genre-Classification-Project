import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "dataset/genres_original"

features = []
labels = []

for genre in os.listdir(DATASET_PATH):
    genre_path = os.path.join(DATASET_PATH, genre)

    if os.path.isdir(genre_path):

        for file in os.listdir(genre_path):

            if file.endswith(".wav"):

                file_path = os.path.join(genre_path, file)

                try:
                    audio, sample_rate = librosa.load(file_path)

                    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

                    mfcc_scaled = np.mean(mfcc.T, axis=0)

                    features.append(mfcc_scaled)
                    labels.append(genre)

                except Exception as e:
                    print("Error processing:", file_path)

data = pd.DataFrame(features)
data["genre"] = labels

print(data.head())

data.to_csv("dataset/music_features.csv", index=False)

print("Dataset created successfully!")