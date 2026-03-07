import os
import librosa
import numpy as np
import pandas as pd

dataset_path = "dataset"

features = []
genres = []


for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            audio, sample_rate = librosa.load(file_path)

            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            mfccs_scaled = np.mean(mfccs.T, axis=0)

            features.append(mfccs_scaled)
            genres.append(os.path.basename(root))

data = pd.DataFrame(features)
data["genre"] = genres

print(data.head())