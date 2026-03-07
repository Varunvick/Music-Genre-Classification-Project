import librosa
import numpy as np

# audio file path
file_path = "dataset/test_song.wav"

# load audio file
audio, sample_rate = librosa.load(file_path)

# extract MFCC features
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

# take mean of features
mfccs_scaled = np.mean(mfccs.T, axis=0)

print("MFCC Features:")
print(mfccs_scaled)