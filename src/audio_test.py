import librosa

# path to the audio file
file_path = "dataset/test_song.wav"

#load audio file
audio, sample_rate = librosa.load(file_path)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Audio length:",len(audio))