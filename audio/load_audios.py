import os
import librosa
import numpy as np

audio_dir = 'audio/Audios/Data'

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs = librosa.util.fix_length(mfccs, size=25860, axis=1)
    return y

for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        reference_file = os.path.join(audio_dir, filename)
        reference_features = extract_features(reference_file)
        np.save("audio/audio_features/" + filename.split('.')[0] + ".npy", reference_features)
