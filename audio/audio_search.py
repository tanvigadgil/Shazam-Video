import librosa
import numpy as np
from scipy import signal

audio_folder_path = 'Audios/Data'
query_audio_file = 'Audios/Query/video1_1.wav'
sr = 22050

# READING THE AUDIO FREATURES NUMPY FILES THAT IS ALREADY CREATED
af1 = np.load('audio_features/video1.npy') 
af2 = np.load('audio_features/video2.npy') 
af3 = np.load('audio_features/video3.npy') 
af4 = np.load('audio_features/video4.npy') 
af5 = np.load('audio_features/video5.npy') 
af6 = np.load('audio_features/video6.npy') 
af7 = np.load('audio_features/video7.npy') 
af8 = np.load('audio_features/video8.npy') 
af9 = np.load('audio_features/video9.npy') 
af11 = np.load('audio_features/video11.npy') 
af10 = np.load('audio_features/video10.npy') 
af12 = np.load('audio_features/video12.npy') 
af13 = np.load('audio_features/video13.npy') 
af14 = np.load('audio_features/video14.npy') 
af15 = np.load('audio_features/video15.npy') 
af16 = np.load('audio_features/video16.npy') 
af17 = np.load('audio_features/video17.npy') 
af18 = np.load('audio_features/video18.npy') 
af19 = np.load('audio_features/video19.npy') 
af20 = np.load('audio_features/video20.npy') 

# HASHMAP TO SAVE THE VIDEOS
afs = {
    "af1": af1,
    "af2": af2,
    "af3": af3,
    "af4": af4,
    "af5": af5,
    "af6": af6,
    "af7": af7,
    "af8": af8,
    "af9": af9,
    "af10": af10,
    "af11": af11,
    "af12": af12,
    "af13": af13,
    "af14": af14,
    "af15": af15,
    "af16": af16,
    "af17": af17,
    "af18": af18,
    "af19": af19,
    "af20": af20,
}

# Function to extract MFCC features from an audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    # print("Sr: " + str(sr))
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs = librosa.util.fix_length(mfccs, size=25860, axis=1)
    return y

# Function to search for similar audio files in a given folder
def search_similar_audio(query_file):
    print("Extracting features from query audio file...")
    query_features = extract_features(query_file)

    similarities = []
    print("Searching for similarities...")
    for af in afs:
        corr_result = signal.correlate(afs[af], query_features, mode='valid', method='fft')
        similarity = np.max(corr_result)
        timestamp = np.argmax(corr_result) / sr

        print(af + ": " + str(similarity))
        similarities.append((af, similarity, timestamp))

    # Sort results by similarity in descending order
    sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)

    print("Search complete.")
    print("Top results: ")

    # Print or return the top results
    top_results = sorted_results[:3]
    for result in top_results:
        print("File: " + result[0] + " Similarity: " + str(result[1]) + " Query Timestamp: " + str(result[2]))

        minutes = int(result[2] // 60)
        seconds = int(result[2] % 60)
        print(f"{minutes:02} minutes {seconds:02} seconds")

# Search for similar audio files within the "Folder" using a query from "Queries"
search_similar_audio(query_audio_file)
