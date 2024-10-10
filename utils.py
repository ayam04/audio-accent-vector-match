import os
import json
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def process_audio_files(audio_folder, output_json):
    data = {}
    for file_name in os.listdir(audio_folder):
        if file_name.endswith(".wav"):
            label = file_name.split("_")[0]
            file_path = os.path.join(audio_folder, file_name)
            vector = extract_features(file_path).tolist()

            if label in data:
                data[label].append(vector)
            else:
                data[label] = [vector]
    
    with open(output_json, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Audio vectors saved to {output_json}")

def match_accent(input_audio, vector_json):
    with open(vector_json, 'r') as f:
        accent_vectors = json.load(f)

    input_vector = extract_features(input_audio)
    input_vector = normalize([input_vector])[0]

    highest_similarity = -1
    matched_accent = None

    for accent, vectors in accent_vectors.items():
        for vector in vectors:
            vector = np.array(vector)
            vector = normalize([vector])[0]
            similarity = 1 - cosine(input_vector, vector)
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                matched_accent = accent

    return matched_accent, highest_similarity

# process_audio_files('samples', 'accent_vectors.json')
input_audio_file = 'Test/punjabi.wav'
matched_accent, similarity_score = match_accent(input_audio_file, 'accent_vectors.json')
print(f"Matched Accent: {matched_accent} with similarity score: {similarity_score:.4f}")
