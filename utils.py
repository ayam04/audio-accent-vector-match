import os
import librosa
import numpy as np
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def process_audio_files(audio_folder):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Drop existing collection and recreate
    collection.drop()
    collection.create_index([("accent", 1)])  # Simple index on accent field

    for file_name in os.listdir(audio_folder):
        if file_name.endswith(".wav"):
            accent = file_name.split(".")[0]  # Remove .wav extension
            file_path = os.path.join(audio_folder, file_name)
            vector = extract_features(file_path).tolist()

            # Insert document with accent and vector
            collection.insert_one({
                "accent": accent,
                "vector": vector
            })

    print(f"Vectors uploaded to MongoDB: {DB_NAME}.{COLLECTION_NAME}")
    client.close()

def match_accent(input_audio):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    input_vector = extract_features(input_audio)
    input_vector = normalize([input_vector])[0]

    # Fetch all vectors from the database
    all_vectors = list(collection.find({}, {"_id": 0, "accent": 1, "vector": 1}))

    # Calculate cosine similarity
    similarities = []
    for doc in all_vectors:
        vec = normalize([doc["vector"]])[0]
        similarity = 1 - cosine(input_vector, vec)
        similarities.append((doc["accent"], similarity))

    # Find the best match
    if similarities:
        matched_accent, similarity_score = max(similarities, key=lambda x: x[1])
    else:
        matched_accent, similarity_score = None, -1

    client.close()
    return matched_accent, similarity_score

# Usage example:
# process_audio_files('samples')
input_audio_file = 'Test/punjabi.wav'
matched_accent, similarity_score = match_accent(input_audio_file)
print(f"Matched Accent: {matched_accent} with similarity score: {similarity_score:.4f}")