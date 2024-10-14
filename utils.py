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

client = MongoClient(MONGO_URI)

def extract_accent_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def add_vector_to_mongodb(accent, vector):
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    collection.insert_one({
        "accent": accent,
        "vector": vector.tolist()
    })

def match_accent(input_audio):
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    input_vector = extract_accent_features(input_audio)
    input_vector = normalize([input_vector])[0]
    all_vectors = list(collection.find({}, {"_id": 0, "accent": 1, "vector": 1}))

    similarities = []
    for doc in all_vectors:
        vec = normalize([doc["vector"]])[0]
        similarity = 1 - cosine(input_vector, vec)
        similarities.append((doc["accent"], similarity))

    if similarities:
        matched_accent, similarity_score = max(similarities, key=lambda x: x[1])
    else:
        matched_accent, similarity_score = None, -1
    return matched_accent, similarity_score

