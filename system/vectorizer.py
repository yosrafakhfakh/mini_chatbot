import os
import urllib.request
import gzip
import shutil
import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pretraitement import preprocess

# === Téléchargement automatique du modèle fastText si nécessaire ===
MODEL_PATH = 'cc.fr.300.bin'
MODEL_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz'

if not os.path.isfile(MODEL_PATH):
    print("🔽 Téléchargement du modèle fastText cc.fr.300.bin...")
    gz_path = MODEL_PATH + '.gz'
    urllib.request.urlretrieve(MODEL_URL, gz_path)
    print("✅ Téléchargement terminé. Décompression...")

    with gzip.open(gz_path, 'rb') as f_in:
        with open(MODEL_PATH, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)
    print("📦 Modèle fastText prêt.")

# Charger le modèle FastText
fasttext_model = fasttext.load_model(MODEL_PATH)

# Variables globales pour les TF-IDF
tfidf_vectorizers = {}

def build_tfidf_vectorizer(questions, lang):
    cleaned_questions = [preprocess(q, lang) for q in questions]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(cleaned_questions)
    tfidf_vectorizers[lang] = vectorizer

def create_embeddings(questions, lang):
    vectorizer = tfidf_vectorizers.get(lang)
    if not vectorizer:
        raise ValueError(f"TF-IDF vectorizer not built for language: {lang}")
    
    cleaned_questions = [preprocess(q, lang) for q in questions]
    embeddings = []

    for question in cleaned_questions:
        words = question.split()
        vector = np.zeros(fasttext_model.get_dimension())
        weights_sum = 0

        for word in words:
            if word in vectorizer.vocabulary_:
                tfidf_weight = vectorizer.idf_[vectorizer.vocabulary_[word]]
                
                # Vérification de la présence du mot dans le vocabulaire FastText
                if word in fasttext_model:
                    word_vector = fasttext_model.get_word_vector(word)
                    vector += word_vector * tfidf_weight
                    weights_sum += tfidf_weight

        if weights_sum > 0:
            vector /= weights_sum

        embeddings.append(vector)

    return embeddings
