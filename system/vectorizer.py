import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pretraitement import preprocess

# Charger le modèle FastText
fasttext_model = fasttext.load_model('cc.fr.300.bin')  # pour fr ou cc.en.300.bin pour en

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
                word_vector = fasttext_model.get_word_vector(word)
                vector += word_vector * tfidf_weight
                weights_sum += tfidf_weight

        if weights_sum > 0:
            vector /= weights_sum

        embeddings.append(vector)

    return embeddings
