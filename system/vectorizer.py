import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pretraitement import preprocess

# Variables globales
tfidf_vectorizers = {}

def build_tfidf_vectorizer(corpus):
    """
    Crée un modèle TF-IDF pour un corpus donné et retourne le vecteur TF-IDF et la matrice transformée.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)  # Transforme le corpus en une matrice TF-IDF
    return vectorizer, tfidf_matrix

def train_vectorizer(questions, lang):
    """
    Entraîne le modèle TF-IDF pour une liste de questions dans une langue donnée.
    """
    # Prétraitement des questions
    cleaned_questions = [preprocess(q, lang) for q in questions]

    # Création du modèle TF-IDF
    tfidf = TfidfVectorizer()
    tfidf.fit(cleaned_questions)  # Apprentissage du modèle sur les questions prétraitées
    tfidf_vectorizers[lang] = tfidf

    # Récupération des embeddings TF-IDF pour chaque question
    tfidf_matrix = tfidf.transform(cleaned_questions).toarray()  # Matrice TF-IDF pour chaque question
    
    return tfidf_matrix

def create_tfidf_embeddings(questions, lang):
    """
    Crée des embeddings TF-IDF pour une liste de questions.
    """
    vectorizer = tfidf_vectorizers.get(lang)
    if not vectorizer:
        raise ValueError(f"Vectorizer not trained for language: {lang}")

    # Créer les embeddings TF-IDF pour les questions
    embeddings = train_vectorizer(questions, lang)
    return embeddings
