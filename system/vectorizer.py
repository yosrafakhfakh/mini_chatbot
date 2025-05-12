import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from pretraitement import preprocess

# === Variables globales ===
tfidf_vectorizers = {}
word2vec_models = {}

def build_tfidf_vectorizer(corpus):
    """
    Prend une liste de textes (corpus) et retourne un vecteur TF-IDF et la matrice transformée.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def train_vectorizer_and_w2v(questions, lang):
    """
    Entraîne les modèles TF-IDF et Word2Vec pour une langue donnée à partir de la liste de questions.
    """
    # Prétraitement
    cleaned_questions = [preprocess(q, lang) for q in questions]
    tokenized_questions = [q.split() for q in cleaned_questions]

    # TF-IDF
    tfidf = TfidfVectorizer()
    tfidf.fit(cleaned_questions)
    tfidf_vectorizers[lang] = tfidf

    # Word2Vec local (léger)
    w2v = Word2Vec(sentences=tokenized_questions, vector_size=50, window=5, min_count=1)
    word2vec_models[lang] = w2v

def create_embeddings(questions, lang):
    """
    Crée des embeddings pour une liste de questions en utilisant les modèles TF-IDF et Word2Vec pour la langue spécifiée.
    """
    vectorizer = tfidf_vectorizers.get(lang)
    w2v_model = word2vec_models.get(lang)

    if not vectorizer or not w2v_model:
        raise ValueError(f"Vectorizer or Word2Vec model not trained for language: {lang}")

    cleaned_questions = [preprocess(q, lang) for q in questions]
    embeddings = []

    for question in cleaned_questions:
        words = question.split()
        vector = np.zeros(w2v_model.vector_size)
        weights_sum = 0

        for word in words:
            if word in vectorizer.vocabulary_ and word in w2v_model.wv:
                tfidf_weight = vectorizer.idf_[vectorizer.vocabulary_[word]]
                vector += w2v_model.wv[word] * tfidf_weight
                weights_sum += tfidf_weight

        if weights_sum > 0:
            vector /= weights_sum

        embeddings.append(vector)

    return embeddings

def get_average_vector(words, model):
    """
    Calcule le vecteur moyen des mots fournis dans la liste 'words' en utilisant le modèle Word2Vec.
    """
    # On extrait les vecteurs de chaque mot
    vectors = [model.wv[word] for word in words if word in model.wv]
    
    # Si aucun mot n'a été trouvé dans le modèle, retourner un vecteur nul
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    
    # Calculer la moyenne des vecteurs
    return np.mean(vectors, axis=0)
