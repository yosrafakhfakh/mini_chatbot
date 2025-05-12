from langdetect import detect
from .loader import load_dataset
from .vectorizer import create_tfidf_embeddings, build_tfidf_vectorizer
from pretraitement import preprocess
import numpy as np

# Charger les données
data = load_dataset()
questions_fr, answers_fr, questions_en, answers_en = [], [], [], []

# Chargement des questions et réponses en français et en anglais
for category in data['qa_categories'].values():
    for item in category['questions']:
        if 'fr' in item['question']:
            questions_fr.append(item['question']['fr'])
            answers_fr.append(item['reponse']['fr'])
        if 'en' in item['question']:
            questions_en.append(item['question']['en'])
            answers_en.append(item['reponse']['en'])

# Préparer les vecteurs TF-IDF
build_tfidf_vectorizer(questions_fr)
embeddings_fr = create_tfidf_embeddings(questions_fr, 'fr')

build_tfidf_vectorizer(questions_en)
embeddings_en = create_tfidf_embeddings(questions_en, 'en')

# Calcul des normes pour éviter de les recalculer
norms_fr = np.linalg.norm(embeddings_fr, axis=1)
norms_en = np.linalg.norm(embeddings_en, axis=1)

def format_answer(answer):
    """
    Formate les réponses pour une meilleure lisibilité HTML.
    """
    if isinstance(answer, str):
        return answer.strip()
    elif isinstance(answer, dict):
        output = ""
        for key, value in answer.items():
            output += f"<strong>{key.capitalize()}:</strong> "
            output += format_answer(value)
            output += "<br>"
        return output.strip()
    elif isinstance(answer, list):
        return "<ul>" + "".join(f"<li>{format_answer(a)}</li>" for a in answer) + "</ul>"
    return str(answer)

def get_answer(user_input):
    """
    Obtient la réponse à la question de l'utilisateur en calculant les similarités avec les questions existantes.
    """
    try:
        # Détection de la langue de la question
        lang = detect(user_input)
        if lang not in ['fr', 'en']:
            lang = 'fr'

        # Prétraitement de la question de l'utilisateur
        cleaned_input = preprocess(user_input, lang)
        # Utilisation du vecteur TF-IDF pour la question de l'utilisateur
        input_vector = create_tfidf_embeddings([cleaned_input], lang)[0]  # Vecteur TF-IDF de l'entrée

        if input_vector is None:
            return "Je n'ai pas pu comprendre votre question. Essayez de reformuler."

        # Calcul des similarités en français
        if lang == 'fr':
            sims = [np.dot(input_vector, emb) / (np.linalg.norm(input_vector) * norm + 1e-8) for emb, norm in zip(embeddings_fr, norms_fr)]
            idx = int(np.argmax(sims))
            return format_answer(answers_fr[idx]) if sims[idx] > 0.3 else "Désolé, je n'ai pas compris votre question."
        # Calcul des similarités en anglais
        else:
            sims = [np.dot(input_vector, emb) / (np.linalg.norm(input_vector) * norm + 1e-8) for emb, norm in zip(embeddings_en, norms_en)]
            idx = int(np.argmax(sims))
            return format_answer(answers_en[idx]) if sims[idx] > 0.3 else "Sorry, I didn’t understand your question."
    except Exception as e:
        print(f"Erreur dans la fonction get_answer : {e}")
        return "Une erreur est survenue. Essayez de reformuler votre question."
