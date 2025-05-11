from langdetect import detect
from .loader import load_dataset
from .vectorizer import create_embeddings, build_tfidf_vectorizer, get_average_vector
from pretraitement import preprocess
import numpy as np

# Charger les données
data = load_dataset()
questions_fr, answers_fr, questions_en, answers_en = [], [], [], []

for category in data['qa_categories'].values():
    for item in category['questions']:
        if 'fr' in item['question']:
            questions_fr.append(item['question']['fr'])
            answers_fr.append(item['reponse']['fr'])
        if 'en' in item['question']:
            questions_en.append(item['question']['en'])
            answers_en.append(item['reponse']['en'])

# Préparer les vecteurs
build_tfidf_vectorizer(questions_fr, 'fr')
embeddings_fr = create_embeddings(questions_fr, 'fr')

build_tfidf_vectorizer(questions_en, 'en')
embeddings_en = create_embeddings(questions_en, 'en')

def format_answer(answer):
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
    try:
        lang = detect(user_input)
        if lang not in ['fr', 'en']:
            lang = 'fr'

        cleaned_input = preprocess(user_input, lang)
        input_vector = get_average_vector(cleaned_input, lang)

        if input_vector is None:
            return "Je n'ai pas pu comprendre votre question. Essayez de reformuler."

        if lang == 'fr':
            sims = [np.dot(input_vector, emb) / (np.linalg.norm(input_vector) * np.linalg.norm(emb) + 1e-8) for emb in embeddings_fr]
            idx = int(np.argmax(sims))
            return format_answer(answers_fr[idx]) if sims[idx] > 0.3 else "Désolé, je n'ai pas compris votre question."
        else:
            sims = [np.dot(input_vector, emb) / (np.linalg.norm(input_vector) * np.linalg.norm(emb) + 1e-8) for emb in embeddings_en]
            idx = int(np.argmax(sims))
            return format_answer(answers_en[idx]) if sims[idx] > 0.3 else "Sorry, I didn’t understand your question."
    except Exception as e:
        print("Erreur get_answer:", e)
        return "Une erreur est survenue."
