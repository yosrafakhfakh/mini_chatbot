import string
import re
import nltk
import unidecode
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Télécharger les stopwords seulement si non présents
try:
    stopwords.words('french')
except LookupError:
    nltk.download('stopwords')

def preprocess(text, lang='fr'):
    if not isinstance(text, str):
        return ''
    
    # 1. Mise en minuscules
    text = text.lower()
    # 2. Suppression des accents
    text = unidecode.unidecode(text)
    # 3. Suppression des élisions courantes
    text = re.sub(r"\b[ldjtmcqs]['’]", "", text, flags=re.IGNORECASE)
    # 4. Suppression de la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 5. Tokenisation
    tokens = text.split()
    # 6. Suppression des stopwords + vérif alpha
    stop_words = stopwords.words('english' if lang == 'en' else 'french')
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # 7. Stemming
    stemmer = SnowballStemmer('english' if lang == 'en' else 'french')
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)
