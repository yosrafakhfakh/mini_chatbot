import string
import re
import nltk
import unidecode
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords')

def preprocess(text, lang='fr'):
    # 1. Mise en minuscules
    text = text.lower()
    
    # 2. Suppression des accents (é, è, à, etc.)
    text = unidecode.unidecode(text)
    
    # 3. Suppression des élisions (l', d', j', qu', etc.)
    text = re.sub(r"\b[ldjtmcqs]['’]", "", text, flags=re.IGNORECASE)
    
    # 4. Suppression de la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 5. Tokenisation simple
    tokens = text.split()
    
    # 6. Suppression des stopwords
    stop_words = stopwords.words('english' if lang == 'en' else 'french')
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # 7. Stemming
    stemmer = SnowballStemmer('english' if lang == 'en' else 'french')
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)
