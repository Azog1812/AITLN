import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# Collecte des données textuelles
text = "Votre texte à nettoyer"

# Nettoyage des données
# Convertir le texte en minuscules
text = text.lower()

# Supprimer les caractères spéciaux et les chiffres
text = text.translate(str.maketrans('', '', string.punctuation))
text = text.translate(str.maketrans('', '', string.digits))

# Tokeniser le texte en mots
tokens = word_tokenize(text)

# Supprimer les mots vides (stop words)
stop_words = set(stopwords.words('french'))
filtered_tokens = [token for token in tokens if token not in stop_words]

# Lemmatisation des mots
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

# Afficher les résultats
print(lemmatized_tokens)
