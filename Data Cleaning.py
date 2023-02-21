import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string


# Collecte des données textuelles
text = "Votre texte à nettoyer. test de vectorization."

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


# Segmentez le texte en phrases
sentences = nltk.sent_tokenize(text)

# Prétraitez chaque phrase
for i in range(len(sentences)):
    # Divisez la phrase en mots
    words = nltk.word_tokenize(sentences[i])

    # Prétraitement des mots
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]

    # Remplacez la phrase d'origine par la phrase prétraitée
    sentences[i] = words


word_dict = {}
for sentence in sentences:
    for word in sentence:
        if word not in word_dict:
            word_dict[word] = 0

for sentence in sentences:
    for word in sentence:
        if word in word_dict:
            word_dict[word] += 1

vectors = []
for sentence in sentences:
    vector = []
    for word in word_dict:
        if word in sentence:
            vector.append(sentence.count(word))
        else:
            vector.append(0)
    vectors.append(vector)

print (sentences)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(lowercase=True, stop_words='french')


# Appliquer le CountVectorizer à votre ensemble de données
X = vectorizer.fit_transform(sentences)
