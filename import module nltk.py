import nltk
from nltk.tokenize import word_tokenize

# Téléchargez le package 'punkt' si ce n'est pas déjà fait
nltk.download('punkt')

# Exemple de texte à découper en tokens
text = "Bonjour, je m'appelle ChatGPT. Je suis une IA développée par OpenAI."

# Découpez le texte en tokens
tokens = word_tokenize(text)

# Affichez les tokens
print(tokens)


import nltk
from nltk.corpus import stopwords

# Téléchargez les mots vides pour la langue française si ce n'est pas déjà fait
nltk.download('stopwords')

# Créez un ensemble de mots vides pour la langue française
stop_words = set(stopwords.words('french'))

# Affichez les mots vides
print(stop_words)


import nltk
nltk.download('wordnet')
