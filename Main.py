import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Etape 1: Charger et nettoyer les données
data = pd.read_csv('phrases.csv')
clean_data = data.copy()
clean_data['text'] = clean_data['text'].apply(lambda x: clean_text(x))

# Etape 2: Convertir le texte en vecteur
sentences = clean_data['text'].tolist()
vectorizer = TfidfVectorizer(stop_words='french')
X = vectorizer.fit_transform(sentences)

# Etape 3: Entraîner un modèle de machine learning
y = clean_data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Etape 4: Tester le modèle et évaluer ses performances
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Etape 5: Déployer le modèle
def predict_label(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    return prediction

# Exemple d'utilisation
text = "Je suis très satisfait de ce produit, je le recommande vivement."
prediction = predict_label(text)
print('Label:', prediction)
