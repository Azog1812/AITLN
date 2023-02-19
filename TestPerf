# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Charger les données d'entraînement et de test
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Initialiser le vectoriseur et le transformer pour les données d'entraînement
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['label']

# Initialiser le modèle de régression logistique et entraîner sur les données d'entraînement
model = LogisticRegression()
model.fit(X_train, y_train)

# Transformer les données de test en vecteurs
X_test = vectorizer.transform(test_data['text'])
y_test = test_data['label']

# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle en utilisant la précision, le rappel et la F-mesure
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
print("Recall: {:.2f}%".format(recall*100))
print("F1-score: {:.2f}%".format(f1*100))
