from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.model_selection import train_test_split

# Séparez vos vecteurs et vos étiquettes en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2)

# Créez un objet de régression logistique
logreg = LogisticRegression()

# Entraînez le modèle de régression logistique sur l'ensemble d'entraînement
logreg.fit(X_train, y_train)

# Obtenez la précision du modèle sur l'ensemble de test
accuracy = logreg.score(X_test, y_test)
print("Accuracy:", accuracy)



from keras.models import Sequential
from keras.layers import Dense

# Créez un modèle séquentiel
model = Sequential()

# Ajoutez une couche dense avec 64 neurones et une fonction d'activation ReLU
model.add(Dense(64, activation='relu', input_dim=len(word_dict)))

# Ajoutez une couche dense de sortie avec une fonction d'activation sigmoïde
model.add(Dense(1, activation='sigmoid'))

# Compilez le modèle avec une fonction de perte de binary_crossentropy et un optimiseur adam
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînez le modèle sur l'ensemble d'entraînement
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32)
