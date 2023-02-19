import pickle

# Sauvegardez le modèle entraîné en utilisant pickle
with open('modele.pickle', 'wb') as f:
    pickle.dump(modele, f)

import pickle

# Chargez le modèle entraîné en utilisant pickle
with open('modele.pickle', 'rb') as f:
    modele = pickle.load(f)

# Utilisez le modèle pour effectuer des prédictions
prediction = modele.predict(X)
