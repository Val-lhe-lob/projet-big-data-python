import pandas as pd
import numpy as np

# Bibliothèques pour l'apprentissage automatique
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Lecture du fichier CSV de données d'entraînement
training_file = pd.read_csv('Training.csv')

# Groupement par maladies et extraction des colonnes de symptômes
prognosis = training_file.groupby(training_file['prognosis']).max()

# Extraction des colonnes de symptômes et des maladies
symptom_list = prognosis.columns[:].to_list()  # Supposons que la dernière colonne est "prognosis"
diseases = training_file.iloc[:, -1].to_list()

print(symptom_list)

# Création de la matrice des symptômes
matrice_symptoms = []
for i in range(len(diseases)):
    matrice_symptoms.append(training_file.iloc[i, :-1].to_list())

# Conversion en type entier (binaire 0/1)
converted_matrice = [[int(value) for value in row] for row in matrice_symptoms]
print(len(symptom_list))
# Encodage des maladies (target) en entiers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(diseases)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    converted_matrice, encoded_labels, test_size=0.2, random_state=42
)

# Vérification des données d'entrée
print("Exemples de Features (X_train):", X_train[:5])
print("Exemples de Labels (y_train):", y_train[:5])

# Entraînement d'un modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation des performances du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy sur l'ensemble de test:", accuracy)
print("Rapport de classification:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Exemple de prédiction avec de nouveaux symptômes
# Génération de test_symptoms basée sur symptom_list
example_symptoms = ["itching", "skin_rash"]  # Liste des symptômes observés (remplace par tes propres valeurs)
test_symptoms = [1 if symptom in example_symptoms else 0 for symptom in symptom_list]

# Validation de la dimension de test_symptoms avant la prédiction
if len(test_symptoms) != len(X_train[0]):
    raise ValueError(f"Le modèle attend {len(X_train[0])} caractéristiques, mais test_symptoms en a {len(test_symptoms)}.")

# Prédiction avec les nouveaux symptômes
predicted_label = model.predict([test_symptoms])[0]
predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

# Résultats de la prédiction
print("Symptômes fournis :", example_symptoms)
print("Vecteur de test_symptoms :", test_symptoms)
print("Maladie prédite :", predicted_disease)



def extract_symptoms_from_sentence(sentence, symptom_list):
    words = sentence.lower().split()  # Découpe la phrase en mots
    symptoms = [symptom for symptom in symptom_list if any(word in symptom.lower() for word in words)]
    return symptoms


