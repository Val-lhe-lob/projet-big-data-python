import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rapidfuzz import fuzz, process
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Télécharger les ressources nécessaires pour le traitement du texte
nltk.download('stopwords')
nltk.download('wordnet')

# Chargement des données d'entraînement
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.isnull().values.any():
            print("Attention : des valeurs manquantes ont été détectées. Elles seront remplies avec 0.")
            data.fillna(0, inplace=True)
        return data
    except Exception as e:
        raise FileNotFoundError(f"Erreur lors du chargement du fichier : {e}")

# Chargement des niveaux de gravité des symptômes
def load_severity_data(file_path):
    try:
        severity_data = pd.read_csv(file_path, header=None)
        severity_dict = dict(zip(severity_data.iloc[:, 0], severity_data.iloc[:, 1]))
        return severity_dict
    except Exception as e:
        raise FileNotFoundError(f"Erreur lors du chargement du fichier Symptom_severity.csv : {e}")

# Fonction pour obtenir la sévérité d'un symptôme
def get_severity(symptom, severity_dict):
    return severity_dict.get(symptom, "Unknown")

# Préparation des données
def prepare_data(data):
    symptom_list = data.columns[:-1].to_list()
    diseases = data.iloc[:, -1].tolist()

    for col in symptom_list:
        if not set(data[col].unique()).issubset({0, 1}):
            data[col] = data[col].apply(lambda x: 1 if x > 0 else 0)

    X = data.iloc[:, :-1].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(diseases)

    return symptom_list, X, y, label_encoder

# Entraînement du modèle
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Prétraitement du texte
def preprocess_text(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())
    tokens = sentence.split()
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens

# Extraction des symptômes depuis une phrase + récupération des niveaux de gravité
def extract_symptoms_from_sentence(sentence, symptom_list, severity_dict, threshold=80):
    words = preprocess_text(sentence)
    ngrams = [' '.join(words[i:i + n]) for n in range(1, 4) for i in range(len(words) - n + 1)]
    identified_symptoms = {}

    for term in ngrams:
        best_match = process.extractOne(term, symptom_list, scorer=fuzz.ratio)
        if best_match and best_match[1] >= threshold:
            symptom = best_match[0]
            severity = get_severity(symptom, severity_dict)
            identified_symptoms[symptom] = severity

    return identified_symptoms

# Prédiction de la maladie
def predict_disease(model, label_encoder, symptom_list, symptoms):
    test_symptoms = [1 if symptom in symptoms else 0 for symptom in symptom_list]
    if not any(test_symptoms):
        raise ValueError("Aucun symptôme valide identifié.")

    predicted_proba = model.predict_proba([test_symptoms])[0]
    predicted_label = predicted_proba.argmax()
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

    return predicted_disease, predicted_proba

# Initialisation du modèle
def initialize_model(file_path):
    data = load_data(file_path)
    symptom_list, X, y, label_encoder = prepare_data(data)
    model = train_random_forest(X, y)
    return model, symptom_list, label_encoder
