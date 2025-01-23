import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rapidfuzz import fuzz, process
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Téléchargement des mots et caractères à supprimer pour pouvoir extraire les mots de la phrase
nltk.download('stopwords')
nltk.download('wordnet')

# Chargement des données
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.isnull().values.any():
            print("Attention : des valeurs manquantes ont été détectées. Elles seront remplies avec 0.")
            data.fillna(0, inplace=True)
        return data
    except Exception as e:
        raise FileNotFoundError(f"Erreur lors du chargement du fichier : {e}")

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

# Entraînement du modèle Random Forest
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Prétraitement du texte
def preprocess_text(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())  # Supprimer la ponctuation et mettre en minuscule
    tokens = sentence.split()
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens

# Extraction des symptômes
def extract_symptoms_from_sentence(sentence, symptom_list, threshold=80):
    words = preprocess_text(sentence)
    ngrams = [' '.join(words[i:i + n]) for n in range(1, 4) for i in range(len(words) - n + 1)]
    identified_symptoms = set()
    for term in ngrams:
        best_match = process.extractOne(term, symptom_list, scorer=fuzz.ratio)
        if best_match and best_match[1] >= threshold:
            identified_symptoms.add(best_match[0])
    return list(identified_symptoms)

# Prédiction de la maladie
def predict_disease(model, label_encoder, symptom_list, symptoms):
    test_symptoms = [1 if symptom in symptoms else 0 for symptom in symptom_list]
    if not any(test_symptoms):
        raise ValueError("Aucun symptôme valide identifié. Veuillez vérifier la saisie utilisateur.")

    predicted_proba = model.predict_proba([test_symptoms])[0]
    predicted_label = predicted_proba.argmax()
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]
    
    return predicted_disease, predicted_proba

# Chargement des données et entraînement du modèle
def initialize_model(file_path):
    data = load_data(file_path)
    symptom_list, X, y, label_encoder = prepare_data(data)
    model, X_test, y_test = train_random_forest(X, y)
    return model, symptom_list, label_encoder