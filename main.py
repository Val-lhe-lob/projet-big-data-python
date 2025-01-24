import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rapidfuzz import fuzz, process
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

# Téléchargement des mots et caractères à supprimer pour pouvoir extraire les mots de la phrase
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(file_path):
    """
    Charge les données à partir d'un fichier CSV.

    Args:
        file_path (str): Le chemin du fichier CSV.

    Returns:
        pd.DataFrame: Les données chargées.

    Raises:
        FileNotFoundError: Si le fichier n'est pas trouvé.
        pd.errors.EmptyDataError: Si le fichier est vide.
        pd.errors.ParserError: Si le fichier contient des erreurs de parsing.
    """
    try:
        data = pd.read_csv(file_path)
        if data.isnull().values.any():
            print("Attention : des valeurs manquantes ont été détectées. Elles seront remplies avec 0.")
            data.fillna(0, inplace=True)
        return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Erreur lors du chargement du fichier : {e}")
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Le fichier est vide : {e}")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Erreur de parsing du fichier : {e}")

def prepare_data(data):
    """
    Prépare les données pour l'entraînement du modèle.

    Args:
        data (pd.DataFrame): Les données à préparer.

    Returns:
        tuple: Une liste des symptômes, les caractéristiques X, les étiquettes y, et le label encoder.
    """
    symptom_list = data.columns[:-1].to_list()
    diseases = data.iloc[:, -1].tolist()

    for col in symptom_list:
        if not set(data[col].unique()).issubset({0, 1}):
            data[col] = data[col].apply(lambda x: 1 if x > 0 else 0)

    X = data.iloc[:, :-1].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(diseases)

    return symptom_list, X, y, label_encoder

def train_random_forest(X, y):
    """
    Entraîne un modèle Random Forest.

    Args:
        X (np.ndarray): Les caractéristiques.
        y (np.ndarray): Les étiquettes.

    Returns:
        tuple: Le modèle entraîné, les données de test X_test et y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def get_synonyms(word):
    """
    Obtient les synonymes d'un mot à partir de WordNet.

    Args:
        word (str): Le mot pour lequel obtenir les synonymes.

    Returns:
        set: Un ensemble de synonymes.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def preprocess_text(sentence):
    """
    Prétraite une phrase en supprimant la ponctuation et les mots vides, et en lemmatisant les mots restants.

    Args:
        sentence (str): La phrase à prétraiter.

    Returns:
        list: Une liste de tokens prétraités.
    """
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())  # Supprimer la ponctuation et mettre en minuscule
    tokens = sentence.split()
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens

def extract_symptoms_from_sentence(sentence, symptom_list, threshold=80):
    """
    Extrait les symptômes d'une phrase en utilisant le fuzzy matching et la correction orthographique.

    Args:
        sentence (str): La phrase à analyser.
        symptom_list (list): La liste des symptômes possibles.
        threshold (int): Le seuil de similarité pour le fuzzy matching.

    Returns:
        list: Une liste des symptômes identifiés.
    """
    words = preprocess_text(sentence)
    print(f"Preprocessed words: {words}")  # Debugging
    ngrams = [' '.join(words[i:i + n]) for n in range(1, 4) for i in range(len(words) - n + 1)]
    print(f"Ngrams: {ngrams}")  # Debugging
    identified_symptoms = set()
    for term in ngrams:
        # Ajouter les synonymes au terme
        synonyms = get_synonyms(term)
        synonyms.add(term)
        for synonym in synonyms:
            best_match = process.extractOne(synonym, symptom_list, scorer=fuzz.ratio)
            if best_match and best_match[1] >= threshold:
                identified_symptoms.add(best_match[0])
    print(f"Identified symptoms: {identified_symptoms}")  # Debugging
    return list(identified_symptoms)

def predict_disease(model, label_encoder, symptom_list, symptoms):
    """
    Prédit la maladie en fonction des symptômes fournis.

    Args:
        model (RandomForestClassifier): Le modèle entraîné.
        label_encoder (LabelEncoder): Le label encoder pour les maladies.
        symptom_list (list): La liste des symptômes possibles.
        symptoms (list): La liste des symptômes identifiés.

    Returns:
        tuple: La maladie prédite, les probabilités associées et les noms des maladies.

    Raises:
        ValueError: Si aucun symptôme valide n'est identifié.
    """
    test_symptoms = [1 if symptom in symptoms else 0 for symptom in symptom_list]
    if not any(test_symptoms):
        raise ValueError("Aucun symptôme valide identifié. Veuillez vérifier la saisie utilisateur.")

    predicted_proba = model.predict_proba([test_symptoms])[0]
    predicted_label = predicted_proba.argmax()
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]
    disease_probabilities = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(predicted_proba)}
    
    return predicted_disease, predicted_proba, disease_probabilities

def initialize_model(file_path):
    """
    Initialise le modèle en chargeant les données et en entraînant le modèle.

    Args:
        file_path (str): Le chemin du fichier CSV contenant les données.

    Returns:
        tuple: Le modèle entraîné, la liste des symptômes et le label encoder.
    """
    data = load_data(file_path)
    symptom_list, X, y, label_encoder = prepare_data(data)
    model, X_test, y_test = train_random_forest(X, y)
    return model, symptom_list, label_encoder