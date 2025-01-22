from flask import Flask, jsonify
from flask_socketio import SocketIO, send
from flask_cors import CORS
from main import initialize_model, extract_symptoms_from_sentence, predict_disease

# Initialisation de l'application Flask et Flask-SocketIO
import json
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")  # Autoriser CORS ici directement

# Initialisation du modèle et des ressources
file_path = 'Training.csv'
model, symptom_list, label_encoder = initialize_model(file_path)

# Route HTTP de base
@app.route('/')
def index():
    return "Bienvenue sur le serveur Flask!"

# Fonction pour traiter le message reçu via WebSocket
@socketio.on('message')
def handle_message(msg):
    print(f"Message brut reçu : {msg}")

    try:
        # Traiter directement le message comme du texte brut
        input_symptoms = msg.split(",")  # Séparer les symptômes par virgule
        print(f"Symptômes reçus : {input_symptoms}")

        # Traiter la prédiction et l'extraction des symptômes
        extracted_symptoms = extract_symptoms_from_sentence(msg, symptom_list)
        print(f"Symptômes extraits : {extracted_symptoms}")

        if extracted_symptoms:
            try:
                predicted_disease, probabilities = predict_disease(model, label_encoder, symptom_list, extracted_symptoms)
                print(f"Maladie prédite : {predicted_disease}")
                response = {
                    "disease": predicted_disease,
                    "treatment": "Prenez des médicaments contre la grippe et reposez-vous.",
                    "probabilities": probabilities.tolist()  # Conversion en liste pour JSON
                }
            except ValueError as e:
                response = {"error": str(e)}
        else:
            response = {"error": "Aucun symptôme valide identifié."}

        # Envoyer la réponse au frontend
        send(response, broadcast=True)

    except Exception as e:
        print(f"Erreur de traitement du message : {e}")
        send({"error": "Erreur de traitement du message."}, broadcast=True)

@socketio.on('connect')
def handle_connect():
    print("Client connecté")

if __name__ == '__main__':
    socketio.run(app, debug=True)
