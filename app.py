from flask import Flask
from flask_socketio import SocketIO, send
from flask_cors import CORS
from main import initialize_model, extract_symptoms_from_sentence, predict_disease, load_severity_data

# Initialisation de l'application Flask et Flask-SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")  # Autoriser CORS

# Initialisation du modèle et des ressources
file_path = 'Training.csv'
model, symptom_list, label_encoder = initialize_model(file_path)
severity_dict = load_severity_data("MasterData/Symptom_severity.csv")  # Charger les niveaux de gravité des symptômes
pending_severity = {}  # Stockage temporaire des symptômes et niveaux de gravité

# Route HTTP de base
@app.route('/')
def index():
    return "Bienvenue sur le serveur Flask!"

# Gestion des messages reçus via WebSocket
@socketio.on('message')
def handle_message(msg):
    global pending_severity

    print(f"Raw message received: {msg}")

    try:
        if msg.startswith("pain-level:"):
            # Mise à jour du niveau de douleur pour un symptôme
            symptom, severity = msg.replace("pain-level:", "").split(":")
            severity = int(severity)

            # Remettre le format du symptôme avec les underscores pour correspondre à la base de données
            for key in pending_severity.keys():
                if key.replace("_", " ") == symptom:
                    pending_severity[key] = severity
                    break

            # Vérifier s'il reste des symptômes sans sévérité définie
            remaining_symptoms = [s for s in pending_severity if pending_severity[s] == -1]
            if remaining_symptoms:
                next_symptom = remaining_symptoms[0].replace("_", " ")  # Supprimer les underscores
                send({"question": f"What is the pain level (1-10) for '{next_symptom}'?"}, broadcast=True)
            else:
                # Prédiction de la maladie une fois toutes les sévérités fournies
                extracted_symptoms = list(pending_severity.keys())
                try:
                    predicted_disease, probabilities = predict_disease(
                        model, label_encoder, symptom_list, extracted_symptoms
                    )
                    response = {
                        "disease": predicted_disease,
                        "treatment": "Take appropriate medications and rest.",
                        "probabilities": probabilities.tolist()
                    }
                    send(response, broadcast=True)
                    pending_severity = {}  # Réinitialiser pour la prochaine interaction
                except ValueError as e:
                    send({"error": str(e)}, broadcast=True)
        else:
            # Extraction des symptômes du message
            extracted_symptoms = extract_symptoms_from_sentence(msg, symptom_list, severity_dict)
            print(f"Extracted symptoms: {extracted_symptoms}")

            if extracted_symptoms:
                # Initialisation des niveaux de douleur avec -1
                pending_severity = {symptom: -1 for symptom in extracted_symptoms.keys()}
                
                # Demander le niveau de douleur pour le premier symptôme
                first_symptom = list(pending_severity.keys())[0].replace("_", " ")  # Supprime les underscores
                send({"question": f"What is the pain level (1-10) for '{first_symptom}'?"}, broadcast=True)
            else:
                send({"error": "I couldn't detect any valid symptoms. Please try again."}, broadcast=True)

    except Exception as e:
        print(f"Error processing message: {e}")
        send({"error": "An error occurred while processing the message."}, broadcast=True)

# Gestion de la connexion WebSocket
@socketio.on('connect')
def handle_connect():
    print("Client connecté")

if __name__ == '__main__':
    socketio.run(app, debug=True)
