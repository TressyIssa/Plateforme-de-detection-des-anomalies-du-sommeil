from flask import Flask, render_template, request, jsonify
import joblib
import os

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du modèle ML
MODEL_PATH = "models/XGBoost_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Le fichier de modèle {MODEL_PATH} est introuvable.")

# Route principale
@app.route('/')
def index():
    return render_template('predict_form.html')

# Route pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupération des données du formulaire
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        sleep_hours = float(request.form['sleep_hours'])
        stress_level = float(request.form['stress_level'])

        # Préparation des données pour la prédiction
        input_data = [[age, bmi, sleep_hours, stress_level]]

        # Prédiction
        prediction = model.predict(input_data)
        prediction_label = {0: "No Disorder", 1: "Sleep Apnea", 2: "Insomnia"}
        result = prediction_label[prediction[0]]

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Lancer l'application Flask
    app.run(debug=True)
