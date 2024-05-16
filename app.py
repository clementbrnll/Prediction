from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

# Charger le modèle sauvegardé
model = joblib.load('new_bike_availability_predictor.pkl')

# Charger les données depuis le fichier JSON
with open('response_001.json') as f:
    data = json.load(f)

# Extraire les données pertinentes
timestamps = data['index']
available_bikes = data['values']

# Créer un DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'available_bikes': available_bikes
})

# Convertir les timestamps en datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Ajouter des colonnes pour l'heure et le jour de la semaine
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Mapper les jours de la semaine en noms
days_mapping = {0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi', 4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'}
df['day_of_week'] = df['day_of_week'].map(days_mapping)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        hour = int(request.args.get('hour'))
        day_of_week = request.args.get('day_of_week')
        day_of_week_num = {v: k for k, v in days_mapping.items()}[day_of_week]

        # Préparer les données pour la prédiction
        prediction_data = pd.DataFrame({
            'hour': [hour],
            'day_of_week': [day_of_week_num]
        })

        # Faire des prédictions
        predictions = model.predict(prediction_data)

        # Préparer les données pour la journée entière
        day_data = df[df['day_of_week'] == day_of_week].groupby('hour').agg({
            'available_bikes': 'mean'
        }).reset_index()

        # Convertir les données pour la réponse JSON
        day_data_json = day_data.to_dict(orient='records')

        return jsonify({
            'hour': hour,
            'day_of_week': day_of_week,
            'predicted_available_bikes': predictions[0],
            'day_data': day_data_json
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
    