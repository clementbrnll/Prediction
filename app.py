from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model
model = joblib.load('new_bike_availability_predictor.pkl')

# Load data from JSON files
def load_data(files):
    dfs = []
    for file in files:
        with open(file) as f:
            data = json.load(f)
            timestamps = data['index']
            available_bikes = data['values']
            entity_id = data['entityId']
            df = pd.DataFrame({
                'timestamp': timestamps,
                'available_bikes': available_bikes,
                'entity_id': entity_id
            })
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# List of JSON files
json_files = ['response_001.json', 'response_002.json', 'response_003.json']
df = load_data(json_files)

# Convert timestamps to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Add columns for hour and day of the week
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Map days of the week to names
days_mapping = {0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi', 4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'}
df['day_of_week'] = df['day_of_week'].map(days_mapping)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        entity_id = request.args.get('entityId')
        hour = int(request.args.get('hour'))
        day_of_week = request.args.get('day_of_week')
        day_of_week_num = {v: k for k, v in days_mapping.items()}[day_of_week]

        if entity_id is None:
            return jsonify({'error': 'entityId is required'}), 400

        # Filter data for the specified station
        station_data = df[df['entity_id'] == entity_id]

        if station_data.empty:
            return jsonify({'error': f'No data found for entity_id: {entity_id}'}), 400

        # Prepare data for prediction
        prediction_data = pd.DataFrame({
            'hour': [hour],
            'day_of_week': [day_of_week_num],
            'entity_id': [station_data['entity_id'].astype('category').cat.codes.iloc[0]]
        })

        # Make predictions
        predictions = model.predict(prediction_data)

        # Prepare data for the entire day
        day_data = station_data[station_data['day_of_week'] == day_of_week].groupby('hour').agg({
            'available_bikes': 'mean'
        }).reset_index()

        if day_data.empty:
            return jsonify({'error': f'No daily data found for entity_id: {entity_id}, day_of_week: {day_of_week}'}), 400

        # Convert day data to JSON format
        day_data_json = day_data.to_dict(orient='records')

        return jsonify({
            'entityId': entity_id,
            'hour': hour,
            'day_of_week': day_of_week,
            'predicted_available_bikes': predictions[0],
            'day_data': day_data_json
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
