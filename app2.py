import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import altair as alt

# Charger les données depuis le fichier JSON
with open('response_002.json') as f:
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

# Charger le modèle sauvegardé
model = joblib.load('new_bike_availability_predictor.pkl')

# Interface Streamlit
st.title('Prédiction de Disponibilité des Vélos à la Station 002 à Montpellier')

# Sélectionner une heure et un jour de la semaine
hour = st.slider('Heure', 0, 23, 12)
day_of_week = st.selectbox('Jour de la semaine', ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])

# Mapper les jours de la semaine en nombres
days_mapping = {'Lundi': 0, 'Mardi': 1, 'Mercredi': 2, 'Jeudi': 3, 'Vendredi': 4, 'Samedi': 5, 'Dimanche': 6}
day_of_week_num = days_mapping[day_of_week]

# Préparer les données pour la prédiction
prediction_data = pd.DataFrame({
    'hour': [hour],
    'day_of_week': [day_of_week_num]
})

# Faire des prédictions
predictions = model.predict(prediction_data)

# Afficher les résultats
st.subheader(f"Prédiction pour l'heure {hour} et le jour {day_of_week}")
st.write(f"Vélos disponibles prédits : {predictions[0]:.2f}")

# Préparer les données pour la journée entière
day_data = df[df['day_of_week'] == day_of_week].groupby('hour').agg({
    'available_bikes': 'mean'
}).reset_index()

# Afficher un tableau avec une seule ligne pour l'heure sélectionnée
selected_data = day_data[day_data['hour'] == hour]
st.subheader("Prédiction pour l'heure sélectionnée")
st.dataframe(selected_data.rename(columns={
    'hour': 'Heure',
    'available_bikes': 'Vélos disponibles'
}))

# Afficher un graphique pour la disponibilité de la journée entière
st.subheader("Graphique de la disponibilité des vélos pour la journée entière")
chart = alt.Chart(day_data).mark_line().encode(
    x='hour:O',
    y='available_bikes:Q'
).properties(
    title='Disponibilité des vélos au cours de la journée'
)

st.altair_chart(chart, use_container_width=True)
