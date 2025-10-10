import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Charger le modèle et le scaler
# -----------------------------
model = joblib.load("notebooks/model.pkl")
scaler = joblib.load("notebooks/scaler.pkl")

st.title("Prédiction du risque avec votre modèle")

# -----------------------------
# Inputs utilisateur
# -----------------------------
Insulin = st.number_input("Insulin", value=100.0)
SkinThickness = st.number_input("Skin Thickness", value=20.0)
Pregnancies_capped = st.number_input("Pregnancies (capped)", value=2.0)
Age_capped = st.number_input("Age (capped)", value=30.0)
BloodPressure_capped = st.number_input("Blood Pressure (capped)", value=70.0)
BMI_capped = st.number_input("BMI (capped)", value=25.0)
Glucose = st.number_input("Glucose", value=120.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", value=0.5)

# -----------------------------
# Appliquer log pour Insulin et SkinThickness
# -----------------------------
Insulin_log_capped = np.log1p(Insulin)  # log(1 + x)
SkinThickness_log_capped = np.log1p(SkinThickness)

# -----------------------------
# Préparer les données
# -----------------------------
input_data = np.array([[Insulin_log_capped, SkinThickness_log_capped, Pregnancies_capped,
                        Age_capped, BloodPressure_capped, BMI_capped, Glucose,
                        DiabetesPedigreeFunction]])

# Appliquer la standardisation
input_scaled = scaler.transform(input_data)

# -----------------------------
# Bouton de prédiction
# -----------------------------
if st.button("Prédire le risque"):
    prediction = model.predict(input_scaled)
    if prediction == 1:
        st.error(f"⚠️ Risque de diabète détecté ")
    else:
        st.success(f"✅ Aucun risque détecté ")