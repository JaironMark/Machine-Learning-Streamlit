import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = 'models/knn_model_iris.pkl'
model = joblib.load(MODEL_PATH)

# Interfaz de usuario
st.title("Predicción de Especies de Iris 🌸")
st.write("Introduce las características de la flor:")

# Inputs del usuario
sepal_length = st.slider("Largo del sépalo (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Ancho del sépalo (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Largo del pétalo (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Ancho del pétalo (cm)", 0.1, 2.5, 0.2)

# Botón para hacer la predicción
if st.button("Predecir especie"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species = load_iris().target_names[prediction]

    st.subheader("Predicción:")
    st.write(f"La flor es de la especie: **{species}**")
