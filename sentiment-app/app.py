import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from utils.preprocessing import preprocess

# Load model
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "svm_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.set_page_config(page_title="Sentiment Analysis Shopee", layout="centered")

st.title("📊 Analisis Sentimen Ulasan Shopee")
st.write("Model: Support Vector Machine + TF-IDF")

# Input user
text = st.text_area("Masukkan ulasan Shopee:")

if st.button("Prediksi Sentimen"):

    if text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        clean_text = preprocess(text)

        vector = vectorizer.transform([clean_text])
        
        st.write("Jumlah fitur vector:", vector.shape)
        st.write("Jumlah fitur model:", model.n_features_in_)

        prediction = model.predict(vector)[0]

        # output
        if prediction == 1:
            st.success("😊 Sentimen: POSITIF")
        else:
            st.error("😡 Sentimen: NEGATIF")
        
        if "prediction" not in st.session_state:
            st.session_state.prediction = None
            
        # optional confidence (jika model support)
        if hasattr(model, "decision_function"):
            score = model.decision_function(vector)[0]
            st.write("Confidence Score:", score)

st.divider()

# Setelah prediction didapat

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.button("Prediksi Sentimen"):

    clean_text = preprocess(text)
    vector = vectorizer.transform([clean_text])

    st.session_state.prediction = model.predict(vector)[0]

if st.session_state.prediction is not None:

    if st.session_state.prediction == 1:
        st.success("😊 POSITIF")
    else:
        st.error("😡 NEGATIF")

st.subheader("📊 Hasil Analisis Sentimen")

fig, ax = plt.subplots(figsize=(4,4))
ax.pie(
    sizes,
    labels=labels,
    autopct='%1.0f%%',
    startangle=90
)

ax.axis('equal')
st.pyplot(fig)
