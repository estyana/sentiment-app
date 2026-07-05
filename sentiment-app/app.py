import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
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

        prediction = model.predict(vector)[0]
        
st.write("Sklearn version:", __import__("sklearn").__version__)
st.write("Model:", model)
st.write("Probability:", getattr(model, "probability", "TIDAK ADA"))
st.write("Has _effective_probability:", hasattr(model, "_effective_probability"))

        # output
        if prediction == 1:
            st.success("😊 Sentimen: POSITIF")
        else:
            st.error("😡 Sentimen: NEGATIF")

        # optional confidence (jika model support)
        if hasattr(model, "decision_function"):
            score = model.decision_function(vector)[0]
            st.write("Confidence Score:", score)

st.divider()

# Visualisasi sederhana
st.subheader("📌 Contoh Distribusi (Dummy Pie Chart)")

labels = ["Negatif", "Positif"]
sizes = [68, 32]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
st.pyplot(fig)
