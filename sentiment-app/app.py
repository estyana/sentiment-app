import streamlit as st

st.set_page_config(
    page_title="Analisis Sentimen Shopee",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
# 🛒 Analisis Sentimen Ulasan Shopee

Aplikasi berbasis Machine Learning untuk mengklasifikasikan ulasan pelanggan menjadi sentimen positif atau negatif menggunakan algoritma Support Vector Machine (SVM) dan TF-IDF.
""")

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

# Inisialisasi session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.button("Prediksi Sentimen"):

    if text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        clean_text = preprocess(text)

        vector = vectorizer.transform([clean_text])

        st.write("Jumlah fitur vector:", vector.shape)
        st.write("Jumlah fitur model:", model.n_features_in_)

        prediction = model.predict(vector)[0]

        # simpan hasil
        st.session_state.prediction = prediction

        # tampilkan hasil
        if prediction == 1:
            st.success("😊 Sentimen: POSITIF")
        else:
            st.error("😡 Sentimen: NEGATIF")

        # confidence score
        if hasattr(model, "decision_function"):
            score = model.decision_function(vector)[0]
            st.write("Confidence Score:", score)

st.divider()

# Tampilkan pie chart setelah prediksi
if st.session_state.prediction is not None:

    st.subheader("📊 Hasil Analisis Sentimen")

    if st.session_state.prediction == 1:
        labels = ["Positif", "Negatif"]
        sizes = [100, 0]
    else:
        labels = ["Positif", "Negatif"]
        sizes = [0, 100]

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.pie(
        sizes,
        labels=labels,
        autopct='%1.0f%%',
        startangle=90
    )

    ax.axis('equal')

    st.pyplot(fig)
