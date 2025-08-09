import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Muat preprocessor, model, dan label encoder
try:
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('best_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    st.error("Model, preprocessor, atau label encoder tidak ditemukan. Pastikan file 'preprocessor.joblib', 'best_model.joblib', dan 'label_encoder.joblib' ada di direktori yang sama.")
    st.stop() # Hentikan eksekusi jika file tidak ditemukan

st.set_page_config(page_title="Prediksi Pemesanan Makanan Online", layout="centered")

st.markdown(
    """
    <style>
    .main-header {
        font-size: 3em;
        color: #2F80ED;
        text-align: center;
        margin-bottom: 0.5em;
        font-weight: bold;
    }
    .subheader {
        font-size: 1.5em;
        color: #555;
        text-align: center;
        margin-bottom: 1.5em;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: bold;
        color: #333;
    }
    .stButton>button {
        background-color: #2F80ED;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1a6ac9;
    }
    .prediction-result {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        margin-top: 1.5em;
        padding: 1em;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .positive {
        color: #28a745;
        background-color: #e6ffe6;
    }
    .negative {
        color: #dc3545;
        background-color: #ffe6e6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-header">Prediksi Pemesanan Makanan Online 🍜</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Aplikasi ini memprediksi apakah pelanggan akan memesan makanan secara online.</p>', unsafe_allow_html=True)

# Input pengguna
st.sidebar.header("Input Data Pelanggan")

age = st.sidebar.slider("Usia", min_value=18, max_value=60, value=25)
gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female", "Prefer not to say"])
marital_status = st.sidebar.selectbox("Status Pernikahan", ["Single", "Married", "Prefer not to say"])
occupation = st.sidebar.selectbox("Pekerjaan", ["Student", "Employee", "Self Employeed", "House wife"])

# Mapping untuk Monthly Income (harus sama dengan preprocessing)
monthly_income_options = ["No Income", "Below Rs.10000", "10001 to 25000", "25001 to 50000", "More than 50000"]
monthly_income = st.sidebar.selectbox("Pendapatan Bulanan", monthly_income_options)
monthly_income_mapped = {
    "No Income": 0, "Below Rs.10000": 1, "10001 to 25000": 2,
    "25001 to 50000": 3, "More than 50000": 4
}[monthly_income]

# Mapping untuk Educational Qualifications (harus sama dengan preprocessing)
edu_qual_options = ["School", "Graduate", "Post Graduate", "Ph.D"]
educational_qualifications = st.sidebar.selectbox("Kualifikasi Pendidikan", edu_qual_options)
edu_qual_mapped = {
    "School": 0, "Graduate": 1, "Post Graduate": 2, "Ph.D": 3
}[educational_qualifications]

family_size = st.sidebar.number_input("Ukuran Keluarga", min_value=1, max_value=10, value=3)
latitude = st.sidebar.number_input("Latitude (contoh: 12.97)", value=12.97, format="%.4f")
longitude = st.sidebar.number_input("Longitude (contoh: 77.59)", value=77.59, format="%.4f")
pin_code = st.sidebar.text_input("Kode Pos (misal: 560001)", value="560001") # Input sebagai teks

feedback = st.sidebar.selectbox("Umpan Balik Sebelumnya", ["Positive", "Negative "]) # Perhatikan spasi di "Negative "

# Buat DataFrame dari input pengguna
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income_mapped, # Gunakan nilai yang sudah di-map
    'Educational Qualifications': edu_qual_mapped, # Gunakan nilai yang sudah di-map
    'Family size': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Pin code': str(pin_code), # Pastikan ini string
    'Feedback': feedback
}])

# Lakukan preprocessing pada input pengguna
# Catatan: Kolom harus dalam urutan yang sama seperti saat pelatihan
# ColumnTransformer akan menangani urutan dan one-hot encoding
try:
    processed_input = preprocessor.transform(input_data)
except Exception as e:
    st.error(f"Error saat memproses input: {e}")
    st.stop()


# Tombol untuk prediksi
if st.button("Prediksi"):
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input) # Untuk probabilitas

    # Mengembalikan prediksi ke label asli ('Yes'/'No')
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    st.write("---")
    if predicted_label == 'Yes':
        st.markdown(f'<div class="prediction-result positive">Pelanggan Cenderung Akan Memesan Makanan Online! 🎉</div>', unsafe_allow_html=True)
        st.write(f"Probabilitas Memesan: **{prediction_proba[0][1]*100:.2f}%**")
    else:
        st.markdown(f'<div class="prediction-result negative">Pelanggan Cenderung Tidak Akan Memesan Makanan Online. 😔</div>', unsafe_allow_html=True)
        st.write(f"Probabilitas Tidak Memesan: **{prediction_proba[0][0]*100:.2f}%**")

st.markdown("---")
st.markdown("Aplikasi dibuat oleh **[Nama Anda/your-username]**") # Ganti dengan nama Anda
