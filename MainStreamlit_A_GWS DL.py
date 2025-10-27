import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==============================================
# 1️⃣ Variabel dan Fungsi
# ==============================================

CLASS_NAMES = ['nasi_liwet', 'panada', 'rawon', 'rendang']
IMG_SIZE = (224, 224) 

# Fungsi untuk memuat SELURUH model (menggunakan cache)
@st.cache_resource
def load_full_model(model_path):
    # Langsung muat seluruh model dari file .h5
    model = load_model(model_path)
    return model

# Fungsi untuk prediksi gambar
def predict_image(model, img):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Pastikan preprocessing ini sama dengan saat training

    prediction = model.predict(img_array)
    return prediction

# ==============================================
# 2️⃣ Tampilan Utama Aplikasi Streamlit
# ==============================================
def main():
    st.title("Klasifikasi Makanan: Nasi Liwet, Panada, Rawon, Rendang")
    uploaded_file = st.file_uploader("Upload gambar makanan Anda", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

        # Ganti dengan path ke file .h5 LENGKAP Anda
        model_path = "model/BestModel_AlexNet_GWS_DL.h5" 

        try:
            # Panggil fungsi yang baru
            model = load_full_model(model_path)
            
            if st.button('Prediksi Gambar'):
                with st.spinner('Model sedang menganalisis...'):
                    prediction = predict_image(model, image)
                    
                    st.write("### Hasil Prediksi:")
                    predicted_class = CLASS_NAMES[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100
                    st.success(f"Prediksi: **{predicted_class}**")
                    st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")

        except Exception as e:
            st.error(f"Gagal memuat model atau melakukan prediksi: {e}")

if __name__ == "__main__":
    main()
