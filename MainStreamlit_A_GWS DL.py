import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# ==============================================
# 1️⃣ Variabel dan Fungsi
# ==============================================

# DIPERBAIKI: Definisikan IMG_SIZE di sini.
# Berdasarkan error Anda sebelumnya, ukurannya harus 224x224.
IMG_SIZE = (224, 224) 
CLASS_NAMES = ['nasi_liwet', 'panada', 'rawon', 'rendang']

# Fungsi untuk membuat arsitektur model
def create_model(num_classes=len(CLASS_NAMES)):
    # Arsitektur ini harus SAMA PERSIS dengan yang ada di notebook Anda
    model = Sequential([
        # DIPERBAIKI: input_shape diatur dengan benar
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    return model

# Fungsi untuk memuat model dan bobotnya
@st.cache_resource
def load_model_and_weights(weight_path):
    model = create_model()
    model.load_weights(weight_path)
    return model

# Fungsi untuk prediksi gambar
def predict_image(model, img):
    # Ukuran ini sekarang cocok dengan input_shape model
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

        # Ganti dengan path ke file bobot Anda (.weights.h5)
        model_path = "model/BestModel_AlexNet_GWS_DL.h5" 

        try:
            model = load_model_and_weights(model_path)
            
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

# DIPERBAIKI: Kesalahan ketik
if __name__ == "__main__":
    main()
