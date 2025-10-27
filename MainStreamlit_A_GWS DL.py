import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization

# ==============================================
# 1️⃣ Variabel dan Fungsi
# ==============================================

# DIPERBAIKI: Sesuaikan dengan 4 kelas Anda
CLASS_NAMES = ['nasi_liwet', 'panada', 'rawon', 'rendang']

# Fungsi untuk membuat arsitektur model
def create_model(num_classes=len(CLASS_NAMES)): # Otomatis menggunakan jumlah kelas yang benar
    # Arsitektur ini harus SAMA PERSIS dengan yang ada di notebook Anda
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),

    # Blok Konvolusi 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Blok Konvolusi 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Ratakan (Flatten) output untuk masuk ke bagian klasifikasi
    Flatten(),

    # Lapisan Klasifikasi (Fully Connected Layer)
    Dense(512, activation='relu'),
    Dropout(0.5), # Dropout sangat penting untuk mencegah overfitting pada data kecil

    # Lapisan Output
    Dense(NUM_CLASSES, activation='softmax')
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
    img = img.resize((224, 224))
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
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        # Ganti dengan path ke file bobot Anda
        model_path = "model/BestModel_AlexNet_GWS-DL.weights.h5" 

        try:
            model = load_model_and_weights(model_path)
            
            if st.button('Prediksi Gambar'):
                with st.spinner('Model sedang menganalisis...'):
                    prediction = predict_image(model, image)
                    
                    st.write("### Hasil Prediksi:")
                    predicted_class = CLASS_NAMES[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100
                    st.success(f"Prediksi: *{predicted_class}*")
                    st.info(f"Tingkat Keyakinan: *{confidence:.2f}%*")

        except Exception as e:
            st.error(f"Gagal memuat model: {e}")

if _name_ == "_main_":
    main()
