import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

CLASS_NAMES = {
    0: 'nasi_liwet',
    1: 'panada',
    2: 'rawon',
    3: 'rendang'
}

def load_model_from_path(model_path):
    model = load_model(model_path)
    return model

def predict_image(image, model):
    img = image.resize((224, 224))
    
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 

    predictions = model.predict(img_array)
    
    # Dapatkan kelas dengan probabilitas tertinggi
    predicted_index = np.argmax(predictions[0])
    
    # Dapatkan nama kelas langsung dari dictionary yang sudah kita buat
    predicted_class_name = CLASS_NAMES[predicted_index]
    
    # Dapatkan confidence score
    confidence = predictions[0][predicted_index] * 100
    
    return predicted_class_name, confidence
    
st.title('Klasifikasi Makanan Daerah')

st.header('Tolong kirim foto makanan kalian (rawon, nasi liwet, rendang, panada)')

st.file_uploader('', type=['jpeg', 'jpg', 'png'])

model = load_model_from_path('model/BestModel_MobileNet_GWS DL.h5')

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang di-upload', use_column_width=True)
    
    # Tombol untuk prediksi
    if st.button('Prediksi Gambar'):
        with st.spinner('Model sedang menganalisis...'):
            # Kita tidak perlu lagi mengirim class_names ke fungsi prediksi
            predicted_class, confidence = predict_image(image, model)
            st.success(f"Prediksi: **{predicted_class}**")
            st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")

