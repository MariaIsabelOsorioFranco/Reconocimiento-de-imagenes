import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
import tensorflow as tf
import platform

st.set_page_config(
    page_title="Reconocimiento Mágico",
    page_icon="🔮",
    layout="centered"
)

st.markdown("""
<style>
.stApp { 
    background-color: #f3e5f5; 
    color: #4a148c !important; 
}

.stApp p, .stApp span, .stApp label, .stApp li, .stApp div {
    color: #4a148c !important;
}

section[data-testid="stSidebar"] { 
    background-color: #e1bee7 !important; 
}
section[data-testid="stSidebar"] * {
    color: #4a148c !important;
}

h1, h2, h3, h4, h5, h6 { 
    color: #7b1fa2 !important; 
}

div.stButton > button {
    background-color: #8e24aa !important; 
    color: white !important;
    border-radius: 12px;
    padding: 10px 24px;
    border: none;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s ease;
}
div.stButton > button * {
    color: white !important;
}
div.stButton > button:hover {
    background-color: #e1bee7 !important; 
    color: #4a148c !important;
}
</style>
""", unsafe_allow_html=True)

st.write("🌌 Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("🔮 Reconocimiento de Imágenes")

try:
    image = Image.open('gatito.jpg')
    st.image(image, width=350)
except:
    st.info("👾 Imagen 'gatito.jpg' no encontrada.")

with st.sidebar:
    st.subheader("🤖 Clasificador Teachable Machine")
    st.write("Sube tu modelo entrenado e identifica objetos en tiempo real usando tu cámara.")

img_file_buffer = st.camera_input("📸 Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    print(prediction)
    
    if prediction[0][0] > 0.5:
        st.header('🐭 Hi Mouse, con Probabilidad: ' + str(round(prediction[0][0], 3)))
    if prediction[0][1] > 0.5:
        st.header('🙀 No Mouse :c, con Probabilidad: ' + str(round(prediction[0][1], 3)))
