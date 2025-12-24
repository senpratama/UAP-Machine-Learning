import tensorflow as tf
from PIL import Image
import numpy as np

def load_and_preprocess_image(uploaded_file):
    # Buka gambar asli untuk ditampilkan di Streamlit
    img = Image.open(uploaded_file).convert('RGB')
    
    # Resize untuk model
    img_resized = img.resize((224, 224))
    
    # Ubah ke array dan normalisasi (Pastikan sama dengan saat training di Colab)
    img_array = np.array(img_resized) / 255.0
    
    # Tambah dimensi batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img