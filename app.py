import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.preprocess import load_and_preprocess_image
from utils.predict import predict_top5

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Vehicle Classification Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI LOAD CSS EKSTERNAL ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

CLASS_NAMES = ['bus', 'car', 'motorcycle', 'train_vehicle', 'truck']
NUM_CLASSES = len(CLASS_NAMES)

# --- FUNGSI BUILD ARSITEKTUR MODEL ---
def build_cnn_base():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def build_mobilenet():
    base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def build_resnet():
    base = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False, weights=None)
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# --- LOAD MODELS ---
@st.cache_resource
def load_all_models():
    cnn = build_cnn_base()
    cnn.load_weights("models/cnn_weights.weights.h5")
    mnet = build_mobilenet()
    mnet.load_weights("models/mobilenet_weights.weights.h5")
    rnet = build_resnet()
    rnet.load_weights("models/resnet_weights.weights.h5")
    return cnn, mnet, rnet

with st.spinner("Memuat model..."):
    cnn_model, mobilenet_model, resnet_model = load_all_models()

# --- SIDEBAR ---
st.sidebar.title("⚙️ Kontrol Panel")
model_choice = st.sidebar.selectbox(
    "Pilih Arsitektur Model:", 
    ("CNN Base (Manual)", "MobileNetV2 (Transfer Learning)", "ResNet50 (Transfer Learning)")
)

active_model = {
    "CNN Base (Manual)": cnn_model,
    "MobileNetV2 (Transfer Learning)": mobilenet_model,
    "ResNet50 (Transfer Learning)": resnet_model
}[model_choice]

st.sidebar.markdown("""
Dashboard Klasifikasi Kendaraan berdasarkan data citra dengan penggunaan 3 model yaitu:
1. **CNN**
2. **MobileNetV2**
3. **ResNet50**

untuk **UAP Machine Learning**.
""")

# --- MAIN UI ---
st.title("🚗 Dashboard Klasifikasi Kendaraan")

tab1, tab2 = st.tabs(["🚀 Prediksi Real-time", "📊 Informasi Evaluasi Model"])

with tab1:
    st.write(f"Model Aktif: **{model_choice}**")
    uploaded_files = st.file_uploader(
        "Unggah gambar kendaraan (Maksimal 5)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        files_to_process = uploaded_files[:5]
        for i, file in enumerate(files_to_process):
            with st.expander(f"Hasil Analisis Gambar {i+1}: {file.name}", expanded=True):
                image_array, original_img = load_and_preprocess_image(file)
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.image(original_img, caption="Gambar Input", use_column_width=True)

                with col2:
                    st.write("### 🔍 Hasil Prediksi Top-5")
                    results = predict_top5(active_model, image_array, CLASS_NAMES)
                    for r in results:
                        st.write(f"**{r['class']}** : {r['confidence']:.2f}%")
                    
                    fig, ax = plt.subplots(figsize=(6, 3))
                    labels = [r["class"] for r in results]
                    scores = [r["confidence"] for r in results]
                    ax.barh(labels[::-1], scores[::-1], color='skyblue')
                    ax.set_xlim(0, 100)
                    st.pyplot(fig)
    else:
        st.info("Silakan unggah gambar pada panel di atas untuk memulai.")

with tab2:
    st.header(f"Evaluasi Model: {model_choice}")
    
    # 1. Classification Report Dinamis
    st.subheader("● Classification Report")
    
    if model_choice == "CNN Base (Manual)":
        report_data = {
            "Class": ["bus", "car", "motorcycle", "train_vehicle", "truck"],
            "Precision": [0.77, 0.74, 0.82, 0.67, 0.58],
            "Recall": [0.82, 0.55, 0.84, 0.81, 0.43],
            "F1-Score": [0.80, 0.63, 0.83, 0.73, 0.49]
        }
        acc_text = "73%"
        cm_path, graph_path = "assets/Confusion_Matrix_CNN.png", "assets/Accuracy_Loss_CNN.png"
    elif model_choice == "MobileNetV2 (Transfer Learning)":
        report_data = {
            "Class": ["bus", "car", "motorcycle", "train_vehicle", "truck"],
            "Precision": [0.92, 0.84, 0.98, 0.95, 0.73],
            "Recall": [0.91, 0.79, 0.99, 0.90, 0.85],
            "F1-Score": [0.92, 0.81, 0.99, 0.92, 0.79]
        }
        acc_text = "90%"
        cm_path, graph_path = "assets/Confusion_Matrix_MobileNetV2.png", "assets/Accuracy_Loss_MobileNetV2.png"
    else: # ResNet50
        report_data = {
            "Class": ["bus", "car", "motorcycle", "train_vehicle", "truck"],
            "Precision": [0.57, 0.00, 0.65, 0.42, 0.50],
            "Recall": [0.48, 0.00, 0.81, 0.79, 0.22],
            "F1-Score": [0.52, 0.00, 0.72, 0.55, 0.31]
        }
        acc_text = "51%"
        cm_path, graph_path = "assets/Confusion_Matrix_ResNet50.png", "assets/Accuracy_Loss_ResNet50.png"

    st.table(report_data)
    st.success(f"**Akurasi Keseluruhan ({model_choice}): {acc_text}**")
    st.markdown("---")

    # 2. Visualisasi Grafik (Safe Check)
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        st.subheader("● Grafik Loss & Accuracy")
        if os.path.exists(graph_path):
            st.image(graph_path, caption=f"Training History {model_choice}", use_column_width=True)
        else:
            st.error(f"⚠️ File tidak ditemukan: {graph_path}")
        
    with col_eval2:
        st.subheader("● Confusion Matrix")
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"Confusion Matrix {model_choice}", use_column_width=True)
        else:
            st.error(f"⚠️ File tidak ditemukan: {cm_path}")

# --- FOOTER ---
st.markdown(
    f"""
    <div class="footer">
        <p>Developed for UAP Machine Learning | Sendy Pratama - Universitas Muhammadiyah Malang</p>
    </div>
    """, 
    unsafe_allow_html=True
)