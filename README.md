# 🚗 Vehicle Classification Dashboard (UAP Machine Learning)

**Nama:** Muhammad Sendy Agung Pratama
**NIM:** 202210370311022
**Kelas:** Machine Learning C

---

## 📚 Deskripsi Project 📚
Proyek ini bertujuan untuk mengembangkan sistem klasifikasi kendaraan otomatis berbasis *Computer Vision*. Dengan memanfaatkan teknologi *Deep Learning*, sistem ini mampu mengenali dan membedakan jenis kendaraan melalui input citra digital. Proyek ini membandingkan performa arsitektur CNN manual dengan model *Transfer Learning* (MobileNetV2 dan ResNet50) untuk mendapatkan hasil prediksi yang paling akurat.

---

## 📊 Sumber Dataset 📊
Dataset berasal dari Kaggle: [5 Vehicles for Multicategory Classification](https://www.kaggle.com/datasets/mrtontrnok/5-vehichles-for-multicategory-classification) dengan total **6.838 file** citra yang mencakup berbagai sudut pandang dan kondisi cahaya.

---

## 🧑‍💻 Preprocessing dan Pemodelan 🧑‍💻

### Pemilihan Atribut & Transformasi Data
| Proses | Detail Transformasi | Deskripsi |
| :--- | :--- | :--- |
| **Resizing** | 150 x 150 piksel | Menyamakan seluruh dimensi gambar input agar konsisten saat masuk ke model. |
| **Rescaling** | 1./255 | Normalisasi nilai pixel menjadi rentang 0-1 untuk mempercepat konvergensi. |
| **Augmentasi** | Horizontal Flip | Menambah variasi data dengan membalik gambar secara horizontal untuk mencegah overfitting. |
| **Data Split** | 80% Train, 20% Val | Pembagian data untuk proses pelatihan dan pengujian validasi performa. |

---

## 🔍 Hasil Analisis Perbandingan Model 🔍
Berikut adalah ringkasan performa dari ketiga model yang diuji sesuai dengan instruksi wajib laporan:

| Nama Model | Akurasi | Hasil Analisis |
| :--- | :--- | :--- |
| **CNN Scratch** | ~73% | Memiliki performa yang cukup stabil namun akurasi masih terbatas karena arsitektur sederhana dalam mengekstraksi fitur kompleks. |
| **MobileNetV2** | **~90%** | **Model Terbaik.** Menunjukkan akurasi tertinggi dan konvergensi yang sangat cepat, sangat efisien untuk klasifikasi kendaraan. |
| **ResNet50** | ~51% | Model mengalami kesulitan belajar (underfitting) pada dataset ini meskipun memiliki arsitektur yang lebih dalam. |

### Visualisasi Performa
* **Kurva Pembelajaran (Accuracy & Loss)**:
  ![MobileNet Plot](assets/Accuracy_Loss_MobileNetV2.png)
* **Confusion Matrix**:
  ![CM MobileNet](assets/Confusion_Matrix_MobileNetV2.png)

---

## 🎓 Sistem Sederhana Streamlit 🎓
Aplikasi web ini dirancang untuk memberikan antarmuka interaktif dalam melakukan klasifikasi kendaraan secara *real-time*.

### Tampilan Dashboard:
1. **Dashboard Utama (Prediksi Real-time)**: Area unggah gambar dan penampilan hasil prediksi otomatis.  
   ![Dashboard Utama](assets/Dashboard_1.png)
2. **Kontrol Panel (Setting Model)**: Pengaturan fungsionalitas dan pemilihan parameter sistem sebelum prediksi.  
   ![Kontrol Panel](assets/Dashboard_3.png) 
3. **Dashboard Evaluasi (Sidebar & Info)**: Menu navigasi pilihan model dan informasi detail mengenai sistem.  
   ![Dashboard Sidebar](assets/Dashboard_2.png)

---

## 🔧 Langkah Instalasi & Penggunaan 🔧

### Software Utama
Proyek ini dapat dijalankan menggunakan **Google Colab** dan **VSCode**. Pastikan **Python 3.10.16** telah terinstal di sistem Anda.

### Dependensi & Menjalankan Sistem
Anda dapat menyiapkan lingkungan kerja dan menjalankan aplikasi dengan mengikuti salah satu cara berikut:

**Cara 1: Instalasi Otomatis & Run**
Jalankan perintah berikut secara berurutan di terminal Anda:
```bash
# Instalasi semua library
pip install -r requirements.txt

# Menjalankan sistem prediksi
streamlit run app.py
