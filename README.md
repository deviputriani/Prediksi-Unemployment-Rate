Prediksi Tingkat Pengangguran Menggunakan Data Ekonomi Makro

Aplikasi ini dibangun untuk melakukan prediksi tingkat pengangguran (Unemployment Rate) berdasarkan indikator ekonomi makro.
Model yang digunakan memadukan pendekatan machine learning (LightGBM) dan metode statistik klasik (ETS).

Fitur Utama
1. Upload dataset CSV ekonomi makro
2. EDA: histogram & heatmap korelasi
3. Preprocessing lengkap:
   cleaning
   missing value
   outlier (IQR)
   normalisasi
   lag-1
   train–validation berdasarkan tahun
4. Prediksi 2025–2030 dengan:
   LightGBM
   Exponential Smoothing (ETS)
5. Tabel & grafik hasil prediksi

Cara Menjalankan (Local)
1. Install dependencies
   pip install -r requirements.txt
2. Jalankan aplikasi
   streamlit run app.py