# 📈 Stock Price Prediction – XGBoost & LightGBM Regression

Project ini merupakan implementasi model regresi **XGBoost** dan **LightGBM** untuk memprediksi harga saham berdasarkan dataset `Stock_Price_Dataset.csv`.  
Model dilatih menggunakan berbagai fitur historis seperti tanggal, harga pembukaan, nama saham, dan tingkat risiko.

---

## 📌 Fitur Utama
- Implementasi **dua model regresi**:
  - XGBoost Regressor  
  - LightGBM Regressor  
- Preprocessing otomatis:
  - Konversi tanggal ke format ordinal  
  - Label Encoding untuk fitur kategorikal  
- Evaluasi lengkap:
  - MAE  
  - MSE  
  - RMSE  
  - MAPE  
  - R² Score  
- Contoh prediksi data aktual
- Code bersih, modular, dan mudah dikembangkan

---

## 📁 Struktur Project
project/
│── regresi_xgboost_lightgbm.py
│── Stock_Price_Dataset.csv
│── README.md


---

## 🚀 Cara Menjalankan

1. Install dependencies
Jika tidak ada `requirements.txt`, install manual:
pip install pandas numpy scikit-learn xgboost lightgbm
2. Jalankan script
python regresi_xgboost_lightgbm.py
Output akan menampilkan evaluasi kedua model serta contoh prediksi 1 data terakhir.

⚙️ Alur Pemodelan
1. Load & Preprocessing Data

Convert kolom Date → ordinal integer

Label Encoding untuk:

Stock

Risk

Menentukan fitur:

["Date_ordinal", "Open", "Stock_enc", "Risk_enc"]

2. Train–Test Split

80% training
20% testing
Dengan random_state=42

3. Model 1 – XGBoost Regressor

Parameter utama:

n_estimators=300
learning_rate=0.05
max_depth=5
subsample=0.8
colsample_bytree=0.8

4. Model 2 – LightGBM Regressor
n_estimators=300
learning_rate=0.05
max_depth=-1   # unlimited
subsample=0.8
colsample_bytree=0.8

5. Evaluasi Model

Fungsi evaluasi menghitung:

MAE

MSE

RMSE

MAPE

R²

📊 Evaluasi & Perbandingan

Gunakan hasil MAE/RMSE/R² untuk menentukan model terbaik.
Secara umum:

XGBoost cocok untuk dataset non-linear

LightGBM lebih cepat dan efisien pada dataset besar

🛠 Teknologi yang Digunakan

Python 3

Pandas

NumPy

Scikit-learn

XGBoost

LightGBM
