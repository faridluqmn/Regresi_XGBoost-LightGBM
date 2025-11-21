# ============================================
# PREDIKSI HARGA SAHAM DENGAN XGBOOST & LIGHTGBM
# Dataset: Stock_Price_Dataset.csv
# Target  : kolom "Price"
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("Stock_Price_Dataset.csv")

# Contoh isi:
#   Stock        Date    Open   Price    Risk
#   AAPL  2023-07-03  143.44  144.81  medium

# ===============================
# 2. PREPROCESSING & FEATURE ENGINEERING
# ===============================

# Ubah Date jadi tipe datetime lalu ke angka (ordinal)
df["Date"] = pd.to_datetime(df["Date"])
df["Date_ordinal"] = df["Date"].map(pd.Timestamp.toordinal)

# Encode kolom kategorikal: Stock dan Risk
le_stock = LabelEncoder()
le_risk = LabelEncoder()

df["Stock_enc"] = le_stock.fit_transform(df["Stock"])
df["Risk_enc"] = le_risk.fit_transform(df["Risk"])

# Fitur yang digunakan untuk prediksi
feature_cols = ["Date_ordinal", "Open", "Stock_enc", "Risk_enc"]
X = df[feature_cols]
y = df["Price"]

# Jika ada missing value, boleh dibersihkan dulu
# df = df.dropna(subset=feature_cols + ["Price"])

# ===============================
# 3. TRAIN–TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ===============================
# 4. FUNGSI EVALUASI
# ===============================
def regression_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Hindari pembagian nol di MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    return mae, mse, rmse, mape, r2

def print_metrics(model_name, y_true, y_pred):
    mae, mse, rmse, mape, r2 = regression_metrics(y_true, y_pred)
    print(f"\n=== {model_name} ===")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

# ===============================
# 5. MODEL 1: XGBOOST REGRESSOR
# ===============================
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ---- TRAINING XGBOOST ----
xgb_model.fit(X_train, y_train)

# ---- TESTING / PREDIKSI ----
y_pred_xgb = xgb_model.predict(X_test)

# ---- EVALUASI ----
print_metrics("XGBoost Regressor", y_test, y_pred_xgb)

# ===============================
# 6. MODEL 2: LIGHTGBM REGRESSOR
# ===============================
lgbm_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,    # -1 artinya tidak dibatasi
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ---- TRAINING LIGHTGBM ----
lgbm_model.fit(X_train, y_train)

# ---- TESTING / PREDIKSI ----
y_pred_lgbm = lgbm_model.predict(X_test)

# ---- EVALUASI ----
print_metrics("LightGBM Regressor", y_test, y_pred_lgbm)

# ===============================
# 7. CONTOH PREDIKSI DATA BARU
# ===============================
# Misal mau prediksi 1 baris data terakhir di dataset:
sample_row = df.iloc[-1]  # baris terakhir
sample_feature = [[
    sample_row["Date_ordinal"],
    sample_row["Open"],
    sample_row["Stock_enc"],
    sample_row["Risk_enc"]
]]

pred_xgb  = xgb_model.predict(sample_feature)[0]
pred_lgbm = lgbm_model.predict(sample_feature)[0]

print("\nContoh prediksi 1 baris terakhir:")
print(f"Price aktual : {sample_row['Price']}")
print(f"Prediksi XGBoost : {pred_xgb:.4f}")
print(f"Prediksi LightGBM: {pred_lgbm:.4f}")
