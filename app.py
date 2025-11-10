import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# STREAMLIT CONFIG
st.set_page_config(page_title="Prediksi Unemployment Rate", layout="wide")

st.title("Sistem Prediksi Pengangguran Menggunakan Data Ekonomi Makro")
st.write("Menggunakan **LightGBM**, **ETS**, dan Preprocessing**")

# 1. UPLOAD DATASET
uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # CLEANING COLUMN NAMES
    df.columns = [
        c.strip()
        .replace(" ", "_")
        .replace("%", "pct")
        .replace("(", "")
        .replace(")", "")
        .lower()
        for c in df.columns
    ]

    for col in df.columns:
        if "unemployment" in col:
            df.rename(columns={col: "unemployment_rate_pct"}, inplace=True)

    st.subheader("Sample Data 10")
    st.dataframe(df.head(10))

    # 2. VALIDASI KOLOM
    required_cols = ["country_name", "year", "unemployment_rate_pct"]

    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset harus memiliki kolom: {required_cols}")
        st.write("Kolom tersedia:", df.columns.tolist())
        st.stop()

    st.success("Dataset valid")
    st.markdown("---")

    # 3. EDA
    st.header("Exploratory Data Analysis (EDA)")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Histogram Fitur Numerik")
    fitur_hist = st.selectbox("Pilih fitur numerik:", numeric_cols)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df[fitur_hist].dropna(), bins=20)
    ax.set_title(f"Histogram: {fitur_hist}")
    st.pyplot(fig)

    st.subheader("Heatmap Korelasi (tanpa seaborn)")
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(corr, cmap="Blues")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im)
    st.pyplot(fig)

    st.markdown("---")

    # 4. PREPROCESSING KOMPLIT
    st.header("PREPROCESSING DATA (Lengkap)")

    negara_list = df["country_name"].unique()
    negara_pilih = st.multiselect("Pilih negara:", negara_list, default=list(negara_list))

    df = df[df["country_name"].isin(negara_pilih)].copy()

    st.subheader("Missing Value Filling (Mean per negara)")
    for col in numeric_cols:
        df[col] = df.groupby("country_name")[col].transform(lambda x: x.fillna(x.mean()))

    st.subheader("Outlier Handling (IQR) - Opsional")
    remove_outliers = st.checkbox("Aktifkan pembuangan outlier (IQR)", value=False)

    if remove_outliers:
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        mask = ~(
            (df[numeric_cols] < (Q1 - 1.5 * IQR)) |
            (df[numeric_cols] > (Q3 + 1.5 * IQR))
        ).any(axis=1)

        df = df[mask]
        st.success("Outlier dibersihkan.")

    st.subheader("Normalisasi (opsional)")
    use_scaler = st.checkbox("Gunakan StandardScaler", value=False)

    if use_scaler:
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        df = df_scaled
        st.info("StandardScaler diterapkan")

    st.subheader("Membuat Lag-1")
    df = df.sort_values(["country_name", "year"])
    df["unemployment_rate_lag1"] = df.groupby("country_name")["unemployment_rate_pct"].shift(1)

    df_lgb = df.dropna(subset=["unemployment_rate_lag1"]).copy()

    st.subheader("Train–Validation Split (Time-based)")
    train = df_lgb[df_lgb["year"] <= 2020]
    val = df_lgb[(df_lgb["year"] > 2020) & (df_lgb["year"] <= 2024)]

    st.write(f"Train: {train.shape[0]} baris")
    st.write(f"Validation: {val.shape[0]} baris")

    st.success("Preprocessing lengkap selesai!")
    st.markdown("---")

    # 5. MODEL LIGHTGBM + PATCH ERROR
    st.header("Prediksi Menggunakan LightGBM")

    feature_cols = st.multiselect(
        "Pilih fitur tambahan:",
        [c for c in numeric_cols if c != "unemployment_rate_pct"]
    )

    all_features = ["unemployment_rate_lag1"] + feature_cols

    X_train = train[all_features]
    y_train = train["unemployment_rate_pct"]

    model_lgb = lgb.LGBMRegressor()
    model_lgb.fit(X_train, y_train)

    st.success("Model LightGBM berhasil dilatih!")

    # ============================================================
    # ✅ 5.1 EVALUASI MODEL PADA DATA VALIDATION
    # ============================================================
    st.subheader("Evaluasi Model pada Data Validation")

    if val.shape[0] == 0:
        st.warning("Tidak ada data validation. Metrik evaluasi tidak dapat dihitung.")
    else:
        X_val = val[all_features]
        y_val = val["unemployment_rate_pct"]

        y_pred_val = model_lgb.predict(X_val)

        MAE = np.mean(np.abs(y_val - y_pred_val))
        RMSE = np.sqrt(np.mean((y_val - y_pred_val)**2))
        MAPE = np.mean(np.abs((y_val - y_pred_val) / y_val)) * 100

        eval_df = pd.DataFrame({
            "Metrik": ["MAE", "RMSE", "MAPE (%)"],
            "Nilai": [round(MAE, 4), round(RMSE, 4), round(MAPE, 4)]
        })

        st.table(eval_df)

    st.markdown("---")

    # MULAI PREDIKSI FUTURE
    pred_years = list(range(2025, 2031))
    pred_dict = {c: [] for c in negara_pilih}

    for year in pred_years:
        rows = []
        valid_countries = []

        for country in negara_pilih:
            df_country = df_lgb[df_lgb["country_name"] == country]

            if df_country.empty:
                st.warning(f"Negara '{country}' tidak punya data cukup. Dilewati.")
                continue

            last_row = df_country.iloc[-1]

            row = {"unemployment_rate_lag1": last_row["unemployment_rate_pct"]}
            for feat in feature_cols:
                row[feat] = last_row[feat]

            rows.append(row)
            valid_countries.append(country)

        if not rows:
            st.error("Tidak ada negara valid untuk prediksi.")
            break

        df_pred_input = pd.DataFrame(rows)
        pred_values = model_lgb.predict(df_pred_input)

        for idx, country in enumerate(valid_countries):
            pred_dict[country].append(pred_values[idx])

    valid_pred_dict = {k: v for k, v in pred_dict.items() if len(v) == len(pred_years)}

    if not valid_pred_dict:
        st.error("Tidak ada negara yang memiliki prediksi lengkap.")
        st.stop()

    df_pred_lgb = pd.DataFrame(valid_pred_dict, index=pred_years)

    st.subheader("Hasil Prediksi LightGBM")
    st.dataframe(df_pred_lgb)

    plt.figure(figsize=(12, 6))
    for country in df_pred_lgb.columns:
        df_c = df[df["country_name"] == country]
        plt.plot(df_c["year"], df_c["unemployment_rate_pct"], marker="o", label=f"{country} Aktual")
        plt.plot(pred_years, df_pred_lgb[country], marker="x", linestyle="--", label=f"{country} Prediksi")

    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.title("Prediksi Unemployment Rate - LightGBM")
    st.pyplot(plt.gcf())
    plt.clf()

    st.markdown("---")

    # 6. ETS MODEL
    st.header("Prediksi Menggunakan ETS")

    forecast_dict = {}

    for country in negara_pilih:
        ts = df[df["country_name"] == country].sort_values("year")["unemployment_rate_pct"]

        if len(ts) < 2:
            st.warning(f"Negara {country} tidak cukup untuk ETS.")
            continue

        ets_model = ExponentialSmoothing(ts, trend="add")
        fit = ets_model.fit()
        forecast = fit.forecast(len(pred_years))

        forecast_dict[country] = forecast.values

    if forecast_dict:
        df_pred_ets = pd.DataFrame(forecast_dict, index=pred_years)

        st.subheader("Hasil Prediksi ETS")
        st.dataframe(df_pred_ets)

        plt.figure(figsize=(12, 6))
        for country in df_pred_ets.columns:
            df_c = df[df["country_name"] == country]
            plt.plot(df_c["year"], df_c["unemployment_rate_pct"], marker="o")
            plt.plot(pred_years, df_pred_ets[country], marker="x", linestyle="--")

        plt.title("Prediksi Unemployment Rate - ETS")
        st.pyplot(plt.gcf())
        plt.clf()
