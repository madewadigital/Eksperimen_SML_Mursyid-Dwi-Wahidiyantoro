"""
Automated Data Preprocessing Module for Heart Disease Dataset.
Author: Mursyid Dwi Wahidiyantoro (mursyiddwiw)
Course: Membangun Sistem Machine Learning (Dicoding)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Memuat raw dataset dari path yang ditentukan."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File raw dataset tidak ditemukan di: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[INFO] Raw dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan pipeline preprocessing:
    1. Handling duplikasi dan missing values.
    2. Deteksi dan capping outlier menggunakan metode IQR.
    3. Standardisasi fitur numerik kontinu.
    4. Memastikan format data siap pakai untuk training model ML.
    """
    df_clean = df.copy()

    # 1. Handling duplicates
    init_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    diff = init_rows - len(df_clean)
    if diff > 0:
        print(f"[INFO] Dihapus {diff} baris duplikat.")

    # 2. Handling missing values
    df_clean = df_clean.dropna()

    # 3. Handling Outliers pada kolom numerik kontinu dengan IQR Capping
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in continuous_features:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)

    # 4. Standardisasi fitur numerik kontinu
    scaler = StandardScaler()
    df_clean[continuous_features] = scaler.fit_transform(df_clean[continuous_features])

    print(f"[INFO] Preprocessing selesai. Output shape: {df_clean.shape}")
    return df_clean


def save_preprocessed_data(df: pd.DataFrame, output_path: str) -> None:
    """Menyimpan data hasil preprocessing ke file CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Dataset preprocessing berhasil disimpan di: {output_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    # Path raw data
    possible_raw_paths = [
        os.path.join(parent_dir, 'heart_raw.csv'),
        os.path.join(base_dir, 'heart_raw.csv'),
        'heart_raw.csv'
    ]
    raw_path = None
    for p in possible_raw_paths:
        if os.path.exists(p):
            raw_path = p
            break

    if raw_path is None:
        raw_path = os.path.join(parent_dir, 'heart_raw.csv')

    output_path = os.path.join(base_dir, 'heart_preprocessing.csv')

    raw_df = load_raw_data(raw_path)
    clean_df = preprocess_data(raw_df)
    save_preprocessed_data(clean_df, output_path)


if __name__ == '__main__':
    main()
