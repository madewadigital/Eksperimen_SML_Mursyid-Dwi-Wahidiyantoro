# Eksperimen SML - Mursyid Dwi Wahidiyantoro

Repository ini memuat eksperimen eksplorasi data analisis (EDA) dan otomatisasi data preprocessing untuk Heart Disease Classification.

## Struktur Repository
```
Eksperimen_SML_Mursyid-Dwi-Wahidiyantoro/
├── .github/workflows/
│   └── preprocessing.yml
├── heart_raw.csv
└── preprocessing/
    ├── Eksperimen_Mursyid-Dwi-Wahidiyantoro.ipynb
    ├── automate_Mursyid-Dwi-Wahidiyantoro.py
    └── heart_preprocessing.csv
```

## Tahapan Eksperimen:
1. **Perkenalan Dataset**: Deskripsi fitur Heart Disease UCI Cleveland dataset.
2. **Import Library**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn.
3. **Memuat Dataset**: Validasi bentuk data 297 baris x 14 kolom.
4. **Exploratory Data Analysis (EDA)**: Distribusi target, histogram fitur, korelasi heatmap.
5. **Data Preprocessing**: Handling outliers (IQR capping), standarisasi fitur numerik, ekspor data bersih.

## Otomatisasi Preprocessing:
Jalankan file otomatisasi melalui terminal:
```bash
python preprocessing/automate_Mursyid-Dwi-Wahidiyantoro.py
```
Output data bersih tersimpan di `preprocessing/heart_preprocessing.csv`.
