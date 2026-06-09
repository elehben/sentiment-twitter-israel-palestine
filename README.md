# 🔍 Analisis Sentimen Tweet — Konflik Israel-Palestine

Aplikasi klasifikasi sentimen tweet berbasis **Multinomial Naive Bayes**, **SVM**, dan **RoBERTa**.  
Abdan Nawwaf El Hibban · UIN Maulana Malik Ibrahim Malang · Skripsi 2024

---

## Struktur Folder

```
sentiment-app/
├── app.py                      # Aplikasi Streamlit (file utama)
├── preprocessing.py            # Modul preprocessing (dipakai app.py)
├── requirements.txt            # Dependencies
├── save_models_notebook.py     # Snippet yang ditambahkan ke notebook
└── model_artifacts/            # Artefak hasil training (dibuat dari notebook)
    ├── tfidf_vectorizer.pkl
    ├── nb_model.pkl
    ├── svm_model.pkl
    ├── label_encoder.pkl
    ├── best_model_bert.pt
    ├── label_classes.json
    └── roberta_model_name.txt
```

---

## Langkah Deploy ke Streamlit Cloud (Gratis)

### Langkah 1 — Simpan artefak dari notebook

Jalankan cell dari `save_models_notebook.py` di **akhir notebook Google Colab** kamu
(setelah training selesai). Ini akan membuat folder `model_artifacts/`.

Download folder `model_artifacts/` ke komputer kamu.

### Langkah 2 — Upload ke GitHub

1. Buat repository GitHub baru (misal: `sentiment-skripsi`)
2. Upload semua file ini ke repo tersebut:
   ```
   app.py
   preprocessing.py
   requirements.txt
   model_artifacts/          ← folder ini ikut diupload
   ```

> ⚠ **Catatan penting ukuran file:**  
> `best_model_bert.pt` bisa berukuran ~500MB.  
> GitHub membatasi file di 100MB. Untuk file besar, gunakan **Git LFS**:
> ```bash
> git lfs install
> git lfs track "model_artifacts/best_model_bert.pt"
> git add .gitattributes
> git commit -m "track large model with LFS"
> ```

### Langkah 3 — Deploy di Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan GitHub
3. Klik **"New app"**
4. Pilih repository dan branch kamu
5. Set **Main file path**: `app.py`
6. Klik **Deploy**

Streamlit akan otomatis install semua dependency dari `requirements.txt`.

---

## Jalankan Lokal (untuk Testing)

```bash
# Install dependencies
pip install -r requirements.txt

# Pastikan folder model_artifacts/ ada di direktori yang sama
# Lalu jalankan:
streamlit run app.py
```

---

## Catatan Penting Sebelum Deploy

### Masalah RAM di Streamlit Cloud (Free Tier)
Streamlit Cloud gratis membatasi RAM ~1GB.  
Model RoBERTa (`best_model_bert.pt`) sendiri bisa butuh ~1.5GB saat inference.

**Solusi jika terkena OOM (Out of Memory):**
- Deploy tanpa RoBERTa dulu (NB + SVM saja) — ubah `use_roberta = False`
- Atau gunakan Hugging Face Spaces yang punya lebih banyak RAM
- Atau gunakan Streamlit Community Cloud dengan resource tambahan

### SVM `predict_proba`
Kode SVM di notebook dilatih tanpa `probability=True`.  
Jika kamu ingin SVM juga menampilkan probabilitas, ubah di notebook:
```python
SVC(random_state=42, probability=True)
```
Dan retrain, lalu simpan ulang.

---

## Alternatif Platform Deploy

| Platform | RAM | GPU | Cocok untuk |
|---|---|---|---|
| Streamlit Cloud | ~1GB | ❌ | Demo ringan (NB+SVM) |
| Hugging Face Spaces | 2-16GB | ✅ (gratis terbatas) | Full pipeline + RoBERTa |
| Railway | 512MB-8GB | ❌ | FastAPI backend |
| Google Cloud Run | Flexible | ❌ | Production |
