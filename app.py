# app.py — Aplikasi Streamlit untuk Klasifikasi Sentimen
# Konflik Israel-Palestine (Twitter)
# Abdan Nawwaf El Hibban — Skripsi UIN Malang 2024
# ============================================================

import os
import json
import pickle
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocessing import preprocess_single_tweet

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen — Konflik Israel-Palestine",
    page_icon="🔍",
    layout="centered",
)

ARTIFACT_DIR = "model_artifacts"
ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# ── Label mapping ────────────────────────────────────────────
LABEL_CONFIG = {
    "Positive": {"emoji": "😊", "color": "#22c55e", "bg": "#f0fdf4"},
    "Negative": {"emoji": "😞", "color": "#ef4444", "bg": "#fef2f2"},
    "Neutral":  {"emoji": "😐", "color": "#f59e0b", "bg": "#fffbeb"},
}


# ── Load artefak dengan caching ─────────────────────────────
@st.cache_resource(show_spinner="Memuat model NB & SVM...")
def load_classical_models():
    """Load TF-IDF, NB, SVM, dan LabelEncoder dari disk."""
    with open(f"{ARTIFACT_DIR}/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(f"{ARTIFACT_DIR}/nb_model.pkl", "rb") as f:
        nb = pickle.load(f)
    with open(f"{ARTIFACT_DIR}/svm_model.pkl", "rb") as f:
        svm = pickle.load(f)
    with open(f"{ARTIFACT_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return tfidf, nb, svm, le


@st.cache_resource(show_spinner="Memuat model RoBERTa (bisa 30-60 detik)...")
def load_roberta():
    """Load RoBERTa tokenizer + model dari HuggingFace + weights lokal."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME, num_labels=3
    )
    weights_path = f"{ARTIFACT_DIR}/best_model_bert.pt"
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        st.warning(
            "⚠ File `best_model_bert.pt` tidak ditemukan. "
            "RoBERTa berjalan dengan pre-trained weights tanpa fine-tuning."
        )
    model.to(device)
    model.eval()
    return tokenizer, model, device


# ── Fungsi inferensi ─────────────────────────────────────────
def predict_nb_svm(text_preprocessed: str, tfidf, nb, svm) -> dict:
    X = tfidf.transform([text_preprocessed])
    
    # NB prediction
    label_nb = nb.predict(X)[0]
    prob_nb = nb.predict_proba(X)[0]

    # SVM prediction — tangani kemungkinan version mismatch
    try:
        label_svm = svm.predict(X)[0]
    except AttributeError:
        # Fallback: gunakan decision_function jika predict gagal
        decision = svm.decision_function(X)[0]
        classes = svm.classes_
        label_svm = classes[decision.argmax()]
    
    return {
        "nb": {"label": label_nb, "probabilities": dict(zip(nb.classes_, prob_nb))},
        "svm": {"label": label_svm},
    }


def predict_roberta(raw_text: str, tokenizer, model, device, le) -> dict:
    """Prediksi dengan RoBERTa."""
    enc = tokenizer(
        raw_text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
    pred_idx = probs.argmax()
    label = le.inverse_transform([pred_idx])[0]
    prob_dict = {le.inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))}
    return {"label": label, "probabilities": prob_dict}


# ── Komponen UI helper ────────────────────────────────────────
def render_sentiment_badge(label: str):
    cfg = LABEL_CONFIG.get(label, {"emoji": "❓", "color": "#6b7280", "bg": "#f9fafb"})
    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background: {cfg['bg']};
            border: 2px solid {cfg['color']};
            border-radius: 8px;
            padding: 8px 20px;
            font-size: 1.1rem;
            font-weight: 600;
            color: {cfg['color']};
            margin: 4px 0;
        ">
            {cfg['emoji']} {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bar(label: str, prob: float):
    cfg = LABEL_CONFIG.get(label, {"emoji": "❓", "color": "#6b7280", "bg": "#f9fafb"})
    st.markdown(
        f"""
        <div style="margin-bottom: 6px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 2px;">
                <span>{cfg['emoji']} {label}</span>
                <span style="font-weight: 600;">{prob:.1%}</span>
            </div>
            <div style="background: #e5e7eb; border-radius: 4px; height: 10px;">
                <div style="
                    width: {prob*100:.1f}%;
                    background: {cfg['color']};
                    height: 10px;
                    border-radius: 4px;
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Layout utama ─────────────────────────────────────────────
st.title("🔍 Analisis Sentimen Tweet")
st.caption("Studi Kasus: Konflik Israel-Palestine — Abdan Nawwaf El Hibban")
st.divider()

# Sidebar info
with st.sidebar:
    st.header("ℹ Tentang Aplikasi")
    st.markdown(
        """
        Aplikasi ini mengklasifikasikan tweet ke dalam tiga kelas sentimen:
        - 😊 **Positive**
        - 😐 **Neutral**
        - 😞 **Negative**

        **Model yang tersedia:**
        - Multinomial Naive Bayes (TF-IDF)
        - Support Vector Machine (TF-IDF)
        - RoBERTa (`twitter-roberta-base-sentiment`)

        **Data training:**  
        ~15.498 tweet (setelah undersampling)

        **Preprocessing:**  
        Cleaning → Tokenisasi → Lemmatisasi → TF-IDF / RoBERTa tokenizer
        """
    )
    st.divider()
    st.caption("UIN Maulana Malik Ibrahim Malang · 2024")

# Pilih model
model_choice = st.selectbox(
    "Pilih model klasifikasi:",
    options=["Semua Model (Bandingkan)", "Naive Bayes", "SVM", "RoBERTa"],
    index=0,
)

use_roberta = model_choice in ("Semua Model (Bandingkan)", "RoBERTa")
use_classical = model_choice in ("Semua Model (Bandingkan)", "Naive Bayes", "SVM")

# Input teks
st.subheader("Masukkan Tweet")
user_input = st.text_area(
    "Teks tweet (bahasa Inggris):",
    height=120,
    placeholder='Contoh: "Free Palestine! The world must stand for justice and human rights."',
)

# Tombol prediksi
predict_btn = st.button("🔎 Analisis Sentimen", type="primary", use_container_width=True)

# Contoh tweet
with st.expander("📋 Contoh tweet untuk dicoba"):
    examples = [
        "We stand with Palestine and pray for peace in the region.",
        "Hamas attacked civilians without mercy. This is terrorism.",
        "The UN is meeting to discuss the ongoing conflict in Gaza.",
        "Innocent children are dying every day. The world must act NOW!",
        "Both sides need to come to the negotiating table immediately.",
    ]
    for i, ex in enumerate(examples):
        if st.button(f"Contoh {i+1}", key=f"ex_{i}"):
            st.session_state["example_text"] = ex
            st.rerun()

# Isi otomatis dari contoh yang dipilih
if "example_text" in st.session_state:
    user_input = st.session_state.pop("example_text")
    st.rerun()  # supaya textarea ter-update — perlu workaround


# ── Prediksi ─────────────────────────────────────────────────
if predict_btn and user_input.strip():
    st.divider()
    st.subheader("📊 Hasil Klasifikasi")

    # Preprocessing — tampilkan ke user agar transparan
    preprocessed = preprocess_single_tweet(user_input)
    with st.expander("🔧 Lihat hasil preprocessing"):
        st.code(preprocessed if preprocessed else "(kosong setelah preprocessing)", language="text")
        if not preprocessed:
            st.warning(
                "Teks menjadi kosong setelah preprocessing. "
                "Coba tweet yang lebih panjang atau cek apakah ada kata substantif."
            )

    if preprocessed:
        # ── Classical models ──
        if use_classical:
            try:
                tfidf, nb, svm, le = load_classical_models()
                results = predict_nb_svm(preprocessed, tfidf, nb, svm)

                if model_choice in ("Semua Model (Bandingkan)", "Naive Bayes"):
                    with st.container(border=True):
                        st.markdown("#### 🔵 Multinomial Naive Bayes")
                        render_sentiment_badge(results["nb"]["label"])
                        st.markdown("**Distribusi Probabilitas:**")
                        for lbl in ["Positive", "Neutral", "Negative"]:
                            render_probability_bar(lbl, results["nb"]["probabilities"].get(lbl, 0.0))

                if model_choice in ("Semua Model (Bandingkan)", "SVM"):
                    with st.container(border=True):
                        st.markdown("#### 🟣 Support Vector Machine")
                        render_sentiment_badge(results["svm"]["label"])
                        st.caption(
                            "SVM tidak menghasilkan probabilitas secara default. "
                            "Hanya label prediksi yang ditampilkan."
                        )

            except FileNotFoundError:
                st.error(
                    "❌ File model klasikal tidak ditemukan di `model_artifacts/`. "
                    "Pastikan kamu sudah menjalankan `save_models_notebook.py` "
                    "di akhir notebook dan mengupload folder tersebut."
                )

        # ── RoBERTa ──
        if use_roberta:
            try:
                with st.spinner("Menjalankan inferensi RoBERTa..."):
                    tok, rob_model, device = load_roberta()
                    _, _, _, le = load_classical_models()  # ambil le dari classical
                    rob_result = predict_roberta(user_input, tok, rob_model, device, le)

                with st.container(border=True):
                    st.markdown("#### 🟢 RoBERTa (`twitter-roberta-base-sentiment`)")
                    render_sentiment_badge(rob_result["label"])
                    st.markdown("**Distribusi Probabilitas:**")
                    for lbl in ["Positive", "Neutral", "Negative"]:
                        render_probability_bar(lbl, rob_result["probabilities"].get(lbl, 0.0))

            except FileNotFoundError:
                st.error(
                    "❌ Artefak RoBERTa tidak ditemukan. "
                    "Pastikan `model_artifacts/best_model_bert.pt` tersedia."
                )
            except Exception as e:
                st.error(f"❌ Error saat menjalankan RoBERTa: {e}")

        # ── Ringkasan perbandingan (hanya jika semua model dipilih) ──
        if model_choice == "Semua Model (Bandingkan)":
            try:
                tfidf, nb, svm, le = load_classical_models()
                results_cl = predict_nb_svm(preprocessed, tfidf, nb, svm)
                tok, rob_model, device = load_roberta()
                rob_result = predict_roberta(user_input, tok, rob_model, device, le)

                st.divider()
                st.subheader("📋 Ringkasan Prediksi")
                cols = st.columns(3)
                for col, (model_name, label) in zip(cols, [
                    ("Naive Bayes", results_cl["nb"]["label"]),
                    ("SVM", results_cl["svm"]["label"]),
                    ("RoBERTa", rob_result["label"]),
                ]):
                    cfg = LABEL_CONFIG.get(label, {"emoji": "❓", "color": "#6b7280"})
                    with col:
                        st.metric(model_name, f"{cfg['emoji']} {label}")

                # Deteksi kesepakatan model
                labels_set = {
                    results_cl["nb"]["label"],
                    results_cl["svm"]["label"],
                    rob_result["label"],
                }
                if len(labels_set) == 1:
                    st.success("✅ Ketiga model **sepakat** pada label yang sama.")
                elif len(labels_set) == 2:
                    st.info("⚠ Dua dari tiga model sepakat. Ada satu model yang berbeda.")
                else:
                    st.warning("❗ Ketiga model memberikan prediksi yang berbeda.")

            except Exception:
                pass  # sudah ditangani di atas

elif predict_btn and not user_input.strip():
    st.warning("Masukkan teks tweet terlebih dahulu.")

import sklearn

st.sidebar.write("sklearn:", sklearn.__version__)
st.sidebar.write(type(svm))