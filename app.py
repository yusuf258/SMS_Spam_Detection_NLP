import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import time

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="SMS Spam Tespiti",
    page_icon="📩",
    layout="wide"
)

# --- GÜVENLİ IMPORT ---
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.layers import TextVectorization
    TF_AVAILABLE = True
except Exception:
    pass

st.title("📩 SMS Spam Tespiti")
st.markdown("---")

# --- MODEL YOLLARI ---
BASE_DIR = os.path.dirname(__file__)
ML_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
DL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dl_model.h5')
VEC_VOCAB_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer_vocab.pkl')

# DEBUG: Show file info
if st.checkbox("🔍 Debug Modu Göster"):
    st.write(f"Model Yolu: {ML_MODEL_PATH}")
    if os.path.exists(ML_MODEL_PATH):
        mod_time = time.ctime(os.path.getmtime(ML_MODEL_PATH))
        st.write(f"Model Dosya Zamanı: {mod_time}")
    else:
        st.error("Model dosyası yok!")

# Clear cache button
if st.button("🔄 Modeli Yeniden Yükle / Önbelleği Temizle"):
    st.cache_resource.clear()
    st.rerun()

@st.cache_resource
def load_all_assets():
    ml_pipe = None
    dl_mod = None
    vec_layer = None
    
    # Load ML Model
    if os.path.exists(ML_MODEL_PATH):
        try:
            ml_pipe = joblib.load(ML_MODEL_PATH)
        except Exception as e:
            st.error(f"ML Model yükleme hatası: {e}")
    
    # Load DL Model & Vectorizer
    if TF_AVAILABLE and os.path.exists(DL_MODEL_PATH) and os.path.exists(VEC_VOCAB_PATH):
        try:
            # 1. Load Vectorizer Config/Vocab
            with open(VEC_VOCAB_PATH, 'rb') as f:
                data = pickle.load(f)
                
            # Recreate layer
            vec_layer = TextVectorization.from_config(data['config'])
            vec_layer.set_vocabulary(data['vocab'])
            
            # 2. Load Model
            dl_mod = tf.keras.models.load_model(DL_MODEL_PATH)
            
        except Exception as e:
            st.error(f"DL Model yükleme hatası: {e}")
            
    return ml_pipe, dl_mod, vec_layer

ml_pipeline, dl_model, vectorizer = load_all_assets()

# --- ARAYÜZ ---
st.subheader("📝 Mesajı Girin")
user_input = st.text_area("Analiz edilecek mesaj:", height=100, placeholder="Örn: Congratulations! You've won a  gift card...")

if st.button("🚀 Analiz Et"):
    if not user_input:
        st.warning("Lütfen bir mesaj girin.")
    else:
        c1, c2 = st.columns(2)
        
        # --- ML Prediction ---
        with c1:
            st.info("🤖 ML Tahmini (SVM - Support Vector Machine)")
            if ml_pipeline:
                try:
                    # Check model type for debugging
                    model_type = type(ml_pipeline.named_steps['classifier']).__name__
                    if st.checkbox("Model Detayı"):
                        st.write(f"Kullanılan Algoritma: {model_type}")
                    
                    pred_class = ml_pipeline.predict([user_input])[0]
                    if hasattr(ml_pipeline, "predict_proba"):
                        proba = ml_pipeline.predict_proba([user_input]).max()
                    else:
                        proba = 1.0
                        
                    label = "SPAM 🚨" if pred_class == 1 else "HAM (Güvenli) ✅"
                    st.metric(label="Sonuç", value=label, delta=f"Güven: %{proba*100:.1f}")
                except Exception as e:
                    st.error(f"Hata: {e}")
            else:
                st.error("ML Modeli bulunamadı.")
        
        # --- DL Prediction ---
        with c2:
            st.info("🧠 DL Tahmini (Neural Network)")
            if dl_model and vectorizer:
                try:
                    # 1. Vectorize
                    text_input = tf.constant([user_input], dtype=tf.string)
                    vec_input = vectorizer(text_input)
                    
                    # 2. Predict
                    pred_prob = dl_model.predict(vec_input, verbose=0)[0][0]
                    
                    is_spam = pred_prob > 0.5
                    label_dl = "SPAM 🚨" if is_spam else "HAM (Güvenli) ✅"
                    
                    # Calculate confidence percentage
                    conf_score = pred_prob if is_spam else 1 - pred_prob
                    
                    st.metric(label="Sonuç", value=label_dl, delta=f"Güven: %{conf_score*100:.1f}")
                except Exception as e:
                    st.error(f"DL Hatası: {e}")
            else:
                if not dl_model:
                    st.warning("DL Model dosyası (h5) bulunamadı.")
                if not vectorizer:
                    st.warning("DL Vektörleştirici (pkl) bulunamadı.")

st.markdown("---")
st.caption("Not: ML modeli SVM (TF-IDF) kullanır. DL modeli Keras tabanlı Embedding + Pooling katmanları içerir.")
