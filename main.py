import sys
import os
import joblib
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from core.dsp_utils import butter_highpass_filter, apply_vad, extract_features

app = FastAPI(title="VocaLink API - Clasificador Emocional")

# 1. CARGAR EL CEREBRO DEL SISTEMA
MODEL_PATH = "models/vocalink_ninos.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    # A. Recibir audio y guardarlo temporalmente
    with open("temp_audio.wav", "wb") as buffer:
        buffer.write(await file.read())
    
    # B. CARGA Y PREPROCESAMIENTO (Iteración 3 del MVP)
    data, sr = librosa.load("temp_audio.wav", sr=22050)
    data = butter_highpass_filter(data) # Limpia ruidos bajos
    data = apply_vad(data)              # Recorta silencios
    
    # C. EXTRAER CARACTERÍSTICAS (Las 162 dimensiones)
    features = extract_features(data)
    
    # D. ESCALAR E INFERIR
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_final = np.expand_dims(features_scaled, axis=2)
    
    prediction = model.predict(features_final)
    idx = np.argmax(prediction)
    
    # Obtener nombre de la emoción y confianza
    emotion = encoder.categories_[0][idx]
    confidence = float(np.max(prediction))

    return {
        "emotion": emotion,
        "confidence": f"{confidence*100:.2f}%",
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)