#!/usr/bin/env python3
"""
watcher.py — Vocalink Inference Watcher
Monitorea una carpeta de entrada de audio y corre el modelo CNN 1D
automáticamente cuando llega un archivo .wav nuevo.
"""

import os
import time
import logging
import threading
import numpy as np
import joblib
import librosa
from keras.models import load_model
from pathlib import Path
from datetime import datetime

# ─── Configuración ───────────────────────────────────────────────────────────
WATCH_DIR   = Path("audio_inbox")          # Carpeta donde llegan los .wav del ESP32
DONE_DIR    = Path("audio_processed")     # Carpeta donde se mueven los archivos procesados
MODEL_PATH  = Path("Models/vocalink_ninos.keras")
SCALER_PATH = Path("Models/scaler.pkl")
ENCODER_PATH= Path("Models/encoder.pkl")
POLL_INTERVAL = 0.5                        # segundos entre checks
LOG_FILE    = "vocalink_watcher.log"

# ─── Setup logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("vocalink")

# ─── Cargar modelo una sola vez al inicio ────────────────────────────────────
log.info("Cargando modelo Vocalink...")
model   = load_model(MODEL_PATH)
scaler  = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
LABELS  = encoder.categories_[0].tolist()
log.info(f"Modelo listo. Clases: {LABELS}")

# ─── Feature extraction (igual que en entrenamiento) ─────────────────────────
def extract_features(data, sample_rate):
    result = np.array([])
    zcr        = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    stft       = np.abs(librosa.stft(data))
    chroma     = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mfcc       = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    rms        = np.mean(librosa.feature.rms(y=data).T, axis=0)
    mel        = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack([result, zcr, chroma, mfcc, rms, mel])
    return result

def preprocess_audio(filepath: Path) -> np.ndarray:
    """Carga, extrae features, escala y devuelve tensor listo para el modelo."""
    data, sr = librosa.load(str(filepath), duration=2.5, offset=0.6)
    if len(data) < sr * 0.5:
        data, sr = librosa.load(str(filepath), duration=2.5, offset=0.0)
    if len(data) < sr * 0.5:
        data = np.pad(data, (0, int(sr * 2.5) - len(data)), mode='constant')
    features = extract_features(data, sr)
    features_scaled = scaler.transform([features])
    features_scaled = np.expand_dims(features_scaled, axis=2)   # (1, 162, 1)
    return features_scaled

# ─── Inferencia ───────────────────────────────────────────────────────────────
def run_inference(filepath: Path) -> dict:
    """Corre el modelo sobre un archivo y devuelve resultado."""
    X = preprocess_audio(filepath)
    probs = model.predict(X, verbose=0)[0]
    pred_label = LABELS[np.argmax(probs)]
    confidence = float(np.max(probs))
    result = {
        "file":        filepath.name,
        "timestamp":   datetime.now().isoformat(),
        "emotion":     pred_label,
        "confidence":  round(confidence, 4),
        "probabilities": {label: round(float(p), 4) for label, p in zip(LABELS, probs)}
    }
    return result

# ─── Estado del watcher ───────────────────────────────────────────────────────
_seen_files: set = set()
_new_file_flag = threading.Event()   # flag que cambia de estado cuando llega archivo nuevo

def _on_new_file(filepath: Path):
    """Callback que se ejecuta cuando llega un archivo nuevo."""
    log.info(f"Archivo nuevo detectado: {filepath.name}")
    try:
        result = run_inference(filepath)
        log.info(
            f"✅ {result['file']} → {result['emotion'].upper()} "
            f"(confianza: {result['confidence']*100:.1f}%)"
        )
        log.info(f"   Probabilidades: {result['probabilities']}")

        # Mover archivo procesado para no reprocesarlo
        DONE_DIR.mkdir(exist_ok=True)
        filepath.rename(DONE_DIR / filepath.name)
        return result

    except Exception as e:
        log.error(f"Error procesando {filepath.name}: {e}")
        return None

# ─── Loop principal de polling ─────────────────────────────────────────────────
def watch_loop():
    """Bucle que monitorea WATCH_DIR y dispara inferencia ante archivos nuevos."""
    WATCH_DIR.mkdir(exist_ok=True)
    log.info(f"Monitoreando carpeta: {WATCH_DIR.resolve()}")
    log.info(f"Intervalo de polling: {POLL_INTERVAL}s")

    while True:
        current_files = {
            f for f in WATCH_DIR.iterdir()
            if f.is_file() and f.suffix.lower() == ".wav"
        }
        new_files = current_files - _seen_files

        for filepath in sorted(new_files):
            _seen_files.add(filepath)
            _new_file_flag.set()         # ← cambia estado: "hay archivo nuevo"
            result = _on_new_file(filepath)
            _new_file_flag.clear()       # ← vuelve a estado en espera

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    watch_loop()
