import numpy as np
import librosa
from scipy.signal import butter, lfilter


# FILTRADO
def butter_highpass_filter(data, cutoff=80, fs=22050):
    if len(data) == 0: return data # Protección contra señales vacías
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

# --- VAD ROBUSTO ---
def apply_vad(data, top_db=25):
    if len(data) == 0: return data
    
    # Intentamos segmentar
    intervals = librosa.effects.split(data, top_db=top_db)
    
    if len(intervals) == 0:
        # Si el VAD fue demasiado agresivo y borró todo, 
        # devolvemos el audio original en lugar de una señal vacía.
        return data 
    
    return np.concatenate([data[start:end] for start, end in intervals])

# --- EXTRACCIÓN DE CARACTERÍSTICAS SEGURA ---

    
def extract_features(data, sr=22050):
    # Garantizar longitud mínima absoluta antes de cualquier operación
    min_length = 2048
    if len(data) < min_length:
        data = np.pad(data, (0, min_length - len(data)), mode='constant')
    
    result = np.array([])
    
    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    
    # Chroma — ahora n_fft siempre >= 1
    stft = np.abs(librosa.stft(data, n_fft=2048))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))
    
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20).T, axis=0)
    result = np.hstack((result, mfcc))
    
    # RMS
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    
    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mel))
    
    return result


def get_features(path):
    # 1. Carga
    data, sample_rate = librosa.load(path, sr=22050, duration=2.5)
    target_length = int(2.5 * sample_rate)

    # 2. Preprocesamiento
    data = butter_highpass_filter(data)
    data = apply_vad(data)

    # 3. Duración fija — SIEMPRE usando mode='constant'
    if len(data) == 0:
        data = np.zeros(target_length)          # audio silencioso → zeros
    elif len(data) < target_length:
        data = np.pad(data, (0, target_length - len(data)), mode='constant')
    else:
        data = data[:target_length]

    # 4. Augmentation
    # Original
    res1 = extract_features(data)
    result = np.array(res1)

    # + Ruido
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data_noise = data + noise_amp * np.random.normal(size=data.shape[0])
    res2 = extract_features(data_noise)
    result = np.vstack((result, res2))

    # + Time stretch
    data_stretch = librosa.effects.time_stretch(data, rate=0.8)
    if len(data_stretch) > target_length:
        data_stretch = data_stretch[:target_length]
    else:
        data_stretch = np.pad(data_stretch, (0, target_length - len(data_stretch)), mode='constant')
    res3 = extract_features(data_stretch)
    result = np.vstack((result, res3))

    return result

