"""Audio: voz -> texto con Whisper (STT) y texto -> voz con pyttsx3 o Piper (TTS). Todo local y gratis."""
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------- 1 · voz -> texto (Whisper)
# "small" (~460 MB) es el minimo razonable en espanol; "base" y "tiny" fallan nombres y numeros.
# Comparativa en extras/paso2_tamano_whisper.py
MODELO_WHISPER = "small"
_whisper = {}


def cargar_audio(ruta: str) -> np.ndarray:
    """Whisper quiere audio mono, a 16 kHz y en float32. whisper.transcribe("archivo.mp3") llama a ffmpeg por
    debajo (y en Windows no suele estar instalado): leyendo el WAV nosotros no hace falta."""
    audio, sr = sf.read(ruta, dtype="float32")
    if audio.ndim > 1:                                    # estereo -> mono
        audio = audio.mean(axis=1)
    if sr != 16000:                                       # re-muestrear a 16 kHz
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio


def transcribir(ruta: str, modelo: str = MODELO_WHISPER) -> dict:
    import whisper
    if modelo not in _whisper:
        _whisper[modelo] = whisper.load_model(modelo)      # la primera vez se descarga el modelo
    t0 = time.time()
    r = _whisper[modelo].transcribe(cargar_audio(ruta), language="es", fp16=False)   # fp16=False: en CPU
    return {"texto": r["text"].strip(), "segundos": round(time.time() - t0, 1),
            "segmentos": [(round(s["start"], 1), round(s["end"], 1), s["text"].strip()) for s in r["segments"]]}


# ---------------------------------------------------------------- 2 · texto -> voz (TTS)
def hablar_pyttsx3(texto: str, salida: str, voz: str = "Helena", velocidad: int = 180) -> str:
    """pyttsx3 usa las voces que ya trae Windows (SAPI5): cero descargas. Helena, Laura y Pablo hablan espanol."""
    import pyttsx3
    motor = pyttsx3.init()
    for v in motor.getProperty("voices"):
        if voz.lower() in v.name.lower():
            motor.setProperty("voice", v.id)
    motor.setProperty("rate", velocidad)
    motor.save_to_file(texto, salida)
    motor.runAndWait()
    return salida


VOZ_PIPER = "es_ES-davefx-medium"      # ~60 MB, voz neuronal: suena bastante mas natural que SAPI


def hablar_piper(texto: str, salida: str, voz: str = VOZ_PIPER) -> str:
    """Piper: TTS neuronal en local. La voz se descarga una vez con: python -m piper.download_voices <voz>"""
    import wave
    from piper import PiperVoice
    modelo = Path("voces") / f"{voz}.onnx"
    v = PiperVoice.load(str(modelo))
    with wave.open(salida, "wb") as w:
        v.synthesize_wav(texto, w)
    return salida


def hablar(texto: str, salida: str, motor: str = "piper") -> dict:
    t0 = time.time()
    (hablar_piper if motor == "piper" else hablar_pyttsx3)(texto, salida)
    info = sf.info(salida)
    return {"archivo": salida, "segundos_generar": round(time.time() - t0, 1), "duracion": round(info.duration, 1)}


def reproducir(ruta: str):
    """Reproduce el WAV por los altavoces (solo Windows, sin librerias extra)."""
    import winsound
    winsound.PlaySound(ruta, winsound.SND_FILENAME)
