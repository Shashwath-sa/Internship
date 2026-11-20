"""Speaker Identification demo (Streamlit).

Two 1D CNN models (Log-STFT and MFCC features) identify which of 20 enrolled
LibriSpeech speakers is talking. Closed-set: an unknown voice is still matched
to one of the 20.
"""
import os
import tempfile

import librosa
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

SPEAKER_IDS = [26, 40, 83, 118, 248, 1098, 1116, 1578, 1594, 1723,
               1743, 1963, 2416, 2893, 4830, 5393, 5808, 6181, 6836, 6848]
TARGET_SR = 16000
TARGET_LENGTH = TARGET_SR * 5
CONFIDENCE_THRESHOLD = 0.60
HERE = os.path.dirname(os.path.abspath(__file__))


class CNN1D(nn.Module):
    def __init__(self, input_channels=257, num_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_models():
    models = {}
    for name, fname, channels in [("Log-STFT", "best_stft_model.pth", 257),
                                  ("MFCC", "best_mfcc_model.pth", 13)]:
        m = CNN1D(input_channels=channels, num_classes=len(SPEAKER_IDS))
        m.load_state_dict(torch.load(os.path.join(HERE, "models", fname), map_location="cpu"))
        m.eval()
        models[name] = m
    return models


def standardize(a):
    std = np.std(a)
    return (a - np.mean(a)) / (std if std > 1e-8 else 1.0)


def log_stft(signal):
    s = librosa.stft(signal, n_fft=512, hop_length=160, win_length=400, window="hann")
    return standardize(librosa.amplitude_to_db(np.abs(s), ref=np.max))


def mfcc(signal):
    m = librosa.feature.mfcc(y=signal, sr=TARGET_SR, n_mfcc=13, n_fft=512, hop_length=160, win_length=400)
    return standardize(m)


EXTRACTORS = {"Log-STFT": log_stft, "MFCC": mfcc}


def load_signal(data: bytes, suffix: str):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        signal, _ = librosa.load(path, sr=TARGET_SR)
    finally:
        os.remove(path)
    return signal


def predict(signal):
    signal = librosa.util.normalize(signal)
    if len(signal) > TARGET_LENGTH:
        signal = signal[:TARGET_LENGTH]
    else:
        signal = np.pad(signal, (0, TARGET_LENGTH - len(signal)))
    models = load_models()
    out = {}
    for name, model in models.items():
        x = torch.from_numpy(EXTRACTORS[name](signal).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out[name] = torch.softmax(model(x), dim=1)[0].numpy()
    return out


def show_results(probs):
    best = {n: int(np.argmax(p)) for n, p in probs.items()}
    a, b = best["Log-STFT"], best["MFCC"]
    conf_min = min(probs["Log-STFT"][a], probs["MFCC"][b])
    if a == b and conf_min >= CONFIDENCE_THRESHOLD:
        st.success(f"Both models agree: **Speaker {SPEAKER_IDS[a]}**")
    elif a == b:
        st.warning(f"Both models say Speaker {SPEAKER_IDS[a]}, but confidence is low. "
                   "The speaker may not be enrolled.")
    else:
        st.warning(f"The models disagree (Log-STFT: Speaker {SPEAKER_IDS[a]}, "
                   f"MFCC: Speaker {SPEAKER_IDS[b]}). Treat this result as uncertain.")

    cols = st.columns(2)
    for col, (name, p) in zip(cols, probs.items()):
        with col:
            st.subheader(name)
            for i in np.argsort(p)[::-1][:3]:
                st.write(f"Speaker {SPEAKER_IDS[i]}: {p[i] * 100:.1f}%")
                st.progress(float(p[i]))


st.set_page_config(page_title="Speaker Identification", page_icon="🎙️")
st.title("Speaker Identification")
st.write("Record or upload a few seconds of speech. Two 1D CNN models (Log-STFT and MFCC features) "
         "predict which of **20 enrolled LibriSpeech speakers** is talking.")
st.info("This is a closed-set system: a voice that is not enrolled will still be matched to one of the 20 "
        "speakers, so read the agreement and confidence message.")

tab_rec, tab_up, tab_ex = st.tabs(["Record", "Upload", "Examples"])
signal = None

with tab_rec:
    rec = st.audio_input("Record 3 to 5 seconds of speech")
    if rec is not None:
        signal = load_signal(rec.getvalue(), ".wav")

with tab_up:
    up = st.file_uploader("Upload a clip", type=["wav", "flac", "mp3", "ogg"])
    if up is not None:
        signal = load_signal(up.getvalue(), os.path.splitext(up.name)[1] or ".wav")

with tab_ex:
    ex_dir = os.path.join(HERE, "examples")
    files = sorted(f for f in os.listdir(ex_dir) if f.lower().endswith((".flac", ".wav"))) if os.path.isdir(ex_dir) else []
    if files:
        choice = st.selectbox("Sample clip", files)
        st.audio(os.path.join(ex_dir, choice))
        if st.button("Identify this sample"):
            signal = librosa.load(os.path.join(ex_dir, choice), sr=TARGET_SR)[0]
    else:
        st.write("No example clips were added.")

if signal is not None:
    if len(signal) < TARGET_SR:
        st.error("The clip is too short. Please use at least 1 second of speech.")
    else:
        with st.spinner("Analysing..."):
            show_results(predict(signal))
