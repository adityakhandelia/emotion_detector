import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import joblib
import numpy as np
import tempfile
import io
torchaudio.set_audio_backend("sox_io")
# ----------------------------
# Load model + label encoder
# ----------------------------
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out

# App config
st.title(" Emotion Detection from Speech (Upload / Record)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
input_dim = 40   # MFCC dimension
hidden_dim = 128
num_layers = 2
num_classes = 7   # adjust based on your training

model = BiLSTM(input_dim, hidden_dim, num_layers, num_classes)
model.load_state_dict(torch.load("bilstm_emotion_model2.pth", map_location=device))
model.to(device)
model.eval()

# Load label encoder
le = joblib.load("label_encoder2.pkl")

# ----------------------------
# Feature extraction function
# ----------------------------
resample_rate = 16000
resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=resample_rate)
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=resample_rate,
    n_mfcc=40,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
)

def extract_features(file_path):
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != resample_rate:
        waveform = resampler(waveform)
    mfcc = mfcc_transform(waveform)
    mfcc = mfcc.squeeze(0).transpose(0,1)  # shape [T, 40]
    return mfcc.unsqueeze(0)  # add batch dim

# ----------------------------
# Option Selector
# ----------------------------
option = st.radio("Choose input method:", ["Upload File", "Record Audio"])

if option == "Upload File":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(uploaded_file.read())
            temp_path = tmpfile.name

        # Predict
        features = extract_features(temp_path).to(device)
        with torch.no_grad():
            outputs = model(features)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            emotion = le.inverse_transform([predicted])[0]

        st.success(f"Predicted Emotion: **{emotion}**")

elif option == "Record Audio":
    from streamlit_mic_recorder import mic_recorder
    import av
    import numpy as np
    import torch

    st.info("Click the button below to record audio")

    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key="recorder"
    )

    if audio is not None:
        st.audio(audio['bytes'], format="audio/webm")  
        container = av.open(io.BytesIO(audio['bytes']))
        frames = []

        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.ndim == 1:  # mono
                arr = arr[np.newaxis, :]
            frames.append(arr)
        audio_np = np.concatenate(frames, axis=1).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(audio_np)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        sr = frame.sample_rate
        if sr != resample_rate:
            waveform = resampler(waveform)
        mfcc = mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0).transpose(0, 1).unsqueeze(0).to(device)  # [1, T, 40]
        with torch.no_grad():
            outputs = model(mfcc)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            emotion = le.inverse_transform([predicted])[0]

        st.success(f"Predicted Emotion: **{emotion}**")

