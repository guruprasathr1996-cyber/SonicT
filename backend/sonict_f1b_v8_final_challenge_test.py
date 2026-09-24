"""
SonicT F1B V8 - Final Ensemble Challenge Test
==============================================
Purpose:
  Load the frozen V7, V7.2 and V7.3 models and apply the validated
  2-of-3 majority-vote rule to the untouched difficult challenge file.

NO TRAINING.
NO MODEL MODIFICATION.
NO SONICT F1-F5 MODIFICATION.
"""

from pathlib import Path
import subprocess, tempfile, warnings
import numpy as np
import librosa
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WavLMModel

warnings.filterwarnings("ignore")

SR = 16000
WIN = 6 * SR
MIN_WIN = SR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 6 if DEVICE.type == "cuda" else 2

CHALLENGE = Path(
    r"E:\Downloads\recording_6ee192c5-aab6-4a46-8afc-ee9985bb9aaf.mp3.mpeg"
)

WAVLM = Path(
    r"C:\Users\Guruprasath\.cache\huggingface\hub\models--microsoft--wavlm-base-plus\snapshots\4c66d4806a428f2e922ccfa1a962776e232d487b"
)

V7 = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_output\wavlm_cross_generator_v7.pt"
)

V72 = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_2_symmetric_codec\wavlm_cross_generator_v7_2_symmetric_codec.pt"
)

V73 = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_3_telephony_branch\wavlm_acoustic_telephony_antispof_v7_3.pt"
)

for p, name in [
    (CHALLENGE, "challenge"),
    (WAVLM, "local WavLM"),
    (V7, "V7"),
    (V72, "V7.2"),
    (V73, "V7.3"),
]:
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")

print("=" * 88)
print("SONICT F1B V8 - FINAL ENSEMBLE CHALLENGE TEST")
print("=" * 88)
print("Challenge:", CHALLENGE)
print("Device   :", DEVICE)
print("Rule     : 2-of-3 majority vote")
print("Training : NONE")

def normalize(y):
    y = np.asarray(y, np.float32)
    peak = np.max(np.abs(y)) if len(y) else 0.0
    return (y / peak if peak > 0 else y).astype(np.float32)

def load_audio(path):
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        return normalize(y)
    except Exception:
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp = Path(f.name)
        f.close()
        try:
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(tmp)
                ],
                check=True
            )
            y, _ = librosa.load(str(tmp), sr=SR, mono=True)
            return normalize(y)
        finally:
            tmp.unlink(missing_ok=True)

def make_windows(y):
    return [
        y[i:i + WIN]
        for i in range(0, len(y), WIN)
        if len(y[i:i + WIN]) >= MIN_WIN
    ]

print("\nLoading local WavLM...")
extractor = AutoFeatureExtractor.from_pretrained(
    str(WAVLM), local_files_only=True
)
wavlm = WavLMModel.from_pretrained(
    str(WAVLM), local_files_only=True
).to(DEVICE)
wavlm.eval()

for p in wavlm.parameters():
    p.requires_grad = False

def wavlm_embeddings(windows):
    outputs = []

    for i in range(0, len(windows), BATCH):
        batch = windows[i:i + BATCH]

        z = extractor(
            batch,
            sampling_rate=SR,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        x = z["input_values"].to(DEVICE)
        mask = z.get("attention_mask")

        if mask is not None:
            mask = mask.to(DEVICE)

        with torch.no_grad():
            h = wavlm(
                input_values=x,
                attention_mask=mask
            ).last_hidden_state

            try:
                fm = (
                    wavlm._get_feature_vector_attention_mask(
                        h.shape[1], mask
                    )
                    if mask is not None
                    else torch.ones(
                        h.shape[:2],
                        dtype=torch.bool,
                        device=DEVICE
                    )
                )
            except Exception:
                fm = torch.ones(
                    h.shape[:2],
                    dtype=torch.bool,
                    device=DEVICE
                )

            m = fm.unsqueeze(-1).float()
            count = m.sum(1).clamp(min=1)

            mean = (h * m).sum(1) / count
            var = (((h - mean[:, None]) ** 2) * m).sum(1) / count

            emb = torch.cat(
                [mean, torch.sqrt(var.clamp(min=1e-8))],
                dim=1
            )

            outputs.append(emb.cpu().numpy())

    return np.concatenate(outputs).astype(np.float32)

class MLP(nn.Module):
    def __init__(self, d=1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(.30),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(.20),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

class TelephonyMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(.35),
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(.30),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(.20),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

def load_standard(path):
    c = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    model = MLP().to(DEVICE)
    model.load_state_dict(c["model_state_dict"])
    model.eval()

    mean = np.asarray(c["feature_mean"], np.float32)
    std = np.asarray(c["feature_std"], np.float32)
    std[std < 1e-6] = 1.0

    return model, mean, std

print("Loading V7...")
m7, mu7, sd7 = load_standard(V7)

print("Loading V7.2...")
m72, mu72, sd72 = load_standard(V72)

print("Loading V7.3...")
c73 = torch.load(
    V73,
    map_location=DEVICE,
    weights_only=False
)

m73 = TelephonyMLP(
    int(c73["combined_feature_size"])
).to(DEVICE)

m73.load_state_dict(c73["model_state_dict"])
m73.eval()

mu73_w = np.asarray(c73["wavlm_mean"], np.float32)
sd73_w = np.asarray(c73["wavlm_std"], np.float32)
mu73_a = np.asarray(c73["acoustic_mean"], np.float32)
sd73_a = np.asarray(c73["acoustic_std"], np.float32)

sd73_w[sd73_w < 1e-6] = 1.0
sd73_a[sd73_a < 1e-6] = 1.0

def band_energy(y, low, high):
    S = np.abs(
        librosa.stft(
            y,
            n_fft=2048,
            hop_length=512
        )
    ) ** 2

    freqs = librosa.fft_frequencies(
        sr=SR,
        n_fft=2048
    )

    return float(
        S[(freqs >= low) & (freqs < high)].sum()
        / (S.sum() + 1e-12)
    )

def acoustic_features(y):
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=SR
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=SR
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=SR,
        roll_percent=.85
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=y
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y
    )[0]

    rms = librosa.feature.rms(
        y=y
    )[0]

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=13
    )

    features = [
        centroid.mean(), centroid.std(),
        bandwidth.mean(), bandwidth.std(),
        rolloff.mean(), rolloff.std(),
        flatness.mean(), flatness.std(),
        zcr.mean(), zcr.std(),
        rms.mean(), rms.std(),
        band_energy(y, 0, 1000),
        band_energy(y, 1000, 3000),
        band_energy(y, 3000, 4000),
        band_energy(y, 4000, 8000)
    ]

    for x in mfcc:
        features += [
            x.mean(),
            x.std()
        ]

    return np.asarray(
        features,
        dtype=np.float32
    )

def probability(model, x):
    with torch.no_grad():
        logits = model(
            torch.tensor(
                x,
                dtype=torch.float32,
                device=DEVICE
            )
        )

        return (
            torch.softmax(logits, dim=1)[:, 1]
            .cpu()
            .numpy()
        )

def summarize(name, probabilities):
    probabilities = np.asarray(probabilities)

    median = float(np.median(probabilities))
    mean = float(np.mean(probabilities))
    p90 = float(np.percentile(probabilities, 90))
    maximum = float(np.max(probabilities))

    prediction = (
        "DEEPFAKE"
        if median >= .50
        else "GENUINE"
    )

    print(f"\n{name}")
    print(f"Prediction : {prediction}")
    print(f"Mean DF    : {mean * 100:.2f}%")
    print(f"Median DF  : {median * 100:.2f}%")
    print(f"P90 DF     : {p90 * 100:.2f}%")
    print(f"Max DF     : {maximum * 100:.2f}%")
    print(
        "Suspicious : "
        f"{np.mean(probabilities >= .50) * 100:.2f}%"
    )

    return prediction, median

# ------------------------------------------------------------
# Run untouched challenge
# ------------------------------------------------------------
y = load_audio(CHALLENGE)
ws = make_windows(y)

print("\nDuration :", f"{len(y) / SR:.2f} sec")
print("Windows  :", len(ws))

E = wavlm_embeddings(ws)

p7 = probability(
    m7,
    (E - mu7) / sd7
)

p72 = probability(
    m72,
    (E - mu72) / sd72
)

A = np.stack(
    [acoustic_features(w) for w in ws]
)

X73 = np.concatenate(
    [
        (E - mu73_w) / sd73_w,
        (A - mu73_a) / sd73_a
    ],
    axis=1
)

p73 = probability(
    m73,
    X73
)

pred7, med7 = summarize("V7", p7)
pred72, med72 = summarize("V7.2", p72)
pred73, med73 = summarize("V7.3", p73)

predictions = [
    pred7,
    pred72,
    pred73
]

votes = sum(
    p == "DEEPFAKE"
    for p in predictions
)

ensemble = (
    "DEEPFAKE"
    if votes >= 2
    else "GENUINE"
)

print("\n" + "=" * 88)
print("FINAL F1B ENSEMBLE RESULT")
print("=" * 88)

print("V7 vote   :", pred7)
print("V7.2 vote :", pred72)
print("V7.3 vote :", pred73)

print(
    f"\nDeepfake votes: {votes}/3"
)

print(
    "FINAL F1B PREDICTION:",
    ensemble
)

if votes == 3:
    agreement = "FULL AGREEMENT"
elif votes == 0:
    agreement = "FULL AGREEMENT"
else:
    agreement = "MODEL DISAGREEMENT"

print(
    "Ensemble status:",
    agreement
)

print("\nNo training performed.")
print("No model files modified.")
print("SonicT F1-F5 were not modified.")
