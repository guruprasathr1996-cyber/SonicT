from pathlib import Path
import json
import math
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoFeatureExtractor, WavLMModel

warnings.filterwarnings("ignore")

# ============================================================
# SONICT F1B V7.3 DIAGNOSTIC ANALYSIS
#
# PURPOSE:
#   Diagnose WHY:
#     - V7 detects the difficult compressed deepfake
#       but falsely flags genuine phone audio.
#     - V7.2 fixes phone genuine audio
#       but misses the same difficult deepfake.
#
# THIS SCRIPT DOES NOT TRAIN A NEW MODEL.
# IT DOES NOT MODIFY F1-F5.
#
# It compares:
#   1. V7 score
#   2. V7.2 score
#   3. WavLM embedding similarities/distances
#   4. Acoustic / spectral / telephony-channel features
#   5. MLAAD genuine vs MLAAD deepfake vs phone genuine
#      vs difficult compressed deepfake
#
# Outputs are saved as CSV + JSON for the V7.3 design decision.
# ============================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SR = 16000
WINDOW_SECONDS = 6.0
WINDOW_SAMPLES = int(SR * WINDOW_SECONDS)
MIN_SECONDS = 1.0
MIN_SAMPLES = int(SR * MIN_SECONDS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

WAVLM_DIR = Path(
    r"C:\Users\Guruprasath\.cache\huggingface\hub\models--microsoft--wavlm-base-plus\snapshots\4c66d4806a428f2e922ccfa1a962776e232d487b"
)

# Original V7.
V7_DIR = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_output"
)
V7_MODEL = V7_DIR / "wavlm_cross_generator_v7.pt"
V7_TEST_CACHE = V7_DIR / "wavlm_test_features_v7.npz"

# V7.2.
V72_DIR = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_2_symmetric_codec"
)
V72_MODEL = V72_DIR / "wavlm_cross_generator_v7_2_symmetric_codec.pt"
V72_PHONE_TRAIN_CACHE = V72_DIR / "phone_train_features_v7_2.npz"
V72_PHONE_VAL_CACHE = V72_DIR / "phone_val_features_v7_2.npz"
V72_PHONE_TEST_CACHE = V72_DIR / "phone_test_features_v7_2.npz"

PERSONAL_GENUINE_DIR = Path(
    r"H:\SonicT_Compression_Test\genuine"
)

DIFFICULT_DEEPFAKE = Path(
    r"E:\Downloads\recording_6ee192c5-aab6-4a46-8afc-ee9985bb9aaf.mp3.mpeg"
)

OUTPUT_DIR = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_3_diagnostic"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS = {
    ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg",
    ".opus", ".wma", ".mpeg", ".mpg", ".amr", ".3gp",
    ".3g2", ".webm", ".mka", ".aiff", ".aif", ".caf",
    ".mov", ".mkv", ".avi", ".ts", ".m2ts",
}

# How many MLAAD test files to use for raw acoustic comparison.
# WavLM score analysis still uses the full cached test set.
ACOUSTIC_SAMPLE_PER_CLASS = 100

# ============================================================
# CHECK FILES
# ============================================================

required = [
    (WAVLM_DIR, "Local WavLM"),
    (V7_MODEL, "V7 model"),
    (V7_TEST_CACHE, "V7 test cache"),
    (V72_MODEL, "V7.2 model"),
    (V72_PHONE_TRAIN_CACHE, "V7.2 phone train cache"),
    (V72_PHONE_VAL_CACHE, "V7.2 phone validation cache"),
    (V72_PHONE_TEST_CACHE, "V7.2 phone test cache"),
    (PERSONAL_GENUINE_DIR, "Personal genuine folder"),
    (DIFFICULT_DEEPFAKE, "Difficult deepfake"),
]

for path, label in required:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

print("=" * 82)
print("SONICT F1B V7.3 DIAGNOSTIC ANALYSIS")
print("=" * 82)
print(f"Device        : {DEVICE}")
print(f"V7 model      : {V7_MODEL}")
print(f"V7.2 model    : {V72_MODEL}")
print(f"Output folder : {OUTPUT_DIR}")

# ============================================================
# MODEL
# ============================================================

class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def load_classifier(checkpoint_path):
    ckpt = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)

    input_dim = int(
        ckpt.get(
            "embedding_size",
            len(mean),
        )
    )

    model = EmbeddingMLP(input_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, mean, std, ckpt


v7_model, v7_mean, v7_std, v7_ckpt = load_classifier(V7_MODEL)
v72_model, v72_mean, v72_std, v72_ckpt = load_classifier(V72_MODEL)

print("\nLoaded both classifiers successfully.")

# ============================================================
# LOAD WAVLM LOCALLY
# ============================================================

print("\nLoading frozen WavLM locally...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    str(WAVLM_DIR),
    local_files_only=True,
)

wavlm = WavLMModel.from_pretrained(
    str(WAVLM_DIR),
    local_files_only=True,
)

wavlm.to(DEVICE)
wavlm.eval()

for p in wavlm.parameters():
    p.requires_grad = False

HIDDEN_SIZE = int(wavlm.config.hidden_size)
EMBED_DIM = HIDDEN_SIZE * 2

print(f"WavLM hidden size : {HIDDEN_SIZE}")
print(f"Embedding size    : {EMBED_DIM}")

# ============================================================
# BASIC HELPERS
# ============================================================

def ffprobe_info(path):
    """
    Read container/codec information. If ffprobe cannot read a field,
    return None rather than failing the diagnostic.
    """
    path = Path(path)

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bit_rate",
        "-show_entries",
        "format=duration,bit_rate,format_name",
        "-of", "json",
        str(path),
    ]

    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
        )
        data = json.loads(out)

        stream = {}
        if data.get("streams"):
            stream = data["streams"][0]

        fmt = data.get("format", {})

        def safe_num(x, cast=float):
            try:
                return cast(x)
            except Exception:
                return None

        return {
            "codec": stream.get("codec_name"),
            "original_sample_rate": safe_num(
                stream.get("sample_rate"),
                int,
            ),
            "channels": safe_num(
                stream.get("channels"),
                int,
            ),
            "stream_bitrate": safe_num(
                stream.get("bit_rate"),
                int,
            ),
            "container_bitrate": safe_num(
                fmt.get("bit_rate"),
                int,
            ),
            "duration_seconds": safe_num(
                fmt.get("duration"),
                float,
            ),
            "format_name": fmt.get("format_name"),
        }

    except Exception:
        return {
            "codec": None,
            "original_sample_rate": None,
            "channels": None,
            "stream_bitrate": None,
            "container_bitrate": None,
            "duration_seconds": None,
            "format_name": None,
        }


def ffmpeg_to_temp_wav(input_path):
    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )
    temp_wav = Path(tmp.name)
    tmp.close()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(SR),
        "-c:a", "pcm_s16le",
        str(temp_wav),
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return temp_wav


def load_audio_robust(path):
    path = Path(path)

    try:
        y, _ = librosa.load(
            str(path),
            sr=SR,
            mono=True,
        )
        return np.asarray(y, dtype=np.float32)

    except Exception:
        tmp = None

        try:
            tmp = ffmpeg_to_temp_wav(path)
            y, _ = librosa.load(
                str(tmp),
                sr=SR,
                mono=True,
            )
            return np.asarray(y, dtype=np.float32)

        finally:
            if tmp is not None and tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass


def normalize_audio(y):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return y

    peak = float(np.max(np.abs(y)))

    if peak > 0:
        y = y / peak

    return y.astype(np.float32)


def make_windows(y):
    y = normalize_audio(y)

    windows = []

    for start in range(0, len(y), WINDOW_SAMPLES):
        chunk = y[start:start + WINDOW_SAMPLES]

        if len(chunk) >= MIN_SAMPLES:
            windows.append(chunk)

    return windows


def center_crop(y):
    y = normalize_audio(y)

    if len(y) > WINDOW_SAMPLES:
        start = (len(y) - WINDOW_SAMPLES) // 2
        y = y[start:start + WINDOW_SAMPLES]

    return y


# ============================================================
# WAVLM EMBEDDINGS
# ============================================================

def wavlm_embeddings(waveforms):
    inputs = feature_extractor(
        waveforms,
        sampling_rate=SR,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    values = inputs["input_values"].to(DEVICE)
    mask = inputs.get("attention_mask")

    if mask is not None:
        mask = mask.to(DEVICE)

    with torch.no_grad():
        output = wavlm(
            input_values=values,
            attention_mask=mask,
        )

        hidden = output.last_hidden_state

        if mask is not None:
            try:
                feat_mask = wavlm._get_feature_vector_attention_mask(
                    hidden.shape[1],
                    mask,
                )
            except Exception:
                feat_mask = torch.ones(
                    hidden.shape[:2],
                    dtype=torch.bool,
                    device=hidden.device,
                )
        else:
            feat_mask = torch.ones(
                hidden.shape[:2],
                dtype=torch.bool,
                device=hidden.device,
            )

        m = feat_mask.unsqueeze(-1).float()
        count = m.sum(dim=1).clamp(min=1.0)

        mean = (hidden * m).sum(dim=1) / count

        var = (
            ((hidden - mean.unsqueeze(1)) ** 2) * m
        ).sum(dim=1) / count

        std = torch.sqrt(
            var.clamp(min=1e-8)
        )

        emb = torch.cat(
            [mean, std],
            dim=1,
        )

    return emb.cpu().numpy().astype(np.float32)


def classifier_prob(model, mean, std, X):
    Xs = (
        (X.astype(np.float32) - mean)
        / (std + 1e-6)
    ).astype(np.float32)

    with torch.no_grad():
        xb = torch.tensor(
            Xs,
            dtype=torch.float32,
            device=DEVICE,
        )

        logits = model(xb)

        probs = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

    return probs.cpu().numpy().astype(np.float32)


# ============================================================
# LOAD CACHED MLAAD + PHONE EMBEDDINGS
# ============================================================

def load_npz_cache(path):
    data = np.load(
        path,
        allow_pickle=True,
    )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    meta = pd.DataFrame(
        json.loads(str(data["meta_json"].item()))
    )

    return X, y, meta


X_test, y_test, test_meta = load_npz_cache(V7_TEST_CACHE)

phone_parts = []

for split_name, cache in [
    ("train", V72_PHONE_TRAIN_CACHE),
    ("validation", V72_PHONE_VAL_CACHE),
    ("test", V72_PHONE_TEST_CACHE),
]:
    Xp, yp, mp = load_npz_cache(cache)

    mp = mp.copy()
    mp["phone_split"] = split_name

    phone_parts.append((Xp, yp, mp))

X_phone = np.concatenate(
    [x[0] for x in phone_parts],
    axis=0,
).astype(np.float32)

y_phone = np.concatenate(
    [x[1] for x in phone_parts],
    axis=0,
).astype(np.int64)

phone_meta = pd.concat(
    [x[2] for x in phone_parts],
    ignore_index=True,
)

print("\nCached embeddings:")
print(f"MLAAD test windows/files : {len(X_test)}")
print(f"Phone windows            : {len(X_phone)}")

# ============================================================
# COMPARE V7 VS V7.2 ON MLAAD AND PHONE
# ============================================================

v7_test_prob = classifier_prob(
    v7_model,
    v7_mean,
    v7_std,
    X_test,
)

v72_test_prob = classifier_prob(
    v72_model,
    v72_mean,
    v72_std,
    X_test,
)

v7_phone_prob = classifier_prob(
    v7_model,
    v7_mean,
    v7_std,
    X_phone,
)

v72_phone_prob = classifier_prob(
    v72_model,
    v72_mean,
    v72_std,
    X_phone,
)

test_scores = test_meta.copy()
test_scores["v7_deepfake_probability"] = v7_test_prob
test_scores["v72_deepfake_probability"] = v72_test_prob
test_scores["score_shift_v72_minus_v7"] = (
    v72_test_prob - v7_test_prob
)

test_scores.to_csv(
    OUTPUT_DIR / "mlaad_test_v7_vs_v72_scores.csv",
    index=False,
)

phone_scores = phone_meta.copy()
phone_scores["v7_deepfake_probability"] = v7_phone_prob
phone_scores["v72_deepfake_probability"] = v72_phone_prob
phone_scores["score_shift_v72_minus_v7"] = (
    v72_phone_prob - v7_phone_prob
)

phone_scores.to_csv(
    OUTPUT_DIR / "phone_v7_vs_v72_window_scores.csv",
    index=False,
)

# ============================================================
# CHALLENGE EMBEDDING + SCORES
# ============================================================

challenge_y = load_audio_robust(DIFFICULT_DEEPFAKE)
challenge_windows = make_windows(challenge_y)

challenge_embeddings = []

batch_size = 2 if DEVICE.type == "cpu" else 8

for i in range(0, len(challenge_windows), batch_size):
    challenge_embeddings.append(
        wavlm_embeddings(
            challenge_windows[i:i + batch_size]
        )
    )

X_challenge = np.concatenate(
    challenge_embeddings,
    axis=0,
).astype(np.float32)

v7_challenge_prob = classifier_prob(
    v7_model,
    v7_mean,
    v7_std,
    X_challenge,
)

v72_challenge_prob = classifier_prob(
    v72_model,
    v72_mean,
    v72_std,
    X_challenge,
)

challenge_score_df = pd.DataFrame({
    "window_index": np.arange(len(X_challenge)),
    "start_seconds": np.arange(len(X_challenge)) * WINDOW_SECONDS,
    "v7_deepfake_probability": v7_challenge_prob,
    "v72_deepfake_probability": v72_challenge_prob,
    "score_shift_v72_minus_v7": (
        v72_challenge_prob - v7_challenge_prob
    ),
})

challenge_score_df.to_csv(
    OUTPUT_DIR / "challenge_v7_vs_v72_window_scores.csv",
    index=False,
)

# ============================================================
# EMBEDDING CENTROIDS
# ============================================================

def cosine_similarity_rows(X, centroid):
    Xn = X / (
        np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    )

    cn = centroid / (
        np.linalg.norm(centroid) + 1e-8
    )

    return np.sum(
        Xn * cn[None, :],
        axis=1,
    )


mlaad_genuine = X_test[y_test == 0]
mlaad_deepfake = X_test[y_test == 1]

centroid_genuine = mlaad_genuine.mean(axis=0)
centroid_deepfake = mlaad_deepfake.mean(axis=0)
centroid_phone = X_phone.mean(axis=0)

def centroid_metrics(X, domain_name):
    return pd.DataFrame({
        "domain": [domain_name] * len(X),
        "cos_to_mlaad_genuine": cosine_similarity_rows(
            X,
            centroid_genuine,
        ),
        "cos_to_mlaad_deepfake": cosine_similarity_rows(
            X,
            centroid_deepfake,
        ),
        "cos_to_phone_genuine": cosine_similarity_rows(
            X,
            centroid_phone,
        ),
    })


embedding_diag = pd.concat(
    [
        centroid_metrics(
            mlaad_genuine,
            "mlaad_genuine",
        ),
        centroid_metrics(
            mlaad_deepfake,
            "mlaad_deepfake",
        ),
        centroid_metrics(
            X_phone,
            "phone_genuine",
        ),
        centroid_metrics(
            X_challenge,
            "challenge_deepfake",
        ),
    ],
    ignore_index=True,
)

embedding_diag.to_csv(
    OUTPUT_DIR / "embedding_centroid_diagnostics.csv",
    index=False,
)

embedding_summary = (
    embedding_diag
    .groupby("domain")
    .agg(
        cos_to_mlaad_genuine_mean=(
            "cos_to_mlaad_genuine",
            "mean",
        ),
        cos_to_mlaad_genuine_median=(
            "cos_to_mlaad_genuine",
            "median",
        ),
        cos_to_mlaad_deepfake_mean=(
            "cos_to_mlaad_deepfake",
            "mean",
        ),
        cos_to_mlaad_deepfake_median=(
            "cos_to_mlaad_deepfake",
            "median",
        ),
        cos_to_phone_genuine_mean=(
            "cos_to_phone_genuine",
            "mean",
        ),
        cos_to_phone_genuine_median=(
            "cos_to_phone_genuine",
            "median",
        ),
    )
    .reset_index()
)

embedding_summary.to_csv(
    OUTPUT_DIR / "embedding_centroid_summary.csv",
    index=False,
)

# ============================================================
# ACOUSTIC FEATURES
# ============================================================

def effective_bandwidth_hz(y, sr=SR, energy_fraction=0.95):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return np.nan

    n_fft = int(2 ** math.ceil(math.log2(max(len(y), 2048))))

    spectrum = np.abs(
        np.fft.rfft(y, n=n_fft)
    ) ** 2

    freqs = np.fft.rfftfreq(
        n_fft,
        d=1.0 / sr,
    )

    total = float(np.sum(spectrum))

    if total <= 0:
        return 0.0

    cumulative = np.cumsum(spectrum) / total

    idx = int(
        np.searchsorted(
            cumulative,
            energy_fraction,
        )
    )

    idx = min(
        idx,
        len(freqs) - 1,
    )

    return float(freqs[idx])


def band_energy_ratio(y, low_hz, high_hz, sr=SR):
    n_fft = 2048

    spec = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=512,
        )
    ) ** 2

    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=n_fft,
    )

    total = float(
        np.sum(spec)
    ) + 1e-12

    mask = (
        (freqs >= low_hz)
        & (freqs < high_hz)
    )

    return float(
        np.sum(spec[mask]) / total
    )


def acoustic_features(path, domain, label_name):
    path = Path(path)
    info = ffprobe_info(path)

    y = load_audio_robust(path)
    y = normalize_audio(y)

    if len(y) < MIN_SAMPLES:
        raise ValueError(
            "Audio shorter than 1 second"
        )

    # Use up to 30 seconds for stable file-level acoustics.
    max_len = int(SR * 30)

    if len(y) > max_len:
        start = (len(y) - max_len) // 2
        ya = y[start:start + max_len]
    else:
        ya = y

    centroid = librosa.feature.spectral_centroid(
        y=ya,
        sr=SR,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=ya,
        sr=SR,
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=ya,
        sr=SR,
        roll_percent=0.85,
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=ya
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        ya
    )[0]

    rms = librosa.feature.rms(
        y=ya
    )[0]

    mfcc = librosa.feature.mfcc(
        y=ya,
        sr=SR,
        n_mfcc=13,
    )

    row = {
        "domain": domain,
        "label_name": label_name,
        "path": str(path),
        "filename": path.name,

        "codec": info["codec"],
        "original_sample_rate": info["original_sample_rate"],
        "channels": info["channels"],
        "stream_bitrate": info["stream_bitrate"],
        "container_bitrate": info["container_bitrate"],
        "duration_seconds": info["duration_seconds"],
        "format_name": info["format_name"],

        "effective_bandwidth_95_hz": effective_bandwidth_hz(ya),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "rolloff85_mean": float(np.mean(rolloff)),
        "rolloff85_std": float(np.std(rolloff)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flatness_std": float(np.std(flatness)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),

        "energy_0_1k": band_energy_ratio(
            ya,
            0,
            1000,
        ),
        "energy_1_3k": band_energy_ratio(
            ya,
            1000,
            3000,
        ),
        "energy_3_4k": band_energy_ratio(
            ya,
            3000,
            4000,
        ),
        "energy_4_8k": band_energy_ratio(
            ya,
            4000,
            8000,
        ),
    }

    for i in range(13):
        row[f"mfcc_{i + 1}_mean"] = float(
            np.mean(mfcc[i])
        )
        row[f"mfcc_{i + 1}_std"] = float(
            np.std(mfcc[i])
        )

    return row


# ============================================================
# SELECT MLAAD RAW FILES FOR ACOUSTIC DIAGNOSTIC
# ============================================================

rng = np.random.default_rng(SEED)

test_meta_work = test_meta.copy()

if "path" not in test_meta_work.columns:
    raise RuntimeError(
        "V7 test cache metadata does not contain raw file paths."
    )

genuine_meta = test_meta_work[
    test_meta_work["label"].astype(int) == 0
].copy()

deepfake_meta = test_meta_work[
    test_meta_work["label"].astype(int) == 1
].copy()

g_n = min(
    ACOUSTIC_SAMPLE_PER_CLASS,
    len(genuine_meta),
)

d_n = min(
    ACOUSTIC_SAMPLE_PER_CLASS,
    len(deepfake_meta),
)

g_sample = genuine_meta.iloc[
    rng.choice(
        len(genuine_meta),
        size=g_n,
        replace=False,
    )
]

d_sample = deepfake_meta.iloc[
    rng.choice(
        len(deepfake_meta),
        size=d_n,
        replace=False,
    )
]

phone_files = sorted([
    p
    for p in PERSONAL_GENUINE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in AUDIO_EXTS
])

acoustic_jobs = []

for _, row in g_sample.iterrows():
    acoustic_jobs.append(
        (
            Path(row["path"]),
            "mlaad_genuine",
            "genuine",
        )
    )

for _, row in d_sample.iterrows():
    acoustic_jobs.append(
        (
            Path(row["path"]),
            "mlaad_deepfake",
            "deepfake",
        )
    )

for p in phone_files:
    acoustic_jobs.append(
        (
            p,
            "phone_genuine",
            "genuine",
        )
    )

acoustic_jobs.append(
    (
        DIFFICULT_DEEPFAKE,
        "challenge_deepfake",
        "deepfake",
    )
)

print("\n" + "=" * 82)
print("RAW ACOUSTIC DIAGNOSTIC")
print("=" * 82)
print(
    f"Files to analyze: {len(acoustic_jobs)} "
    f"({g_n} MLAAD genuine + {d_n} MLAAD deepfake "
    f"+ {len(phone_files)} phone + 1 challenge)"
)

acoustic_rows = []
acoustic_failures = []

for i, (path, domain, label_name) in enumerate(
    acoustic_jobs,
    start=1,
):
    try:
        row = acoustic_features(
            path,
            domain,
            label_name,
        )

        acoustic_rows.append(row)

    except Exception as exc:
        acoustic_failures.append({
            "path": str(path),
            "domain": domain,
            "error": str(exc),
        })

    if i % 25 == 0 or i == len(acoustic_jobs):
        print(
            f"Acoustic analysis: "
            f"{i}/{len(acoustic_jobs)}"
        )

acoustic_df = pd.DataFrame(acoustic_rows)

acoustic_df.to_csv(
    OUTPUT_DIR / "acoustic_file_features.csv",
    index=False,
)

if acoustic_failures:
    pd.DataFrame(
        acoustic_failures
    ).to_csv(
        OUTPUT_DIR / "acoustic_failures.csv",
        index=False,
    )

# ============================================================
# ACOUSTIC DOMAIN SUMMARY
# ============================================================

numeric_cols = [
    c
    for c in acoustic_df.columns
    if c not in {
        "domain",
        "label_name",
        "path",
        "filename",
        "codec",
        "format_name",
    }
    and pd.api.types.is_numeric_dtype(acoustic_df[c])
]

domain_summary_rows = []

for domain, group in acoustic_df.groupby("domain"):
    row = {
        "domain": domain,
        "n_files": len(group),
    }

    for col in numeric_cols:
        vals = pd.to_numeric(
            group[col],
            errors="coerce",
        ).dropna()

        if len(vals):
            row[f"{col}_median"] = float(
                vals.median()
            )
            row[f"{col}_mean"] = float(
                vals.mean()
            )

    domain_summary_rows.append(row)

domain_summary = pd.DataFrame(
    domain_summary_rows
)

domain_summary.to_csv(
    OUTPUT_DIR / "acoustic_domain_summary.csv",
    index=False,
)

# ============================================================
# CHALLENGE Z-SCORE AGAINST PHONE GENUINE
# ============================================================

phone_acoustic = acoustic_df[
    acoustic_df["domain"] == "phone_genuine"
].copy()

challenge_acoustic = acoustic_df[
    acoustic_df["domain"] == "challenge_deepfake"
].copy()

z_rows = []

if len(challenge_acoustic) == 1:
    challenge_row = challenge_acoustic.iloc[0]

    for col in numeric_cols:
        phone_vals = pd.to_numeric(
            phone_acoustic[col],
            errors="coerce",
        ).dropna()

        try:
            challenge_value = float(
                challenge_row[col]
            )
        except Exception:
            continue

        if len(phone_vals) < 3:
            continue

        mean = float(phone_vals.mean())
        std = float(phone_vals.std(ddof=0))

        if std <= 1e-12:
            z = np.nan
        else:
            z = (
                challenge_value - mean
            ) / std

        z_rows.append({
            "feature": col,
            "challenge_value": challenge_value,
            "phone_mean": mean,
            "phone_std": std,
            "z_score_vs_phone": float(z)
            if np.isfinite(z)
            else np.nan,
            "abs_z_score": float(abs(z))
            if np.isfinite(z)
            else np.nan,
        })

z_df = pd.DataFrame(z_rows)

if len(z_df):
    z_df = z_df.sort_values(
        "abs_z_score",
        ascending=False,
    )

z_df.to_csv(
    OUTPUT_DIR / "challenge_vs_phone_acoustic_zscores.csv",
    index=False,
)

# ============================================================
# SCORE SUMMARIES
# ============================================================

def probability_summary(name, probs):
    probs = np.asarray(probs, dtype=np.float32)

    return {
        "domain": name,
        "n": int(len(probs)),
        "mean_df_probability": float(np.mean(probs)),
        "median_df_probability": float(np.median(probs)),
        "p75_df_probability": float(np.percentile(probs, 75)),
        "p90_df_probability": float(np.percentile(probs, 90)),
        "max_df_probability": float(np.max(probs)),
        "ratio_ge_0_50": float(np.mean(probs >= 0.50)),
        "ratio_ge_0_70": float(np.mean(probs >= 0.70)),
    }


score_summary_rows = []

for model_name, probs, labels in [
    ("V7_MLAAD_genuine", v7_test_prob[y_test == 0], None),
    ("V7_MLAAD_deepfake", v7_test_prob[y_test == 1], None),
    ("V7_phone_genuine", v7_phone_prob, None),
    ("V7_challenge", v7_challenge_prob, None),

    ("V7.2_MLAAD_genuine", v72_test_prob[y_test == 0], None),
    ("V7.2_MLAAD_deepfake", v72_test_prob[y_test == 1], None),
    ("V7.2_phone_genuine", v72_phone_prob, None),
    ("V7.2_challenge", v72_challenge_prob, None),
]:
    score_summary_rows.append(
        probability_summary(
            model_name,
            probs,
        )
    )

score_summary_df = pd.DataFrame(
    score_summary_rows
)

score_summary_df.to_csv(
    OUTPUT_DIR / "v7_vs_v72_score_summary.csv",
    index=False,
)

# ============================================================
# PRINT HIGH-VALUE FINDINGS
# ============================================================

print("\n" + "=" * 82)
print("V7 vs V7.2 SCORE COMPARISON")
print("=" * 82)

print(
    f"V7 phone median DF       : "
    f"{np.median(v7_phone_prob) * 100:.2f}%"
)
print(
    f"V7.2 phone median DF     : "
    f"{np.median(v72_phone_prob) * 100:.2f}%"
)

print(
    f"V7 challenge median DF   : "
    f"{np.median(v7_challenge_prob) * 100:.2f}%"
)
print(
    f"V7.2 challenge median DF : "
    f"{np.median(v72_challenge_prob) * 100:.2f}%"
)

print(
    f"V7 challenge P90 DF      : "
    f"{np.percentile(v7_challenge_prob, 90) * 100:.2f}%"
)
print(
    f"V7.2 challenge P90 DF    : "
    f"{np.percentile(v72_challenge_prob, 90) * 100:.2f}%"
)

print("\n" + "=" * 82)
print("EMBEDDING CENTROID SUMMARY")
print("=" * 82)
print(
    embedding_summary.to_string(
        index=False
    )
)

print("\n" + "=" * 82)
print("TOP CHALLENGE-vs-PHONE ACOUSTIC DIFFERENCES")
print("=" * 82)

if len(z_df):
    display_cols = [
        "feature",
        "challenge_value",
        "phone_mean",
        "phone_std",
        "z_score_vs_phone",
    ]

    print(
        z_df[
            display_cols
        ]
        .head(15)
        .to_string(index=False)
    )
else:
    print("No valid z-score features available.")

# ============================================================
# SIMPLE DIAGNOSTIC INTERPRETATION
# ============================================================

interpretation = []

phone_shift = float(
    np.median(v72_phone_prob)
    - np.median(v7_phone_prob)
)

challenge_shift = float(
    np.median(v72_challenge_prob)
    - np.median(v7_challenge_prob)
)

interpretation.append(
    {
        "finding":
        "phone_score_shift_v72_minus_v7",
        "value": phone_shift,
        "meaning":
        "Negative means V7.2 moved genuine phone audio toward Genuine.",
    }
)

interpretation.append(
    {
        "finding":
        "challenge_score_shift_v72_minus_v7",
        "value": challenge_shift,
        "meaning":
        "Negative means V7.2 also moved the difficult deepfake toward Genuine.",
    }
)

if len(z_df):
    for _, row in z_df.head(10).iterrows():
        interpretation.append(
            {
                "finding":
                f"challenge_phone_difference::{row['feature']}",
                "value":
                None
                if pd.isna(row["z_score_vs_phone"])
                else float(row["z_score_vs_phone"]),
                "meaning":
                "Challenge acoustic z-score relative to personal genuine phone audio.",
            }
        )

with open(
    OUTPUT_DIR / "diagnostic_interpretation.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        interpretation,
        f,
        indent=2,
        ensure_ascii=False,
    )

# ============================================================
# SAVE MASTER SUMMARY
# ============================================================

master_summary = {
    "experiment":
    "SonicT F1B V7.3 diagnostic only",

    "purpose":
    "Compare V7 and V7.2 and identify evidence separating genuine phone audio from the difficult compressed deepfake.",

    "v7_phone_median_df_probability":
    float(np.median(v7_phone_prob)),

    "v72_phone_median_df_probability":
    float(np.median(v72_phone_prob)),

    "v7_challenge_median_df_probability":
    float(np.median(v7_challenge_prob)),

    "v72_challenge_median_df_probability":
    float(np.median(v72_challenge_prob)),

    "v7_challenge_p90_df_probability":
    float(np.percentile(v7_challenge_prob, 90)),

    "v72_challenge_p90_df_probability":
    float(np.percentile(v72_challenge_prob, 90)),

    "challenge_windows":
    int(len(X_challenge)),

    "phone_embedding_windows":
    int(len(X_phone)),

    "mlaad_test_embeddings":
    int(len(X_test)),

    "raw_acoustic_files_analyzed":
    int(len(acoustic_df)),

    "output_folder":
    str(OUTPUT_DIR),
}

with open(
    OUTPUT_DIR / "v7_3_master_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        master_summary,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n" + "=" * 82)
print("V7.3 DIAGNOSTIC COMPLETE")
print("=" * 82)

print(
    "No new model was trained. "
    "V7 and V7.2 checkpoints were not modified."
)

print("\nMost important output files:")
print(
    OUTPUT_DIR / "v7_vs_v72_score_summary.csv"
)
print(
    OUTPUT_DIR / "embedding_centroid_summary.csv"
)
print(
    OUTPUT_DIR / "challenge_vs_phone_acoustic_zscores.csv"
)
print(
    OUTPUT_DIR / "acoustic_domain_summary.csv"
)
print(
    OUTPUT_DIR / "v7_3_master_summary.json"
)

print("\nNext decision:")
print(
    "Use these diagnostics to decide whether V7.3 should be "
    "a dual-branch detector, a calibrated fusion model, or "
    "a new telephony-specific anti-spoof branch."
)
