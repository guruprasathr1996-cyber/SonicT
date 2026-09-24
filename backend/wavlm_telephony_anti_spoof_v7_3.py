from pathlib import Path
import json
import math
import re
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

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

# ============================================================
# SONICT F1B V7.3
# TELEPHONY-SPECIFIC ANTI-SPOOF BRANCH
# ============================================================
#
# PURPOSE
# -------
# Train a SEPARATE binary Genuine / Deepfake branch for
# narrowband / telephone / heavily compressed audio.
#
# This branch combines:
#   1. Frozen WavLM mean + std embedding  -> 1536 dims
#   2. Acoustic / spectral features       -> 42 dims
#
# IMPORTANT
# ---------
# - Does NOT overwrite V7 / V7.1 / V7.2.
# - Does NOT modify SonicT F1-F5.
# - Does NOT use the difficult challenge file for training.
# - Does NOT use phone validation/test speakers for training.
# - Does NOT use bitrate/codec/sample-rate as classifier inputs.
#
# TRAINING
# --------
# - MLAAD TRAIN genuine -> telephony transform
# - MLAAD TRAIN deepfake -> same telephony transform
# - Personal phone TRAIN speakers -> genuine
#
# VALIDATION
# ----------
# - MLAAD VALIDATION genuine/deepfake -> same telephony transform
# - Personal phone VALIDATION speakers -> genuine
#
# FINAL TEST
# ----------
# - MLAAD FINAL TEST unseen generators -> same telephony transform
# - Personal phone TEST speakers -> genuine
# - Difficult compressed deepfake remains completely untouched
#
# ============================================================

# ============================================================
# REPRODUCIBILITY / DEVICE
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# AUDIO / MODEL SETTINGS
# ============================================================

SR = 16000

WINDOW_SECONDS = 6.0
WINDOW_SAMPLES = int(
    SR * WINDOW_SECONDS
)

MIN_SECONDS = 1.0
MIN_SAMPLES = int(
    SR * MIN_SECONDS
)

# Phone files can be long.
PHONE_MAX_WINDOWS_PER_FILE = 8

# WavLM extraction batch size.
WAVLM_BATCH_SIZE = (
    6 if DEVICE.type == "cuda" else 2
)

# Training.
TRAIN_BATCH_SIZE = 64
EPOCHS = 35
PATIENCE = 7
LEARNING_RATE = 8e-4

# Personal phone genuine training influence.
# Keep conservative: V7.1/V7.2 showed that excessive phone
# adaptation can move the difficult deepfake toward Genuine.
PHONE_TRAIN_REPEAT = 1

# ============================================================
# TELEPHONY DATASET SIZES
# ============================================================

# V7.2 already created a balanced ~600-file TRAIN telephony set.
# Reuse its WavLM embeddings and regenerate acoustic features.
#
# Validation/test are freshly created from generator-disjoint
# MLAAD validation/test sources.

VAL_GENUINE_FILES = 150
VAL_DEEPFAKE_FILES = 150

TEST_GENUINE_FILES = 150
TEST_DEEPFAKE_FILES = 150

# ============================================================
# PATHS
# ============================================================

WAVLM_DIR = Path(
    r"C:\Users\Guruprasath\.cache\huggingface\hub\models--microsoft--wavlm-base-plus\snapshots\4c66d4806a428f2e922ccfa1a962776e232d487b"
)

V7_CACHE_DIR = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_output"
)

V7_TRAIN_CACHE = (
    V7_CACHE_DIR / "wavlm_train_features_v7.npz"
)

V7_VAL_CACHE = (
    V7_CACHE_DIR / "wavlm_validation_features_v7.npz"
)

V7_TEST_CACHE = (
    V7_CACHE_DIR / "wavlm_test_features_v7.npz"
)

V72_DIR = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_2_symmetric_codec"
)

V72_CODEC_CACHE = (
    V72_DIR / "symmetric_codec_aug_features_v7_2.npz"
)

V72_CODEC_MANIFEST = (
    V72_DIR / "symmetric_codec_aug_manifest_v7_2.csv"
)

V72_PHONE_SPLIT = (
    V72_DIR / "phone_speaker_split_v7_2.csv"
)

V72_PHONE_TRAIN_CACHE = (
    V72_DIR / "phone_train_features_v7_2.npz"
)

V72_PHONE_VAL_CACHE = (
    V72_DIR / "phone_val_features_v7_2.npz"
)

V72_PHONE_TEST_CACHE = (
    V72_DIR / "phone_test_features_v7_2.npz"
)

PERSONAL_GENUINE_DIR = Path(
    r"H:\SonicT_Compression_Test\genuine"
)

DIFFICULT_DEEPFAKE = Path(
    r"E:\Downloads\recording_6ee192c5-aab6-4a46-8afc-ee9985bb9aaf.mp3.mpeg"
)

OUTPUT_DIR = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_3_telephony_branch"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    OUTPUT_DIR
    / "wavlm_acoustic_telephony_antispof_v7_3.pt"
)

TRAIN_FEATURE_CACHE = (
    OUTPUT_DIR
    / "telephony_train_combined_features_v7_3.npz"
)

VAL_FEATURE_CACHE = (
    OUTPUT_DIR
    / "telephony_validation_combined_features_v7_3.npz"
)

TEST_FEATURE_CACHE = (
    OUTPUT_DIR
    / "telephony_test_combined_features_v7_3.npz"
)

PHONE_TRAIN_COMBINED_CACHE = (
    OUTPUT_DIR
    / "phone_train_combined_features_v7_3.npz"
)

PHONE_VAL_COMBINED_CACHE = (
    OUTPUT_DIR
    / "phone_validation_combined_features_v7_3.npz"
)

PHONE_TEST_COMBINED_CACHE = (
    OUTPUT_DIR
    / "phone_test_combined_features_v7_3.npz"
)

VAL_SELECTION_CSV = (
    OUTPUT_DIR
    / "telephony_validation_selection_v7_3.csv"
)

TEST_SELECTION_CSV = (
    OUTPUT_DIR
    / "telephony_test_selection_v7_3.csv"
)

FAILED_CSV = (
    OUTPUT_DIR
    / "failed_files_v7_3.csv"
)

# ============================================================
# INPUT CHECKS
# ============================================================

required_paths = [
    (WAVLM_DIR, "Local WavLM"),
    (V7_TRAIN_CACHE, "V7 train cache"),
    (V7_VAL_CACHE, "V7 validation cache"),
    (V7_TEST_CACHE, "V7 test cache"),
    (V72_CODEC_CACHE, "V7.2 codec cache"),
    (V72_CODEC_MANIFEST, "V7.2 codec manifest"),
    (V72_PHONE_SPLIT, "V7.2 phone split"),
    (V72_PHONE_TRAIN_CACHE, "V7.2 phone train cache"),
    (V72_PHONE_VAL_CACHE, "V7.2 phone validation cache"),
    (V72_PHONE_TEST_CACHE, "V7.2 phone test cache"),
    (PERSONAL_GENUINE_DIR, "Personal genuine folder"),
    (DIFFICULT_DEEPFAKE, "Difficult deepfake challenge"),
]

for path, label in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

print("=" * 82)
print("SONICT F1B V7.3 - TELEPHONY-SPECIFIC ANTI-SPOOF BRANCH")
print("=" * 82)

print(f"Device          : {DEVICE}")
print(f"Local WavLM     : {WAVLM_DIR}")
print(f"V7 cache folder : {V7_CACHE_DIR}")
print(f"V7.2 folder     : {V72_DIR}")
print(f"Output folder   : {OUTPUT_DIR}")

# ============================================================
# COMMON HELPERS
# ============================================================

failed_rows = []


def normalize_audio(y):
    y = np.asarray(
        y,
        dtype=np.float32,
    )

    if len(y) == 0:
        return y

    peak = float(
        np.max(np.abs(y))
    )

    if peak > 0:
        y = y / peak

    return y.astype(
        np.float32
    )


def ffmpeg_to_temp_wav(input_path):
    """
    Decode any FFmpeg-supported audio/container into
    16 kHz mono PCM16 WAV.
    """
    input_path = Path(input_path)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )

    temp_wav = Path(
        tmp.name
    )

    tmp.close()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-c:a",
        "pcm_s16le",
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
    """
    librosa first; FFmpeg fallback.
    """
    path = Path(path)

    try:
        y, _ = librosa.load(
            str(path),
            sr=SR,
            mono=True,
        )

        y = normalize_audio(y)

        if len(y) < MIN_SAMPLES:
            raise ValueError(
                "Audio shorter than 1 second"
            )

        return y

    except Exception:
        tmp = None

        try:
            tmp = ffmpeg_to_temp_wav(
                path
            )

            y, _ = librosa.load(
                str(tmp),
                sr=SR,
                mono=True,
            )

            y = normalize_audio(y)

            if len(y) < MIN_SAMPLES:
                raise ValueError(
                    "Audio shorter than 1 second"
                )

            return y

        finally:
            if (
                tmp is not None
                and tmp.exists()
            ):
                try:
                    tmp.unlink()
                except Exception:
                    pass


def center_crop_6s(y):
    y = normalize_audio(y)

    if len(y) > WINDOW_SAMPLES:
        start = (
            len(y) - WINDOW_SAMPLES
        ) // 2

        y = y[
            start:
            start + WINDOW_SAMPLES
        ]

    return y


def make_phone_windows(
    y,
    max_windows=PHONE_MAX_WINDOWS_PER_FILE,
):
    """
    6-second non-overlap windows.
    Keep final partial only when >=1 sec.

    If too many windows, deterministically select evenly
    spaced windows across the recording.
    """
    y = normalize_audio(y)

    windows = []

    for start in range(
        0,
        len(y),
        WINDOW_SAMPLES,
    ):
        chunk = y[
            start:
            start + WINDOW_SAMPLES
        ]

        if len(chunk) >= MIN_SAMPLES:
            windows.append(chunk)

    if (
        max_windows is not None
        and len(windows) > max_windows
    ):
        indices = np.linspace(
            0,
            len(windows) - 1,
            num=max_windows,
            dtype=int,
        )

        indices = np.unique(
            indices
        )

        windows = [
            windows[int(i)]
            for i in indices
        ]

    return windows


# ============================================================
# TELEPHONY TRANSFORM
# ============================================================

def telephone_codec_transform(input_path):
    """
    Apply exactly the same general telephone/channel idea
    symmetrically to Genuine and Deepfake examples:

        input
          -> mono 8 kHz
          -> highpass 300 Hz
          -> lowpass 3400 Hz
          -> MP3 16 kbps
          -> decode to 16 kHz mono PCM16

    NOTE:
    Codec / bitrate are NOT classifier input features.
    """
    input_path = Path(input_path)

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="sonict_v73_"
        )
    )

    band_wav = (
        temp_dir / "telephone_8k.wav"
    )

    mp3_file = (
        temp_dir / "telephone_16k.mp3"
    )

    final_wav = (
        temp_dir / "telephone_final_16k.wav"
    )

    try:
        cmd1 = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-af",
            "highpass=f=300,lowpass=f=3400",
            "-c:a",
            "pcm_s16le",
            str(band_wav),
        ]

        subprocess.run(
            cmd1,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        cmd2 = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(band_wav),
            "-ac",
            "1",
            "-ar",
            "8000",
            "-b:a",
            "16k",
            str(mp3_file),
        ]

        subprocess.run(
            cmd2,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        cmd3 = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(mp3_file),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-c:a",
            "pcm_s16le",
            str(final_wav),
        ]

        subprocess.run(
            cmd3,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        y, _ = librosa.load(
            str(final_wav),
            sr=SR,
            mono=True,
        )

        y = normalize_audio(y)

        if len(y) < MIN_SAMPLES:
            raise ValueError(
                "Telephony-transformed audio shorter than 1 second"
            )

        return center_crop_6s(y)

    finally:
        for p in [
            band_wav,
            mp3_file,
            final_wav,
        ]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

        try:
            temp_dir.rmdir()
        except Exception:
            pass


# ============================================================
# LOAD WAVLM LOCALLY
# ============================================================

print("\nLoading WavLM locally...")

feature_extractor = (
    AutoFeatureExtractor.from_pretrained(
        str(WAVLM_DIR),
        local_files_only=True,
    )
)

wavlm = WavLMModel.from_pretrained(
    str(WAVLM_DIR),
    local_files_only=True,
)

wavlm.to(DEVICE)
wavlm.eval()

for p in wavlm.parameters():
    p.requires_grad = False

HIDDEN_SIZE = int(
    wavlm.config.hidden_size
)

WAVLM_EMBED_SIZE = (
    HIDDEN_SIZE * 2
)

print(
    f"WavLM hidden size : {HIDDEN_SIZE}"
)

print(
    f"WavLM embedding   : {WAVLM_EMBED_SIZE}"
)


# ============================================================
# WAVLM MEAN + STD EMBEDDING
# ============================================================

def wavlm_batch_embeddings(
    waveforms,
):
    inputs = feature_extractor(
        waveforms,
        sampling_rate=SR,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_values = (
        inputs["input_values"]
        .to(DEVICE)
    )

    attention_mask = (
        inputs.get(
            "attention_mask"
        )
    )

    if attention_mask is not None:
        attention_mask = (
            attention_mask.to(DEVICE)
        )

    with torch.no_grad():
        output = wavlm(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden = (
            output.last_hidden_state
        )

        if attention_mask is not None:
            try:
                feature_mask = (
                    wavlm
                    ._get_feature_vector_attention_mask(
                        hidden.shape[1],
                        attention_mask,
                    )
                )
            except Exception:
                feature_mask = torch.ones(
                    hidden.shape[:2],
                    dtype=torch.bool,
                    device=hidden.device,
                )
        else:
            feature_mask = torch.ones(
                hidden.shape[:2],
                dtype=torch.bool,
                device=hidden.device,
            )

        m = (
            feature_mask
            .unsqueeze(-1)
            .float()
        )

        count = (
            m.sum(dim=1)
            .clamp(min=1.0)
        )

        mean = (
            (hidden * m)
            .sum(dim=1)
            / count
        )

        var = (
            (
                (
                    hidden
                    - mean.unsqueeze(1)
                ) ** 2
            )
            * m
        ).sum(dim=1) / count

        std = torch.sqrt(
            var.clamp(
                min=1e-8
            )
        )

        emb = torch.cat(
            [
                mean,
                std,
            ],
            dim=1,
        )

    return (
        emb
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# ACOUSTIC FEATURES
# ============================================================

ACOUSTIC_FEATURE_NAMES = [
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "rolloff85_mean",
    "rolloff85_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "zcr_mean",
    "zcr_std",
    "rms_mean",
    "rms_std",
    "energy_0_1k",
    "energy_1_3k",
    "energy_3_4k",
    "energy_4_8k",
]

for i in range(1, 14):
    ACOUSTIC_FEATURE_NAMES.append(
        f"mfcc_{i}_mean"
    )
    ACOUSTIC_FEATURE_NAMES.append(
        f"mfcc_{i}_std"
    )

ACOUSTIC_FEATURE_SIZE = len(
    ACOUSTIC_FEATURE_NAMES
)

print(
    f"Acoustic features : {ACOUSTIC_FEATURE_SIZE}"
)


def band_energy_ratio(
    y,
    low_hz,
    high_hz,
):
    spec = np.abs(
        librosa.stft(
            y,
            n_fft=2048,
            hop_length=512,
        )
    ) ** 2

    freqs = (
        librosa.fft_frequencies(
            sr=SR,
            n_fft=2048,
        )
    )

    total = (
        float(np.sum(spec))
        + 1e-12
    )

    mask = (
        (freqs >= low_hz)
        & (freqs < high_hz)
    )

    return float(
        np.sum(spec[mask])
        / total
    )


def extract_acoustic_vector(y):
    """
    42-dimensional telephony acoustic descriptor.

    Explicitly excludes:
      - codec name
      - bitrate
      - original sample rate

    because those are channel metadata, not proof of deepfake.
    """
    y = normalize_audio(y)

    if len(y) < MIN_SAMPLES:
        raise ValueError(
            "Audio shorter than 1 second for acoustic extraction"
        )

    centroid = (
        librosa.feature
        .spectral_centroid(
            y=y,
            sr=SR,
        )[0]
    )

    bandwidth = (
        librosa.feature
        .spectral_bandwidth(
            y=y,
            sr=SR,
        )[0]
    )

    rolloff = (
        librosa.feature
        .spectral_rolloff(
            y=y,
            sr=SR,
            roll_percent=0.85,
        )[0]
    )

    flatness = (
        librosa.feature
        .spectral_flatness(
            y=y
        )[0]
    )

    zcr = (
        librosa.feature
        .zero_crossing_rate(
            y
        )[0]
    )

    rms = (
        librosa.feature
        .rms(
            y=y
        )[0]
    )

    mfcc = (
        librosa.feature
        .mfcc(
            y=y,
            sr=SR,
            n_mfcc=13,
        )
    )

    values = [
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.mean(bandwidth)),
        float(np.std(bandwidth)),
        float(np.mean(rolloff)),
        float(np.std(rolloff)),
        float(np.mean(flatness)),
        float(np.std(flatness)),
        float(np.mean(zcr)),
        float(np.std(zcr)),
        float(np.mean(rms)),
        float(np.std(rms)),
        band_energy_ratio(
            y,
            0,
            1000,
        ),
        band_energy_ratio(
            y,
            1000,
            3000,
        ),
        band_energy_ratio(
            y,
            3000,
            4000,
        ),
        band_energy_ratio(
            y,
            4000,
            8000,
        ),
    ]

    for i in range(13):
        values.append(
            float(
                np.mean(
                    mfcc[i]
                )
            )
        )

        values.append(
            float(
                np.std(
                    mfcc[i]
                )
            )
        )

    arr = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(arr) != ACOUSTIC_FEATURE_SIZE:
        raise RuntimeError(
            "Acoustic feature size mismatch"
        )

    return arr


# ============================================================
# NPZ HELPERS
# ============================================================

def load_npz_cache(path):
    data = np.load(
        path,
        allow_pickle=True,
    )

    X = data[
        "X"
    ].astype(
        np.float32
    )

    y = data[
        "y"
    ].astype(
        np.int64
    )

    meta = pd.DataFrame(
        json.loads(
            str(
                data[
                    "meta_json"
                ].item()
            )
        )
    )

    return (
        X,
        y,
        meta,
    )


def save_combined_cache(
    path,
    X_wavlm,
    X_acoustic,
    y,
    meta,
):
    np.savez_compressed(
        path,
        X_wavlm=(
            X_wavlm
            .astype(np.float16)
        ),
        X_acoustic=(
            X_acoustic
            .astype(np.float32)
        ),
        y=(
            y
            .astype(np.int64)
        ),
        meta_json=np.array(
            json.dumps(
                meta.to_dict(
                    "records"
                )
            ),
            dtype=object,
        ),
    )


def load_combined_cache(
    path,
):
    data = np.load(
        path,
        allow_pickle=True,
    )

    return (
        data[
            "X_wavlm"
        ].astype(
            np.float32
        ),
        data[
            "X_acoustic"
        ].astype(
            np.float32
        ),
        data[
            "y"
        ].astype(
            np.int64
        ),
        pd.DataFrame(
            json.loads(
                str(
                    data[
                        "meta_json"
                    ].item()
                )
            )
        ),
    )


# ============================================================
# LOAD ORIGINAL MLAAD META
# ============================================================

_, _, mlaad_train_meta = (
    load_npz_cache(
        V7_TRAIN_CACHE
    )
)

_, _, mlaad_val_meta = (
    load_npz_cache(
        V7_VAL_CACHE
    )
)

_, _, mlaad_test_meta = (
    load_npz_cache(
        V7_TEST_CACHE
    )
)

print("\nMLAAD metadata:")
print(
    f"Train      : {len(mlaad_train_meta)}"
)

print(
    f"Validation : {len(mlaad_val_meta)}"
)

print(
    f"Test       : {len(mlaad_test_meta)}"
)


# ============================================================
# BALANCED / GENERATOR-AWARE SELECTION
# ============================================================

def balanced_deepfake_selection(
    df,
    n_total,
    seed,
):
    """
    Distribute selected fake files as evenly as possible
    across available generator names.
    """
    fake = df[
        df["label"].astype(int) == 1
    ].copy()

    if len(fake) <= n_total:
        return fake.reset_index(
            drop=True
        )

    generator_col = (
        "generator"
        if "generator" in fake.columns
        else None
    )

    if (
        generator_col is None
        or fake[generator_col].nunique() <= 1
    ):
        return (
            fake.sample(
                n=n_total,
                random_state=seed,
            )
            .reset_index(
                drop=True
            )
        )

    groups = [
        (
            str(name),
            group.copy(),
        )
        for name, group
        in fake.groupby(
            generator_col
        )
    ]

    rng = np.random.default_rng(
        seed
    )

    # Shuffle group order deterministically.
    rng.shuffle(groups)

    selected_parts = []
    selected_count = 0

    base = max(
        1,
        n_total // len(groups)
    )

    for i, (
        name,
        group,
    ) in enumerate(groups):
        remaining = (
            n_total
            - selected_count
        )

        groups_left = (
            len(groups)
            - i
        )

        if remaining <= 0:
            break

        take = min(
            len(group),
            max(
                base,
                math.ceil(
                    remaining
                    / groups_left
                ),
            ),
        )

        selected_parts.append(
            group.sample(
                n=take,
                random_state=(
                    seed + i
                ),
            )
        )

        selected_count += take

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    if len(selected) > n_total:
        selected = selected.sample(
            n=n_total,
            random_state=seed,
        )

    elif len(selected) < n_total:
        used_paths = set(
            selected[
                "path"
            ].astype(str)
        )

        remainder = fake[
            ~fake[
                "path"
            ]
            .astype(str)
            .isin(
                used_paths
            )
        ]

        needed = (
            n_total
            - len(selected)
        )

        if needed > 0:
            extra = remainder.sample(
                n=min(
                    needed,
                    len(remainder),
                ),
                random_state=(
                    seed + 1000
                ),
            )

            selected = pd.concat(
                [
                    selected,
                    extra,
                ],
                ignore_index=True,
            )

    return selected.reset_index(
        drop=True
    )


def balanced_genuine_selection(
    df,
    n_total,
    seed,
):
    genuine = df[
        df["label"].astype(int) == 0
    ].copy()

    if len(genuine) <= n_total:
        return genuine.reset_index(
            drop=True
        )

    # Prefer diversity by source group if available.
    source_col = (
        "source_group"
        if "source_group" in genuine.columns
        else None
    )

    if (
        source_col is None
        or genuine[
            source_col
        ].nunique() <= 1
    ):
        return (
            genuine.sample(
                n=n_total,
                random_state=seed,
            )
            .reset_index(
                drop=True
            )
        )

    groups = list(
        genuine.groupby(
            source_col
        )
    )

    rng = np.random.default_rng(
        seed
    )

    rng.shuffle(groups)

    selected_rows = []

    # First pass: one per source group.
    for _, group in groups:
        if (
            len(selected_rows)
            >= n_total
        ):
            break

        selected_rows.append(
            group.sample(
                n=1,
                random_state=(
                    seed
                    + len(
                        selected_rows
                    )
                ),
            )
        )

    selected = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    if len(selected) < n_total:
        used_paths = set(
            selected[
                "path"
            ].astype(str)
        )

        remainder = genuine[
            ~genuine[
                "path"
            ]
            .astype(str)
            .isin(
                used_paths
            )
        ]

        needed = (
            n_total
            - len(selected)
        )

        extra = remainder.sample(
            n=min(
                needed,
                len(remainder),
            ),
            random_state=(
                seed + 900
            ),
        )

        selected = pd.concat(
            [
                selected,
                extra,
            ],
            ignore_index=True,
        )

    return selected.reset_index(
        drop=True
    )


def make_telephony_selection(
    df,
    n_genuine,
    n_deepfake,
    seed,
):
    g = balanced_genuine_selection(
        df,
        n_genuine,
        seed,
    )

    d = balanced_deepfake_selection(
        df,
        n_deepfake,
        seed + 1,
    )

    selected = pd.concat(
        [
            g,
            d,
        ],
        ignore_index=True,
    )

    selected = selected.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(
        drop=True
    )

    return selected


# ============================================================
# TRAIN FEATURES
# ============================================================

def build_train_combined_features():
    if TRAIN_FEATURE_CACHE.exists():
        print(
            "\nLoading cached V7.3 telephony TRAIN combined features:"
        )
        print(
            TRAIN_FEATURE_CACHE
        )

        return load_combined_cache(
            TRAIN_FEATURE_CACHE
        )

    print("\n" + "=" * 82)
    print("V7.3 TELEPHONY TRAIN FEATURE BUILD")
    print("=" * 82)

    X_codec_wavlm, y_codec, codec_meta = (
        load_npz_cache(
            V72_CODEC_CACHE
        )
    )

    print(
        f"Reusing V7.2 telephony WavLM embeddings: "
        f"{X_codec_wavlm.shape}"
    )

    acoustic_rows = []
    keep_indices = []
    meta_rows = []

    for i, row in codec_meta.reset_index(
        drop=True
    ).iterrows():
        path = Path(
            row["path"]
        )

        try:
            y_audio = (
                telephone_codec_transform(
                    path
                )
            )

            acoustic = (
                extract_acoustic_vector(
                    y_audio
                )
            )

            acoustic_rows.append(
                acoustic
            )

            keep_indices.append(
                i
            )

            meta_rows.append(
                row.to_dict()
            )

        except Exception as exc:
            failed_rows.append({
                "split":
                "telephony_train",
                "path":
                str(path),
                "error":
                str(exc),
            })

        done = i + 1

        if (
            done % 50 == 0
            or done == len(codec_meta)
        ):
            print(
                f"Train acoustic features: "
                f"{done}/{len(codec_meta)}"
            )

    keep_indices = np.asarray(
        keep_indices,
        dtype=np.int64,
    )

    X_wavlm = (
        X_codec_wavlm[
            keep_indices
        ]
        .astype(
            np.float32
        )
    )

    X_acoustic = np.stack(
        acoustic_rows
    ).astype(
        np.float32
    )

    y = y_codec[
        keep_indices
    ].astype(
        np.int64
    )

    meta = pd.DataFrame(
        meta_rows
    )

    save_combined_cache(
        TRAIN_FEATURE_CACHE,
        X_wavlm,
        X_acoustic,
        y,
        meta,
    )

    print(
        "\nTelephony TRAIN combined features:"
    )
    print(
        "WavLM   :",
        X_wavlm.shape,
    )
    print(
        "Acoustic:",
        X_acoustic.shape,
    )
    print(
        "Labels  :",
        {
            "genuine":
            int(
                np.sum(
                    y == 0
                )
            ),
            "deepfake":
            int(
                np.sum(
                    y == 1
                )
            ),
        },
    )

    return (
        X_wavlm,
        X_acoustic,
        y,
        meta,
    )


# ============================================================
# VALIDATION / TEST TELEPHONY FEATURES
# ============================================================

def build_transformed_split_features(
    selected_df,
    cache_path,
    split_name,
):
    if cache_path.exists():
        print(
            f"\nLoading cached {split_name} combined features:"
        )
        print(
            cache_path
        )

        return load_combined_cache(
            cache_path
        )

    print("\n" + "=" * 82)
    print(
        f"BUILD {split_name.upper()} TELEPHONY FEATURES"
    )
    print("=" * 82)

    X_wavlm_parts = []
    X_acoustic_parts = []
    labels = []
    meta_rows = []

    pending_waveforms = []
    pending_acoustics = []
    pending_rows = []

    def flush_pending():
        nonlocal pending_waveforms, pending_acoustics, pending_rows

        if not pending_waveforms:
            return

        embeddings = (
            wavlm_batch_embeddings(
                pending_waveforms
            )
        )

        X_wavlm_parts.append(
            embeddings
        )

        X_acoustic_parts.append(
            np.stack(
                pending_acoustics
            ).astype(
                np.float32
            )
        )

        for row in pending_rows:
            labels.append(
                int(
                    row["label"]
                )
            )

            meta_rows.append(
                row.to_dict()
            )

        pending_waveforms = []
        pending_acoustics = []
        pending_rows = []

    for i, row in selected_df.reset_index(
        drop=True
    ).iterrows():
        path = Path(
            row["path"]
        )

        try:
            y_audio = (
                telephone_codec_transform(
                    path
                )
            )

            acoustic = (
                extract_acoustic_vector(
                    y_audio
                )
            )

            pending_waveforms.append(
                y_audio
            )

            pending_acoustics.append(
                acoustic
            )

            pending_rows.append(
                row
            )

            if (
                len(
                    pending_waveforms
                )
                >= WAVLM_BATCH_SIZE
            ):
                flush_pending()

        except Exception as exc:
            failed_rows.append({
                "split":
                split_name,
                "path":
                str(path),
                "error":
                str(exc),
            })

        done = i + 1

        if (
            done % 25 == 0
            or done == len(
                selected_df
            )
        ):
            print(
                f"{split_name}: "
                f"{done}/{len(selected_df)}"
            )

    flush_pending()

    if not X_wavlm_parts:
        raise RuntimeError(
            f"No valid features for {split_name}"
        )

    X_wavlm = np.concatenate(
        X_wavlm_parts,
        axis=0,
    ).astype(
        np.float32
    )

    X_acoustic = np.concatenate(
        X_acoustic_parts,
        axis=0,
    ).astype(
        np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    meta = pd.DataFrame(
        meta_rows
    )

    save_combined_cache(
        cache_path,
        X_wavlm,
        X_acoustic,
        y,
        meta,
    )

    print(
        f"\n{split_name} feature shapes:"
    )
    print(
        "WavLM   :",
        X_wavlm.shape,
    )
    print(
        "Acoustic:",
        X_acoustic.shape,
    )

    return (
        X_wavlm,
        X_acoustic,
        y,
        meta,
    )


# ============================================================
# SELECT GENERATOR-DISJOINT VALIDATION / TEST
# ============================================================

if VAL_SELECTION_CSV.exists():
    val_selection = pd.read_csv(
        VAL_SELECTION_CSV
    )
else:
    val_selection = (
        make_telephony_selection(
            mlaad_val_meta,
            VAL_GENUINE_FILES,
            VAL_DEEPFAKE_FILES,
            SEED + 100,
        )
    )

    val_selection.to_csv(
        VAL_SELECTION_CSV,
        index=False,
    )


if TEST_SELECTION_CSV.exists():
    test_selection = pd.read_csv(
        TEST_SELECTION_CSV
    )
else:
    test_selection = (
        make_telephony_selection(
            mlaad_test_meta,
            TEST_GENUINE_FILES,
            TEST_DEEPFAKE_FILES,
            SEED + 200,
        )
    )

    test_selection.to_csv(
        TEST_SELECTION_CSV,
        index=False,
    )


print("\nTelephony validation selection:")
print(
    val_selection[
        "label_name"
    ].value_counts()
)

if "generator" in val_selection.columns:
    fake_val = val_selection[
        val_selection[
            "label"
        ].astype(int) == 1
    ]

    print(
        "Validation fake generators:",
        fake_val[
            "generator"
        ].nunique(),
    )


print("\nTelephony final test selection:")
print(
    test_selection[
        "label_name"
    ].value_counts()
)

if "generator" in test_selection.columns:
    fake_test = test_selection[
        test_selection[
            "label"
        ].astype(int) == 1
    ]

    print(
        "Test fake generators:",
        fake_test[
            "generator"
        ].nunique(),
    )


# ============================================================
# BUILD MLAAD TELEPHONY TRAIN / VAL / TEST
# ============================================================

(
    X_train_wavlm,
    X_train_acoustic,
    y_train_codec,
    train_meta,
) = build_train_combined_features()


(
    X_val_wavlm,
    X_val_acoustic,
    y_val_codec,
    val_meta,
) = build_transformed_split_features(
    val_selection,
    VAL_FEATURE_CACHE,
    "telephony_validation",
)


(
    X_test_wavlm,
    X_test_acoustic,
    y_test_codec,
    test_meta,
) = build_transformed_split_features(
    test_selection,
    TEST_FEATURE_CACHE,
    "telephony_test",
)


# ============================================================
# PHONE COMBINED FEATURES
# ============================================================

def build_phone_combined_features(
    wavlm_cache_path,
    output_cache_path,
    split_name,
):
    if output_cache_path.exists():
        print(
            f"\nLoading cached phone {split_name} combined features:"
        )

        print(
            output_cache_path
        )

        return load_combined_cache(
            output_cache_path
        )

    X_phone_wavlm, y_phone, phone_meta = (
        load_npz_cache(
            wavlm_cache_path
        )
    )

    acoustic_rows = []
    keep_indices = []
    meta_rows = []

    print(
        f"\nBuilding phone {split_name} acoustic features..."
    )

    grouped = phone_meta.groupby(
        "path",
        sort=False,
    )

    # Cache each source file's selected windows once.
    file_window_cache = {}

    for path_str, group in grouped:
        path = Path(
            path_str
        )

        try:
            y_audio = load_audio_robust(
                path
            )

            windows = make_phone_windows(
                y_audio,
                max_windows=PHONE_MAX_WINDOWS_PER_FILE,
            )

            file_window_cache[
                path_str
            ] = windows

        except Exception as exc:
            failed_rows.append({
                "split":
                f"phone_{split_name}",
                "path":
                str(path),
                "error":
                str(exc),
            })

            file_window_cache[
                path_str
            ] = []

    for i, row in phone_meta.reset_index(
        drop=True
    ).iterrows():
        path_str = str(
            row["path"]
        )

        windows = file_window_cache.get(
            path_str,
            [],
        )

        window_index = int(
            row[
                "window_index"
            ]
        )

        if (
            window_index < 0
            or window_index >= len(
                windows
            )
        ):
            failed_rows.append({
                "split":
                f"phone_{split_name}",
                "path":
                path_str,
                "error":
                (
                    "Window index missing during "
                    "acoustic reconstruction"
                ),
            })

            continue

        try:
            acoustic = (
                extract_acoustic_vector(
                    windows[
                        window_index
                    ]
                )
            )

            acoustic_rows.append(
                acoustic
            )

            keep_indices.append(
                i
            )

            meta_rows.append(
                row.to_dict()
            )

        except Exception as exc:
            failed_rows.append({
                "split":
                f"phone_{split_name}",
                "path":
                path_str,
                "error":
                str(exc),
            })

    keep_indices = np.asarray(
        keep_indices,
        dtype=np.int64,
    )

    X_wavlm = (
        X_phone_wavlm[
            keep_indices
        ]
        .astype(
            np.float32
        )
    )

    X_acoustic = np.stack(
        acoustic_rows
    ).astype(
        np.float32
    )

    y = y_phone[
        keep_indices
    ].astype(
        np.int64
    )

    meta = pd.DataFrame(
        meta_rows
    )

    save_combined_cache(
        output_cache_path,
        X_wavlm,
        X_acoustic,
        y,
        meta,
    )

    print(
        f"Phone {split_name} shape: "
        f"WavLM={X_wavlm.shape}, "
        f"Acoustic={X_acoustic.shape}"
    )

    return (
        X_wavlm,
        X_acoustic,
        y,
        meta,
    )


(
    X_phone_train_wavlm,
    X_phone_train_acoustic,
    y_phone_train,
    phone_train_meta,
) = build_phone_combined_features(
    V72_PHONE_TRAIN_CACHE,
    PHONE_TRAIN_COMBINED_CACHE,
    "train",
)


(
    X_phone_val_wavlm,
    X_phone_val_acoustic,
    y_phone_val,
    phone_val_meta,
) = build_phone_combined_features(
    V72_PHONE_VAL_CACHE,
    PHONE_VAL_COMBINED_CACHE,
    "validation",
)


(
    X_phone_test_wavlm,
    X_phone_test_acoustic,
    y_phone_test,
    phone_test_meta,
) = build_phone_combined_features(
    V72_PHONE_TEST_CACHE,
    PHONE_TEST_COMBINED_CACHE,
    "test",
)


# ============================================================
# NORMALIZATION
# ============================================================

# Fit BOTH normalizers only on training data.
# Phone validation/test are NOT used in normalization.

phone_repeat_wavlm = np.repeat(
    X_phone_train_wavlm,
    PHONE_TRAIN_REPEAT,
    axis=0,
)

phone_repeat_acoustic = np.repeat(
    X_phone_train_acoustic,
    PHONE_TRAIN_REPEAT,
    axis=0,
)

phone_repeat_y = np.repeat(
    y_phone_train,
    PHONE_TRAIN_REPEAT,
    axis=0,
)

X_wavlm_train_raw = np.concatenate(
    [
        X_train_wavlm,
        phone_repeat_wavlm,
    ],
    axis=0,
).astype(
    np.float32
)

X_acoustic_train_raw = np.concatenate(
    [
        X_train_acoustic,
        phone_repeat_acoustic,
    ],
    axis=0,
).astype(
    np.float32
)

y_train = np.concatenate(
    [
        y_train_codec,
        phone_repeat_y,
    ],
    axis=0,
).astype(
    np.int64
)

wavlm_mean = np.mean(
    X_wavlm_train_raw,
    axis=0,
).astype(
    np.float32
)

wavlm_std = np.std(
    X_wavlm_train_raw,
    axis=0,
).astype(
    np.float32
)

wavlm_std[
    wavlm_std < 1e-6
] = 1.0


acoustic_mean = np.mean(
    X_acoustic_train_raw,
    axis=0,
).astype(
    np.float32
)

acoustic_std = np.std(
    X_acoustic_train_raw,
    axis=0,
).astype(
    np.float32
)

acoustic_std[
    acoustic_std < 1e-6
] = 1.0


def standardize_wavlm(X):
    return (
        (
            X.astype(
                np.float32
            )
            - wavlm_mean
        )
        / wavlm_std
    ).astype(
        np.float32
    )


def standardize_acoustic(X):
    return (
        (
            X.astype(
                np.float32
            )
            - acoustic_mean
        )
        / acoustic_std
    ).astype(
        np.float32
    )


def combine_features(
    Xw,
    Xa,
):
    return np.concatenate(
        [
            standardize_wavlm(
                Xw
            ),
            standardize_acoustic(
                Xa
            ),
        ],
        axis=1,
    ).astype(
        np.float32
    )


X_train = combine_features(
    X_wavlm_train_raw,
    X_acoustic_train_raw,
)

X_val = combine_features(
    X_val_wavlm,
    X_val_acoustic,
)

X_test = combine_features(
    X_test_wavlm,
    X_test_acoustic,
)

X_phone_val = combine_features(
    X_phone_val_wavlm,
    X_phone_val_acoustic,
)

X_phone_test = combine_features(
    X_phone_test_wavlm,
    X_phone_test_acoustic,
)

COMBINED_FEATURE_SIZE = (
    X_train.shape[1]
)

print("\n" + "=" * 82)
print("V7.3 TRAINING DATA")
print("=" * 82)

print(
    f"Telephony MLAAD train       : {len(X_train_wavlm)}"
)

print(
    f"Personal phone train windows: {len(X_phone_train_wavlm)}"
)

print(
    f"Phone repeat factor         : {PHONE_TRAIN_REPEAT}"
)

print(
    f"Combined training rows      : {len(X_train)}"
)

print(
    "Training class counts      : "
    f"genuine={int(np.sum(y_train == 0))}, "
    f"deepfake={int(np.sum(y_train == 1))}"
)

print(
    f"Combined feature size       : {COMBINED_FEATURE_SIZE}"
)


# ============================================================
# MODEL
# ============================================================

class TelephonyFusionMLP(
    nn.Module
):
    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                384,
            ),
            nn.BatchNorm1d(
                384
            ),
            nn.ReLU(),
            nn.Dropout(
                0.35
            ),

            nn.Linear(
                384,
                128,
            ),
            nn.BatchNorm1d(
                128
            ),
            nn.ReLU(),
            nn.Dropout(
                0.30
            ),

            nn.Linear(
                128,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.20
            ),

            nn.Linear(
                32,
                2,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


model = TelephonyFusionMLP(
    COMBINED_FEATURE_SIZE
).to(
    DEVICE
)


class_counts = np.bincount(
    y_train,
    minlength=2,
).astype(
    np.float32
)

class_weights = (
    class_counts.sum()
    / (
        2.0
        * np.maximum(
            class_counts,
            1.0,
        )
    )
)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=2e-4,
)

scheduler = (
    torch.optim.lr_scheduler
    .ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
)


def make_loader(
    X,
    y,
    shuffle,
):
    return DataLoader(
        TensorDataset(
            torch.tensor(
                X,
                dtype=torch.float32,
            ),
            torch.tensor(
                y,
                dtype=torch.long,
            ),
        ),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


train_loader = make_loader(
    X_train,
    y_train,
    True,
)

val_loader = make_loader(
    X_val,
    y_val_codec,
    False,
)

test_loader = make_loader(
    X_test,
    y_test_codec,
    False,
)

phone_val_loader = make_loader(
    X_phone_val,
    y_phone_val,
    False,
)

phone_test_loader = make_loader(
    X_phone_test,
    y_phone_test,
    False,
)


def predict_loader(
    loader,
):
    model.eval()

    truth = []
    pred = []
    probs = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(
                DEVICE
            )

            logits = model(
                xb
            )

            p = torch.softmax(
                logits,
                dim=1,
            )

            truth.extend(
                yb.numpy().tolist()
            )

            pred.extend(
                torch.argmax(
                    p,
                    dim=1,
                )
                .cpu()
                .numpy()
                .tolist()
            )

            probs.extend(
                p[
                    :,
                    1
                ]
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        np.asarray(
            truth
        ),
        np.asarray(
            pred
        ),
        np.asarray(
            probs
        ),
    )


# ============================================================
# TRAIN V7.3
# ============================================================

print("\n" + "=" * 82)
print("TRAINING V7.3 TELEPHONY WAVLM + ACOUSTIC MLP")
print("=" * 82)

best_selection_score = -1.0
best_val_f1 = -1.0
best_phone_val_acc = -1.0

epochs_without_improvement = 0

history = []


for epoch in range(
    1,
    EPOCHS + 1,
):
    model.train()

    total_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(
            DEVICE
        )

        yb = yb.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            xb
        )

        loss = criterion(
            logits,
            yb,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += (
            float(
                loss.item()
            )
            * len(yb)
        )

    train_loss = (
        total_loss
        / len(
            y_train
        )
    )

    (
        val_true,
        val_pred,
        val_probs,
    ) = predict_loader(
        val_loader
    )

    val_acc = accuracy_score(
        val_true,
        val_pred,
    )

    (
        val_precision,
        val_recall,
        val_f1,
        _,
    ) = (
        precision_recall_fscore_support(
            val_true,
            val_pred,
            average="macro",
            zero_division=0,
        )
    )

    (
        phone_val_true,
        phone_val_pred,
        phone_val_probs,
    ) = predict_loader(
        phone_val_loader
    )

    phone_val_acc = (
        accuracy_score(
            phone_val_true,
            phone_val_pred,
        )
    )

    # Primary objective: telephony anti-spoof F1.
    # Secondary objective: do not falsely flag real phone audio.
    #
    # The challenge file is NEVER used here.
    selection_score = (
        0.75
        * float(
            val_f1
        )
        + 0.25
        * float(
            phone_val_acc
        )
    )

    scheduler.step(
        selection_score
    )

    history.append({
        "epoch":
        int(epoch),

        "train_loss":
        float(train_loss),

        "telephony_validation_accuracy":
        float(val_acc),

        "telephony_validation_macro_precision":
        float(
            val_precision
        ),

        "telephony_validation_macro_recall":
        float(
            val_recall
        ),

        "telephony_validation_macro_f1":
        float(
            val_f1
        ),

        "phone_validation_genuine_accuracy":
        float(
            phone_val_acc
        ),

        "phone_validation_mean_df_probability":
        float(
            np.mean(
                phone_val_probs
            )
        ),

        "phone_validation_median_df_probability":
        float(
            np.median(
                phone_val_probs
            )
        ),

        "selection_score":
        float(
            selection_score
        ),
    })

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {train_loss:.4f} | "
        f"Telephony Val F1 {val_f1 * 100:.2f}% | "
        f"Phone Val Genuine {phone_val_acc * 100:.2f}% | "
        f"Score {selection_score * 100:.2f}%"
    )

    if (
        selection_score
        > best_selection_score
        + 1e-4
    ):
        best_selection_score = float(
            selection_score
        )

        best_val_f1 = float(
            val_f1
        )

        best_phone_val_acc = float(
            phone_val_acc
        )

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                model.state_dict(),

                "wavlm_mean":
                wavlm_mean,

                "wavlm_std":
                wavlm_std,

                "acoustic_mean":
                acoustic_mean,

                "acoustic_std":
                acoustic_std,

                "acoustic_feature_names":
                ACOUSTIC_FEATURE_NAMES,

                "wavlm_hidden_size":
                HIDDEN_SIZE,

                "wavlm_embedding_size":
                WAVLM_EMBED_SIZE,

                "acoustic_feature_size":
                ACOUSTIC_FEATURE_SIZE,

                "combined_feature_size":
                COMBINED_FEATURE_SIZE,

                "best_selection_score":
                best_selection_score,

                "best_telephony_validation_macro_f1":
                best_val_f1,

                "best_phone_validation_genuine_accuracy":
                best_phone_val_acc,

                "config": {
                    "version":
                    "F1B_V7_3_TELEPHONY_WAVLM_ACOUSTIC",

                    "sr":
                    SR,

                    "window_seconds":
                    WINDOW_SECONDS,

                    "wavlm_path":
                    str(
                        WAVLM_DIR
                    ),

                    "wavlm_pooling":
                    "temporal_mean_plus_std",

                    "telephone_transform":
                    (
                        "mono 8kHz, "
                        "300-3400Hz, "
                        "MP3 16kbps, "
                        "decode to 16kHz PCM16"
                    ),

                    "phone_train_repeat":
                    PHONE_TRAIN_REPEAT,

                    "selection_score":
                    (
                        "0.75*telephony_validation_macro_f1 "
                        "+ 0.25*phone_validation_genuine_accuracy"
                    ),

                    "metadata_features_excluded":
                    [
                        "codec",
                        "bitrate",
                        "original_sample_rate",
                    ],
                },
            },
            MODEL_PATH,
        )

    else:
        epochs_without_improvement += 1

    if (
        epochs_without_improvement
        >= PATIENCE
    ):
        print(
            "Early stopping."
        )
        break


pd.DataFrame(
    history
).to_csv(
    OUTPUT_DIR
    / "training_history_v7_3.csv",
    index=False,
)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

wavlm_mean = np.asarray(
    checkpoint[
        "wavlm_mean"
    ],
    dtype=np.float32,
)

wavlm_std = np.asarray(
    checkpoint[
        "wavlm_std"
    ],
    dtype=np.float32,
)

acoustic_mean = np.asarray(
    checkpoint[
        "acoustic_mean"
    ],
    dtype=np.float32,
)

acoustic_std = np.asarray(
    checkpoint[
        "acoustic_std"
    ],
    dtype=np.float32,
)


# ============================================================
# FINAL TELEPHONY TEST
# ============================================================

print("\n" + "=" * 82)
print("FINAL TELEPHONY MLAAD UNSEEN-GENERATOR TEST - V7.3")
print("=" * 82)

(
    test_true,
    test_pred,
    test_probs,
) = predict_loader(
    test_loader
)

test_acc = accuracy_score(
    test_true,
    test_pred,
)

(
    test_precision,
    test_recall,
    test_f1,
    _,
) = (
    precision_recall_fscore_support(
        test_true,
        test_pred,
        average="macro",
        zero_division=0,
    )
)

print(
    f"Accuracy        : {test_acc * 100:.2f}%"
)

print(
    f"Macro Precision : {test_precision * 100:.2f}%"
)

print(
    f"Macro Recall    : {test_recall * 100:.2f}%"
)

print(
    f"Macro F1        : {test_f1 * 100:.2f}%"
)

print(
    "Confusion [genuine=0, deepfake=1]:"
)

print(
    confusion_matrix(
        test_true,
        test_pred,
        labels=[
            0,
            1,
        ],
    )
)

print(
    classification_report(
        test_true,
        test_pred,
        target_names=[
            "genuine",
            "deepfake",
        ],
        digits=4,
        zero_division=0,
    )
)


test_result_df = (
    test_meta.copy()
)

test_result_df[
    "deepfake_probability_v7_3"
] = test_probs

test_result_df[
    "prediction_v7_3"
] = np.where(
    test_pred == 1,
    "deepfake",
    "genuine",
)

test_result_df.to_csv(
    OUTPUT_DIR
    / "telephony_unseen_generator_test_results_v7_3.csv",
    index=False,
)


# ============================================================
# PER UNSEEN GENERATOR
# ============================================================

print("\n" + "=" * 82)
print("UNSEEN GENERATOR PERFORMANCE - V7.3 TELEPHONY")
print("=" * 82)

fake_test_df = (
    test_result_df[
        test_result_df[
            "label"
        ].astype(int) == 1
    ]
    .copy()
)

generator_rows = []

if (
    "generator"
    in fake_test_df.columns
):
    for generator, group in (
        fake_test_df.groupby(
            "generator"
        )
    ):
        probs = (
            group[
                "deepfake_probability_v7_3"
            ]
            .astype(float)
            .to_numpy()
        )

        preds = (
            probs >= 0.5
        ).astype(
            np.int64
        )

        generator_rows.append({
            "generator":
            str(generator),

            "n_files":
            int(
                len(group)
            ),

            "deepfake_detection_rate":
            float(
                np.mean(
                    preds == 1
                )
            ),

            "mean_deepfake_probability":
            float(
                np.mean(
                    probs
                )
            ),

            "median_deepfake_probability":
            float(
                np.median(
                    probs
                )
            ),
        })


generator_df = pd.DataFrame(
    generator_rows
)

if len(generator_df):
    print(
        generator_df
        .sort_values(
            "generator"
        )
        .to_string(
            index=False
        )
    )

    generator_df.to_csv(
        OUTPUT_DIR
        / "unseen_generator_performance_v7_3.csv",
        index=False,
    )
else:
    print(
        "Generator column unavailable."
    )


# ============================================================
# FINAL UNSEEN PHONE TEST
# ============================================================

print("\n" + "=" * 82)
print("FINAL UNSEEN-SPEAKER PERSONAL PHONE TEST - V7.3")
print("=" * 82)

(
    phone_test_true,
    phone_test_pred,
    phone_test_probs,
) = predict_loader(
    phone_test_loader
)

phone_window_acc = accuracy_score(
    phone_test_true,
    phone_test_pred,
)

phone_result_df = (
    phone_test_meta.copy()
)

phone_result_df[
    "deepfake_probability_v7_3"
] = phone_test_probs

phone_file_rows = []

for (
    path,
    group
) in phone_result_df.groupby(
    "path",
    sort=False,
):
    probs = (
        group[
            "deepfake_probability_v7_3"
        ]
        .astype(float)
        .to_numpy()
    )

    median_prob = float(
        np.median(
            probs
        )
    )

    mean_prob = float(
        np.mean(
            probs
        )
    )

    max_prob = float(
        np.max(
            probs
        )
    )

    prediction = (
        "DEEPFAKE"
        if median_prob >= 0.50
        else "GENUINE"
    )

    speaker_id = (
        str(
            group[
                "speaker_id"
            ].iloc[0]
        )
        if "speaker_id"
        in group.columns
        else "unknown"
    )

    phone_file_rows.append({
        "path":
        str(path),

        "filename":
        Path(
            path
        ).name,

        "speaker_id":
        speaker_id,

        "prediction":
        prediction,

        "mean_deepfake_probability":
        mean_prob,

        "median_deepfake_probability":
        median_prob,

        "max_deepfake_probability":
        max_prob,

        "n_windows":
        int(
            len(group)
        ),
    })


phone_file_df = pd.DataFrame(
    phone_file_rows
)

for i, row in phone_file_df.iterrows():
    print(
        f"[{i + 1}/{len(phone_file_df)}] "
        f"{row['filename']} | "
        f"speaker={row['speaker_id']} | "
        f"{row['prediction']} | "
        f"median DF {row['median_deepfake_probability'] * 100:.2f}% | "
        f"max {row['max_deepfake_probability'] * 100:.2f}%"
    )


phone_file_accuracy = float(
    np.mean(
        phone_file_df[
            "prediction"
        ] == "GENUINE"
    )
)

phone_false_df_rate = (
    1.0
    - phone_file_accuracy
)

phone_median_df = float(
    np.median(
        phone_file_df[
            "median_deepfake_probability"
        ]
    )
)

print(
    "\nUnseen phone speaker genuine accuracy : "
    f"{phone_file_accuracy * 100:.2f}%"
)

print(
    "Unseen phone false deepfake rate     : "
    f"{phone_false_df_rate * 100:.2f}%"
)

print(
    "Median unseen-phone DF score         : "
    f"{phone_median_df * 100:.2f}%"
)

print(
    "Phone TEST window-level genuine acc  : "
    f"{phone_window_acc * 100:.2f}%"
)


phone_file_df.to_csv(
    OUTPUT_DIR
    / "phone_unseen_speaker_results_v7_3.csv",
    index=False,
)


# ============================================================
# EXTERNAL / CHALLENGE INFERENCE
# ============================================================

def build_external_combined_features(
    path,
):
    y_audio = load_audio_robust(
        path
    )

    windows = make_phone_windows(
        y_audio,
        max_windows=None,
    )

    if not windows:
        raise RuntimeError(
            "No valid >=1 second windows in external audio"
        )

    Xw_parts = []
    Xa_parts = []

    for start in range(
        0,
        len(windows),
        WAVLM_BATCH_SIZE,
    ):
        batch = windows[
            start:
            start + WAVLM_BATCH_SIZE
        ]

        Xw_parts.append(
            wavlm_batch_embeddings(
                batch
            )
        )

        Xa_parts.append(
            np.stack([
                extract_acoustic_vector(
                    w
                )
                for w in batch
            ]).astype(
                np.float32
            )
        )

    Xw = np.concatenate(
        Xw_parts,
        axis=0,
    ).astype(
        np.float32
    )

    Xa = np.concatenate(
        Xa_parts,
        axis=0,
    ).astype(
        np.float32
    )

    X = combine_features(
        Xw,
        Xa,
    )

    return (
        windows,
        X,
    )


def predict_feature_matrix(
    X,
):
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(
            X,
            dtype=torch.float32,
            device=DEVICE,
        )

        logits = model(
            xb
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

    return (
        probs
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


print("\n" + "=" * 82)
print("DIFFICULT KNOWN DEEPFAKE - UNSEEN CHALLENGE - V7.3")
print("=" * 82)

challenge_windows, X_challenge = (
    build_external_combined_features(
        DIFFICULT_DEEPFAKE
    )
)

challenge_probs = (
    predict_feature_matrix(
        X_challenge
    )
)

challenge_mean = float(
    np.mean(
        challenge_probs
    )
)

challenge_median = float(
    np.median(
        challenge_probs
    )
)

challenge_p75 = float(
    np.percentile(
        challenge_probs,
        75,
    )
)

challenge_p90 = float(
    np.percentile(
        challenge_probs,
        90,
    )
)

challenge_max = float(
    np.max(
        challenge_probs
    )
)

challenge_suspicious_ratio = float(
    np.mean(
        challenge_probs >= 0.50
    )
)

challenge_high_ratio = float(
    np.mean(
        challenge_probs >= 0.70
    )
)

challenge_prediction = (
    "DEEPFAKE"
    if challenge_median >= 0.50
    else "GENUINE"
)

print(
    f"File             : {DIFFICULT_DEEPFAKE.name}"
)

print(
    f"Windows          : {len(challenge_probs)}"
)

print(
    f"Prediction       : {challenge_prediction}"
)

print(
    f"Mean Deepfake    : {challenge_mean * 100:.2f}%"
)

print(
    f"Median Deepfake  : {challenge_median * 100:.2f}%"
)

print(
    f"P75 Deepfake     : {challenge_p75 * 100:.2f}%"
)

print(
    f"P90 Deepfake     : {challenge_p90 * 100:.2f}%"
)

print(
    f"Maximum Evidence : {challenge_max * 100:.2f}%"
)

print(
    f"Suspicious Ratio : {challenge_suspicious_ratio * 100:.2f}%"
)

print(
    f"High-Risk Ratio  : {challenge_high_ratio * 100:.2f}%"
)


challenge_df = pd.DataFrame({
    "window_index":
    np.arange(
        len(
            challenge_probs
        )
    ),

    "start_seconds":
    np.arange(
        len(
            challenge_probs
        )
    )
    * WINDOW_SECONDS,

    "deepfake_probability_v7_3":
    challenge_probs,
})

challenge_df.to_csv(
    OUTPUT_DIR
    / "challenge_window_evidence_v7_3.csv",
    index=False,
)


# ============================================================
# FAILED FILES
# ============================================================

if failed_rows:
    pd.DataFrame(
        failed_rows
    ).to_csv(
        FAILED_CSV,
        index=False,
    )


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "experiment":
    "SonicT F1B V7.3 Telephony WavLM + Acoustic branch",

    "model_path":
    str(
        MODEL_PATH
    ),

    "best_telephony_validation_macro_f1":
    float(
        checkpoint[
            "best_telephony_validation_macro_f1"
        ]
    ),

    "best_phone_validation_genuine_accuracy":
    float(
        checkpoint[
            "best_phone_validation_genuine_accuracy"
        ]
    ),

    "telephony_unseen_generator_test_accuracy":
    float(
        test_acc
    ),

    "telephony_unseen_generator_test_macro_precision":
    float(
        test_precision
    ),

    "telephony_unseen_generator_test_macro_recall":
    float(
        test_recall
    ),

    "telephony_unseen_generator_test_macro_f1":
    float(
        test_f1
    ),

    "unseen_phone_genuine_file_accuracy":
    float(
        phone_file_accuracy
    ),

    "unseen_phone_false_deepfake_rate":
    float(
        phone_false_df_rate
    ),

    "unseen_phone_median_deepfake_probability":
    float(
        phone_median_df
    ),

    "unseen_phone_window_genuine_accuracy":
    float(
        phone_window_acc
    ),

    "challenge_prediction":
    challenge_prediction,

    "challenge_mean_deepfake_probability":
    challenge_mean,

    "challenge_median_deepfake_probability":
    challenge_median,

    "challenge_p75_deepfake_probability":
    challenge_p75,

    "challenge_p90_deepfake_probability":
    challenge_p90,

    "challenge_max_deepfake_probability":
    challenge_max,

    "challenge_suspicious_ratio":
    challenge_suspicious_ratio,

    "challenge_high_risk_ratio":
    challenge_high_ratio,

    "wavlm_embedding_size":
    int(
        WAVLM_EMBED_SIZE
    ),

    "acoustic_feature_size":
    int(
        ACOUSTIC_FEATURE_SIZE
    ),

    "combined_feature_size":
    int(
        COMBINED_FEATURE_SIZE
    ),

    "phone_train_repeat":
    int(
        PHONE_TRAIN_REPEAT
    ),

    "metadata_features_excluded":
    [
        "codec",
        "bitrate",
        "original_sample_rate",
    ],

    "challenge_used_for_training":
    False,

    "phone_test_speakers_used_for_training":
    False,

    "output_folder":
    str(
        OUTPUT_DIR
    ),
}


with open(
    OUTPUT_DIR
    / "experiment_results_v7_3.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 82)
print("F1B V7.3 TELEPHONY BRANCH COMPLETE")
print("=" * 82)

print(
    "Best telephony validation Macro F1 : "
    f"{checkpoint['best_telephony_validation_macro_f1'] * 100:.2f}%"
)

print(
    "Best phone validation accuracy     : "
    f"{checkpoint['best_phone_validation_genuine_accuracy'] * 100:.2f}%"
)

print(
    "Telephony unseen-generator test F1 : "
    f"{test_f1 * 100:.2f}%"
)

print(
    "Unseen phone genuine accuracy      : "
    f"{phone_file_accuracy * 100:.2f}%"
)

print(
    "Unseen phone false DF rate         : "
    f"{phone_false_df_rate * 100:.2f}%"
)

print(
    "Challenge median deepfake          : "
    f"{challenge_median * 100:.2f}%"
)

print(
    "Challenge P90 deepfake             : "
    f"{challenge_p90 * 100:.2f}%"
)

print(
    "Challenge suspicious windows       : "
    f"{challenge_suspicious_ratio * 100:.2f}%"
)

print(
    "Challenge prediction               : "
    f"{challenge_prediction}"
)

print("\nOutputs:")
print(
    OUTPUT_DIR
)

print(
    MODEL_PATH
)

print(
    OUTPUT_DIR
    / "experiment_results_v7_3.json"
)

print("\nIMPORTANT:")
print(
    "V7.3 is still experimental. "
    "Do NOT integrate it into SonicT F1-F5 yet."
)

print(
    "The difficult compressed deepfake was NOT used for training."
)

print(
    "Phone validation/test speakers were NOT used for model training."
)

print(
    "Next decision should be based on all three: "
    "telephony unseen-generator F1, unseen-phone false positives, "
    "and the untouched challenge result."
)
