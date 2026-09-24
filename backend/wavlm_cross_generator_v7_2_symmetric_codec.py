from pathlib import Path
import json
import warnings
import re
import subprocess
import tempfile

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformers import AutoFeatureExtractor, WavLMModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ============================================================
# SONICT F1B V7.2
# Frozen WavLM + MLP with speaker-disjoint genuine-phone
# domain adaptation.
#
# Base:
#   - V7 MLAAD generator-disjoint experiment remains unchanged
#   - Existing cached MLAAD WavLM embeddings are reused
#
# V7.2 adds:
#   - Real phone genuine audio adaptation
#   - Speaker-disjoint phone train / validation / test split
#   - FFmpeg fallback for M4A / AAC / MP3 / MPEG / etc.
#   - Window-level phone embeddings
#   - Phone-train oversampling only
#
# IMPORTANT:
#   - Existing SonicT F1-F5 are NOT modified
#   - Difficult challenge deepfake is NEVER used for training
#   - Phone TEST speakers are NEVER used for adaptation
#   - MLAAD final TEST split remains unchanged
# ============================================================

SEED = 42

SR = 16000
MAX_SECONDS = 6.0
MAX_SAMPLES = int(SR * MAX_SECONDS)
MIN_SECONDS = 1.0
MIN_SAMPLES = int(SR * MIN_SECONDS)

FEATURE_BATCH_GPU = 8
FEATURE_BATCH_CPU = 2

TRAIN_BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = 1e-3

# Phone-domain adaptation controls.
PHONE_MAX_WINDOWS_PER_FILE = 8
PHONE_TRAIN_REPEAT = 2

# Symmetric codec/channel augmentation controls.
# Same number of genuine and deepfake MLAAD TRAIN files are augmented.
CODEC_AUG_FILES_PER_CLASS = 300
CODEC_AUG_SEED = 31415

# Speaker split ratios.
PHONE_TRAIN_RATIO = 0.60
PHONE_VAL_RATIO = 0.20
# Remaining speakers become phone TEST.

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

FAKE_MANIFEST = Path(
    r"H:\SonicT_Compression_Test\v7_mlaad_split\fake_manifest_v7.csv"
)

GENUINE_MANIFEST = Path(
    r"H:\SonicT_Compression_Test\v7_mlaad_split\genuine_manifest_v7_1_grouped.csv"
)

WAVLM_DIR = Path(
    r"C:\Users\Guruprasath\.cache\huggingface\hub\models--microsoft--wavlm-base-plus\snapshots\4c66d4806a428f2e922ccfa1a962776e232d487b"
)

PERSONAL_GENUINE_DIR = Path(
    r"H:\SonicT_Compression_Test\genuine"
)

DIFFICULT_DEEPFAKE = Path(
    r"E:\Downloads\recording_6ee192c5-aab6-4a46-8afc-ee9985bb9aaf.mp3.mpeg"
)

# IMPORTANT:
# V7.2 uses a NEW output folder so V7/V7.1 baseline files are preserved.
OUTPUT_DIR = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_2_symmetric_codec"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "wavlm_cross_generator_v7_2_symmetric_codec.pt"

# Reuse ORIGINAL V7 MLAAD feature caches.
V7_CACHE_DIR = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_output"
)

CACHE_TRAIN = V7_CACHE_DIR / "wavlm_train_features_v7.npz"
CACHE_VAL = V7_CACHE_DIR / "wavlm_validation_features_v7.npz"
CACHE_TEST = V7_CACHE_DIR / "wavlm_test_features_v7.npz"

# New phone-domain caches.
PHONE_SPLIT_CSV = OUTPUT_DIR / "phone_speaker_split_v7_2.csv"
PHONE_TRAIN_CACHE = OUTPUT_DIR / "phone_train_features_v7_2.npz"
PHONE_VAL_CACHE = OUTPUT_DIR / "phone_val_features_v7_2.npz"
PHONE_TEST_CACHE = OUTPUT_DIR / "phone_test_features_v7_2.npz"

FAILED_CSV = OUTPUT_DIR / "failed_feature_files_v7_2.csv"

CODEC_AUG_CACHE = OUTPUT_DIR / "symmetric_codec_aug_features_v7_2.npz"
CODEC_AUG_MANIFEST = OUTPUT_DIR / "symmetric_codec_aug_manifest_v7_2.csv"

AUDIO_EXTS = {
    ".wav", ".flac", ".mp3", ".m4a",
    ".aac", ".ogg", ".opus", ".wma",
    ".mpeg", ".mpg", ".amr", ".3gp",
    ".3g2", ".webm", ".mka", ".aiff",
    ".aif", ".caf", ".mov", ".mkv",
}

# ============================================================
# INPUT CHECKS
# ============================================================

for required_path, label in [
    (FAKE_MANIFEST, "fake manifest"),
    (GENUINE_MANIFEST, "grouped genuine manifest"),
    (WAVLM_DIR, "local WavLM folder"),
    (CACHE_TRAIN, "V7 train feature cache"),
    (CACHE_VAL, "V7 validation feature cache"),
    (CACHE_TEST, "V7 test feature cache"),
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"{label} not found: {required_path}"
        )

if not PERSONAL_GENUINE_DIR.exists():
    raise FileNotFoundError(
        f"Personal genuine folder not found: {PERSONAL_GENUINE_DIR}"
    )

# ============================================================
# LOAD MLAAD MANIFESTS
# ============================================================

fake_df = pd.read_csv(FAKE_MANIFEST)
genuine_df = pd.read_csv(GENUINE_MANIFEST)

required_fake = {
    "path", "filename", "generator",
    "label", "label_name", "split"
}

required_genuine = {
    "path", "filename", "group",
    "label", "label_name", "split"
}

if not required_fake.issubset(fake_df.columns):
    raise RuntimeError(
        f"Fake manifest missing: "
        f"{sorted(required_fake - set(fake_df.columns))}"
    )

if not required_genuine.issubset(genuine_df.columns):
    raise RuntimeError(
        f"Genuine manifest missing: "
        f"{sorted(required_genuine - set(genuine_df.columns))}"
    )

fake_df = fake_df.copy()
genuine_df = genuine_df.copy()

fake_df["source_group"] = (
    "generator::" + fake_df["generator"].astype(str)
)

genuine_df["source_group"] = (
    "genuine::" + genuine_df["group"].astype(str)
)

all_df = pd.concat(
    [
        fake_df[
            [
                "path", "filename", "label",
                "label_name", "split",
                "source_group", "generator"
            ]
        ],
        genuine_df.assign(
            generator="genuine_original"
        )[
            [
                "path", "filename", "label",
                "label_name", "split",
                "source_group", "generator"
            ]
        ],
    ],
    ignore_index=True,
)

all_df["label"] = all_df["label"].astype(int)

exists_mask = all_df["path"].apply(
    lambda x: Path(x).exists()
)

missing_files = all_df.loc[
    ~exists_mask,
    ["path", "split", "label_name", "generator"]
]

if len(missing_files):
    missing_files.to_csv(
        OUTPUT_DIR / "missing_manifest_files_v7_1.csv",
        index=False,
    )
    print(
        f"WARNING: {len(missing_files)} MLAAD manifest files "
        "are missing and will be skipped."
    )

all_df = all_df[exists_mask].reset_index(drop=True)

train_df = all_df[all_df["split"] == "train"].reset_index(drop=True)
val_df = all_df[all_df["split"] == "validation"].reset_index(drop=True)
test_df = all_df[all_df["split"] == "test"].reset_index(drop=True)

print("=" * 78)
print("SONICT F1B V7.2 - WAVLM + PHONE-DOMAIN ADAPTATION")
print("=" * 78)
print(f"Device: {DEVICE}")
print(f"Local WavLM: {WAVLM_DIR}")
print(f"V7 cache folder: {V7_CACHE_DIR}")
print(f"V7.1 output folder: {OUTPUT_DIR}")

for name, df in [
    ("MLAAD TRAIN", train_df),
    ("MLAAD VALIDATION", val_df),
    ("MLAAD FINAL TEST", test_df),
]:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Files: {len(df)}")
    print(df["label_name"].value_counts().to_string())
    print(
        "Fake generators: "
        f"{df.loc[df['label'] == 1, 'generator'].nunique()}"
    )
    print(
        "Source groups: "
        f"{df['source_group'].nunique()}"
    )

# ============================================================
# LOAD WAVLM LOCALLY
# ============================================================

print("\nLoading WavLM locally...")

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

for param in wavlm.parameters():
    param.requires_grad = False

HIDDEN_SIZE = int(wavlm.config.hidden_size)
EMBEDDING_SIZE = HIDDEN_SIZE * 2

print(f"WavLM hidden size : {HIDDEN_SIZE}")
print(f"Embedding size    : {EMBEDDING_SIZE}")

feature_batch_size = (
    FEATURE_BATCH_GPU
    if DEVICE.type == "cuda"
    else FEATURE_BATCH_CPU
)

# ============================================================
# AUDIO HELPERS
# ============================================================

failed_rows = []


def ffmpeg_to_temp_wav(input_path):
    """
    Convert any FFmpeg-readable audio/video file to temporary
    16 kHz mono PCM16 WAV. Caller must delete returned file.
    """
    input_path = Path(input_path)

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

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception:
        if temp_wav.exists():
            try:
                temp_wav.unlink()
            except Exception:
                pass
        raise

    return temp_wav


def load_audio_robust(path, sr=SR):
    """
    Try librosa first.
    If decoding fails, normalize through FFmpeg and load WAV.
    """
    path = Path(path)

    try:
        y, _ = librosa.load(
            str(path),
            sr=sr,
            mono=True,
        )
        return np.asarray(y, dtype=np.float32)

    except Exception as first_exc:
        temp_wav = None

        try:
            temp_wav = ffmpeg_to_temp_wav(path)

            y, _ = librosa.load(
                str(temp_wav),
                sr=sr,
                mono=True,
            )

            return np.asarray(y, dtype=np.float32)

        except Exception as second_exc:
            raise RuntimeError(
                f"Audio decode failed. "
                f"librosa={first_exc}; ffmpeg={second_exc}"
            )

        finally:
            if temp_wav is not None and temp_wav.exists():
                try:
                    temp_wav.unlink()
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


def load_audio_for_embedding(path):
    y = load_audio_robust(path)
    y = normalize_audio(y)

    if len(y) < MIN_SAMPLES:
        raise ValueError("Audio shorter than 1 second")

    # Deterministic center crop for MLAAD dataset files.
    if len(y) > MAX_SAMPLES:
        start = (len(y) - MAX_SAMPLES) // 2
        y = y[start:start + MAX_SAMPLES]

    return y


def wavlm_batch_embeddings(waveforms):
    inputs = feature_extractor(
        waveforms,
        sampling_rate=SR,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_values = inputs["input_values"].to(DEVICE)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        output = wavlm(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden = output.last_hidden_state

        if attention_mask is not None:
            try:
                feature_mask = (
                    wavlm._get_feature_vector_attention_mask(
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

        mask = feature_mask.unsqueeze(-1).float()
        count = mask.sum(dim=1).clamp(min=1.0)

        mean = (hidden * mask).sum(dim=1) / count

        variance = (
            ((hidden - mean.unsqueeze(1)) ** 2) * mask
        ).sum(dim=1) / count

        std = torch.sqrt(
            variance.clamp(min=1e-8)
        )

        embedding = torch.cat(
            [mean, std],
            dim=1,
        )

    return embedding.cpu().numpy().astype(np.float32)


# ============================================================
# LOAD EXISTING V7 MLAAD CACHES
# ============================================================

def load_v7_cache(cache_path, split_name):
    print(f"\nLoading V7 cached {split_name} features:")
    print(cache_path)

    data = np.load(
        cache_path,
        allow_pickle=True,
    )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    meta_df = pd.DataFrame(
        json.loads(str(data["meta_json"].item()))
    )

    return X, y, meta_df


print("\n" + "=" * 78)
print("LOAD ORIGINAL V7 MLAAD FEATURES")
print("=" * 78)

X_train_mlaad, y_train_mlaad, train_meta = load_v7_cache(
    CACHE_TRAIN,
    "train",
)

X_val_mlaad, y_val_mlaad, val_meta = load_v7_cache(
    CACHE_VAL,
    "validation",
)

X_test_mlaad, y_test_mlaad, test_meta = load_v7_cache(
    CACHE_TEST,
    "test",
)

print("\nMLAAD feature shapes:")
print("Train      :", X_train_mlaad.shape)
print("Validation :", X_val_mlaad.shape)
print("Test       :", X_test_mlaad.shape)

# ============================================================
# PHONE SPEAKER IDENTIFICATION
# ============================================================

def infer_phone_speaker_id(path):
    """
    Derive a stable speaker/contact group from the call filename.

    Examples:
      Call Mathanprasath NEC_260722_090553.m4a
        -> call mathanprasath nec

      Call Prithiv Krishna_260713_091330.m4a
        -> call prithiv krishna

    The trailing YYMMDD_HHMMSS timestamp is removed.
    """
    stem = Path(path).stem.strip()

    # Remove common trailing date/time:
    # _260722_090553
    stem = re.sub(
        r"[_\-\s]*\d{6}[_\-\s]*\d{6}$",
        "",
        stem,
    )

    # Also handle 8-digit dates if present.
    stem = re.sub(
        r"[_\-\s]*\d{8}[_\-\s]*\d{6}$",
        "",
        stem,
    )

    speaker = re.sub(r"\s+", " ", stem).strip().lower()

    if not speaker:
        speaker = Path(path).stem.lower()

    return speaker


phone_files = sorted([
    p
    for p in PERSONAL_GENUINE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in AUDIO_EXTS
])

if not phone_files:
    raise RuntimeError(
        f"No phone audio found in {PERSONAL_GENUINE_DIR}"
    )

phone_manifest = pd.DataFrame({
    "path": [str(p) for p in phone_files],
    "filename": [p.name for p in phone_files],
})

phone_manifest["speaker_id"] = phone_manifest["path"].apply(
    infer_phone_speaker_id
)

speaker_counts = (
    phone_manifest
    .groupby("speaker_id")
    .size()
    .sort_values(ascending=False)
)

print("\n" + "=" * 78)
print("PERSONAL PHONE SPEAKER GROUPS")
print("=" * 78)
print(f"Phone files   : {len(phone_manifest)}")
print(f"Speaker groups: {phone_manifest['speaker_id'].nunique()}")
print(speaker_counts.to_string())

# ============================================================
# SPEAKER-DISJOINT PHONE SPLIT
# ============================================================

speakers = sorted(
    phone_manifest["speaker_id"].unique().tolist()
)

if len(speakers) < 3:
    raise RuntimeError(
        "Need at least 3 distinct phone speaker groups "
        "for train/validation/test speaker-disjoint splitting."
    )

rng = np.random.default_rng(SEED)
shuffled_speakers = speakers.copy()
rng.shuffle(shuffled_speakers)

n_speakers = len(shuffled_speakers)

n_train_speakers = max(
    1,
    int(round(n_speakers * PHONE_TRAIN_RATIO))
)

n_val_speakers = max(
    1,
    int(round(n_speakers * PHONE_VAL_RATIO))
)

# Ensure at least one test speaker.
if n_train_speakers + n_val_speakers >= n_speakers:
    n_val_speakers = max(
        1,
        n_speakers - n_train_speakers - 1
    )

if n_train_speakers + n_val_speakers >= n_speakers:
    n_train_speakers = max(
        1,
        n_speakers - n_val_speakers - 1
    )

train_speakers = set(
    shuffled_speakers[:n_train_speakers]
)

val_speakers = set(
    shuffled_speakers[
        n_train_speakers:
        n_train_speakers + n_val_speakers
    ]
)

test_speakers = set(
    shuffled_speakers[
        n_train_speakers + n_val_speakers:
    ]
)

def phone_split_name(speaker_id):
    if speaker_id in train_speakers:
        return "train"
    if speaker_id in val_speakers:
        return "validation"
    return "test"


phone_manifest["split"] = phone_manifest["speaker_id"].apply(
    phone_split_name
)

phone_manifest.to_csv(
    PHONE_SPLIT_CSV,
    index=False,
)

# Leakage checks.
assert not (train_speakers & val_speakers)
assert not (train_speakers & test_speakers)
assert not (val_speakers & test_speakers)

print("\n" + "=" * 78)
print("PHONE SPEAKER-DISJOINT SPLIT")
print("=" * 78)

for split_name in ["train", "validation", "test"]:
    subset = phone_manifest[
        phone_manifest["split"] == split_name
    ]

    print(
        f"{split_name.upper():10s} | "
        f"speakers={subset['speaker_id'].nunique():2d} | "
        f"files={len(subset):2d}"
    )

    for spk in sorted(subset["speaker_id"].unique()):
        n_files = int(
            (subset["speaker_id"] == spk).sum()
        )
        print(f"  - {spk} ({n_files} files)")

# ============================================================
# PHONE WINDOWING
# ============================================================

def make_nonoverlap_windows(
    y,
    max_windows=None,
):
    """
    6-second non-overlapping windows.
    Final partial window is kept only if >= 1 second.

    If max_windows is set and a file has more windows,
    choose deterministic evenly-spaced windows across the call.
    """
    y = normalize_audio(y)

    windows = []

    if len(y) < MIN_SAMPLES:
        return windows

    for start in range(
        0,
        len(y),
        MAX_SAMPLES,
    ):
        chunk = y[start:start + MAX_SAMPLES]

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

        # Remove any duplicate indices caused by rounding.
        indices = np.unique(indices)

        windows = [
            windows[int(i)]
            for i in indices
        ]

    return windows


def extract_phone_features(
    split_df,
    split_name,
    cache_path,
):
    if cache_path.exists():
        print(f"\nLoading cached phone {split_name} features:")
        print(cache_path)

        data = np.load(
            cache_path,
            allow_pickle=True,
        )

        return (
            data["X"].astype(np.float32),
            data["y"].astype(np.int64),
            pd.DataFrame(
                json.loads(str(data["meta_json"].item()))
            ),
        )

    X_parts = []
    y_parts = []
    meta_rows = []

    print(f"\nExtracting phone {split_name} WavLM features...")

    for file_index, row in split_df.reset_index(drop=True).iterrows():
        path = Path(row["path"])

        try:
            y_audio = load_audio_robust(path)

            windows = make_nonoverlap_windows(
                y_audio,
                max_windows=PHONE_MAX_WINDOWS_PER_FILE,
            )

            if not windows:
                raise ValueError(
                    "No valid >=1 second windows"
                )

            for batch_start in range(
                0,
                len(windows),
                feature_batch_size,
            ):
                batch = windows[
                    batch_start:
                    batch_start + feature_batch_size
                ]

                embeddings = wavlm_batch_embeddings(
                    batch
                )

                X_parts.append(embeddings)

                for local_i in range(len(embeddings)):
                    global_window_i = batch_start + local_i

                    # Personal phone recordings are known genuine.
                    y_parts.append(0)

                    meta_rows.append({
                        "path": str(path),
                        "filename": str(row["filename"]),
                        "speaker_id": str(row["speaker_id"]),
                        "split": split_name,
                        "window_index": int(global_window_i),
                        "label": 0,
                        "label_name": "genuine_phone",
                    })

            print(
                f"[{file_index + 1}/{len(split_df)}] "
                f"{path.name} | windows={len(windows)}"
            )

        except Exception as exc:
            failed_rows.append({
                "split": f"phone_{split_name}",
                "path": str(path),
                "error": str(exc),
            })

            print(
                f"[{file_index + 1}/{len(split_df)}] "
                f"{path.name} | ERROR: {exc}"
            )

    if not X_parts:
        raise RuntimeError(
            f"No phone embeddings extracted for {split_name}"
        )

    X = np.concatenate(
        X_parts,
        axis=0,
    ).astype(np.float32)

    y = np.asarray(
        y_parts,
        dtype=np.int64,
    )

    meta_df = pd.DataFrame(meta_rows)

    np.savez_compressed(
        cache_path,
        X=X.astype(np.float16),
        y=y,
        meta_json=np.array(
            json.dumps(
                meta_df.to_dict("records")
            ),
            dtype=object,
        ),
    )

    return X, y, meta_df


phone_train_df = phone_manifest[
    phone_manifest["split"] == "train"
].reset_index(drop=True)

phone_val_df = phone_manifest[
    phone_manifest["split"] == "validation"
].reset_index(drop=True)

phone_test_df = phone_manifest[
    phone_manifest["split"] == "test"
].reset_index(drop=True)

print("\n" + "=" * 78)
print("PHONE WAVLM FEATURE EXTRACTION")
print("=" * 78)

X_phone_train, y_phone_train, phone_train_meta = extract_phone_features(
    phone_train_df,
    "train",
    PHONE_TRAIN_CACHE,
)

X_phone_val, y_phone_val, phone_val_meta = extract_phone_features(
    phone_val_df,
    "validation",
    PHONE_VAL_CACHE,
)

X_phone_test, y_phone_test, phone_test_meta = extract_phone_features(
    phone_test_df,
    "test",
    PHONE_TEST_CACHE,
)

print("\nPhone feature shapes:")
print("Train      :", X_phone_train.shape)
print("Validation :", X_phone_val.shape)
print("Test       :", X_phone_test.shape)

if failed_rows:
    pd.DataFrame(failed_rows).to_csv(
        FAILED_CSV,
        index=False,
    )

# ============================================================
# V7.2 SYMMETRIC TELEPHONE / CODEC AUGMENTATION
# ============================================================

def telephone_codec_transform(input_path):
    """
    Create a temporary telephony-like version:
      source -> mono 8 kHz -> 300-3400 Hz band-limit -> MP3 16 kbps
             -> decode/resample to 16 kHz mono PCM16 WAV

    IMPORTANT:
    The exact SAME transform is applied to selected genuine AND
    selected deepfake MLAAD training files. This prevents channel
    characteristics from becoming a one-class shortcut.
    """
    input_path = Path(input_path)

    temp_dir = Path(
        tempfile.mkdtemp(prefix="sonict_v72_")
    )

    band_wav = temp_dir / "telephone_8k.wav"
    mp3_file = temp_dir / "telephone_16k.mp3"
    final_wav = temp_dir / "telephone_final_16k.wav"

    try:
        cmd1 = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(input_path),
            "-vn",
            "-ac", "1",
            "-ar", "8000",
            "-af", "highpass=f=300,lowpass=f=3400",
            "-c:a", "pcm_s16le",
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
            "-loglevel", "error",
            "-y",
            "-i", str(band_wav),
            "-ac", "1",
            "-ar", "8000",
            "-b:a", "16k",
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
            "-loglevel", "error",
            "-y",
            "-i", str(mp3_file),
            "-vn",
            "-ac", "1",
            "-ar", str(SR),
            "-c:a", "pcm_s16le",
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
                "Codec-augmented audio shorter than 1 second"
            )

        # Same deterministic center-crop rule as MLAAD baseline.
        if len(y) > MAX_SAMPLES:
            start = (len(y) - MAX_SAMPLES) // 2
            y = y[start:start + MAX_SAMPLES]

        return np.asarray(y, dtype=np.float32)

    finally:
        # Clean all transform intermediates.
        for p in [band_wav, mp3_file, final_wav]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

        try:
            temp_dir.rmdir()
        except Exception:
            pass


def build_symmetric_codec_manifest():
    """
    Select equal numbers of genuine and deepfake files ONLY from
    MLAAD TRAIN. Validation/test generators are never touched.
    """
    rng = np.random.default_rng(CODEC_AUG_SEED)

    genuine_train = train_df[
        train_df["label"] == 0
    ].copy()

    fake_train = train_df[
        train_df["label"] == 1
    ].copy()

    n_each = min(
        CODEC_AUG_FILES_PER_CLASS,
        len(genuine_train),
        len(fake_train),
    )

    genuine_idx = rng.choice(
        len(genuine_train),
        size=n_each,
        replace=False,
    )

    fake_idx = rng.choice(
        len(fake_train),
        size=n_each,
        replace=False,
    )

    selected = pd.concat(
        [
            genuine_train.iloc[genuine_idx],
            fake_train.iloc[fake_idx],
        ],
        ignore_index=True,
    )

    # Shuffle selected augmentation manifest deterministically.
    selected = selected.sample(
        frac=1.0,
        random_state=CODEC_AUG_SEED,
    ).reset_index(drop=True)

    selected["augmentation"] = "telephone_mp3_8k_to_16k"

    selected.to_csv(
        CODEC_AUG_MANIFEST,
        index=False,
    )

    return selected


def extract_symmetric_codec_features():
    if CODEC_AUG_CACHE.exists():
        print("\nLoading cached symmetric codec features:")
        print(CODEC_AUG_CACHE)

        data = np.load(
            CODEC_AUG_CACHE,
            allow_pickle=True,
        )

        return (
            data["X"].astype(np.float32),
            data["y"].astype(np.int64),
            pd.DataFrame(
                json.loads(str(data["meta_json"].item()))
            ),
        )

    selected = build_symmetric_codec_manifest()

    print("\n" + "=" * 78)
    print("V7.2 SYMMETRIC TELEPHONE/CODEC FEATURE EXTRACTION")
    print("=" * 78)
    print(
        f"Selected genuine : "
        f"{int(np.sum(selected['label'].values == 0))}"
    )
    print(
        f"Selected deepfake: "
        f"{int(np.sum(selected['label'].values == 1))}"
    )
    print(
        "Transform         : "
        "8 kHz + 300-3400 Hz + MP3 16 kbps + 16 kHz decode"
    )

    X_parts = []
    y_parts = []
    meta_rows = []

    # Process one transformed file at a time because each needs
    # temporary FFmpeg intermediates. Embedding batches are kept
    # small and safe for CPU.
    pending_waveforms = []
    pending_rows = []

    def flush_pending():
        nonlocal pending_waveforms, pending_rows

        if not pending_waveforms:
            return

        embeddings = wavlm_batch_embeddings(
            pending_waveforms
        )

        X_parts.append(embeddings)

        for row in pending_rows:
            y_parts.append(int(row["label"]))

            meta_rows.append({
                "path": str(row["path"]),
                "filename": str(row["filename"]),
                "label": int(row["label"]),
                "label_name": str(row["label_name"]),
                "generator": str(row["generator"]),
                "source_group": str(row["source_group"]),
                "augmentation": "telephone_mp3_8k_to_16k",
            })

        pending_waveforms = []
        pending_rows = []

    for index, row in selected.iterrows():
        path = Path(row["path"])

        try:
            y_aug = telephone_codec_transform(path)

            pending_waveforms.append(y_aug)
            pending_rows.append(row)

            if len(pending_waveforms) >= feature_batch_size:
                flush_pending()

        except Exception as exc:
            failed_rows.append({
                "split": "codec_aug_train",
                "path": str(path),
                "error": str(exc),
            })

        done = index + 1

        if done % 50 == 0 or done == len(selected):
            print(
                f"Codec augmentation: "
                f"{done}/{len(selected)}"
            )

    flush_pending()

    if not X_parts:
        raise RuntimeError(
            "No symmetric codec embeddings were extracted."
        )

    X = np.concatenate(
        X_parts,
        axis=0,
    ).astype(np.float32)

    y = np.asarray(
        y_parts,
        dtype=np.int64,
    )

    meta_df = pd.DataFrame(meta_rows)

    np.savez_compressed(
        CODEC_AUG_CACHE,
        X=X.astype(np.float16),
        y=y,
        meta_json=np.array(
            json.dumps(
                meta_df.to_dict("records")
            ),
            dtype=object,
        ),
    )

    print("\nSymmetric codec feature shape:", X.shape)
    print(
        "Successful class counts: "
        f"genuine={int(np.sum(y == 0))}, "
        f"deepfake={int(np.sum(y == 1))}"
    )

    return X, y, meta_df


X_codec_aug, y_codec_aug, codec_meta = (
    extract_symmetric_codec_features()
)

# ============================================================
# BUILD V7.2 TRAINING SET
# ============================================================

# Repeat only PHONE TRAIN embeddings.
# This gives the small real-phone domain enough influence while
# preserving the large balanced MLAAD training set.
X_phone_train_repeated = np.repeat(
    X_phone_train,
    PHONE_TRAIN_REPEAT,
    axis=0,
)

y_phone_train_repeated = np.repeat(
    y_phone_train,
    PHONE_TRAIN_REPEAT,
    axis=0,
)

X_train_raw = np.concatenate(
    [
        X_train_mlaad,
        X_codec_aug,
        X_phone_train_repeated,
    ],
    axis=0,
).astype(np.float32)

y_train = np.concatenate(
    [
        y_train_mlaad,
        y_codec_aug,
        y_phone_train_repeated,
    ],
    axis=0,
).astype(np.int64)

# Validation used for model selection:
# MLAAD validation + speaker-disjoint phone validation.
X_val_combined_raw = np.concatenate(
    [
        X_val_mlaad,
        X_phone_val,
    ],
    axis=0,
).astype(np.float32)

y_val_combined = np.concatenate(
    [
        y_val_mlaad,
        y_phone_val,
    ],
    axis=0,
).astype(np.int64)

print("\n" + "=" * 78)
print("V7.2 TRAINING DATA")
print("=" * 78)

print(f"Original MLAAD train embeddings : {len(X_train_mlaad)}")
print(f"Symmetric codec embeddings      : {len(X_codec_aug)}")
print(f"  codec genuine                 : {int(np.sum(y_codec_aug == 0))}")
print(f"  codec deepfake                : {int(np.sum(y_codec_aug == 1))}")
print(f"Phone train embeddings          : {len(X_phone_train)}")
print(f"Phone repeat factor             : {PHONE_TRAIN_REPEAT}")
print(f"Repeated phone train embeddings : {len(X_phone_train_repeated)}")
print(f"Combined training embeddings    : {len(X_train_raw)}")
print(
    "Combined train class counts    : "
    f"genuine={int(np.sum(y_train == 0))}, "
    f"deepfake={int(np.sum(y_train == 1))}"
)

# ============================================================
# STANDARDIZE USING V7.1 TRAIN ONLY
# ============================================================

feature_mean = X_train_raw.mean(
    axis=0
).astype(np.float32)

feature_std = X_train_raw.std(
    axis=0
).astype(np.float32) + 1e-6

def standardize(X):
    return (
        (X - feature_mean) / feature_std
    ).astype(np.float32)


X_train = standardize(X_train_raw)
X_val_combined = standardize(X_val_combined_raw)

X_val_mlaad_std = standardize(X_val_mlaad)
X_test_mlaad_std = standardize(X_test_mlaad)

X_phone_val_std = standardize(X_phone_val)
X_phone_test_std = standardize(X_phone_test)

# ============================================================
# MLP CLASSIFIER
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


model = EmbeddingMLP(
    EMBEDDING_SIZE
).to(DEVICE)

class_counts = np.bincount(
    y_train,
    minlength=2,
).astype(np.float32)

class_weights = (
    class_counts.sum()
    / (
        2.0 * np.maximum(
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
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)


def make_loader(X, y, shuffle):
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

val_combined_loader = make_loader(
    X_val_combined,
    y_val_combined,
    False,
)

val_mlaad_loader = make_loader(
    X_val_mlaad_std,
    y_val_mlaad,
    False,
)

test_mlaad_loader = make_loader(
    X_test_mlaad_std,
    y_test_mlaad,
    False,
)

phone_val_loader = make_loader(
    X_phone_val_std,
    y_phone_val,
    False,
)

phone_test_loader = make_loader(
    X_phone_test_std,
    y_phone_test,
    False,
)


def predict_loader(loader):
    model.eval()

    truth = []
    pred = []
    probs = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)

            logits = model(xb)

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
                p[:, 1]
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        np.asarray(truth),
        np.asarray(pred),
        np.asarray(probs),
    )


# ============================================================
# TRAIN V7.1
# ============================================================

best_selection_score = -1.0
best_mlaad_val_f1 = -1.0
best_phone_val_acc = -1.0
epochs_without_improvement = 0
history = []

print("\n" + "=" * 78)
print("TRAINING V7.2 WAVLM-EMBEDDING MLP")
print("=" * 78)

for epoch in range(1, EPOCHS + 1):
    model.train()

    total_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_loss += (
            float(loss.item()) * len(yb)
        )

    train_loss = total_loss / len(y_train)

    # MLAAD validation.
    mlaad_val_true, mlaad_val_pred, _ = predict_loader(
        val_mlaad_loader
    )

    mlaad_val_acc = accuracy_score(
        mlaad_val_true,
        mlaad_val_pred,
    )

    (
        mlaad_val_precision,
        mlaad_val_recall,
        mlaad_val_f1,
        _,
    ) = precision_recall_fscore_support(
        mlaad_val_true,
        mlaad_val_pred,
        average="macro",
        zero_division=0,
    )

    # Speaker-disjoint phone validation.
    phone_val_true, phone_val_pred, phone_val_probs = predict_loader(
        phone_val_loader
    )

    phone_val_acc = accuracy_score(
        phone_val_true,
        phone_val_pred,
    )

    # Composite selection score:
    # Preserve generator performance while rewarding phone generalization.
    selection_score = (
        0.70 * float(mlaad_val_f1)
        + 0.30 * float(phone_val_acc)
    )

    scheduler.step(selection_score)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "mlaad_validation_accuracy": float(mlaad_val_acc),
        "mlaad_validation_macro_precision": float(mlaad_val_precision),
        "mlaad_validation_macro_recall": float(mlaad_val_recall),
        "mlaad_validation_macro_f1": float(mlaad_val_f1),
        "phone_validation_accuracy": float(phone_val_acc),
        "phone_validation_mean_deepfake_probability": float(
            np.mean(phone_val_probs)
        ),
        "phone_validation_median_deepfake_probability": float(
            np.median(phone_val_probs)
        ),
        "selection_score": float(selection_score),
    })

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {train_loss:.4f} | "
        f"MLAAD Val F1 {mlaad_val_f1 * 100:.2f}% | "
        f"Phone Val Genuine {phone_val_acc * 100:.2f}% | "
        f"Score {selection_score * 100:.2f}%"
    )

    if selection_score > best_selection_score + 1e-4:
        best_selection_score = float(selection_score)
        best_mlaad_val_f1 = float(mlaad_val_f1)
        best_phone_val_acc = float(phone_val_acc)
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "hidden_size": HIDDEN_SIZE,
                "embedding_size": EMBEDDING_SIZE,
                "best_selection_score": best_selection_score,
                "best_mlaad_validation_macro_f1": best_mlaad_val_f1,
                "best_phone_validation_accuracy": best_phone_val_acc,
                "config": {
                    "version": "F1B_V7_2_SYMMETRIC_CODEC_PHONE_ADAPT",
                    "sr": SR,
                    "max_seconds": MAX_SECONDS,
                    "wavlm_path": str(WAVLM_DIR),
                    "pooling": "mean_plus_std",
                    "phone_max_windows_per_file": PHONE_MAX_WINDOWS_PER_FILE,
                    "phone_train_repeat": PHONE_TRAIN_REPEAT,
                    "selection_score": "0.70*MLAAD_macroF1 + 0.30*phone_genuine_accuracy",
                },
            },
            MODEL_PATH,
        )

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break


pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "training_history_v7_2.csv",
    index=False,
)

# Trusted local checkpoint generated by this script.
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

# Re-load saved normalization exactly.
feature_mean = np.asarray(
    checkpoint["feature_mean"],
    dtype=np.float32,
)

feature_std = np.asarray(
    checkpoint["feature_std"],
    dtype=np.float32,
)

# ============================================================
# FINAL MLAAD UNSEEN-GENERATOR TEST
# ============================================================

test_true, test_pred, test_probs = predict_loader(
    test_mlaad_loader
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
) = precision_recall_fscore_support(
    test_true,
    test_pred,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 78)
print("FINAL MLAAD UNSEEN-GENERATOR TEST - V7.2")
print("=" * 78)

print(f"Accuracy        : {test_acc * 100:.2f}%")
print(f"Macro Precision : {test_precision * 100:.2f}%")
print(f"Macro Recall    : {test_recall * 100:.2f}%")
print(f"Macro F1        : {test_f1 * 100:.2f}%")

print("Confusion [genuine=0, deepfake=1]:")
print(
    confusion_matrix(
        test_true,
        test_pred,
        labels=[0, 1],
    )
)

print(
    classification_report(
        test_true,
        test_pred,
        labels=[0, 1],
        target_names=[
            "genuine",
            "deepfake",
        ],
        digits=4,
        zero_division=0,
    )
)

test_meta = test_meta.copy()
test_meta["deepfake_probability_v7_2"] = test_probs
test_meta["prediction_v7_2"] = test_pred

test_meta.to_csv(
    OUTPUT_DIR / "unseen_generator_test_results_v7_2.csv",
    index=False,
)

# ============================================================
# PER-UNSEEN-GENERATOR PERFORMANCE
# ============================================================

generator_rows = []

print("\n" + "=" * 78)
print("UNSEEN GENERATOR PERFORMANCE - V7.2")
print("=" * 78)

for generator in sorted(
    test_meta.loc[
        test_meta["label"] == 1,
        "generator"
    ].unique()
):
    subset = test_meta[
        test_meta["generator"] == generator
    ]

    detection_rate = float(
        np.mean(
            subset["prediction_v7_2"].values == 1
        )
    )

    mean_prob = float(
        subset["deepfake_probability_v7_2"].mean()
    )

    median_prob = float(
        subset["deepfake_probability_v7_2"].median()
    )

    generator_rows.append({
        "generator": generator,
        "n_files": len(subset),
        "deepfake_detection_rate": detection_rate,
        "mean_deepfake_probability": mean_prob,
        "median_deepfake_probability": median_prob,
    })

generator_df = pd.DataFrame(generator_rows)

print(
    generator_df.to_string(
        index=False
    )
)

generator_df.to_csv(
    OUTPUT_DIR / "unseen_generator_performance_v7_2.csv",
    index=False,
)

# ============================================================
# EXTERNAL AUDIO / WINDOW ANALYSIS
# ============================================================

def load_full_audio(path):
    y = load_audio_robust(path)
    return normalize_audio(y)


def external_windows(y):
    return make_nonoverlap_windows(
        y,
        max_windows=None,
    )


def predict_waveforms(waveforms):
    if not waveforms:
        raise RuntimeError(
            "No valid external audio windows."
        )

    all_embeddings = []

    for start in range(
        0,
        len(waveforms),
        feature_batch_size,
    ):
        batch = waveforms[
            start:start + feature_batch_size
        ]

        emb = wavlm_batch_embeddings(batch)
        all_embeddings.append(emb)

    X = np.concatenate(
        all_embeddings,
        axis=0,
    ).astype(np.float32)

    X = (
        (X - feature_mean) / feature_std
    ).astype(np.float32)

    loader = make_loader(
        X,
        np.zeros(len(X), dtype=np.int64),
        False,
    )

    _, _, probs = predict_loader(loader)

    return probs


def analyze_external(path):
    y = load_full_audio(path)
    windows = external_windows(y)

    probs = predict_waveforms(windows)

    median_prob = float(
        np.median(probs)
    )

    return {
        "file": Path(path).name,
        "windows": int(len(probs)),
        "prediction": (
            "DEEPFAKE"
            if median_prob >= 0.50
            else "GENUINE"
        ),
        "mean_deepfake_probability": float(
            np.mean(probs)
        ),
        "median_deepfake_probability": median_prob,
        "p75_deepfake_probability": float(
            np.percentile(probs, 75)
        ),
        "p90_deepfake_probability": float(
            np.percentile(probs, 90)
        ),
        "max_deepfake_probability": float(
            np.max(probs)
        ),
        "suspicious_ratio": float(
            np.mean(probs >= 0.50)
        ),
        "high_ratio": float(
            np.mean(probs >= 0.70)
        ),
        "window_probabilities": [
            float(x) for x in probs
        ],
    }


# ============================================================
# FINAL SPEAKER-DISJOINT PHONE TEST
# ============================================================

print("\n" + "=" * 78)
print("FINAL UNSEEN-SPEAKER PERSONAL PHONE TEST - V7.2")
print("=" * 78)

phone_test_results = []
phone_file_correct = 0

for index, row in phone_test_df.iterrows():
    path = Path(row["path"])

    try:
        result = analyze_external(path)
        result["speaker_id"] = row["speaker_id"]

        phone_test_results.append(result)

        if result["prediction"] == "GENUINE":
            phone_file_correct += 1

        print(
            f"[{index + 1}/{len(phone_test_df)}] "
            f"{path.name} | "
            f"speaker={row['speaker_id']} | "
            f"{result['prediction']} | "
            f"median DF "
            f"{result['median_deepfake_probability'] * 100:.2f}% | "
            f"max "
            f"{result['max_deepfake_probability'] * 100:.2f}%"
        )

    except Exception as exc:
        print(
            f"[{index + 1}/{len(phone_test_df)}] "
            f"{path.name} | ERROR: {exc}"
        )


phone_file_accuracy = None
phone_false_deepfake_rate = None

if phone_test_results:
    phone_file_accuracy = (
        phone_file_correct / len(phone_test_results)
    )

    phone_false_deepfake_rate = (
        1.0 - phone_file_accuracy
    )

    median_phone_df = float(
        np.median([
            x["median_deepfake_probability"]
            for x in phone_test_results
        ])
    )

    print(
        "\nUnseen phone speaker genuine accuracy : "
        f"{phone_file_accuracy * 100:.2f}%"
    )

    print(
        "Unseen phone false deepfake rate     : "
        f"{phone_false_deepfake_rate * 100:.2f}%"
    )

    print(
        "Median unseen-phone DF score         : "
        f"{median_phone_df * 100:.2f}%"
    )

with open(
    OUTPUT_DIR / "phone_unseen_speaker_results_v7_2.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        phone_test_results,
        f,
        indent=2,
        ensure_ascii=False,
    )

# Window-level phone test diagnostic.
phone_window_true, phone_window_pred, phone_window_probs = predict_loader(
    phone_test_loader
)

phone_window_accuracy = accuracy_score(
    phone_window_true,
    phone_window_pred,
)

print(
    "Phone TEST window-level genuine acc  : "
    f"{phone_window_accuracy * 100:.2f}%"
)

# ============================================================
# DIFFICULT UNSEEN DEEPFAKE CHALLENGE
# ============================================================

challenge = None

print("\n" + "=" * 78)
print("DIFFICULT KNOWN DEEPFAKE - UNSEEN CHALLENGE - V7.2")
print("=" * 78)

if DIFFICULT_DEEPFAKE.exists():
    challenge = analyze_external(
        DIFFICULT_DEEPFAKE
    )

    print(f"File             : {challenge['file']}")
    print(f"Windows          : {challenge['windows']}")
    print(f"Prediction       : {challenge['prediction']}")

    print(
        "Mean Deepfake    : "
        f"{challenge['mean_deepfake_probability'] * 100:.2f}%"
    )

    print(
        "Median Deepfake  : "
        f"{challenge['median_deepfake_probability'] * 100:.2f}%"
    )

    print(
        "P75 Deepfake     : "
        f"{challenge['p75_deepfake_probability'] * 100:.2f}%"
    )

    print(
        "P90 Deepfake     : "
        f"{challenge['p90_deepfake_probability'] * 100:.2f}%"
    )

    print(
        "Maximum Evidence : "
        f"{challenge['max_deepfake_probability'] * 100:.2f}%"
    )

    print(
        "Suspicious Ratio : "
        f"{challenge['suspicious_ratio'] * 100:.2f}%"
    )

    print(
        "High-Risk Ratio  : "
        f"{challenge['high_ratio'] * 100:.2f}%"
    )

    probs = challenge["window_probabilities"]

    challenge_df = pd.DataFrame({
        "window_index": np.arange(len(probs)),
        "start_seconds": np.arange(len(probs)) * MAX_SECONDS,
        "end_seconds": (
            np.arange(len(probs)) + 1
        ) * MAX_SECONDS,
        "deepfake_probability": probs,
    })

    challenge_df.to_csv(
        OUTPUT_DIR / "challenge_window_evidence_v7_2.csv",
        index=False,
    )

else:
    print(
        f"Challenge file not found: "
        f"{DIFFICULT_DEEPFAKE}"
    )

# ============================================================
# SAVE EXPERIMENT SUMMARY
# ============================================================

summary = {
    "version": "F1B_V7_2_WAVLM_SYMMETRIC_CODEC_PHONE_ADAPT",
    "device": str(DEVICE),

    "mlaad_train_files": int(len(train_df)),
    "mlaad_validation_files": int(len(val_df)),
    "mlaad_test_files": int(len(test_df)),

    "phone_total_files": int(len(phone_manifest)),
    "phone_total_speakers": int(
        phone_manifest["speaker_id"].nunique()
    ),

    "phone_train_speakers": sorted(
        list(train_speakers)
    ),
    "phone_validation_speakers": sorted(
        list(val_speakers)
    ),
    "phone_test_speakers": sorted(
        list(test_speakers)
    ),

    "phone_train_files": int(len(phone_train_df)),
    "phone_validation_files": int(len(phone_val_df)),
    "phone_test_files": int(len(phone_test_df)),

    "phone_train_windows": int(len(X_phone_train)),
    "phone_validation_windows": int(len(X_phone_val)),
    "phone_test_windows": int(len(X_phone_test)),
    "phone_train_repeat": int(PHONE_TRAIN_REPEAT),

    "codec_aug_files_requested_per_class": int(
        CODEC_AUG_FILES_PER_CLASS
    ),
    "codec_aug_successful_embeddings": int(
        len(X_codec_aug)
    ),
    "codec_aug_genuine_embeddings": int(
        np.sum(y_codec_aug == 0)
    ),
    "codec_aug_deepfake_embeddings": int(
        np.sum(y_codec_aug == 1)
    ),

    "best_selection_score": float(
        checkpoint["best_selection_score"]
    ),
    "best_mlaad_validation_macro_f1": float(
        checkpoint["best_mlaad_validation_macro_f1"]
    ),
    "best_phone_validation_accuracy": float(
        checkpoint["best_phone_validation_accuracy"]
    ),

    "unseen_generator_test_accuracy": float(
        test_acc
    ),
    "unseen_generator_test_macro_f1": float(
        test_f1
    ),

    "unseen_phone_file_genuine_accuracy": (
        None
        if phone_file_accuracy is None
        else float(phone_file_accuracy)
    ),

    "unseen_phone_false_deepfake_rate": (
        None
        if phone_false_deepfake_rate is None
        else float(phone_false_deepfake_rate)
    ),

    "unseen_phone_window_genuine_accuracy": float(
        phone_window_accuracy
    ),

    "challenge": challenge,
    "model_path": str(MODEL_PATH),
}

with open(
    OUTPUT_DIR / "experiment_results_v7_2.json",
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

print("\n" + "=" * 78)
print("F1B V7.2 SYMMETRIC CODEC + PHONE ADAPTATION COMPLETE")
print("=" * 78)

print(
    f"Best MLAAD validation Macro F1 : "
    f"{checkpoint['best_mlaad_validation_macro_f1'] * 100:.2f}%"
)

print(
    f"Best phone validation accuracy : "
    f"{checkpoint['best_phone_validation_accuracy'] * 100:.2f}%"
)

print(
    f"MLAAD unseen-generator test F1 : "
    f"{test_f1 * 100:.2f}%"
)

if phone_file_accuracy is not None:
    print(
        f"Unseen phone genuine accuracy  : "
        f"{phone_file_accuracy * 100:.2f}%"
    )

    print(
        f"Unseen phone false DF rate     : "
        f"{phone_false_deepfake_rate * 100:.2f}%"
    )

if challenge is not None:
    print(
        f"Challenge median deepfake      : "
        f"{challenge['median_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"Challenge P90 deepfake         : "
        f"{challenge['p90_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"Challenge suspicious windows   : "
        f"{challenge['suspicious_ratio'] * 100:.2f}%"
    )

print("\nOutputs:")
print(OUTPUT_DIR)
print(MODEL_PATH)
print(PHONE_SPLIT_CSV)

print("\nIMPORTANT:")
print(
    "V7.2 is still experimental. Do NOT integrate it into "
    "SonicT F1-F5 until the unseen-phone and challenge "
    "results are both acceptable."
)

print(
    "The difficult challenge deepfake and the phone TEST "
    "speakers were not used for training."
)
