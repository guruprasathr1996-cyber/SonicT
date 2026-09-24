from pathlib import Path
import subprocess
import tempfile
import shutil
import warnings
import json
import re
import random

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ============================================================
# SONICT F1B V6.1
# Large-data, speaker-disjoint, codec-robust Log-Mel CNN
#
# Uses:
#   1000 bonafide + 1000 spoof manifest from V6 selector
#
# Important design:
# - split by SPEAKER first: train / validation / test
# - same channel transforms for genuine and spoof
# - challenge deepfake NEVER used in training
# - personal phone speakers split separately
# - 3-second windows
# - only 2 windows per file/condition to control RAM/runtime
# - 4 focused channel conditions
# ============================================================

SEED = 42

SR = 16000
WINDOW_SECONDS = 3.0
WINDOW_SAMPLES = int(SR * WINDOW_SECONDS)

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

MAX_WINDOWS_PER_FILE = 2

EPOCHS = 20
BATCH_SIZE = 48
LEARNING_RATE = 8e-4
PATIENCE = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

MANIFEST = Path(
    r"H:\SonicT_Compression_Test\asvspoof_v6_balanced\manifest_v6.csv"
)

PERSONAL_GENUINE_DIR = Path(
    r"H:\SonicT_Compression_Test\genuine"
)

DIFFICULT_DEEPFAKE = Path(
    r"E:\Downloads\recording_6ee192c5-aab6-4a46-8afc-ee9985bb9aaf.mp3.mpeg"
)

OUTPUT_DIR = Path(
    r"H:\SonicT_Compression_Test\experiment_v6_1_output"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_ROOT = Path(
    tempfile.mkdtemp(prefix="sonict_f1b_v6_")
)

BEST_MODEL_PATH = (
    OUTPUT_DIR / "codec_robust_large_cnn_v6_1.pt"
)

# ------------------------------------------------------------
# FOCUSED CHANNEL CONDITIONS
# ------------------------------------------------------------

CONDITIONS = {
    "clean16k": [],

    "mp3_8k_16k": [
        "-ar", "8000",
        "-ac", "1",
        "-b:a", "16k",
        "-c:a", "libmp3lame",
    ],

    "telephone_bandpass": [
        "-af", "highpass=f=300,lowpass=f=3400",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
    ],

    "telephone_mp3_8k_16k": [
        "-af", "highpass=f=300,lowpass=f=3400",
        "-ar", "8000",
        "-ac", "1",
        "-b:a", "16k",
        "-c:a", "libmp3lame",
    ],
}

PHONE_TRAIN_CONDITIONS = list(CONDITIONS.keys())

AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".flac",
    ".aac", ".ogg", ".opus", ".mpeg",
    ".mpg", ".wma", ".amr", ".3gp",
}

# ============================================================
# UTILITIES
# ============================================================

def run_ffmpeg(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return result.returncode == 0, result.stderr


def cleanup(path):
    try:
        p = Path(path)

        if p.exists():
            p.unlink()

    except Exception:
        pass


def output_suffix(condition):
    if "mp3" in condition:
        return ".mp3"

    return ".wav"


def transform_audio(
    input_file,
    condition,
    output_wav,
):
    """
    Apply channel condition, then always decode to
    16-kHz mono PCM WAV before feature extraction.

    V6.1 FIX:
    The previous V6 used output_wav itself as the temporary
    WAV for telephone_bandpass. FFmpeg then attempted to read
    and write the same file during normalization, causing that
    condition to fail. V6.1 always uses a DISTINCT stage file.
    """

    stage_file = None

    try:
        if condition == "clean16k":
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel", "error",
                "-i", str(input_file),
                "-ar", str(SR),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(output_wav),
            ]

            return run_ffmpeg(cmd)

        suffix = output_suffix(condition)

        # Always create a different file from output_wav.
        stage_file = output_wav.with_name(
            output_wav.stem
            + "_stage"
            + suffix
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", str(input_file),
        ] + CONDITIONS[condition] + [
            str(stage_file)
        ]

        ok, error = run_ffmpeg(cmd)

        if not ok:
            return False, error

        decode_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", str(stage_file),
            "-ar", str(SR),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_wav),
        ]

        return run_ffmpeg(decode_cmd)

    finally:
        if stage_file is not None:
            cleanup(stage_file)

def phone_speaker_id(path):
    name = Path(path).stem

    name = re.sub(
        r"_\d{6}_\d{6}$",
        "",
        name,
    )

    if name.lower().startswith("call "):
        name = name[5:]

    name = name.lower().strip()
    name = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        name,
    )
    name = re.sub(
        r"\s+",
        "_",
        name,
    ).strip("_")

    return "phone::" + (
        name or Path(path).stem.lower()
    )


def load_audio(path):
    y, _ = librosa.load(
        path,
        sr=SR,
        mono=True,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    if len(y) < SR:
        raise ValueError(
            "Audio shorter than 1 second"
        )

    peak = float(
        np.max(np.abs(y))
    )

    if peak > 0:
        y = y / peak

    return y


def make_windows(
    y,
    max_windows=None,
):
    windows = []

    for start in range(
        0,
        len(y) - WINDOW_SAMPLES + 1,
        WINDOW_SAMPLES,
    ):
        windows.append(
            y[start:start + WINDOW_SAMPLES]
        )

    if not windows and len(y) >= SR:
        padded = np.zeros(
            WINDOW_SAMPLES,
            dtype=np.float32,
        )

        padded[:len(y)] = y[:WINDOW_SAMPLES]
        windows = [padded]

    if (
        max_windows is not None
        and len(windows) > max_windows
    ):
        indices = np.linspace(
            0,
            len(windows) - 1,
            max_windows,
            dtype=int,
        )

        windows = [
            windows[i]
            for i in indices
        ]

    return windows


def logmel(window):
    mel = librosa.feature.melspectrogram(
        y=window,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=SR // 2,
        power=2.0,
    )

    x = librosa.power_to_db(
        mel + 1e-10,
        ref=np.max,
    )

    # Normalize each 3-second window.
    x = (
        x - float(np.mean(x))
    ) / (
        float(np.std(x)) + 1e-6
    )

    return x.astype(np.float16)


def extract_logmels(
    wav_path,
    max_windows=None,
):
    y = load_audio(wav_path)

    windows = make_windows(
        y,
        max_windows=max_windows,
    )

    return [
        logmel(window)
        for window in windows
    ]


# ============================================================
# LOAD V6 MANIFEST
# ============================================================

if not MANIFEST.exists():
    raise FileNotFoundError(
        f"V6 manifest not found: {MANIFEST}"
    )

manifest = pd.read_csv(MANIFEST)

required_columns = {
    "speaker",
    "file_id",
    "label",
    "source_path",
}

missing = (
    required_columns
    - set(manifest.columns)
)

if missing:
    raise RuntimeError(
        f"Manifest missing columns: "
        f"{sorted(missing)}"
    )

manifest["speaker"] = (
    manifest["speaker"]
    .astype(str)
)

manifest["file_id"] = (
    manifest["file_id"]
    .astype(str)
)

manifest["label_name"] = (
    manifest["label"]
    .astype(str)
    .str.lower()
)

manifest["label"] = (
    manifest["label_name"]
    .map({
        "bonafide": 0,
        "spoof": 1,
    })
)

if manifest["label"].isna().any():
    raise RuntimeError(
        "Unexpected label in V6 manifest."
    )

manifest["label"] = (
    manifest["label"]
    .astype(int)
)

manifest["path"] = (
    manifest["source_path"]
    .apply(Path)
)

exists_mask = manifest["path"].apply(
    lambda p: p.exists()
)

missing_count = int(
    (~exists_mask).sum()
)

if missing_count:
    print(
        f"WARNING: {missing_count} manifest "
        f"audio files are missing locally."
    )

manifest = (
    manifest[exists_mask]
    .reset_index(drop=True)
)

# ============================================================
# SPEAKER-DISJOINT TRAIN / VAL / TEST SPLIT
# ============================================================

def valid_binary_split(
    df,
    test_size,
    seed_start,
):
    labels = df["label"].values
    groups = df["speaker"].values
    dummy = np.zeros(
        (len(df), 1)
    )

    best = None

    for seed in range(
        seed_start,
        seed_start + 300,
    ):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )

        train_idx, test_idx = next(
            splitter.split(
                dummy,
                labels,
                groups,
            )
        )

        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        if (
            len(np.unique(train_labels)) < 2
            or len(np.unique(test_labels)) < 2
        ):
            continue

        test_ratio = float(
            np.mean(test_labels)
        )

        train_ratio = float(
            np.mean(train_labels)
        )

        # Prefer reasonably balanced class ratios.
        score = (
            abs(test_ratio - 0.5)
            + abs(train_ratio - 0.5)
        )

        if (
            best is None
            or score < best[0]
        ):
            best = (
                score,
                seed,
                train_idx,
                test_idx,
            )

    if best is None:
        raise RuntimeError(
            "Could not create valid "
            "speaker-disjoint split."
        )

    _, seed, train_idx, test_idx = best

    return (
        df.iloc[train_idx]
        .reset_index(drop=True),
        df.iloc[test_idx]
        .reset_index(drop=True),
        seed,
    )


# First hold out final TEST speakers.
train_val_df, test_df, test_seed = (
    valid_binary_split(
        manifest,
        test_size=0.20,
        seed_start=SEED,
    )
)

# Then hold out VALIDATION speakers from remaining.
train_df, val_df, val_seed = (
    valid_binary_split(
        train_val_df,
        test_size=0.20,
        seed_start=SEED + 500,
    )
)

train_speakers = set(
    train_df["speaker"]
)

val_speakers = set(
    val_df["speaker"]
)

test_speakers = set(
    test_df["speaker"]
)

if (
    train_speakers & val_speakers
    or train_speakers & test_speakers
    or val_speakers & test_speakers
):
    raise RuntimeError(
        "Speaker leakage detected!"
    )

# ============================================================
# PERSONAL PHONE SPEAKER SPLIT
# ============================================================

phone_files = sorted([
    p
    for p in PERSONAL_GENUINE_DIR.iterdir()
    if (
        p.is_file()
        and p.suffix.lower() in AUDIO_EXTS
    )
])

phone_df = pd.DataFrame([
    {
        "path": p,
        "file_id": p.stem,
        "speaker": phone_speaker_id(p),
        "label": 0,
        "label_name": "genuine",
    }
    for p in phone_files
])

if phone_df.empty:
    raise RuntimeError(
        "No personal genuine phone "
        "recordings found."
    )

phone_speakers = sorted(
    phone_df["speaker"].unique()
)

rng = np.random.default_rng(SEED)
rng.shuffle(phone_speakers)

phone_test_count = max(
    1,
    int(
        round(
            len(phone_speakers) * 0.35
        )
    ),
)

phone_test_count = min(
    phone_test_count,
    len(phone_speakers) - 1,
)

phone_test_speakers = set(
    phone_speakers[
        :phone_test_count
    ]
)

phone_train_df = (
    phone_df[
        ~phone_df["speaker"]
        .isin(phone_test_speakers)
    ]
    .reset_index(drop=True)
)

phone_test_df = (
    phone_df[
        phone_df["speaker"]
        .isin(phone_test_speakers)
    ]
    .reset_index(drop=True)
)

# ============================================================
# SPLIT SUMMARY
# ============================================================

def split_summary(name, df):
    print("\n" + name)
    print("-" * len(name))

    print(
        f"Files            : {len(df)}"
    )

    print(
        f"Speakers         : "
        f"{df['speaker'].nunique()}"
    )

    print("Class counts:")
    print(
        df["label_name"]
        .value_counts()
    )


print("=" * 78)
print(
    "SONICT F1B V6.1 - "
    "LARGE-DATA CODEC-ROBUST CNN"
)
print("=" * 78)

print(f"Device            : {DEVICE}")
print(f"Test split seed   : {test_seed}")
print(f"Val split seed    : {val_seed}")
print(
    f"Missing source files: "
    f"{missing_count}"
)

split_summary(
    "ASV TRAIN",
    train_df,
)

split_summary(
    "ASV VALIDATION",
    val_df,
)

split_summary(
    "ASV FINAL TEST",
    test_df,
)

print("\nPHONE DOMAIN")
print("------------")

print(
    f"Train files       : "
    f"{len(phone_train_df)}"
)

print(
    f"Test files        : "
    f"{len(phone_test_df)}"
)

print(
    f"Train speakers    : "
    f"{phone_train_df['speaker'].nunique()}"
)

print(
    f"Test speakers     : "
    f"{phone_test_df['speaker'].nunique()}"
)

# Save exact split assignments.
train_df.to_csv(
    OUTPUT_DIR / "asv_train_split_v6_1.csv",
    index=False,
)

val_df.to_csv(
    OUTPUT_DIR / "asv_validation_split_v6_1.csv",
    index=False,
)

test_df.to_csv(
    OUTPUT_DIR / "asv_test_split_v6_1.csv",
    index=False,
)

# ============================================================
# V6.1 PRE-FLIGHT CONDITION CHECK
# ============================================================

print("\n" + "=" * 78)
print("PRE-FLIGHT CHANNEL CONDITION CHECK")
print("=" * 78)

preflight_source = Path(train_df.iloc[0]["path"])
preflight_failures = []

for condition in CONDITIONS:
    test_wav = (
        TEMP_ROOT
        / f"preflight_{condition}.wav"
    )

    try:
        ok, error = transform_audio(
            preflight_source,
            condition,
            test_wav,
        )

        if not ok:
            preflight_failures.append(
                (condition, error)
            )
            print(
                f"{condition:28s}: FAILED"
            )
        else:
            # Also verify librosa can actually decode the result.
            _ = extract_logmels(
                test_wav,
                max_windows=1,
            )
            print(
                f"{condition:28s}: OK"
            )

    except Exception as exc:
        preflight_failures.append(
            (condition, str(exc))
        )
        print(
            f"{condition:28s}: FAILED"
        )

    finally:
        cleanup(test_wav)

if preflight_failures:
    print("\nPre-flight errors:")
    for condition, error in preflight_failures:
        print(
            f"\n[{condition}]\n{error}"
        )

    shutil.rmtree(
        TEMP_ROOT,
        ignore_errors=True,
    )

    raise RuntimeError(
        "One or more channel conditions failed "
        "pre-flight. Training was stopped before "
        "feature extraction."
    )

print(
    "\nAll channel conditions passed pre-flight."
)

# ============================================================
# LOG-MEL EXTRACTION
# ============================================================

failed = []


def build_windows(
    df,
    conditions,
    split_name,
):
    X = []
    y = []
    meta = []

    total = len(df)

    for row_index, row in df.iterrows():
        source = Path(row["path"])

        print(
            f"[{split_name} "
            f"{row_index + 1}/{total}] "
            f"{source.name}"
        )

        for condition in conditions:
            temp_wav = (
                TEMP_ROOT
                / (
                    f"{split_name}_"
                    f"{row_index}_"
                    f"{condition}.wav"
                )
            )

            try:
                ok, error = transform_audio(
                    source,
                    condition,
                    temp_wav,
                )

                if not ok:
                    failed.append({
                        "split": split_name,
                        "file": source.name,
                        "condition": condition,
                        "error": error.strip(),
                    })
                    continue

                feature_windows = (
                    extract_logmels(
                        temp_wav,
                        max_windows=(
                            MAX_WINDOWS_PER_FILE
                        ),
                    )
                )

                for (
                    window_index,
                    feature,
                ) in enumerate(
                    feature_windows
                ):
                    X.append(feature)

                    y.append(
                        int(row["label"])
                    )

                    meta.append({
                        "file_id": str(
                            row["file_id"]
                        ),
                        "speaker": str(
                            row["speaker"]
                        ),
                        "label": int(
                            row["label"]
                        ),
                        "label_name": str(
                            row["label_name"]
                        ),
                        "condition": condition,
                        "window_index": (
                            window_index
                        ),
                        "split": split_name,
                    })

            except Exception as exc:
                failed.append({
                    "split": split_name,
                    "file": source.name,
                    "condition": condition,
                    "error": str(exc),
                })

            finally:
                cleanup(temp_wav)

    if not X:
        raise RuntimeError(
            f"No windows extracted for "
            f"{split_name}."
        )

    return (
        np.asarray(
            X,
            dtype=np.float16,
        ),
        np.asarray(
            y,
            dtype=np.int64,
        ),
        pd.DataFrame(meta),
    )


print("\n" + "=" * 78)
print("EXTRACTING ASV TRAIN WINDOWS")
print("=" * 78)

X_train, y_train, train_meta = (
    build_windows(
        train_df,
        list(CONDITIONS.keys()),
        "asv_train",
    )
)

# Add small real-phone genuine anchor.
print("\n" + "=" * 78)
print("EXTRACTING PHONE TRAIN WINDOWS")
print("=" * 78)

X_phone, y_phone, phone_meta = (
    build_windows(
        phone_train_df,
        PHONE_TRAIN_CONDITIONS,
        "phone_train",
    )
)

X_train = np.concatenate(
    [X_train, X_phone],
    axis=0,
)

y_train = np.concatenate(
    [y_train, y_phone],
    axis=0,
)

train_meta = pd.concat(
    [train_meta, phone_meta],
    ignore_index=True,
)

del X_phone
del y_phone

print("\n" + "=" * 78)
print("EXTRACTING ASV VALIDATION WINDOWS")
print("=" * 78)

X_val, y_val, val_meta = (
    build_windows(
        val_df,
        list(CONDITIONS.keys()),
        "asv_validation",
    )
)

print("\n" + "=" * 78)
print("EXTRACTING ASV FINAL TEST WINDOWS")
print("=" * 78)

X_test, y_test, test_meta = (
    build_windows(
        test_df,
        list(CONDITIONS.keys()),
        "asv_test",
    )
)

train_meta.to_csv(
    OUTPUT_DIR / "train_window_manifest_v6_1.csv",
    index=False,
)

val_meta.to_csv(
    OUTPUT_DIR / "validation_window_manifest_v6_1.csv",
    index=False,
)

test_meta.to_csv(
    OUTPUT_DIR / "test_window_manifest_v6_1.csv",
    index=False,
)

failed_df = pd.DataFrame(failed)

failed_df.to_csv(
    OUTPUT_DIR / "failed_samples_v6_1.csv",
    index=False,
)

print("\nWINDOW SUMMARY")
print("--------------")

print(
    f"Train windows      : "
    f"{len(X_train)}"
)

print(
    f"Validation windows : "
    f"{len(X_val)}"
)

print(
    f"Test windows       : "
    f"{len(X_test)}"
)

print(
    f"Failed transforms  : "
    f"{len(failed_df)}"
)

if not failed_df.empty:
    print("\nFailure breakdown by condition:")
    print(
        failed_df["condition"]
        .value_counts()
        .to_string()
    )

# ============================================================
# PYTORCH DATASET
# ============================================================

class MelDataset(Dataset):

    def __init__(
        self,
        X,
        y,
    ):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = torch.tensor(
            self.X[index],
            dtype=torch.float32,
        ).unsqueeze(0)

        label = torch.tensor(
            self.y[index],
            dtype=torch.long,
        )

        return x, label


train_loader = DataLoader(
    MelDataset(
        X_train,
        y_train,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    MelDataset(
        X_val,
        y_val,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    MelDataset(
        X_test,
        y_test,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# ============================================================
# CNN
# ============================================================

class CodecRobustCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                96,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (4, 4)
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                96 * 4 * 4,
                128,
            ),

            nn.ReLU(),
            nn.Dropout(0.40),

            nn.Linear(
                128,
                2,
            ),
        )

    def forward(self, x):
        return self.classifier(
            self.features(x)
        )


model = CodecRobustCNN().to(
    DEVICE
)

# Window-level class balance.
class_counts = np.bincount(
    y_train,
    minlength=2,
).astype(np.float32)

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


def predict_loader(loader):
    model.eval()

    true_labels = []
    predictions = []
    deepfake_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(
                DEVICE
            )

            logits = model(X_batch)

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )
            )

            true_labels.extend(
                y_batch.numpy().tolist()
            )

            predictions.extend(
                torch.argmax(
                    probabilities,
                    dim=1,
                )
                .cpu()
                .numpy()
                .tolist()
            )

            deepfake_probs.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        np.asarray(true_labels),
        np.asarray(predictions),
        np.asarray(deepfake_probs),
    )


# ============================================================
# TRAINING
# ============================================================

best_val_f1 = -1.0
epochs_without_improvement = 0
history = []

print("\n" + "=" * 78)
print("TRAINING V6 CNN")
print("=" * 78)

for epoch in range(
    1,
    EPOCHS + 1,
):
    model.train()

    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            X_batch
        )

        loss = criterion(
            logits,
            y_batch,
        )

        loss.backward()
        optimizer.step()

        running_loss += (
            float(loss.item())
            * len(y_batch)
        )

    train_loss = (
        running_loss
        / len(y_train)
    )

    val_true, val_pred, _ = (
        predict_loader(
            val_loader
        )
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
    ) = precision_recall_fscore_support(
        val_true,
        val_pred,
        average="macro",
        zero_division=0,
    )

    scheduler.step(
        val_f1
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_accuracy": (
            val_acc
        ),
        "validation_macro_precision": (
            val_precision
        ),
        "validation_macro_recall": (
            val_recall
        ),
        "validation_macro_f1": (
            val_f1
        ),
    })

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {train_loss:.4f} | "
        f"Val Acc "
        f"{val_acc * 100:.2f}% | "
        f"Val F1 "
        f"{val_f1 * 100:.2f}%"
    )

    if (
        val_f1
        > best_val_f1 + 1e-4
    ):
        best_val_f1 = val_f1

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "best_validation_macro_f1":
                    float(best_val_f1),

                "config": {
                    "sr": SR,
                    "window_seconds":
                        WINDOW_SECONDS,
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "hop_length":
                        HOP_LENGTH,
                },
            },
            BEST_MODEL_PATH,
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

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "training_history_v6_1.csv",
    index=False,
)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

# ============================================================
# FINAL ASV TEST - WINDOW LEVEL
# ============================================================

test_true, test_pred, test_probs = (
    predict_loader(
        test_loader
    )
)

window_accuracy = accuracy_score(
    test_true,
    test_pred,
)

(
    window_precision,
    window_recall,
    window_f1,
    _,
) = precision_recall_fscore_support(
    test_true,
    test_pred,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 78)
print("FINAL UNSEEN-SPEAKER ASV WINDOW TEST")
print("=" * 78)

print(
    f"Accuracy        : "
    f"{window_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision : "
    f"{window_precision * 100:.2f}%"
)

print(
    f"Macro Recall    : "
    f"{window_recall * 100:.2f}%"
)

print(
    f"Macro F1        : "
    f"{window_f1 * 100:.2f}%"
)

print(
    "Confusion "
    "[genuine=0, deepfake=1]:"
)

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

# ============================================================
# FILE-LEVEL AGGREGATION
# ============================================================

test_meta = test_meta.copy()

test_meta[
    "deepfake_probability"
] = test_probs

file_rows = []

for (
    keys,
    group,
) in test_meta.groupby(
    [
        "file_id",
        "speaker",
        "label",
        "label_name",
        "condition",
    ]
):
    (
        file_id,
        speaker,
        label,
        label_name,
        condition,
    ) = keys

    probabilities = (
        group[
            "deepfake_probability"
        ]
        .values
    )

    mean_probability = float(
        np.mean(probabilities)
    )

    median_probability = float(
        np.median(probabilities)
    )

    max_probability = float(
        np.max(probabilities)
    )

    final_probability = (
        median_probability
    )

    final_prediction = int(
        final_probability >= 0.50
    )

    file_rows.append({
        "file_id": file_id,
        "speaker": speaker,
        "label": int(label),
        "label_name": label_name,
        "condition": condition,
        "windows": len(probabilities),

        "mean_deepfake_probability":
            mean_probability,

        "median_deepfake_probability":
            median_probability,

        "max_deepfake_probability":
            max_probability,

        "suspicious_ratio": float(
            np.mean(
                probabilities >= 0.50
            )
        ),

        "high_ratio": float(
            np.mean(
                probabilities >= 0.70
            )
        ),

        "prediction":
            final_prediction,
    })

file_df = pd.DataFrame(
    file_rows
)

file_accuracy = accuracy_score(
    file_df["label"],
    file_df["prediction"],
)

(
    file_precision,
    file_recall,
    file_f1,
    _,
) = precision_recall_fscore_support(
    file_df["label"],
    file_df["prediction"],
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 78)
print(
    "FINAL ASV FILE TEST - "
    "MEDIAN WINDOW AGGREGATION"
)
print("=" * 78)

print(
    f"Accuracy        : "
    f"{file_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision : "
    f"{file_precision * 100:.2f}%"
)

print(
    f"Macro Recall    : "
    f"{file_recall * 100:.2f}%"
)

print(
    f"Macro F1        : "
    f"{file_f1 * 100:.2f}%"
)

print(
    "Confusion "
    "[genuine=0, deepfake=1]:"
)

print(
    confusion_matrix(
        file_df["label"],
        file_df["prediction"],
        labels=[0, 1],
    )
)

file_df.to_csv(
    OUTPUT_DIR / "asv_file_results_v6_1.csv",
    index=False,
)

# Per-condition final test.
condition_rows = []

print("\n" + "=" * 78)
print(
    "PER-CONDITION FINAL TEST"
)
print("=" * 78)

for condition in sorted(
    file_df["condition"].unique()
):
    subset = file_df[
        file_df["condition"]
        == condition
    ]

    accuracy = accuracy_score(
        subset["label"],
        subset["prediction"],
    )

    _, _, macro_f1, _ = (
        precision_recall_fscore_support(
            subset["label"],
            subset["prediction"],
            average="macro",
            zero_division=0,
        )
    )

    condition_rows.append({
        "condition": condition,
        "n_files": len(subset),
        "accuracy": float(
            accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
    })

condition_df = pd.DataFrame(
    condition_rows
)

print(
    condition_df.to_string(
        index=False
    )
)

condition_df.to_csv(
    OUTPUT_DIR / "per_condition_results_v6_1.csv",
    index=False,
)

# ============================================================
# EXTERNAL FILE ANALYSIS
# ============================================================

def analyze_external_file(path):
    normalized = (
        TEMP_ROOT
        / (
            "external_"
            + str(
                abs(
                    hash(str(path))
                )
            )
            + ".wav"
        )
    )

    try:
        ok, error = transform_audio(
            path,
            "clean16k",
            normalized,
        )

        if not ok:
            raise RuntimeError(
                error
            )

        feature_windows = (
            extract_logmels(
                normalized,
                max_windows=None,
            )
        )

        X = np.asarray(
            feature_windows,
            dtype=np.float16,
        )

        dummy_labels = np.zeros(
            len(X),
            dtype=np.int64,
        )

        loader = DataLoader(
            MelDataset(
                X,
                dummy_labels,
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        _, _, probabilities = (
            predict_loader(
                loader
            )
        )

        median_probability = float(
            np.median(probabilities)
        )

        return {
            "file": Path(path).name,
            "windows": int(
                len(probabilities)
            ),

            "prediction": (
                "DEEPFAKE"
                if median_probability >= 0.50
                else "GENUINE"
            ),

            "mean_deepfake_probability":
                float(
                    np.mean(
                        probabilities
                    )
                ),

            "median_deepfake_probability":
                median_probability,

            "max_deepfake_probability":
                float(
                    np.max(
                        probabilities
                    )
                ),

            "p75_deepfake_probability":
                float(
                    np.percentile(
                        probabilities,
                        75,
                    )
                ),

            "p90_deepfake_probability":
                float(
                    np.percentile(
                        probabilities,
                        90,
                    )
                ),

            "suspicious_ratio":
                float(
                    np.mean(
                        probabilities >= 0.50
                    )
                ),

            "high_ratio":
                float(
                    np.mean(
                        probabilities >= 0.70
                    )
                ),

            "window_probabilities": [
                float(value)
                for value in probabilities
            ],
        }

    finally:
        cleanup(
            normalized
        )


# ============================================================
# HELD-OUT PERSONAL GENUINE TEST
# ============================================================

print("\n" + "=" * 78)
print(
    "HELD-OUT PERSONAL GENUINE "
    "SPEAKER TEST"
)
print("=" * 78)

phone_results = []
phone_correct = 0

for index, row in (
    phone_test_df.iterrows()
):
    path = Path(
        row["path"]
    )

    print(
        f"[{index + 1}/"
        f"{len(phone_test_df)}] "
        f"{path.name}"
    )

    try:
        result = (
            analyze_external_file(
                path
            )
        )

        result["speaker"] = (
            row["speaker"]
        )

        phone_results.append(
            result
        )

        if (
            result["prediction"]
            == "GENUINE"
        ):
            phone_correct += 1

        print(
            f"   Prediction : "
            f"{result['prediction']}"
        )

        print(
            f"   Median DF  : "
            f"{result['median_deepfake_probability'] * 100:.2f}%"
        )

        print(
            f"   Max DF     : "
            f"{result['max_deepfake_probability'] * 100:.2f}%"
        )

        print(
            f"   Suspicious : "
            f"{result['suspicious_ratio'] * 100:.2f}%"
        )

        print(
            f"   High-Risk  : "
            f"{result['high_ratio'] * 100:.2f}%"
        )

    except Exception as exc:
        print(
            f"   ERROR: {exc}"
        )

phone_accuracy = (
    phone_correct
    / len(phone_results)
    if phone_results
    else None
)

if phone_accuracy is not None:
    print(
        "\nHeld-out personal "
        "genuine accuracy: "
        f"{phone_accuracy * 100:.2f}%"
    )

with open(
    OUTPUT_DIR
    / "heldout_personal_results_v6_1.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        phone_results,
        file,
        indent=2,
    )

# ============================================================
# HELD-OUT GENUINE EVIDENCE BASELINE
# ============================================================

if phone_results:
    phone_suspicious = np.asarray(
        [
            x["suspicious_ratio"]
            for x in phone_results
        ],
        dtype=float,
    )

    phone_high = np.asarray(
        [
            x["high_ratio"]
            for x in phone_results
        ],
        dtype=float,
    )

    phone_max = np.asarray(
        [
            x["max_deepfake_probability"]
            for x in phone_results
        ],
        dtype=float,
    )

    print("\n" + "=" * 78)
    print("HELD-OUT GENUINE WINDOW-EVIDENCE BASELINE")
    print("=" * 78)

    print(
        f"Median suspicious ratio : "
        f"{np.median(phone_suspicious) * 100:.2f}%"
    )

    print(
        f"Max suspicious ratio    : "
        f"{np.max(phone_suspicious) * 100:.2f}%"
    )

    print(
        f"Median high-risk ratio  : "
        f"{np.median(phone_high) * 100:.2f}%"
    )

    print(
        f"Max high-risk ratio     : "
        f"{np.max(phone_high) * 100:.2f}%"
    )

    print(
        f"Median max-evidence     : "
        f"{np.median(phone_max) * 100:.2f}%"
    )

    print(
        f"Max max-evidence        : "
        f"{np.max(phone_max) * 100:.2f}%"
    )

# ============================================================
# DIFFICULT CHALLENGE DEEPFAKE
# ============================================================

print("\n" + "=" * 78)
print(
    "DIFFICULT KNOWN DEEPFAKE - "
    "UNSEEN CHALLENGE"
)
print("=" * 78)

challenge = None

if DIFFICULT_DEEPFAKE.exists():
    challenge = (
        analyze_external_file(
            DIFFICULT_DEEPFAKE
        )
    )

    print(
        f"File             : "
        f"{challenge['file']}"
    )

    print(
        f"Windows          : "
        f"{challenge['windows']}"
    )

    print(
        f"Prediction       : "
        f"{challenge['prediction']}"
    )

    print(
        f"Mean Deepfake    : "
        f"{challenge['mean_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"Median Deepfake  : "
        f"{challenge['median_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"P75 Deepfake     : "
        f"{challenge['p75_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"P90 Deepfake     : "
        f"{challenge['p90_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"Maximum Evidence : "
        f"{challenge['max_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"Suspicious Ratio : "
        f"{challenge['suspicious_ratio'] * 100:.2f}%"
    )

    print(
        f"High-Risk Ratio  : "
        f"{challenge['high_ratio'] * 100:.2f}%"
    )

    challenge_probabilities = np.asarray(
        challenge[
            "window_probabilities"
        ]
    )

    challenge_windows = pd.DataFrame({
        "window_index":
            np.arange(
                len(
                    challenge_probabilities
                )
            ),

        "start_seconds":
            np.arange(
                len(
                    challenge_probabilities
                )
            )
            * WINDOW_SECONDS,

        "end_seconds":
            (
                np.arange(
                    len(
                        challenge_probabilities
                    )
                )
                + 1
            )
            * WINDOW_SECONDS,

        "deepfake_probability":
            challenge_probabilities,
    })

    challenge_windows.to_csv(
        OUTPUT_DIR
        / "challenge_window_evidence_v6_1.csv",
        index=False,
    )

else:
    print(
        f"Challenge file not found: "
        f"{DIFFICULT_DEEPFAKE}"
    )

# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "version": "F1B_V6_1",

    "device": str(
        DEVICE
    ),

    "manifest_files": int(
        len(manifest)
    ),

    "train_files": int(
        len(train_df)
    ),

    "validation_files": int(
        len(val_df)
    ),

    "test_files": int(
        len(test_df)
    ),

    "train_speakers": int(
        len(train_speakers)
    ),

    "validation_speakers": int(
        len(val_speakers)
    ),

    "test_speakers": int(
        len(test_speakers)
    ),

    "best_validation_macro_f1":
        float(
            best_val_f1
        ),

    "final_test_window_accuracy":
        float(
            window_accuracy
        ),

    "final_test_window_macro_f1":
        float(
            window_f1
        ),

    "final_test_file_accuracy":
        float(
            file_accuracy
        ),

    "final_test_file_macro_f1":
        float(
            file_f1
        ),

    "heldout_phone_genuine_accuracy": (
        None
        if phone_accuracy is None
        else float(
            phone_accuracy
        )
    ),

    "challenge": challenge,

    "failed_transform_count":
        int(
            len(failed_df)
        ),
}

with open(
    OUTPUT_DIR
    / "experiment_results_v6_1.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
    )

shutil.rmtree(
    TEMP_ROOT,
    ignore_errors=True,
)

print("\n" + "=" * 78)
print(
    "F1B V6.1 EXPERIMENT COMPLETE"
)
print("=" * 78)

print(
    f"Best validation Macro F1 : "
    f"{best_val_f1 * 100:.2f}%"
)

print(
    f"Final ASV window Macro F1: "
    f"{window_f1 * 100:.2f}%"
)

print(
    f"Final ASV file Macro F1  : "
    f"{file_f1 * 100:.2f}%"
)

if phone_accuracy is not None:
    print(
        f"Held-out phone genuine   : "
        f"{phone_accuracy * 100:.2f}%"
    )

if challenge is not None:
    print(
        f"Challenge median DF      : "
        f"{challenge['median_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"Challenge P90 DF         : "
        f"{challenge['p90_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"Challenge max evidence   : "
        f"{challenge['max_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"Challenge suspicious     : "
        f"{challenge['suspicious_ratio'] * 100:.2f}% windows"
    )

print("\nOutputs:")
print(OUTPUT_DIR)
print(BEST_MODEL_PATH)

print("\nIMPORTANT:")
print(
    "V6.1 is experimental. "
    "Do NOT integrate it into SonicT yet."
)

print(
    "The challenge deepfake remained "
    "completely unseen during training."
)


print(
    "Do not use maximum window evidence alone "
    "for a final decision; genuine calls can also "
    "contain isolated high-probability windows."
)
