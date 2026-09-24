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
# SONICT F1B V7
# Frozen WavLM + MLP cross-generator deepfake detector
#
# Training data:
#   MLAAD-tiny English
#   - fake generators are disjoint across train/val/test
#   - genuine source groups are disjoint across train/val/test
#
# Important:
#   - existing SonicT F1-F5 are NOT modified
#   - difficult challenge deepfake is NEVER used for training
#   - WavLM is loaded locally/offline
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
EPOCHS = 25
PATIENCE = 5
LEARNING_RATE = 1e-3

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

OUTPUT_DIR = Path(
    r"H:\SonicT_Compression_Test\experiment_v7_output"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "wavlm_cross_generator_v7.pt"

CACHE_TRAIN = OUTPUT_DIR / "wavlm_train_features_v7.npz"
CACHE_VAL = OUTPUT_DIR / "wavlm_validation_features_v7.npz"
CACHE_TEST = OUTPUT_DIR / "wavlm_test_features_v7.npz"

FAILED_CSV = OUTPUT_DIR / "failed_feature_files_v7.csv"

AUDIO_EXTS = {
    ".wav", ".flac", ".mp3", ".m4a",
    ".aac", ".ogg", ".opus", ".wma",
    ".mpeg", ".mpg", ".amr", ".3gp",
    ".mp4", ".webm", ".mkv", ".mov",
    ".aiff", ".aif", ".caf", ".3g2"
}

# FFmpeg must be available from CMD as: ffmpeg -version
FFMPEG_BIN = "ffmpeg"

# ============================================================
# CHECK INPUTS
# ============================================================

for required_path, label in [
    (FAKE_MANIFEST, "fake manifest"),
    (GENUINE_MANIFEST, "grouped genuine manifest"),
    (WAVLM_DIR, "local WavLM folder"),
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"{label} not found: {required_path}"
        )

# ============================================================
# LOAD MANIFESTS
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

# Verify paths now, before expensive inference.
exists_mask = all_df["path"].apply(
    lambda x: Path(x).exists()
)

missing_files = all_df.loc[
    ~exists_mask,
    ["path", "split", "label_name", "generator"]
]

if len(missing_files):
    missing_files.to_csv(
        OUTPUT_DIR / "missing_manifest_files_v7.csv",
        index=False,
    )
    print(
        f"WARNING: {len(missing_files)} manifest files "
        "are missing and will be skipped."
    )

all_df = all_df[exists_mask].reset_index(drop=True)

train_df = all_df[all_df["split"] == "train"].reset_index(drop=True)
val_df = all_df[all_df["split"] == "validation"].reset_index(drop=True)
test_df = all_df[all_df["split"] == "test"].reset_index(drop=True)

print("=" * 78)
print("SONICT F1B V7 - WAVLM CROSS-GENERATOR DETECTOR")
print("=" * 78)
print(f"Device: {DEVICE}")
print(f"Local WavLM: {WAVLM_DIR}")

for name, df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("FINAL TEST", test_df),
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
# AUDIO / EMBEDDING FUNCTIONS
# ============================================================


def convert_to_wav_ffmpeg(input_path):
    """
    Convert any FFmpeg-readable audio/video file to a temporary
    16 kHz, mono, PCM16 WAV file. The caller must delete the
    returned temporary file after use.
    """
    input_path = Path(input_path)

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )
    temp_wav = Path(temp_file.name)
    temp_file.close()

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(SR),
        "-c:a", "pcm_s16le",
        str(temp_wav),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_text = (result.stderr or "").strip()
            if len(error_text) > 1000:
                error_text = error_text[-1000:]

            raise RuntimeError(
                f"FFmpeg conversion failed for {input_path.name}. "
                f"{error_text}"
            )

        if not temp_wav.exists() or temp_wav.stat().st_size == 0:
            raise RuntimeError(
                f"FFmpeg created no valid WAV for {input_path.name}"
            )

        return temp_wav

    except Exception:
        try:
            if temp_wav.exists():
                temp_wav.unlink()
        except Exception:
            pass
        raise


def load_audio_robust(path):
    """
    Load audio at 16 kHz mono.

    First try librosa directly. If the container/codec is not
    supported (for example some .m4a phone recordings), fall
    back to FFmpeg conversion and then load the temporary WAV.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        y, _ = librosa.load(
            str(path),
            sr=SR,
            mono=True,
        )
        return np.asarray(y, dtype=np.float32)

    except Exception as direct_error:
        temp_wav = None

        try:
            temp_wav = convert_to_wav_ffmpeg(path)

            y, _ = librosa.load(
                str(temp_wav),
                sr=SR,
                mono=True,
            )

            return np.asarray(y, dtype=np.float32)

        except Exception as ffmpeg_error:
            raise RuntimeError(
                f"Could not decode {path.name}. "
                f"Direct loader error: {direct_error}. "
                f"FFmpeg fallback error: {ffmpeg_error}"
            ) from ffmpeg_error

        finally:
            if temp_wav is not None:
                try:
                    temp_wav = Path(temp_wav)
                    if temp_wav.exists():
                        temp_wav.unlink()
                except Exception:
                    pass


failed_rows = []


def load_audio_for_embedding(path):
    y = load_audio_robust(path)

    if len(y) < MIN_SAMPLES:
        raise ValueError("Audio shorter than 1 second")

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    # Deterministic center crop for dataset files.
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


def extract_manifest_features(df, split_name, cache_path):
    if cache_path.exists():
        print(f"\nLoading cached {split_name} features:")
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

    total = len(df)

    for batch_start in range(
        0,
        total,
        feature_batch_size,
    ):
        batch_df = df.iloc[
            batch_start:
            batch_start + feature_batch_size
        ]

        waveforms = []
        valid_rows = []

        for _, row in batch_df.iterrows():
            path = Path(row["path"])

            try:
                y = load_audio_for_embedding(path)
                waveforms.append(y)
                valid_rows.append(row)

            except Exception as exc:
                failed_rows.append({
                    "split": split_name,
                    "path": str(path),
                    "error": str(exc),
                })

        if waveforms:
            try:
                embeddings = wavlm_batch_embeddings(
                    waveforms
                )

                X_parts.append(embeddings)

                for row in valid_rows:
                    y_parts.append(int(row["label"]))

                    meta_rows.append({
                        "path": str(row["path"]),
                        "filename": str(row["filename"]),
                        "label": int(row["label"]),
                        "label_name": str(row["label_name"]),
                        "split": str(row["split"]),
                        "source_group": str(row["source_group"]),
                        "generator": str(row["generator"]),
                    })

            except Exception as exc:
                for row in valid_rows:
                    failed_rows.append({
                        "split": split_name,
                        "path": str(row["path"]),
                        "error": "batch_embedding: " + str(exc),
                    })

        done = min(
            batch_start + feature_batch_size,
            total
        )

        if (
            done % 100 == 0
            or done == total
        ):
            print(
                f"{split_name}: "
                f"{done}/{total}"
            )

    if not X_parts:
        raise RuntimeError(
            f"No embeddings extracted for {split_name}"
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


# ============================================================
# EXTRACT / LOAD CACHED FEATURES
# ============================================================

print("\n" + "=" * 78)
print("WAVLM FEATURE EXTRACTION")
print("=" * 78)

X_train, y_train, train_meta = extract_manifest_features(
    train_df,
    "train",
    CACHE_TRAIN,
)

X_val, y_val, val_meta = extract_manifest_features(
    val_df,
    "validation",
    CACHE_VAL,
)

X_test, y_test, test_meta = extract_manifest_features(
    test_df,
    "test",
    CACHE_TEST,
)

if failed_rows:
    pd.DataFrame(failed_rows).to_csv(
        FAILED_CSV,
        index=False,
    )

print("\nFeature shapes:")
print("Train      :", X_train.shape)
print("Validation :", X_val.shape)
print("Test       :", X_test.shape)
print("Failures   :", len(failed_rows))

# ============================================================
# STANDARDIZE USING TRAIN ONLY
# ============================================================

feature_mean = X_train.mean(axis=0).astype(np.float32)
feature_std = X_train.std(axis=0).astype(np.float32) + 1e-6

X_train = (
    (X_train - feature_mean) / feature_std
).astype(np.float32)

X_val = (
    (X_val - feature_mean) / feature_std
).astype(np.float32)

X_test = (
    (X_test - feature_mean) / feature_std
).astype(np.float32)

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

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
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

val_loader = make_loader(
    X_val,
    y_val,
    False,
)

test_loader = make_loader(
    X_test,
    y_test,
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


best_val_f1 = -1.0
epochs_without_improvement = 0
history = []

print("\n" + "=" * 78)
print("TRAINING WAVLM-EMBEDDING MLP")
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

    val_true, val_pred, _ = predict_loader(
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
    ) = precision_recall_fscore_support(
        val_true,
        val_pred,
        average="macro",
        zero_division=0,
    )

    scheduler.step(val_f1)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_accuracy": float(val_acc),
        "validation_macro_precision": float(val_precision),
        "validation_macro_recall": float(val_recall),
        "validation_macro_f1": float(val_f1),
    })

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {train_loss:.4f} | "
        f"Val Acc {val_acc * 100:.2f}% | "
        f"Val F1 {val_f1 * 100:.2f}%"
    )

    if val_f1 > best_val_f1 + 1e-4:
        best_val_f1 = float(val_f1)
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "hidden_size": HIDDEN_SIZE,
                "embedding_size": EMBEDDING_SIZE,
                "best_validation_macro_f1": best_val_f1,
                "config": {
                    "sr": SR,
                    "max_seconds": MAX_SECONDS,
                    "wavlm_path": str(WAVLM_DIR),
                    "pooling": "mean_plus_std",
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
    OUTPUT_DIR / "training_history_v7.csv",
    index=False,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

# ============================================================
# FINAL UNSEEN-GENERATOR TEST
# ============================================================

test_true, test_pred, test_probs = predict_loader(
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
) = precision_recall_fscore_support(
    test_true,
    test_pred,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 78)
print("FINAL MLAAD UNSEEN-GENERATOR TEST")
print("=" * 78)

print(f"Accuracy        : {test_acc * 100:.2f}%")
print(f"Macro Precision : {test_precision * 100:.2f}%")
print(f"Macro Recall    : {test_recall * 100:.2f}%")
print(f"Macro F1        : {test_f1 * 100:.2f}%")

print(
    "Confusion [genuine=0, deepfake=1]:"
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

test_meta = test_meta.copy()
test_meta["deepfake_probability"] = test_probs
test_meta["prediction"] = test_pred

test_meta.to_csv(
    OUTPUT_DIR / "unseen_generator_test_results_v7.csv",
    index=False,
)

# Per unseen generator performance.
generator_rows = []

print("\n" + "=" * 78)
print("UNSEEN GENERATOR PERFORMANCE")
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

    # All are deepfake, so recall = fraction correctly detected.
    detection_rate = float(
        np.mean(
            subset["prediction"].values == 1
        )
    )

    mean_prob = float(
        subset["deepfake_probability"].mean()
    )

    median_prob = float(
        subset["deepfake_probability"].median()
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
    OUTPUT_DIR / "unseen_generator_performance_v7.csv",
    index=False,
)

# ============================================================
# EXTERNAL LONG-FILE WINDOW ANALYSIS
# ============================================================

def load_full_audio(path):
    y = load_audio_robust(path)

    peak = float(np.max(np.abs(y))) if len(y) else 0.0

    if peak > 0:
        y = y / peak

    return y


def external_windows(y):
    windows = []

    if len(y) < MIN_SAMPLES:
        return windows

    if len(y) <= MAX_SAMPLES:
        windows.append(y)
        return windows

    step = MAX_SAMPLES

    for start in range(
        0,
        len(y) - MIN_SAMPLES + 1,
        step,
    ):
        chunk = y[start:start + MAX_SAMPLES]

        if len(chunk) >= MIN_SAMPLES:
            windows.append(chunk)

    return windows


def predict_waveforms(waveforms):
    if not waveforms:
        raise RuntimeError("No valid external audio windows.")

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

    median_prob = float(np.median(probs))

    return {
        "file": Path(path).name,
        "windows": int(len(probs)),
        "prediction": (
            "DEEPFAKE"
            if median_prob >= 0.50
            else "GENUINE"
        ),
        "mean_deepfake_probability": float(np.mean(probs)),
        "median_deepfake_probability": median_prob,
        "p75_deepfake_probability": float(
            np.percentile(probs, 75)
        ),
        "p90_deepfake_probability": float(
            np.percentile(probs, 90)
        ),
        "max_deepfake_probability": float(np.max(probs)),
        "suspicious_ratio": float(np.mean(probs >= 0.50)),
        "high_ratio": float(np.mean(probs >= 0.70)),
        "window_probabilities": [
            float(x) for x in probs
        ],
    }


# ============================================================
# HELD-OUT PERSONAL GENUINE TEST
# ============================================================

phone_results = []
phone_accuracy = None

if PERSONAL_GENUINE_DIR.exists():
    phone_files = sorted([
        p
        for p in PERSONAL_GENUINE_DIR.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() in AUDIO_EXTS
        )
    ])

    print("\n" + "=" * 78)
    print("EXTERNAL PERSONAL GENUINE PHONE TEST")
    print("=" * 78)

    correct = 0

    for index, path in enumerate(
        phone_files,
        start=1,
    ):
        try:
            result = analyze_external(path)
            phone_results.append(result)

            if result["prediction"] == "GENUINE":
                correct += 1

            print(
                f"[{index}/{len(phone_files)}] "
                f"{path.name} | "
                f"{result['prediction']} | "
                f"median DF "
                f"{result['median_deepfake_probability'] * 100:.2f}% | "
                f"max "
                f"{result['max_deepfake_probability'] * 100:.2f}%"
            )

        except Exception as exc:
            print(
                f"[{index}/{len(phone_files)}] "
                f"{path.name} | ERROR: {exc}"
            )

    if phone_results:
        phone_accuracy = (
            correct / len(phone_results)
        )

        false_deepfake_rate = 1.0 - phone_accuracy

        phone_medians = np.asarray(
            [
                row["median_deepfake_probability"]
                for row in phone_results
            ],
            dtype=np.float32,
        )

        print(
            "\nPersonal genuine accuracy: "
            f"{phone_accuracy * 100:.2f}%"
        )

        print(
            "False deepfake rate      : "
            f"{false_deepfake_rate * 100:.2f}%"
        )

        print(
            "Median phone DF score    : "
            f"{np.median(phone_medians) * 100:.2f}%"
        )

with open(
    OUTPUT_DIR / "personal_genuine_results_v7.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        phone_results,
        f,
        indent=2,
    )

# ============================================================
# DIFFICULT CHALLENGE
# ============================================================

challenge = None

print("\n" + "=" * 78)
print("DIFFICULT KNOWN DEEPFAKE - UNSEEN CHALLENGE")
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
        OUTPUT_DIR / "challenge_window_evidence_v7.csv",
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
    "version": "F1B_V7_WAVLM_CROSS_GENERATOR",
    "device": str(DEVICE),
    "train_files": int(len(train_df)),
    "validation_files": int(len(val_df)),
    "test_files": int(len(test_df)),
    "best_validation_macro_f1": float(best_val_f1),
    "unseen_generator_test_accuracy": float(test_acc),
    "unseen_generator_test_macro_f1": float(test_f1),
    "personal_genuine_accuracy": (
        None
        if phone_accuracy is None
        else float(phone_accuracy)
    ),
    "personal_false_deepfake_rate": (
        None
        if phone_accuracy is None
        else float(1.0 - phone_accuracy)
    ),
    "challenge": challenge,
    "model_path": str(MODEL_PATH),
}

with open(
    OUTPUT_DIR / "experiment_results_v7.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print("\n" + "=" * 78)
print("F1B V7 EXPERIMENT COMPLETE")
print("=" * 78)

print(
    f"Best validation Macro F1     : "
    f"{best_val_f1 * 100:.2f}%"
)

print(
    f"Unseen-generator test F1     : "
    f"{test_f1 * 100:.2f}%"
)

if phone_accuracy is not None:
    print(
        f"Personal genuine accuracy    : "
        f"{phone_accuracy * 100:.2f}%"
    )

if challenge is not None:
    print(
        f"Challenge median deepfake    : "
        f"{challenge['median_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"Challenge P90 deepfake       : "
        f"{challenge['p90_deepfake_probability'] * 100:.2f}%"
    )

    print(
        f"Challenge suspicious windows : "
        f"{challenge['suspicious_ratio'] * 100:.2f}%"
    )

print("\nOutputs:")
print(OUTPUT_DIR)
print(MODEL_PATH)

print("\nIMPORTANT:")
print(
    "V7 is experimental. Do NOT integrate it into "
    "SonicT F1-F5 yet."
)

print(
    "The final MLAAD fake test generators and the difficult "
    "challenge file were unseen during training."
)
