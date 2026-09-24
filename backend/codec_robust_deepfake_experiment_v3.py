from pathlib import Path
import subprocess
import tempfile
import shutil
import warnings
import json
import re

import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

SR = 16000
MAX_DURATION = 15
RANDOM_STATE = 42

ASV_ROOT = Path(r"H:\SonicT_Compression_Test\asvspoof_balanced")
BONAFIDE_DIR = ASV_ROOT / "bonafide"
SPOOF_DIR = ASV_ROOT / "spoof"
MANIFEST = ASV_ROOT / "manifest.csv"

PERSONAL_GENUINE_DIR = Path(r"H:\SonicT_Compression_Test\genuine")

DIFFICULT_DEEPFAKE = Path(
    r"E:\Downloads\recording_6ee192c5-aab6-4a46-8afc-ee9985bb9aaf.mp3.mpeg"
)

OUTPUT_DIR = Path(r"H:\SonicT_Compression_Test\experiment_v3_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_ROOT = Path(tempfile.mkdtemp(prefix="sonict_codec_v3_"))

ASV_CODECS = {
    "wav16k": [
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
    ],
    "mp3_8k_16k": [
        "-ar", "8000",
        "-ac", "1",
        "-b:a", "16k",
        "-c:a", "libmp3lame",
    ],
    "mp3_8k_32k": [
        "-ar", "8000",
        "-ac", "1",
        "-b:a", "32k",
        "-c:a", "libmp3lame",
    ],
    "mp3_16k_32k": [
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "32k",
        "-c:a", "libmp3lame",
    ],
    "aac_16k_32k": [
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "32k",
        "-c:a", "aac",
    ],
    "opus_16k_24k": [
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "24k",
        "-c:a", "libopus",
    ],
}

PHONE_AUGS = {
    "phone_original": "original",
    "phone_mp3_8k_16k": "mp3_8k_16k",
    "phone_bandpass": "bandpass",
    "phone_bandpass_mp3": "bandpass_mp3",
    "phone_low_gain": "low_gain",
    "phone_light_noise": "light_noise",
}

FEATURE_NAMES = [
    "spectral_centroid",
    "spectral_bandwidth",
    "rolloff_95",
    "spectral_flatness",
    "zcr",
    "f0_median_hz",
    "f0_std_hz",
    "voiced_ratio",
    "jitter",
    "shimmer",
    "tonality_ratio",
    "noise_ratio",
    "effective_bandwidth",
    "energy_0_1khz_ratio",
    "energy_1_3khz_ratio",
    "energy_3_4khz_ratio",
    "mfcc_std_mean",
    "mfcc_delta_abs_mean",
    "mfcc_delta2_abs_mean",
]

for i in range(1, 21):
    FEATURE_NAMES.append(f"mfcc_{i}_mean")


def cleanup_file(path):
    try:
        if path is not None and Path(path).exists():
            Path(path).unlink()
    except Exception:
        pass


def run_ffmpeg(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0, result.stderr


def decode_to_wav(input_file, output_wav):
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(input_file),
        "-t", str(MAX_DURATION),
        "-ar", str(SR),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]
    return run_ffmpeg(cmd)


def encode_codec(input_file, output_file, codec_args):
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(input_file),
        "-t", str(MAX_DURATION),
    ]
    cmd.extend(codec_args)
    cmd.append(str(output_file))
    return run_ffmpeg(cmd)


def codec_suffix(codec_name):
    if "mp3" in codec_name:
        return ".mp3"
    if "aac" in codec_name:
        return ".aac"
    if "opus" in codec_name:
        return ".opus"
    return ".wav"


def load_audio(path):
    y, _ = librosa.load(
        path,
        sr=SR,
        mono=True,
        duration=MAX_DURATION,
    )

    y = np.asarray(y, dtype=np.float32)

    if len(y) < SR:
        raise ValueError("Audio shorter than 1 second")

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    return y


def extract_features(path):
    y = load_audio(path)
    eps = 1e-10

    centroid = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=SR,
        roll_percent=0.95,
    )[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    mean_flatness = float(np.mean(flatness))

    try:
        f0 = librosa.yin(
            y,
            fmin=60,
            fmax=500,
            sr=SR,
            frame_length=1024,
            hop_length=512,
        )

        f0 = np.asarray(f0, dtype=np.float32)
        valid = np.isfinite(f0) & (f0 >= 60) & (f0 <= 500)
        valid_f0 = f0[valid]

        if len(valid_f0) >= 2:
            f0_median = float(np.median(valid_f0))
            f0_std = float(np.std(valid_f0))
            jitter = float(
                np.mean(np.abs(np.diff(valid_f0))) /
                (np.mean(valid_f0) + eps)
            )
        else:
            f0_median = 0.0
            f0_std = 0.0
            jitter = 0.0

        rms_v = librosa.feature.rms(
            y=y,
            frame_length=1024,
            hop_length=512,
        )[0]

        if len(rms_v) > 0:
            threshold = max(
                float(np.median(rms_v) * 0.5),
                1e-5,
            )
            voiced_ratio = float(np.mean(rms_v > threshold))
        else:
            voiced_ratio = 0.0

    except Exception:
        f0_median = 0.0
        f0_std = 0.0
        jitter = 0.0
        voiced_ratio = 0.0

    rms = librosa.feature.rms(y=y)[0]

    if len(rms) > 1:
        shimmer = float(
            np.mean(np.abs(np.diff(rms))) /
            (np.mean(rms) + eps)
        )
    else:
        shimmer = 0.0

    tonality_ratio = float(
        np.clip(1.0 - mean_flatness, 0.0, 1.0)
    )
    noise_ratio = float(
        np.clip(mean_flatness, 0.0, 1.0)
    )

    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    spectrum_total = np.sum(spectrum) + eps

    def band_ratio(low, high):
        mask = (freqs >= low) & (freqs < high)
        return float(
            np.sum(spectrum[mask]) /
            spectrum_total
        )

    energy_0_1 = band_ratio(0, 1000)
    energy_1_3 = band_ratio(1000, 3000)
    energy_3_4 = band_ratio(3000, 4000)

    cumulative = np.cumsum(spectrum)

    if cumulative[-1] > 0:
        target = 0.95 * cumulative[-1]
        idx = int(np.searchsorted(cumulative, target))
        idx = min(idx, len(freqs) - 1)
        effective_bandwidth = float(freqs[idx])
    else:
        effective_bandwidth = 0.0

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=20,
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_means = np.mean(mfcc, axis=1)

    mfcc_std_mean = float(
        np.mean(np.std(mfcc, axis=1))
    )
    mfcc_delta_abs_mean = float(
        np.mean(np.abs(delta))
    )
    mfcc_delta2_abs_mean = float(
        np.mean(np.abs(delta2))
    )

    features = [
        float(np.mean(centroid)),
        float(np.mean(bandwidth)),
        float(np.mean(rolloff)),
        mean_flatness,
        float(np.mean(zcr)),
        f0_median,
        f0_std,
        voiced_ratio,
        jitter,
        shimmer,
        tonality_ratio,
        noise_ratio,
        effective_bandwidth,
        energy_0_1,
        energy_1_3,
        energy_3_4,
        mfcc_std_mean,
        mfcc_delta_abs_mean,
        mfcc_delta2_abs_mean,
    ]

    features.extend(mfcc_means.tolist())

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def personal_speaker_id(path):
    name = Path(path).stem

    name = re.sub(
        r"_\d{6}_\d{6}$",
        "",
        name,
    )

    if name.lower().startswith("call "):
        name = name[5:]

    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    name = re.sub(r"\s+", "_", name).strip("_")

    if not name:
        name = Path(path).stem.lower()

    return f"phone::{name}"


if not MANIFEST.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST}"
    )

manifest = pd.read_csv(MANIFEST)

required_columns = {
    "speaker",
    "file_id",
    "codec",
    "source",
    "attack",
    "label",
}

missing = required_columns - set(manifest.columns)

if missing:
    raise RuntimeError(
        f"Manifest missing columns: {sorted(missing)}"
    )

manifest["file_id"] = manifest["file_id"].astype(str)
manifest["speaker"] = manifest["speaker"].astype(str)

metadata = {
    str(row["file_id"]): row
    for _, row in manifest.iterrows()
}

asv_records = []

for label_name, label_value, folder in [
    ("genuine", 0, BONAFIDE_DIR),
    ("deepfake", 1, SPOOF_DIR),
]:
    for path in sorted(folder.glob("*.flac")):
        file_id = path.stem

        if file_id not in metadata:
            continue

        info = metadata[file_id]

        asv_records.append({
            "path": path,
            "file_id": file_id,
            "speaker": f"asv::{info['speaker']}",
            "label": label_value,
            "label_name": label_name,
            "original_codec": str(info["codec"]),
            "source": str(info["source"]),
            "attack": str(info["attack"]),
        })

asv_df = pd.DataFrame(asv_records)

if asv_df.empty:
    raise RuntimeError("No ASVspoof source files found.")

asv_X_dummy = np.zeros((len(asv_df), 1), dtype=np.float32)
asv_y = asv_df["label"].values
asv_groups = asv_df["speaker"].values

asv_split_found = False

for seed in range(RANDOM_STATE, RANDOM_STATE + 100):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=seed,
    )

    train_idx, test_idx = next(
        splitter.split(
            asv_X_dummy,
            asv_y,
            asv_groups,
        )
    )

    y_train_candidate = asv_y[train_idx]
    y_test_candidate = asv_y[test_idx]

    if (
        len(np.unique(y_train_candidate)) == 2
        and
        len(np.unique(y_test_candidate)) == 2
    ):
        asv_split_found = True
        ASV_SPLIT_SEED = seed
        break

if not asv_split_found:
    raise RuntimeError(
        "Could not create a valid ASV speaker split."
    )

asv_train_sources = asv_df.iloc[train_idx].reset_index(drop=True)
asv_test_sources = asv_df.iloc[test_idx].reset_index(drop=True)

allowed_extensions = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".mpeg",
    ".mpg",
    ".wma",
    ".amr",
    ".3gp",
}

personal_files = sorted([
    p for p in PERSONAL_GENUINE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in allowed_extensions
])

personal_records = []

for p in personal_files:
    personal_records.append({
        "path": p,
        "speaker": personal_speaker_id(p),
    })

personal_df = pd.DataFrame(personal_records)

if personal_df.empty:
    raise RuntimeError(
        f"No personal genuine recordings found in {PERSONAL_GENUINE_DIR}"
    )

unique_phone_speakers = sorted(
    personal_df["speaker"].unique().tolist()
)

if len(unique_phone_speakers) < 2:
    raise RuntimeError(
        "Need at least two personal speakers for unseen-speaker validation."
    )

rng = np.random.default_rng(RANDOM_STATE)
shuffled_speakers = unique_phone_speakers.copy()
rng.shuffle(shuffled_speakers)

test_speaker_count = max(
    1,
    int(round(len(shuffled_speakers) * 0.35))
)

test_speaker_count = min(
    test_speaker_count,
    len(shuffled_speakers) - 1,
)

personal_test_speakers = set(
    shuffled_speakers[:test_speaker_count]
)

personal_train_speakers = set(
    shuffled_speakers[test_speaker_count:]
)

personal_train_df = personal_df[
    personal_df["speaker"].isin(personal_train_speakers)
].reset_index(drop=True)

personal_test_df = personal_df[
    personal_df["speaker"].isin(personal_test_speakers)
].reset_index(drop=True)

print("=" * 78)
print("SONICT F1B V3 - DOMAIN-AWARE CODEC-ROBUST EXPERIMENT")
print("=" * 78)

print("\nASVspoof source split")
print("---------------------")
print(f"Split seed       : {ASV_SPLIT_SEED}")
print(f"Train originals  : {len(asv_train_sources)}")
print(f"Test originals   : {len(asv_test_sources)}")
print(f"Train speakers   : {asv_train_sources['speaker'].nunique()}")
print(f"Test speakers    : {asv_test_sources['speaker'].nunique()}")

print("\nASV train class counts:")
print(
    asv_train_sources.groupby("label_name")["file_id"].nunique()
)

print("\nASV test class counts:")
print(
    asv_test_sources.groupby("label_name")["file_id"].nunique()
)

print("\nPersonal phone split")
print("--------------------")
print(
    f"Total phone speakers : "
    f"{personal_df['speaker'].nunique()}"
)
print(
    f"Train speakers       : "
    f"{personal_train_df['speaker'].nunique()}"
)
print(
    f"Test speakers        : "
    f"{personal_test_df['speaker'].nunique()}"
)
print(
    f"Train recordings     : "
    f"{len(personal_train_df)}"
)
print(
    f"Test recordings      : "
    f"{len(personal_test_df)}"
)

print("\nHeld-out personal test speakers:")
for speaker in sorted(personal_test_speakers):
    print(f"  {speaker}")


def make_phone_variant(input_file, variant_name, output_wav):
    if variant_name == "phone_original":
        return decode_to_wav(
            input_file,
            output_wav,
        )

    if variant_name == "phone_mp3_8k_16k":
        compressed = output_wav.with_suffix(".mp3")

        try:
            ok, err = encode_codec(
                input_file,
                compressed,
                ASV_CODECS["mp3_8k_16k"],
            )

            if not ok:
                return False, err

            return decode_to_wav(
                compressed,
                output_wav,
            )

        finally:
            cleanup_file(compressed)

    if variant_name == "phone_bandpass":
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", str(input_file),
            "-t", str(MAX_DURATION),
            "-af", "highpass=f=300,lowpass=f=3400",
            "-ar", str(SR),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_wav),
        ]

        return run_ffmpeg(cmd)

    if variant_name == "phone_bandpass_mp3":
        filtered = output_wav.with_name(
            output_wav.stem + "_filtered.wav"
        )
        compressed = output_wav.with_suffix(".mp3")

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel", "error",
                "-i", str(input_file),
                "-t", str(MAX_DURATION),
                "-af", "highpass=f=300,lowpass=f=3400",
                "-ar", str(SR),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(filtered),
            ]

            ok, err = run_ffmpeg(cmd)

            if not ok:
                return False, err

            ok, err = encode_codec(
                filtered,
                compressed,
                ASV_CODECS["mp3_8k_16k"],
            )

            if not ok:
                return False, err

            return decode_to_wav(
                compressed,
                output_wav,
            )

        finally:
            cleanup_file(filtered)
            cleanup_file(compressed)

    if variant_name == "phone_low_gain":
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", str(input_file),
            "-t", str(MAX_DURATION),
            "-filter:a", "volume=0.55",
            "-ar", str(SR),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_wav),
        ]

        return run_ffmpeg(cmd)

    if variant_name == "phone_light_noise":
        clean_wav = output_wav.with_name(
            output_wav.stem + "_clean.wav"
        )

        try:
            ok, err = decode_to_wav(
                input_file,
                clean_wav,
            )

            if not ok:
                return False, err

            y, _ = librosa.load(
                clean_wav,
                sr=SR,
                mono=True,
                duration=MAX_DURATION,
            )

            y = np.asarray(
                y,
                dtype=np.float32,
            )

            local_rng = np.random.default_rng(
                abs(hash(str(input_file))) % (2**32)
            )

            rms = float(
                np.sqrt(
                    np.mean(y ** 2)
                    + 1e-12
                )
            )

            noise_rms = rms / (10 ** (30 / 20))

            noise = local_rng.normal(
                0.0,
                noise_rms,
                size=y.shape,
            ).astype(np.float32)

            out = np.clip(
                y + noise,
                -1.0,
                1.0,
            )

            import soundfile as sf

            sf.write(
                output_wav,
                out,
                SR,
                subtype="PCM_16",
            )

            return True, ""

        finally:
            cleanup_file(clean_wav)

    return False, f"Unknown phone augmentation: {variant_name}"


train_rows = []
test_rows = []
failed_rows = []


def add_feature_row(
    collection,
    features,
    *,
    file_id,
    speaker,
    label,
    label_name,
    domain,
    condition,
    original_codec,
    source,
    attack,
):
    row = {
        "file_id": file_id,
        "speaker": speaker,
        "label": label,
        "label_name": label_name,
        "domain": domain,
        "condition": condition,
        "original_codec": original_codec,
        "source": source,
        "attack": attack,
    }

    for name, value in zip(
        FEATURE_NAMES,
        features,
    ):
        row[name] = float(value)

    collection.append(row)


print("\n" + "=" * 78)
print("EXTRACTING ASVSPOOF TRAIN FEATURES")
print("=" * 78)

for index, record in asv_train_sources.iterrows():
    source_file = Path(record["path"])

    print(
        f"[{index + 1}/{len(asv_train_sources)}] "
        f"{source_file.name}"
    )

    for codec_name, codec_args in ASV_CODECS.items():
        suffix = codec_suffix(codec_name)

        compressed = (
            TEMP_ROOT /
            f"train_{record['file_id']}_{codec_name}{suffix}"
        )

        decoded = (
            TEMP_ROOT /
            f"train_{record['file_id']}_{codec_name}_decoded.wav"
        )

        try:
            ok, err = encode_codec(
                source_file,
                compressed,
                codec_args,
            )

            if not ok:
                failed_rows.append({
                    "stage": "asv_train_encode",
                    "file": source_file.name,
                    "condition": codec_name,
                    "error": err.strip(),
                })
                continue

            ok, err = decode_to_wav(
                compressed,
                decoded,
            )

            if not ok:
                failed_rows.append({
                    "stage": "asv_train_decode",
                    "file": source_file.name,
                    "condition": codec_name,
                    "error": err.strip(),
                })
                continue

            features = extract_features(decoded)

            add_feature_row(
                train_rows,
                features,
                file_id=str(record["file_id"]),
                speaker=str(record["speaker"]),
                label=int(record["label"]),
                label_name=str(record["label_name"]),
                domain="asvspoof",
                condition=codec_name,
                original_codec=str(record["original_codec"]),
                source=str(record["source"]),
                attack=str(record["attack"]),
            )

        except Exception as exc:
            failed_rows.append({
                "stage": "asv_train_feature",
                "file": source_file.name,
                "condition": codec_name,
                "error": str(exc),
            })

        finally:
            cleanup_file(compressed)
            cleanup_file(decoded)


print("\n" + "=" * 78)
print("EXTRACTING PERSONAL PHONE TRAIN FEATURES")
print("=" * 78)

for index, record in personal_train_df.iterrows():
    source_file = Path(record["path"])
    speaker = str(record["speaker"])
    file_id = source_file.stem

    print(
        f"[{index + 1}/{len(personal_train_df)}] "
        f"{source_file.name}"
    )

    for variant_name in PHONE_AUGS:
        output_wav = (
            TEMP_ROOT /
            f"phone_train_{index}_{variant_name}.wav"
        )

        try:
            ok, err = make_phone_variant(
                source_file,
                variant_name,
                output_wav,
            )

            if not ok:
                failed_rows.append({
                    "stage": "phone_train",
                    "file": source_file.name,
                    "condition": variant_name,
                    "error": err.strip(),
                })
                continue

            features = extract_features(
                output_wav
            )

            add_feature_row(
                train_rows,
                features,
                file_id=file_id,
                speaker=speaker,
                label=0,
                label_name="genuine",
                domain="personal_phone",
                condition=variant_name,
                original_codec=source_file.suffix.lower(),
                source="personal_phone",
                attack="-",
            )

        except Exception as exc:
            failed_rows.append({
                "stage": "phone_train_feature",
                "file": source_file.name,
                "condition": variant_name,
                "error": str(exc),
            })

        finally:
            cleanup_file(output_wav)


print("\n" + "=" * 78)
print("EXTRACTING ASVSPOOF TEST FEATURES")
print("=" * 78)

for index, record in asv_test_sources.iterrows():
    source_file = Path(record["path"])

    print(
        f"[{index + 1}/{len(asv_test_sources)}] "
        f"{source_file.name}"
    )

    for codec_name, codec_args in ASV_CODECS.items():
        suffix = codec_suffix(codec_name)

        compressed = (
            TEMP_ROOT /
            f"test_{record['file_id']}_{codec_name}{suffix}"
        )

        decoded = (
            TEMP_ROOT /
            f"test_{record['file_id']}_{codec_name}_decoded.wav"
        )

        try:
            ok, err = encode_codec(
                source_file,
                compressed,
                codec_args,
            )

            if not ok:
                failed_rows.append({
                    "stage": "asv_test_encode",
                    "file": source_file.name,
                    "condition": codec_name,
                    "error": err.strip(),
                })
                continue

            ok, err = decode_to_wav(
                compressed,
                decoded,
            )

            if not ok:
                failed_rows.append({
                    "stage": "asv_test_decode",
                    "file": source_file.name,
                    "condition": codec_name,
                    "error": err.strip(),
                })
                continue

            features = extract_features(decoded)

            add_feature_row(
                test_rows,
                features,
                file_id=str(record["file_id"]),
                speaker=str(record["speaker"]),
                label=int(record["label"]),
                label_name=str(record["label_name"]),
                domain="asvspoof",
                condition=codec_name,
                original_codec=str(record["original_codec"]),
                source=str(record["source"]),
                attack=str(record["attack"]),
            )

        except Exception as exc:
            failed_rows.append({
                "stage": "asv_test_feature",
                "file": source_file.name,
                "condition": codec_name,
                "error": str(exc),
            })

        finally:
            cleanup_file(compressed)
            cleanup_file(decoded)


train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)
failed_df = pd.DataFrame(failed_rows)

if train_df.empty or test_df.empty:
    shutil.rmtree(TEMP_ROOT, ignore_errors=True)
    raise RuntimeError(
        "Training or test feature table is empty."
    )

TRAIN_DATASET_FILE = (
    OUTPUT_DIR /
    "training_features_v3.csv"
)

TEST_DATASET_FILE = (
    OUTPUT_DIR /
    "asv_test_features_v3.csv"
)

FAILED_FILE = (
    OUTPUT_DIR /
    "failed_samples_v3.csv"
)

train_df.to_csv(
    TRAIN_DATASET_FILE,
    index=False,
)

test_df.to_csv(
    TEST_DATASET_FILE,
    index=False,
)

failed_df.to_csv(
    FAILED_FILE,
    index=False,
)

print("\nTraining feature summary")
print("------------------------")
print(f"Rows             : {len(train_df)}")
print(f"Original files   : {train_df['file_id'].nunique()}")
print("\nRows by domain/class:")
print(
    train_df.groupby(
        ["domain", "label_name"]
    ).size()
)

print("\nASV test feature summary")
print("------------------------")
print(f"Rows             : {len(test_df)}")
print(f"Original files   : {test_df['file_id'].nunique()}")
print(f"Failed samples   : {len(failed_df)}")

X_train = train_df[FEATURE_NAMES].values
y_train = train_df["label"].values

X_test = test_df[FEATURE_NAMES].values
y_test = test_df["label"].values

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=600,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=600,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    ),
}

results = {}
best_model = None
best_name = None
best_f1 = -1.0

print("\n" + "=" * 78)
print("MODEL EVALUATION - UNSEEN ASVSPOOF SPEAKERS")
print("=" * 78)

for name, model in models.items():
    model.fit(
        X_train,
        y_train,
    )

    pred = model.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        pred,
    )

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            pred,
            average="macro",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        pred,
        labels=[0, 1],
    )

    print("\n" + "-" * 60)
    print(name)
    print(f"Accuracy        : {acc * 100:.2f}%")
    print(f"Macro Precision : {precision * 100:.2f}%")
    print(f"Macro Recall    : {recall * 100:.2f}%")
    print(f"Macro F1        : {macro_f1 * 100:.2f}%")
    print("Confusion [genuine=0, deepfake=1]:")
    print(matrix)

    print(
        classification_report(
            y_test,
            pred,
            labels=[0, 1],
            target_names=[
                "genuine",
                "deepfake",
            ],
            digits=4,
            zero_division=0,
        )
    )

    results[name] = {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": matrix.tolist(),
    }

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model = model
        best_name = name


print("\n" + "=" * 78)
print(f"BEST MODEL: {best_name}")
print(f"Macro F1  : {best_f1 * 100:.2f}%")
print("=" * 78)

importance = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "importance": best_model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
)

IMPORTANCE_FILE = (
    OUTPUT_DIR /
    "feature_importance_v3.csv"
)

importance.to_csv(
    IMPORTANCE_FILE,
    index=False,
)

print("\nTop 15 features:")
print(
    importance.head(15).to_string(index=False)
)

codec_rows = []

print("\n" + "=" * 78)
print("PER-CODEC ASVSPOOF TEST PERFORMANCE")
print("=" * 78)

for condition in sorted(
    test_df["condition"].unique()
):
    subset = test_df[
        test_df["condition"] == condition
    ]

    X_sub = subset[FEATURE_NAMES].values
    y_sub = subset["label"].values

    pred = best_model.predict(X_sub)

    acc = accuracy_score(
        y_sub,
        pred,
    )

    _, _, f1, _ = (
        precision_recall_fscore_support(
            y_sub,
            pred,
            average="macro",
            zero_division=0,
        )
    )

    codec_rows.append({
        "condition": condition,
        "n": int(len(subset)),
        "accuracy": float(acc),
        "macro_f1": float(f1),
    })

codec_df = pd.DataFrame(codec_rows)

print(
    codec_df.to_string(index=False)
)

CODEC_FILE = (
    OUTPUT_DIR /
    "per_codec_results_v3.csv"
)

codec_df.to_csv(
    CODEC_FILE,
    index=False,
)


def predict_external(path):
    decoded = (
        TEMP_ROOT /
        f"external_{abs(hash(str(path)))}.wav"
    )

    try:
        ok, err = decode_to_wav(
            path,
            decoded,
        )

        if not ok:
            raise RuntimeError(
                f"FFmpeg decode failed: {err.strip()}"
            )

        features = extract_features(
            decoded
        )

        probs = best_model.predict_proba(
            [features]
        )[0]

        genuine_prob = float(probs[0])
        deepfake_prob = float(probs[1])

        prediction = (
            "DEEPFAKE"
            if deepfake_prob >= 0.5
            else "GENUINE"
        )

        return {
            "file": Path(path).name,
            "prediction": prediction,
            "genuine_probability": genuine_prob,
            "deepfake_probability": deepfake_prob,
        }

    finally:
        cleanup_file(decoded)


print("\n" + "=" * 78)
print("HELD-OUT PERSONAL GENUINE SPEAKER TEST")
print("=" * 78)

personal_test_results = []
phone_correct = 0

for index, record in personal_test_df.iterrows():
    path = Path(record["path"])

    print(
        f"[{index + 1}/{len(personal_test_df)}] "
        f"{path.name}"
    )

    try:
        result = predict_external(
            path
        )

        result["speaker"] = str(
            record["speaker"]
        )

        personal_test_results.append(
            result
        )

        if result["prediction"] == "GENUINE":
            phone_correct += 1

        print(
            f"   Prediction : "
            f"{result['prediction']}"
        )
        print(
            f"   Genuine    : "
            f"{result['genuine_probability'] * 100:.2f}%"
        )
        print(
            f"   Deepfake   : "
            f"{result['deepfake_probability'] * 100:.2f}%"
        )

    except Exception as exc:
        print(f"   ERROR: {exc}")


if len(personal_test_results) > 0:
    personal_accuracy = (
        phone_correct /
        len(personal_test_results)
    )

    print(
        "\nHeld-out personal genuine accuracy: "
        f"{personal_accuracy * 100:.2f}%"
    )
else:
    personal_accuracy = None
    print(
        "\nNo held-out personal files were successfully evaluated."
    )


print("\n" + "=" * 78)
print("DIFFICULT KNOWN DEEPFAKE - UNSEEN CHALLENGE FILE")
print("=" * 78)

challenge_result = None

if DIFFICULT_DEEPFAKE.exists():
    try:
        challenge_result = predict_external(
            DIFFICULT_DEEPFAKE
        )

        print(
            f"File       : "
            f"{challenge_result['file']}"
        )
        print(
            f"Prediction : "
            f"{challenge_result['prediction']}"
        )
        print(
            f"Genuine    : "
            f"{challenge_result['genuine_probability'] * 100:.2f}%"
        )
        print(
            f"Deepfake   : "
            f"{challenge_result['deepfake_probability'] * 100:.2f}%"
        )

    except Exception as exc:
        print(f"ERROR: {exc}")

else:
    print(
        f"Challenge file not found: "
        f"{DIFFICULT_DEEPFAKE}"
    )

MODEL_FILE = (
    OUTPUT_DIR /
    "codec_robust_experimental_model_v3.pkl"
)

FEATURE_FILE = (
    OUTPUT_DIR /
    "codec_robust_features_v3.pkl"
)

joblib.dump(
    best_model,
    MODEL_FILE,
)

joblib.dump(
    FEATURE_NAMES,
    FEATURE_FILE,
)

PERSONAL_RESULT_FILE = (
    OUTPUT_DIR /
    "heldout_personal_results_v3.csv"
)

pd.DataFrame(
    personal_test_results
).to_csv(
    PERSONAL_RESULT_FILE,
    index=False,
)

RESULT_JSON = (
    OUTPUT_DIR /
    "experiment_results_v3.json"
)

with open(
    RESULT_JSON,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "version": "F1B_V3",
            "analysis_duration_seconds": MAX_DURATION,
            "asv_split_seed": ASV_SPLIT_SEED,
            "best_model": best_name,
            "best_macro_f1": float(best_f1),
            "models": results,
            "personal_train_speakers": sorted(
                personal_train_speakers
            ),
            "personal_test_speakers": sorted(
                personal_test_speakers
            ),
            "personal_train_files": int(
                len(personal_train_df)
            ),
            "personal_test_files": int(
                len(personal_test_df)
            ),
            "heldout_personal_genuine_accuracy": (
                None
                if personal_accuracy is None
                else float(personal_accuracy)
            ),
            "challenge_result": challenge_result,
            "failed_sample_count": int(
                len(failed_df)
            ),
        },
        f,
        indent=4,
    )

shutil.rmtree(
    TEMP_ROOT,
    ignore_errors=True,
)

print("\n" + "=" * 78)
print("F1B V3 EXPERIMENT COMPLETE")
print("=" * 78)

print(f"Best model                    : {best_name}")
print(f"ASV speaker-grouped Macro F1 : {best_f1 * 100:.2f}%")

if personal_accuracy is not None:
    print(
        f"Held-out phone genuine acc    : "
        f"{personal_accuracy * 100:.2f}%"
    )

if challenge_result is not None:
    print(
        f"Challenge deepfake probability: "
        f"{challenge_result['deepfake_probability'] * 100:.2f}%"
    )

print("\nOutputs:")
print(TRAIN_DATASET_FILE)
print(TEST_DATASET_FILE)
print(FAILED_FILE)
print(IMPORTANCE_FILE)
print(CODEC_FILE)
print(PERSONAL_RESULT_FILE)
print(MODEL_FILE)
print(FEATURE_FILE)
print(RESULT_JSON)

print("\nIMPORTANT:")
print("This is still experimental F1B.")
print("Do NOT integrate it into SonicT production yet.")
print("Keep the difficult deepfake unseen until evaluation.")
