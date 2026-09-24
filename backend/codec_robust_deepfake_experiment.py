import subprocess
from pathlib import Path
import tempfile
import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(r"H:\SonicT_Compression_Test")
GENUINE_DIR = ROOT / "genuine"
DEEPFAKE_DIR = ROOT / "deepfake"
OUTPUT_DIR = ROOT / "experiment_output"
AUG_DIR = OUTPUT_DIR / "augmented"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUG_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".mpeg", ".mpg", ".m4a", ".flac", ".aac",
    ".ogg", ".wma", ".opus", ".webm", ".mov", ".mkv", ".avi", ".3gp",
    ".3g2", ".ts", ".m2ts", ".mka", ".aiff", ".aif", ".caf", ".amr"
}

CODEC_CONDITIONS = [
    {"name": "wav16k", "ext": ".wav", "args": ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]},
    {"name": "mp3_8k_16k", "ext": ".mp3", "args": ["-ac", "1", "-ar", "8000", "-c:a", "libmp3lame", "-b:a", "16k"]},
    {"name": "mp3_8k_32k", "ext": ".mp3", "args": ["-ac", "1", "-ar", "8000", "-c:a", "libmp3lame", "-b:a", "32k"]},
    {"name": "mp3_16k_32k", "ext": ".mp3", "args": ["-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k"]},
    {"name": "aac_16k_32k", "ext": ".m4a", "args": ["-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "32k"]},
    {"name": "opus_16k_24k", "ext": ".opus", "args": ["-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k"]},
]

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

def collect(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)

def safe_name(path):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem)[:60]

def convert(src, dst, args):
    dst.parent.mkdir(parents=True, exist_ok=True)
    p = run(["ffmpeg", "-y", "-i", str(src), "-vn", *args, str(dst)])
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-1200:])

def analysis_wav(src, dst):
    p = run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)])
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-1200:])

def mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else 0.0

def std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) else 0.0

def med(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else 0.0

def band_ratio(y, sr, lo, hi):
    n_fft = 2048
    s = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (f >= lo) & (f < hi)
    return float(np.sum(s[mask]) / (np.sum(s) + 1e-12))

def bandwidth99(y, sr):
    n_fft = 2048
    s = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=512)) ** 2
    p = np.mean(s, axis=1)
    c = np.cumsum(p)
    idx = int(np.searchsorted(c, 0.99 * (c[-1] + 1e-12)))
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return float(f[min(idx, len(f)-1)])

def pitch_features(y, sr):
    try:
        f0, _, _ = librosa.pyin(y, fmin=70, fmax=450, sr=sr, frame_length=2048, hop_length=256)
    except Exception:
        return {"f0_median_hz": 0.0, "f0_std_hz": 0.0, "voiced_ratio": 0.0, "approx_jitter_pct": 0.0}
    valid = f0[np.isfinite(f0)]
    if len(valid) < 3:
        return {"f0_median_hz": med(valid), "f0_std_hz": std(valid), "voiced_ratio": float(np.mean(np.isfinite(f0))), "approx_jitter_pct": 0.0}
    periods = 1.0 / np.maximum(valid, 1e-12)
    jitter = np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-12) * 100.0
    return {"f0_median_hz": med(valid), "f0_std_hz": std(valid), "voiced_ratio": float(np.mean(np.isfinite(f0))), "approx_jitter_pct": float(jitter)}

def shimmer(y):
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    if len(rms) < 3:
        return 0.0
    rms = rms[rms > np.percentile(rms, 25)]
    if len(rms) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-12) * 100.0)

def extract_features(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if len(y) < sr:
        raise ValueError("Audio shorter than 1 second")
    yt, _ = librosa.effects.trim(y, top_db=35)
    if len(yt) < int(0.5 * sr):
        yt = y

    centroid = librosa.feature.spectral_centroid(y=yt, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=yt, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=yt, sr=sr, roll_percent=0.95)[0]
    flatness = librosa.feature.spectral_flatness(y=yt)[0]
    zcr = librosa.feature.zero_crossing_rate(yt)[0]
    harmonic, _ = librosa.effects.hpss(yt)
    total = float(np.sum(yt**2)) + 1e-12
    residual = yt - harmonic

    mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=20, n_fft=1024, hop_length=256)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    feats = {
        "spectral_centroid_hz": mean(centroid),
        "spectral_bandwidth_hz": mean(bandwidth),
        "spectral_rolloff95_hz": mean(rolloff),
        "spectral_flatness": mean(flatness),
        "zero_crossing_rate": mean(zcr),
        "approx_shimmer_pct": shimmer(yt),
        "harmonic_energy_ratio": float(np.sum(harmonic**2) / total),
        "residual_energy_ratio": float(np.sum(residual**2) / total),
        "mfcc_std_mean": float(np.mean(np.std(mfcc, axis=1))),
        "mfcc_delta_abs_mean": float(np.mean(np.abs(d1))),
        "mfcc_delta2_abs_mean": float(np.mean(np.abs(d2))),
        "effective_bandwidth_99_hz": bandwidth99(yt, sr),
        "energy_0_1khz_ratio": band_ratio(yt, sr, 0, 1000),
        "energy_1_3khz_ratio": band_ratio(yt, sr, 1000, 3000),
        "energy_3_4khz_ratio": band_ratio(yt, sr, 3000, 4000),
        "energy_above_4khz_ratio": band_ratio(yt, sr, 4000, 8000),
    }
    feats.update(pitch_features(yt, sr))
    for i in range(20):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
    return feats

def build_dataset(genuine_files, deepfake_files):
    rows = []
    with tempfile.TemporaryDirectory(prefix="sonict_codec_exp_") as td:
        td = Path(td)
        for class_name, label, files in [("genuine", 0, genuine_files), ("deepfake", 1, deepfake_files)]:
            print(f"\nProcessing {class_name}: {len(files)} originals")
            for idx, src in enumerate(files, start=1):
                sid = f"{class_name}_{idx:04d}_{safe_name(src)}"
                print(f"  [{idx}/{len(files)}] {src.name}")
                for cond in CODEC_CONDITIONS:
                    try:
                        variant = AUG_DIR / class_name / sid / f"{cond['name']}{cond['ext']}"
                        if not variant.exists():
                            convert(src, variant, cond["args"])
                        wav = td / f"{sid}_{cond['name']}.wav"
                        analysis_wav(variant, wav)
                        feats = extract_features(wav)
                        row = {
                            "class_name": class_name,
                            "label": label,
                            "source_id": sid,
                            "source_file": str(src),
                            "codec_condition": cond["name"],
                            "variant_file": str(variant),
                        }
                        row.update(feats)
                        rows.append(row)
                    except Exception as e:
                        print(f"    SKIP {cond['name']}: {e}")
    return pd.DataFrame(rows)

def evaluate(name, model, train, test, features):
    model.fit(train[features], train["label"])
    pred = model.predict(test[features])
    return {
        "name": name,
        "model": model,
        "accuracy": accuracy_score(test["label"], pred),
        "precision": precision_score(test["label"], pred, average="macro", zero_division=0),
        "recall": recall_score(test["label"], pred, average="macro", zero_division=0),
        "f1": f1_score(test["label"], pred, average="macro", zero_division=0),
        "cm": confusion_matrix(test["label"], pred, labels=[0, 1]),
        "report": classification_report(test["label"], pred, target_names=["genuine", "deepfake"], digits=4, zero_division=0),
    }

def main():
    print("="*76)
    print("SONICT CODEC-ROBUST DEEPFAKE EXPERIMENT")
    print("="*76)

    genuine_files = collect(GENUINE_DIR)
    deepfake_files = collect(DEEPFAKE_DIR)

    print(f"Genuine originals : {len(genuine_files)}")
    print(f"Deepfake originals: {len(deepfake_files)}")

    if len(genuine_files) < 5:
        raise RuntimeError("Need at least 5 genuine files.")
    if len(deepfake_files) < 5:
        raise RuntimeError("Need at least 5 deepfake files. 10-20+ is recommended.")

    df = build_dataset(genuine_files, deepfake_files)
    dataset_path = OUTPUT_DIR / "codec_feature_dataset.csv"
    df.to_csv(dataset_path, index=False)

    metadata = {"class_name", "label", "source_id", "source_file", "codec_condition", "variant_file"}
    features = [c for c in df.columns if c not in metadata and pd.api.types.is_numeric_dtype(df[c])]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    groups = df["source_id"].values
    train_idx, test_idx = next(splitter.split(df[features], df["label"], groups=groups))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    print("\nGrouped split:")
    print("Train originals:", train["source_id"].nunique())
    print("Test originals :", test["source_id"].nunique())

    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)

    results = [
        evaluate("Random Forest", rf, train, test, features),
        evaluate("Extra Trees", et, train, test, features),
    ]

    for r in results:
        print(f"\n{r['name']}")
        print(f"Accuracy       : {r['accuracy']*100:.2f}%")
        print(f"Macro Precision: {r['precision']*100:.2f}%")
        print(f"Macro Recall   : {r['recall']*100:.2f}%")
        print(f"Macro F1       : {r['f1']*100:.2f}%")
        print("Confusion [genuine=0, deepfake=1]:")
        print(r["cm"])
        print(r["report"])

    best = max(results, key=lambda x: x["f1"])
    model = best["model"]

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_path = OUTPUT_DIR / "codec_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    per_codec_rows = []
    for codec, subset in test.groupby("codec_condition"):
        pred = model.predict(subset[features])
        per_codec_rows.append({
            "codec_condition": codec,
            "n": len(subset),
            "accuracy": accuracy_score(subset["label"], pred),
            "macro_f1": f1_score(subset["label"], pred, average="macro", zero_division=0),
        })

    per_codec = pd.DataFrame(per_codec_rows)
    per_codec_path = OUTPUT_DIR / "codec_condition_results.csv"
    per_codec.to_csv(per_codec_path, index=False)

    model_path = OUTPUT_DIR / "codec_robust_experimental_model.pkl"
    features_path = OUTPUT_DIR / "codec_robust_features.pkl"
    joblib.dump(model, model_path)
    joblib.dump(features, features_path)

    report_path = OUTPUT_DIR / "codec_experiment_results.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("SonicT Codec-Robust Deepfake Experiment\n")
        f.write("="*76 + "\n")
        f.write(f"Genuine originals: {len(genuine_files)}\n")
        f.write(f"Deepfake originals: {len(deepfake_files)}\n")
        f.write(f"Dataset rows: {len(df)}\n\n")
        for r in results:
            f.write(f"{r['name']}\n")
            f.write(f"Accuracy: {r['accuracy']*100:.2f}%\n")
            f.write(f"Macro Precision: {r['precision']*100:.2f}%\n")
            f.write(f"Macro Recall: {r['recall']*100:.2f}%\n")
            f.write(f"Macro F1: {r['f1']*100:.2f}%\n")
            f.write(f"Confusion:\n{r['cm']}\n")
            f.write(r["report"] + "\n\n")
        f.write("Top 15 features\n")
        f.write(importance.head(15).to_string(index=False))
        f.write("\n\nPer-codec performance\n")
        f.write(per_codec.to_string(index=False))

    print("\nBEST MODEL:", best["name"])
    print(f"Macro F1: {best['f1']*100:.2f}%")
    print("\nTop 15 features:")
    print(importance.head(15).to_string(index=False))
    print("\nPer-codec performance:")
    print(per_codec.round(4).to_string(index=False))

    print("\nOutputs:")
    print(dataset_path)
    print(report_path)
    print(importance_path)
    print(per_codec_path)
    print(model_path)
    print(features_path)
    print("\nDo NOT integrate this model into SonicT yet. First review these results.")

if __name__ == "__main__":
    main()
