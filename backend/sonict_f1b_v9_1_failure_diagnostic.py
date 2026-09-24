"""
SonicT F1B V9.1 - Failure Case Diagnostic
==========================================
NO TRAINING. NO MODEL MODIFICATION.

Analyzes:
- V7.2 false negatives among 100 Wukong/EnvSDD deepfakes
- V7.2 false positives among 50 genuine-phone recordings
- the difficult compressed challenge
- correctly classified comparison samples

Uses the already-generated V9 CSV and re-extracts diagnostic acoustic features
from only a small number of selected files.
"""

from pathlib import Path
import subprocess, tempfile, warnings, json
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore")

SR = 16000
RESULTS = Path(r"H:\SonicT_F1B_V9\results")
V9_CSV = RESULTS / "v9_external_cross_corpus_results.csv"
CHALLENGE = Path(r"E:\Downloads\recording_6ee192c5-aab6-4a46-8afc-ee9985bb9aaf.mp3.mpeg")
OUT = RESULTS / "v9_1_failure_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

if not V9_CSV.exists():
    raise FileNotFoundError(V9_CSV)

df = pd.read_csv(V9_CSV)

required = {
    "path","filename","truth","source","condition",
    "v7_prediction","v7_median_df","v7_p90_df","v7_max_df",
    "v72_prediction","v72_median_df","v72_p90_df","v72_max_df",
    "v73_prediction","v73_median_df","v73_p90_df","v73_max_df"
}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns: {sorted(missing)}")

# Important V7.2 failures
fn = df[(df.truth == 1) & (df.v72_prediction == 0)].copy()
fp = df[(df.truth == 0) & (df.v72_prediction == 1)].copy()

# Deterministic comparison samples:
# strongest correctly detected deepfakes and safest correctly accepted genuine phones.
correct_df = df[(df.truth == 1) & (df.v72_prediction == 1)].copy()
correct_g = df[(df.truth == 0) & (df.v72_prediction == 0)].copy()

compare_df = correct_df.sort_values("v72_median_df", ascending=False).head(10)
compare_g = correct_g.sort_values("v72_median_df", ascending=True).head(10)

selected = pd.concat([
    fn.assign(case_type="V72_FALSE_NEGATIVE"),
    fp.assign(case_type="V72_FALSE_POSITIVE"),
    compare_df.assign(case_type="CORRECT_DEEPFAKE_REFERENCE"),
    compare_g.assign(case_type="CORRECT_GENUINE_REFERENCE")
], ignore_index=True)

def normalize(y):
    y = np.asarray(y, np.float32)
    peak = np.max(np.abs(y)) if len(y) else 0.0
    return (y / peak if peak > 0 else y).astype(np.float32)

def load_audio(path):
    path = Path(path)
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        return normalize(y)
    except Exception:
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp = Path(f.name)
        f.close()
        try:
            subprocess.run([
                "ffmpeg","-hide_banner","-loglevel","error","-y",
                "-i",str(path),"-vn","-ac","1","-ar","16000",
                "-c:a","pcm_s16le",str(tmp)
            ], check=True)
            y, _ = librosa.load(str(tmp), sr=SR, mono=True)
            return normalize(y)
        finally:
            tmp.unlink(missing_ok=True)

def band_energy(y, low, high):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    return float(S[(f >= low) & (f < high)].sum() / (S.sum() + 1e-12))

def features(path):
    y = load_audio(path)

    centroid = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)

    out = {
        "duration_sec": len(y) / SR,
        "centroid_mean": float(centroid.mean()),
        "centroid_std": float(centroid.std()),
        "bandwidth_mean": float(bandwidth.mean()),
        "bandwidth_std": float(bandwidth.std()),
        "rolloff_mean": float(rolloff.mean()),
        "rolloff_std": float(rolloff.std()),
        "flatness_mean": float(flatness.mean()),
        "flatness_std": float(flatness.std()),
        "zcr_mean": float(zcr.mean()),
        "zcr_std": float(zcr.std()),
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std()),
        "energy_0_1k": band_energy(y,0,1000),
        "energy_1_3k": band_energy(y,1000,3000),
        "energy_3_4k": band_energy(y,3000,4000),
        "energy_4_8k": band_energy(y,4000,8000),
    }

    for i, x in enumerate(mfcc, 1):
        out[f"mfcc{i}_mean"] = float(x.mean())
        out[f"mfcc{i}_std"] = float(x.std())

    return out

rows = []
print("=" * 100)
print("SONICT F1B V9.1 - FAILURE CASE DIAGNOSTIC")
print("=" * 100)
print("V7.2 false-negative deepfakes:", len(fn))
print("V7.2 false-positive genuine  :", len(fp))
print("Correct DF references         :", len(compare_df))
print("Correct genuine references    :", len(compare_g))
print("Training                      : NONE")

print("\nExtracting diagnostic features...")
for i, r in selected.iterrows():
    try:
        f = features(r["path"])
        base = r.to_dict()
        base.update(f)
        rows.append(base)
        print(f"  {len(rows)}/{len(selected)} {r['case_type']}: {r['filename']}")
    except Exception as e:
        print("FAILED:", r["filename"], "->", e)

diag = pd.DataFrame(rows)

# Add challenge as separate diagnostic case.
challenge_row = None
if CHALLENGE.exists():
    print("\nAnalyzing difficult challenge...")
    f = features(CHALLENGE)
    challenge_row = {
        "path": str(CHALLENGE),
        "filename": CHALLENGE.name,
        "truth": 1,
        "source": "special_challenge",
        "condition": "compressed_narrowband",
        "case_type": "DIFFICULT_CHALLENGE",
        "v7_prediction": 1,
        "v7_median_df": 1.0,
        "v7_p90_df": 1.0,
        "v7_max_df": 1.0,
        "v72_prediction": 0,
        "v72_median_df": 0.0024,
        "v72_p90_df": 0.0676,
        "v72_max_df": 0.4780,
        "v73_prediction": 0,
        "v73_median_df": 0.0183,
        "v73_p90_df": 0.5016,
        "v73_max_df": 0.8870,
    }
    challenge_row.update(f)
    diag = pd.concat([diag, pd.DataFrame([challenge_row])], ignore_index=True)

diag.to_csv(OUT / "v9_1_selected_case_features.csv", index=False)

# Compact failure table
score_cols = [
    "filename","truth","source","condition","case_type",
    "v7_median_df","v72_median_df","v73_median_df",
    "v7_p90_df","v72_p90_df","v73_p90_df"
]
failure_table = diag[
    diag.case_type.isin([
        "V72_FALSE_NEGATIVE",
        "V72_FALSE_POSITIVE",
        "DIFFICULT_CHALLENGE"
    ])
][score_cols].copy()

failure_table.to_csv(OUT / "v9_1_important_failures.csv", index=False)

# Compare acoustic group means.
numeric_feature_cols = [
    c for c in diag.columns
    if c in {
        "duration_sec","centroid_mean","centroid_std","bandwidth_mean","bandwidth_std",
        "rolloff_mean","rolloff_std","flatness_mean","flatness_std","zcr_mean","zcr_std",
        "rms_mean","rms_std","energy_0_1k","energy_1_3k","energy_3_4k","energy_4_8k"
    } or c.startswith("mfcc")
]

group_means = diag.groupby("case_type")[numeric_feature_cols].mean(numeric_only=True).T
group_means.to_csv(OUT / "v9_1_acoustic_group_means.csv")

# Standardize challenge relative to correct genuine reference and correct deepfake reference.
challenge_z = {}
if challenge_row is not None:
    ch = diag[diag.case_type == "DIFFICULT_CHALLENGE"].iloc[0]
    g = diag[diag.case_type == "CORRECT_GENUINE_REFERENCE"]
    d = diag[diag.case_type == "CORRECT_DEEPFAKE_REFERENCE"]

    for col in numeric_feature_cols:
        gv = g[col].astype(float)
        dv = d[col].astype(float)

        gsd = float(gv.std(ddof=0))
        dsd = float(dv.std(ddof=0))

        challenge_z[col] = {
            "challenge_value": float(ch[col]),
            "z_vs_correct_genuine": (
                (float(ch[col]) - float(gv.mean())) / gsd
                if gsd > 1e-12 else None
            ),
            "z_vs_correct_deepfake": (
                (float(ch[col]) - float(dv.mean())) / dsd
                if dsd > 1e-12 else None
            )
        }

    zdf = pd.DataFrame([
        {"feature": k, **v}
        for k, v in challenge_z.items()
    ])

    zdf["abs_z_vs_genuine"] = zdf["z_vs_correct_genuine"].abs()
    zdf["abs_z_vs_deepfake"] = zdf["z_vs_correct_deepfake"].abs()
    zdf["max_abs_z"] = zdf[["abs_z_vs_genuine","abs_z_vs_deepfake"]].max(axis=1)
    zdf = zdf.sort_values("max_abs_z", ascending=False)
    zdf.to_csv(OUT / "v9_1_challenge_acoustic_zscores.csv", index=False)

print("\n" + "=" * 100)
print("IMPORTANT V7.2 FAILURE CASES")
print("=" * 100)

for _, r in failure_table.iterrows():
    label = "DEEPFAKE" if int(r.truth) == 1 else "GENUINE"
    print(
        f"\n{r['case_type']} | {label}"
        f"\n  {r['filename']}"
        f"\n  condition: {r['condition']}"
        f"\n  V7 median : {float(r['v7_median_df'])*100:6.2f}%"
        f"\n  V7.2 median: {float(r['v72_median_df'])*100:6.2f}%"
        f"\n  V7.3 median: {float(r['v73_median_df'])*100:6.2f}%"
    )

if challenge_row is not None:
    zdf = pd.read_csv(OUT / "v9_1_challenge_acoustic_zscores.csv")
    print("\n" + "=" * 100)
    print("TOP CHALLENGE ACOUSTIC DEVIATIONS")
    print("=" * 100)
    for _, r in zdf.head(12).iterrows():
        print(
            f"{r['feature']:20s} | "
            f"z vs genuine {r['z_vs_correct_genuine']:7.2f} | "
            f"z vs deepfake {r['z_vs_correct_deepfake']:7.2f}"
        )

print("\nOutputs:", OUT)
print("No training performed. No model files modified.")
print("NOTE: This is diagnostic analysis, not threshold calibration.")
