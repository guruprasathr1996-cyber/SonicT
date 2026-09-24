"""
SonicT Compressed-Deepfake Diagnostic
====================================

Goal
----
Investigate failure cases where a known deepfake is classified as Genuine
after heavy telephony / low-bitrate compression.

This is a DIAGNOSTIC tool only:
- Does NOT retrain F1-F5
- Does NOT modify app.py
- Does NOT modify inference.py
- Does NOT modify trained model files
- Does NOT change production thresholds

What it measures
----------------
1. Container / codec information via ffprobe
2. Effective bandwidth and high-frequency loss
3. Spectral flatness / centroid / rolloff / ZCR
4. Harmonic-vs-residual energy
5. Pitch stability (F0 statistics)
6. Approximate jitter / shimmer
7. MFCC temporal-delta irregularity
8. Segment-by-segment stability
9. Existing SonicT prediction, if inference.py can be imported
10. Optional comparison against a folder of KNOWN-GENUINE phone recordings

Why this helps
--------------
A compressed deepfake can lose many synthesis artifacts.
This script checks what forensic information survives after compression and
whether the known deepfake sits unusually close to genuine phone recordings.

Place:
    H:\SonicT_API\compressed_deepfake_diagnostic.py

Example:
    cd /d H:\SonicT_API
    python compressed_deepfake_diagnostic.py "E:\Downloads\sample.mpeg"

Optional comparison:
    python compressed_deepfake_diagnostic.py "E:\Downloads\sample.mpeg" --genuine-folder "H:\AudioProject\genuine_phone"

Output:
    <input>_compressed_deepfake_diagnostic.csv
    <input>_compressed_deepfake_summary.txt
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


TARGET_SR = 16000
ANALYSIS_SR = 16000
SEGMENT_SECONDS = 3.0
MIN_SEGMENT_SECONDS = 1.0

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".mpeg", ".mpg", ".m4a",
    ".flac", ".aac", ".ogg", ".wma", ".opus", ".webm",
    ".mov", ".mkv", ".avi", ".3gp", ".3g2", ".ts",
    ".m2ts", ".mka", ".aiff", ".aif", ".caf", ".amr"
}


def run_command(cmd):
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return completed


def ffprobe_metadata(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=format_name,duration,bit_rate:"
        "stream=codec_name,codec_type,sample_rate,channels,bit_rate",
        "-of", "json",
        str(path),
    ]

    try:
        completed = run_command(cmd)
    except FileNotFoundError:
        return {
            "ffprobe_available": False,
            "error": "ffprobe not found in PATH"
        }

    if completed.returncode != 0:
        return {
            "ffprobe_available": True,
            "error": completed.stderr.strip()
        }

    try:
        data = json.loads(completed.stdout)
    except Exception:
        data = {}

    streams = data.get("streams", [])
    audio_stream = None
    for stream in streams:
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    fmt = data.get("format", {})

    def maybe_int(v):
        try:
            return int(v)
        except Exception:
            return None

    def maybe_float(v):
        try:
            return float(v)
        except Exception:
            return None

    return {
        "ffprobe_available": True,
        "format_name": fmt.get("format_name"),
        "codec_name": audio_stream.get("codec_name") if audio_stream else None,
        "sample_rate": maybe_int(audio_stream.get("sample_rate")) if audio_stream else None,
        "channels": maybe_int(audio_stream.get("channels")) if audio_stream else None,
        "stream_bit_rate": maybe_int(audio_stream.get("bit_rate")) if audio_stream else None,
        "container_bit_rate": maybe_int(fmt.get("bit_rate")),
        "duration_seconds": maybe_float(fmt.get("duration")),
    }


def ffmpeg_normalize(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]

    try:
        completed = run_command(cmd)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg is not available in PATH. Confirm `ffmpeg -version` works."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg normalization failed:\n" + completed.stderr[-3000:]
        )


def db(x):
    return 20.0 * np.log10(np.maximum(x, 1e-12))


def safe_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else 0.0


def safe_std(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) else 0.0


def safe_median(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else 0.0


def band_energy_ratio(y, sr, low_hz, high_hz, n_fft=2048, hop_length=512):
    spec = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann",
            center=True,
        )
    ) ** 2

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band_mask = (freqs >= low_hz) & (freqs < high_hz)
    total_mask = freqs >= 0

    band = float(np.sum(spec[band_mask]))
    total = float(np.sum(spec[total_mask])) + 1e-12

    return band / total


def effective_bandwidth_hz(y, sr, percentile=0.99, n_fft=2048, hop_length=512):
    spec = np.abs(
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    ) ** 2

    avg_power = np.mean(spec, axis=1)
    cumulative = np.cumsum(avg_power)
    total = cumulative[-1] + 1e-12
    threshold = percentile * total

    index = int(np.searchsorted(cumulative, threshold))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    index = min(index, len(freqs) - 1)
    return float(freqs[index])


def pitch_features(y, sr):
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=70,
            fmax=450,
            sr=sr,
            frame_length=2048,
            hop_length=256,
        )
    except Exception:
        return {
            "f0_median_hz": 0.0,
            "f0_std_hz": 0.0,
            "voiced_ratio": 0.0,
            "approx_jitter_pct": 0.0,
        }

    valid = f0[np.isfinite(f0)]

    if len(valid) < 3:
        return {
            "f0_median_hz": safe_median(valid),
            "f0_std_hz": safe_std(valid),
            "voiced_ratio": float(np.mean(np.isfinite(f0))) if len(f0) else 0.0,
            "approx_jitter_pct": 0.0,
        }

    periods = 1.0 / np.maximum(valid, 1e-12)
    jitter = (
        np.mean(np.abs(np.diff(periods)))
        / (np.mean(periods) + 1e-12)
    ) * 100.0

    return {
        "f0_median_hz": safe_median(valid),
        "f0_std_hz": safe_std(valid),
        "voiced_ratio": float(np.mean(np.isfinite(f0))),
        "approx_jitter_pct": float(jitter),
    }


def shimmer_feature(y, sr):
    frame_length = 1024
    hop = 256

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop,
        center=True,
    )[0]

    rms = rms[rms > np.percentile(rms, 25)] if len(rms) else rms

    if len(rms) < 3:
        return 0.0

    shimmer = (
        np.mean(np.abs(np.diff(rms)))
        / (np.mean(rms) + 1e-12)
    ) * 100.0

    return float(shimmer)


def harmonic_residual_features(y):
    try:
        harmonic, percussive = librosa.effects.hpss(y)
    except Exception:
        return {
            "harmonic_energy_ratio": 0.0,
            "residual_energy_ratio": 0.0,
        }

    total_energy = float(np.sum(y ** 2)) + 1e-12
    harmonic_energy = float(np.sum(harmonic ** 2))
    residual_energy = float(np.sum((y - harmonic) ** 2))

    return {
        "harmonic_energy_ratio": harmonic_energy / total_energy,
        "residual_energy_ratio": residual_energy / total_energy,
    }


def mfcc_temporal_features(y, sr):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20,
        n_fft=1024,
        hop_length=256,
    )

    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    return {
        "mfcc_std_mean": float(np.mean(np.std(mfcc, axis=1))),
        "mfcc_delta_abs_mean": float(np.mean(np.abs(d1))),
        "mfcc_delta2_abs_mean": float(np.mean(np.abs(d2))),
    }


def spectral_features(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        roll_percent=0.95
    )[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    return {
        "spectral_centroid_hz": safe_mean(centroid),
        "spectral_bandwidth_hz": safe_mean(bandwidth),
        "spectral_rolloff95_hz": safe_mean(rolloff),
        "spectral_flatness": safe_mean(flatness),
        "zero_crossing_rate": safe_mean(zcr),
        "rms_dbfs": safe_mean(db(rms)),
    }


def analyze_signal(y, sr):
    if len(y) < int(0.5 * sr):
        raise ValueError("Audio segment is too short for stable analysis.")

    # Avoid leading/trailing silence from dominating.
    yt, _ = librosa.effects.trim(y, top_db=35)

    if len(yt) < int(0.4 * sr):
        yt = y

    features = {}

    features.update(spectral_features(yt, sr))
    features.update(pitch_features(yt, sr))
    features["approx_shimmer_pct"] = shimmer_feature(yt, sr)
    features.update(harmonic_residual_features(yt))
    features.update(mfcc_temporal_features(yt, sr))

    nyquist = sr / 2.0

    features["effective_bandwidth_99_hz"] = effective_bandwidth_hz(
        yt, sr, percentile=0.99
    )

    # Useful bands for telephony / codec analysis after normalization to 16 kHz.
    features["energy_0_1khz_ratio"] = band_energy_ratio(yt, sr, 0, 1000)
    features["energy_1_3khz_ratio"] = band_energy_ratio(yt, sr, 1000, 3000)
    features["energy_3_4khz_ratio"] = band_energy_ratio(yt, sr, 3000, 4000)

    if nyquist > 4000:
        features["energy_above_4khz_ratio"] = band_energy_ratio(
            yt, sr, 4000, nyquist
        )
    else:
        features["energy_above_4khz_ratio"] = 0.0

    return features


def try_sonict_prediction(path):
    try:
        from inference import analyze_audio
    except Exception as exc:
        return {
            "available": False,
            "error": f"Could not import SonicT inference: {exc}"
        }

    try:
        result = analyze_audio(str(path))
    except Exception as exc:
        return {
            "available": True,
            "error": f"SonicT analyze_audio failed: {exc}"
        }

    return {
        "available": True,
        "classification": result.get("classification"),
        "confidence": result.get("confidence"),
        "class_probabilities": result.get("class_probabilities"),
    }


def segment_analysis(y, sr):
    seg_samples = int(SEGMENT_SECONDS * sr)
    rows = []
    segment_idx = 0

    for start in range(0, len(y), seg_samples):
        end = min(start + seg_samples, len(y))
        seg = y[start:end]

        duration = len(seg) / sr
        if duration < MIN_SEGMENT_SECONDS:
            continue

        segment_idx += 1

        try:
            feats = analyze_signal(seg, sr)
        except Exception:
            continue

        row = {
            "segment": segment_idx,
            "start_sec": round(start / sr, 3),
            "end_sec": round(end / sr, 3),
            "duration_sec": round(duration, 3),
        }
        row.update(feats)
        rows.append(row)

    return rows


def summarize_segments(rows):
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    summary = {}

    numeric_cols = [
        c for c in df.columns
        if c not in {"segment", "start_sec", "end_sec", "duration_sec"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values):
            summary[f"{col}_median"] = float(values.median())
            summary[f"{col}_std"] = float(values.std(ddof=0))

    return summary


def codec_risk_interpretation(meta, feats):
    notes = []

    source_sr = meta.get("sample_rate")
    bitrate = meta.get("stream_bit_rate") or meta.get("container_bit_rate")
    codec = (meta.get("codec_name") or "").lower()

    if source_sr and source_sr <= 8000:
        notes.append(
            "Very narrow-band source: sample rate <= 8 kHz. "
            "Frequencies above ~4 kHz are absent before SonicT analysis."
        )
    elif source_sr and source_sr <= 16000:
        notes.append(
            "Narrow-band / wideband speech source: source sample rate <= 16 kHz."
        )

    if bitrate and bitrate <= 24000:
        notes.append(
            "Very low bitrate compression detected (<= 24 kbps). "
            "Codec processing may suppress synthesis artifacts."
        )
    elif bitrate and bitrate <= 64000:
        notes.append(
            "Low bitrate compression detected (<= 64 kbps)."
        )

    if codec in {"mp3", "aac", "opus", "amr", "vorbis"}:
        notes.append(
            f"Lossy codec detected: {codec.upper()}."
        )

    bw = feats.get("effective_bandwidth_99_hz", 0.0)
    if bw and bw < 3800:
        notes.append(
            f"Most signal energy is contained below about {bw:.0f} Hz, "
            "which is consistent with strong bandwidth limitation."
        )

    high = feats.get("energy_above_4khz_ratio", 0.0)
    if high < 0.005:
        notes.append(
            "Almost no energy survives above 4 kHz after normalization."
        )

    return notes


def collect_genuine_comparison(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(
            f"Genuine comparison folder not found: {folder}"
        )

    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files)


def compare_against_genuine(target_features, genuine_files):
    rows = []

    with tempfile.TemporaryDirectory(prefix="sonict_genuine_compare_") as temp_dir:
        temp_dir = Path(temp_dir)

        for idx, file in enumerate(genuine_files, start=1):
            out = temp_dir / f"genuine_{idx:04d}.wav"

            try:
                ffmpeg_normalize(file, out)
                y, sr = librosa.load(out, sr=ANALYSIS_SR, mono=True)
                feats = analyze_signal(y, sr)
            except Exception as exc:
                rows.append({
                    "file": str(file),
                    "error": str(exc),
                })
                continue

            row = {
                "file": str(file),
                "error": "",
            }
            row.update(feats)
            rows.append(row)

    df = pd.DataFrame(rows)
    valid = df[df["error"] == ""].copy() if "error" in df.columns else df.copy()

    if valid.empty:
        return df, {}

    feature_names = [
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff95_hz",
        "spectral_flatness",
        "zero_crossing_rate",
        "f0_median_hz",
        "f0_std_hz",
        "voiced_ratio",
        "approx_jitter_pct",
        "approx_shimmer_pct",
        "harmonic_energy_ratio",
        "residual_energy_ratio",
        "mfcc_std_mean",
        "mfcc_delta_abs_mean",
        "mfcc_delta2_abs_mean",
        "effective_bandwidth_99_hz",
        "energy_0_1khz_ratio",
        "energy_1_3khz_ratio",
        "energy_3_4khz_ratio",
        "energy_above_4khz_ratio",
    ]

    comparison = {}

    z_scores = []

    for feature in feature_names:
        if feature not in valid.columns or feature not in target_features:
            continue

        vals = pd.to_numeric(valid[feature], errors="coerce").dropna()

        if len(vals) < 2:
            continue

        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        target = float(target_features[feature])

        if std > 1e-12:
            z = (target - mean) / std
            z_scores.append(abs(z))
        else:
            z = 0.0

        comparison[feature] = {
            "target": target,
            "genuine_mean": mean,
            "genuine_std": std,
            "z_score": float(z),
        }

    comparison["aggregate"] = {
        "mean_absolute_z_score": float(np.mean(z_scores)) if z_scores else 0.0,
        "median_absolute_z_score": float(np.median(z_scores)) if z_scores else 0.0,
        "features_compared": len(z_scores),
    }

    return df, comparison


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a suspected compressed deepfake and optionally compare it "
            "against known genuine phone recordings."
        )
    )

    parser.add_argument(
        "audio_file",
        help="Known/suspected deepfake audio file."
    )

    parser.add_argument(
        "--genuine-folder",
        default=None,
        help=(
            "Optional folder containing KNOWN-GENUINE recordings for comparison."
        )
    )

    args = parser.parse_args()

    input_path = Path(args.audio_file)

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"ERROR: Unsupported extension: {input_path.suffix}")
        sys.exit(1)

    print("\n" + "=" * 76)
    print("SONICT COMPRESSED-DEEPFAKE DIAGNOSTIC")
    print("=" * 76)
    print(f"Input: {input_path}")
    print("Mode : Diagnostic only; production models remain unchanged.")

    metadata = ffprobe_metadata(input_path)

    print("\n[1/5] Source codec / container")
    print("-" * 76)
    for key, value in metadata.items():
        print(f"{key:24}: {value}")

    with tempfile.TemporaryDirectory(prefix="sonict_compressed_df_") as temp_dir:
        normalized = Path(temp_dir) / "normalized_16k.wav"

        print("\n[2/5] Normalizing with FFmpeg...")
        ffmpeg_normalize(input_path, normalized)

        y, sr = librosa.load(
            normalized,
            sr=ANALYSIS_SR,
            mono=True,
        )

        duration = len(y) / sr
        print(f"Normalized duration : {duration:.2f} s")
        print(f"Analysis sample rate: {sr} Hz")

        print("\n[3/5] Whole-file forensic features...")
        whole = analyze_signal(y, sr)

        for key, value in whole.items():
            if isinstance(value, float):
                print(f"{key:32}: {value:.6f}")
            else:
                print(f"{key:32}: {value}")

        print("\nCompression/channel observations:")
        observations = codec_risk_interpretation(metadata, whole)

        if observations:
            for item in observations:
                print(f"  - {item}")
        else:
            print("  - No strong compression warning detected.")

        print("\nExisting SonicT result:")
        sonict = try_sonict_prediction(normalized)

        if sonict.get("classification") is not None:
            print(f"  Classification : {sonict.get('classification')}")
            print(f"  Confidence     : {sonict.get('confidence')}")
            print(f"  Probabilities  : {sonict.get('class_probabilities')}")
        else:
            print(f"  {sonict.get('error', 'Unavailable')}")

        print("\n[4/5] Segment-level stability...")
        segments = segment_analysis(y, sr)
        segment_df = pd.DataFrame(segments)

        if not segment_df.empty:
            display_cols = [
                "segment",
                "start_sec",
                "end_sec",
                "spectral_flatness",
                "f0_std_hz",
                "approx_jitter_pct",
                "approx_shimmer_pct",
                "harmonic_energy_ratio",
                "mfcc_delta_abs_mean",
                "effective_bandwidth_99_hz",
            ]

            existing_cols = [
                c for c in display_cols
                if c in segment_df.columns
            ]

            print(
                segment_df[existing_cols]
                .round(4)
                .to_string(index=False)
            )

        segment_summary = summarize_segments(segments)

        comparison_df = None
        comparison = None

        print("\n[5/5] Genuine-reference comparison")

        if args.genuine_folder:
            genuine_files = collect_genuine_comparison(args.genuine_folder)

            print(
                f"Known-genuine files found: {len(genuine_files)}"
            )

            comparison_df, comparison = compare_against_genuine(
                whole,
                genuine_files,
            )

            agg = comparison.get("aggregate", {}) if comparison else {}

            print(
                "Mean absolute feature z-score : "
                f"{agg.get('mean_absolute_z_score', 0.0):.3f}"
            )
            print(
                "Median absolute feature z-score: "
                f"{agg.get('median_absolute_z_score', 0.0):.3f}"
            )

            print("\nLargest deviations from genuine reference:")
            feature_items = [
                (name, info)
                for name, info in comparison.items()
                if name != "aggregate"
            ]

            feature_items.sort(
                key=lambda x: abs(x[1].get("z_score", 0.0)),
                reverse=True,
            )

            for name, info in feature_items[:8]:
                print(
                    f"  {name:<30} "
                    f"z={info['z_score']:+.2f} | "
                    f"target={info['target']:.6f} | "
                    f"genuine mean={info['genuine_mean']:.6f}"
                )

        else:
            print(
                "No genuine folder supplied. "
                "Run again with --genuine-folder for direct comparison."
            )

    output_base = input_path.with_suffix("")

    segment_csv = Path(
        str(output_base)
        + "_compressed_deepfake_diagnostic.csv"
    )

    summary_txt = Path(
        str(output_base)
        + "_compressed_deepfake_summary.txt"
    )

    if not segment_df.empty:
        segment_df.to_csv(segment_csv, index=False)

    lines = []
    lines.append("SonicT Compressed-Deepfake Diagnostic")
    lines.append("=" * 60)
    lines.append(f"Input: {input_path}")
    lines.append("")
    lines.append("SOURCE METADATA")
    lines.append("-" * 60)

    for key, value in metadata.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("WHOLE-FILE FEATURES")
    lines.append("-" * 60)

    for key, value in whole.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("COMPRESSION / CHANNEL OBSERVATIONS")
    lines.append("-" * 60)

    for item in observations:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("EXISTING SONICT RESULT")
    lines.append("-" * 60)
    lines.append(json.dumps(sonict, indent=2, default=str))

    lines.append("")
    lines.append("SEGMENT SUMMARY")
    lines.append("-" * 60)

    for key, value in segment_summary.items():
        lines.append(f"{key}: {value}")

    if comparison:
        lines.append("")
        lines.append("GENUINE-REFERENCE COMPARISON")
        lines.append("-" * 60)
        lines.append(json.dumps(comparison, indent=2, default=str))

    summary_txt.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    if comparison_df is not None:
        genuine_csv = Path(
            str(output_base)
            + "_genuine_reference_features.csv"
        )
        comparison_df.to_csv(genuine_csv, index=False)
        print(f"\nGenuine reference CSV: {genuine_csv}")

    print("\n" + "=" * 76)
    print("OUTPUT")
    print("=" * 76)

    if segment_csv.exists():
        print(f"Segment diagnostic CSV : {segment_csv}")

    print(f"Summary text           : {summary_txt}")

    print("\nHOW TO INTERPRET")
    print("-" * 76)
    print(
        "1. Very low sample rate / bitrate confirms a codec-laundering challenge."
    )
    print(
        "2. If the deepfake is close to genuine references across many features, "
        "simple handcrafted thresholds will probably not solve the problem."
    )
    print(
        "3. If specific features deviate strongly from genuine references, "
        "those features become candidates for a compression-robust detector."
    )
    print(
        "4. Do not change F1-F5 from this one recording. "
        "Repeat with multiple compressed genuine + deepfake recordings."
    )


if __name__ == "__main__":
    main()
