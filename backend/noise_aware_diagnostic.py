"""
SonicT Noise-Aware F1-F5 Diagnostic
-----------------------------------
Purpose:
1. Keep all trained SonicT F1-F5 models unchanged.
2. Split one recording into 5-second chunks.
3. Estimate an approximate single-channel SNR and speech activity per chunk.
4. Run the existing SonicT analyze_audio() pipeline on every usable chunk.
5. Compare:
   - all valid chunks
   - cleaner speech chunks
6. Save a CSV report.

IMPORTANT:
This script is diagnostic only. It does NOT modify app.py,
inference.py, model files, thresholds, or fusion training.

Place this file in:
    H:\\SonicT_API\\noise_aware_diagnostic.py

Run from CMD:
    cd /d H:\\SonicT_API
    python noise_aware_diagnostic.py "E:\\Downloads\\your_audio.mp4"
"""

import argparse
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

from inference import analyze_audio


TARGET_SR = 16000
CHUNK_SECONDS = 5.0
MIN_FINAL_CHUNK_SECONDS = 1.0

# Conservative diagnostic gates.
# These are NOT SonicT decision thresholds.
MIN_SPEECH_RATIO = 0.35
MIN_APPROX_SNR_DB = 8.0

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".mpeg", ".mpg", ".m4a",
    ".flac", ".aac", ".ogg", ".wma", ".opus", ".webm",
    ".mov", ".mkv", ".avi", ".3gp", ".3g2", ".ts",
    ".m2ts", ".mka", ".aiff", ".aif", ".caf", ".amr"
}


def ffmpeg_to_wav(input_path: str, output_path: str) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-acodec", "pcm_s16le",
        output_path,
    ]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg was not found in PATH. Open CMD and confirm `ffmpeg -version` works."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg conversion failed.\n\n"
            + completed.stderr[-3000:]
        )


def frame_rms(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512):
    if len(audio) < frame_length:
        pad = frame_length - len(audio)
        audio = np.pad(audio, (0, pad))

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    return np.asarray(rms, dtype=np.float64)


def estimate_chunk_quality(audio: np.ndarray, sr: int):
    """
    Approximate quality estimate from one audio channel.

    approx_snr_db:
        Uses RMS distribution. It is a relative diagnostic estimate,
        not laboratory-grade SNR.

    speech_ratio:
        Fraction of chunk samples inside non-silent regions detected
        by librosa.effects.split().
    """
    if audio.size == 0:
        return {
            "approx_snr_db": -99.0,
            "speech_ratio": 0.0,
            "rms_dbfs": -99.0,
            "noise_floor_dbfs": -99.0,
            "quality_label": "INVALID",
        }

    eps = 1e-10

    rms_values = frame_rms(audio)
    rms_values = np.maximum(rms_values, eps)

    # Robust energy estimates.
    noise_rms = float(np.percentile(rms_values, 20))
    speech_rms = float(np.percentile(rms_values, 75))

    approx_snr_db = 20.0 * math.log10(
        max(speech_rms, eps) / max(noise_rms, eps)
    )

    total_rms = float(np.sqrt(np.mean(np.square(audio))) + eps)
    rms_dbfs = 20.0 * math.log10(total_rms)
    noise_floor_dbfs = 20.0 * math.log10(max(noise_rms, eps))

    intervals = librosa.effects.split(
        audio,
        top_db=30,
        frame_length=2048,
        hop_length=512,
    )

    active_samples = 0
    for start, end in intervals:
        active_samples += max(0, int(end) - int(start))

    speech_ratio = (
        active_samples / len(audio)
        if len(audio) > 0
        else 0.0
    )

    if speech_ratio < 0.15:
        quality_label = "LOW_SPEECH"
    elif approx_snr_db < 4.0:
        quality_label = "VERY_NOISY"
    elif approx_snr_db < MIN_APPROX_SNR_DB:
        quality_label = "NOISY"
    elif speech_ratio < MIN_SPEECH_RATIO:
        quality_label = "LIMITED_SPEECH"
    else:
        quality_label = "CLEANER_SPEECH"

    return {
        "approx_snr_db": float(approx_snr_db),
        "speech_ratio": float(speech_ratio),
        "rms_dbfs": float(rms_dbfs),
        "noise_floor_dbfs": float(noise_floor_dbfs),
        "quality_label": quality_label,
    }


def normalize_probabilities(probabilities: dict):
    labels = ["genuine", "deepfake", "tampered", "replay"]

    values = np.array(
        [float(probabilities.get(label, 0.0)) for label in labels],
        dtype=np.float64,
    )

    values = np.clip(values, 0.0, None)

    total = float(values.sum())
    if total <= 0:
        return {label: 0.0 for label in labels}

    values /= total

    return {
        label: float(value)
        for label, value in zip(labels, values)
    }


def aggregate_probabilities(rows, use_clean_only=False):
    if use_clean_only:
        selected = [
            row for row in rows
            if row["use_for_clean_aggregation"]
        ]
    else:
        selected = list(rows)

    if not selected:
        return None

    labels = ["genuine", "deepfake", "tampered", "replay"]

    medians = {}
    for label in labels:
        medians[label] = float(
            np.median([
                row[f"prob_{label}"]
                for row in selected
            ])
        )

    normalized = normalize_probabilities(medians)

    best_label = max(
        normalized,
        key=normalized.get,
    )

    return {
        "chunks_used": len(selected),
        "probabilities": normalized,
        "classification": best_label,
        "confidence": normalized[best_label],
    }


def print_probability_block(title, aggregate):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if aggregate is None:
        print("No chunks available for this aggregation.")
        return

    probs = aggregate["probabilities"]

    print(f"Chunks used : {aggregate['chunks_used']}")
    print(f"Class       : {aggregate['classification'].upper()}")
    print(f"Confidence  : {aggregate['confidence'] * 100:.2f}%")
    print()
    print(f"Genuine     : {probs['genuine'] * 100:.2f}%")
    print(f"Replay      : {probs['replay'] * 100:.2f}%")
    print(f"Tampered    : {probs['tampered'] * 100:.2f}%")
    print(f"Deepfake    : {probs['deepfake'] * 100:.2f}%")


def analyze_file(input_path: str):
    input_path = str(Path(input_path).resolve())

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input file was not found:\n{input_path}"
        )

    extension = Path(input_path).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported input extension: {extension}"
        )

    print("\nSonicT Noise-Aware F1-F5 Diagnostic")
    print("-" * 70)
    print(f"Input : {input_path}")
    print("Mode  : Diagnostic only; no model files will be changed.")

    rows = []

    with tempfile.TemporaryDirectory(
        prefix="sonict_noise_diag_"
    ) as temp_dir:

        normalized_path = os.path.join(
            temp_dir,
            "normalized_16k.wav",
        )

        print("\n[1/4] Converting input to 16 kHz mono WAV...")
        ffmpeg_to_wav(
            input_path,
            normalized_path,
        )

        print("[2/4] Loading normalized audio...")
        audio, sr = librosa.load(
            normalized_path,
            sr=TARGET_SR,
            mono=True,
        )

        duration = len(audio) / sr
        print(f"      Duration: {duration:.2f} seconds")

        chunk_samples = int(
            CHUNK_SECONDS * sr
        )

        print("[3/4] Running quality diagnostics + SonicT F1-F5...")

        chunk_number = 0

        for start_sample in range(
            0,
            len(audio),
            chunk_samples,
        ):
            end_sample = min(
                start_sample + chunk_samples,
                len(audio),
            )

            chunk = audio[
                start_sample:end_sample
            ]

            chunk_duration = len(chunk) / sr

            if chunk_duration < MIN_FINAL_CHUNK_SECONDS:
                continue

            chunk_number += 1

            start_time = start_sample / sr
            end_time = end_sample / sr

            quality = estimate_chunk_quality(
                chunk,
                sr,
            )

            chunk_path = os.path.join(
                temp_dir,
                f"chunk_{chunk_number:04d}.wav",
            )

            sf.write(
                chunk_path,
                chunk,
                sr,
                subtype="PCM_16",
            )

            try:
                result = analyze_audio(
                    chunk_path
                )
            except Exception as exc:
                print(
                    f"      Chunk {chunk_number:03d}: SonicT error: {exc}"
                )
                continue

            probs = normalize_probabilities(
                result.get(
                    "class_probabilities",
                    {}
                )
            )

            use_clean = (
                quality["speech_ratio"] >= MIN_SPEECH_RATIO
                and quality["approx_snr_db"] >= MIN_APPROX_SNR_DB
            )

            rows.append({
                "chunk": chunk_number,
                "start_sec": round(start_time, 3),
                "end_sec": round(end_time, 3),
                "duration_sec": round(chunk_duration, 3),

                "approx_snr_db": round(
                    quality["approx_snr_db"],
                    3
                ),
                "speech_ratio": round(
                    quality["speech_ratio"],
                    4
                ),
                "rms_dbfs": round(
                    quality["rms_dbfs"],
                    3
                ),
                "noise_floor_dbfs": round(
                    quality["noise_floor_dbfs"],
                    3
                ),
                "quality_label":
                    quality["quality_label"],

                "use_for_clean_aggregation":
                    use_clean,

                "classification":
                    result.get(
                        "classification",
                        ""
                    ),

                "confidence":
                    float(
                        result.get(
                            "confidence",
                            0.0
                        )
                    ),

                "prob_genuine":
                    probs["genuine"],
                "prob_deepfake":
                    probs["deepfake"],
                "prob_tampered":
                    probs["tampered"],
                "prob_replay":
                    probs["replay"],
            })

            print(
                f"      Chunk {chunk_number:03d} "
                f"{start_time:6.1f}-{end_time:6.1f}s | "
                f"SNR~ {quality['approx_snr_db']:6.2f} dB | "
                f"Speech {quality['speech_ratio']*100:5.1f}% | "
                f"{quality['quality_label']:<14} | "
                f"G {probs['genuine']*100:5.1f}% "
                f"R {probs['replay']*100:5.1f}% "
                f"T {probs['tampered']*100:5.1f}% "
                f"D {probs['deepfake']*100:5.1f}%"
            )

    if not rows:
        raise RuntimeError(
            "No valid audio chunks were analyzed."
        )

    all_aggregate = aggregate_probabilities(
        rows,
        use_clean_only=False,
    )

    clean_aggregate = aggregate_probabilities(
        rows,
        use_clean_only=True,
    )

    print("\n[4/4] Aggregating diagnostic results...")

    print_probability_block(
        "A) ROBUST MEDIAN — ALL VALID CHUNKS",
        all_aggregate,
    )

    print_probability_block(
        "B) ROBUST MEDIAN — CLEANER SPEECH CHUNKS ONLY",
        clean_aggregate,
    )

    if clean_aggregate is None:
        print(
            "\nNOTE: No chunk passed the conservative cleaner-speech gate."
        )
        print(
            "Do NOT reduce the thresholds yet. First inspect the CSV."
        )
    else:
        all_probs = all_aggregate["probabilities"]
        clean_probs = clean_aggregate["probabilities"]

        replay_change = (
            clean_probs["replay"]
            - all_probs["replay"]
        ) * 100

        tamper_change = (
            clean_probs["tampered"]
            - all_probs["tampered"]
        ) * 100

        genuine_change = (
            clean_probs["genuine"]
            - all_probs["genuine"]
        ) * 100

        print("\n" + "=" * 70)
        print("NOISE-SENSITIVITY COMPARISON")
        print("=" * 70)
        print(
            f"Genuine change  : {genuine_change:+.2f} percentage points"
        )
        print(
            f"Replay change   : {replay_change:+.2f} percentage points"
        )
        print(
            f"Tampered change : {tamper_change:+.2f} percentage points"
        )

        if (
            genuine_change > 2.0
            and (
                replay_change < -2.0
                or tamper_change < -2.0
            )
        ):
            print(
                "\nDiagnostic observation:"
            )
            print(
                "Cleaner speech segments reduce at least one suspicious "
                "class while increasing Genuine probability."
            )
            print(
                "This recording shows evidence that background/channel "
                "noise is influencing SonicT predictions."
            )
        else:
            print(
                "\nDiagnostic observation:"
            )
            print(
                "This recording alone does not show a strong enough "
                "noise effect to justify changing production inference."
            )

    df = pd.DataFrame(rows)

    output_csv = str(
        Path(input_path).with_name(
            Path(input_path).stem
            + "_sonict_noise_diagnostic.csv"
        )
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    print("\nCSV report saved:")
    print(output_csv)

    print("\nIMPORTANT:")
    print(
        "- Approximate SNR is a diagnostic estimate, not a calibrated acoustic measurement."
    )
    print(
        "- Do not change F1-F5 model weights or production probabilities from one file."
    )
    print(
        "- Test several known Genuine, Replay and Tampered recordings before integration."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run noise-aware diagnostic analysis using "
            "the existing SonicT F1-F5 inference pipeline."
        )
    )

    parser.add_argument(
        "audio_file",
        help="Path to an audio/video evidence file.",
    )

    args = parser.parse_args()

    try:
        analyze_file(
            args.audio_file
        )
    except Exception as exc:
        print("\nERROR:")
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
