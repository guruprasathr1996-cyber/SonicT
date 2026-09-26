import os
import json
import hashlib
import hmac
import uuid
import shutil
import subprocess
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import librosa
from fastapi.staticfiles import StaticFiles
from speaker_verification import register_speaker, verify_speaker
from forensic_report import build_forensic_report
from forensic_pdf import generate_forensic_pdf
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Body,
    Depends,
    Header
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from fastapi.responses import StreamingResponse
from fastapi.responses import Response

from inference import analyze_audio

from database import (
    init_db,
    save_analysis,
    get_reports,
    create_incident,
    get_incidents,
    get_incident_by_uuid,
    get_incident_verification_secret,
    save_otp_challenge,
    update_verification_result,
    get_incident_audit_events,
    update_incident_lifecycle,
    create_user,
    get_user_by_email,
    mark_user_login
)

from verification_service import (
    create_otp_record,
    verify_otp_value
)

from auth_service import (
    create_access_token,
    hash_password,
    verify_access_token,
    verify_password,
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="SonicT Audio Forensic API",
    version="1.4"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://sonic-t.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# =========================================================
# API SECURITY
# =========================================================

API_KEY = os.getenv(
    "SONICT_API_KEY",
    "sonict-demo-2026"
)

# OpenAPI / Swagger API-key security scheme.
# Use the Authorize button in /docs once per Swagger session.
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    scheme_name="SonicTAPIKey"
)


def verify_api_key(
    x_api_key: str = Depends(api_key_header),
    authorization: str = Header(default="")
):
    """
    Protect SonicT integration endpoints with an API key.

    For local development, the default key is:
        sonict-demo-2026

    For deployment, set the SONICT_API_KEY environment variable.
    """

    if x_api_key and hmac.compare_digest(str(x_api_key), str(API_KEY)):
        return {"auth_type": "api_key"}

    if authorization.startswith("Bearer "):
        payload = verify_access_token(authorization[7:].strip())
        if payload:
            return {"auth_type": "user", "user": payload}

    raise HTTPException(
        status_code=401,
        detail="A valid SonicT access token or API key is required."
    )


# =========================================================
# TEMP DIRECTORY
# =========================================================

TEMP_DIR = "temp"

os.makedirs(
    TEMP_DIR,
    exist_ok=True
)


# =========================================================
# INITIALIZE DATABASE
# =========================================================

init_db()


# =========================================================
# SUPPORTED AUDIO FORMATS
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".wma",
    ".opus",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
    ".3gp",
    ".3g2",
    ".ts",
    ".m2ts",
    ".mka",
    ".aiff",
    ".aif",
    ".caf",
    ".amr"
}


# =========================================================
# HUMAN MIMICRY DETECTION - V2
# =========================================================

MIMICRY_MODEL_PATH = (
    r"H:\SonicT_Mimicry\models\sonict_human_mimicry_v2.pkl"
)

MIMICRY_FEATURE_PATH = (
    r"H:\SonicT_Mimicry\models\sonict_human_mimicry_v2_features.pkl"
)

_mimicry_model = None
_mimicry_feature_columns = None


def mimicry_model_available():
    """
    Check whether the standalone human-mimicry model assets
    are available on the local machine.
    """
    return (
        os.path.exists(MIMICRY_MODEL_PATH)
        and os.path.exists(MIMICRY_FEATURE_PATH)
    )


def load_mimicry_assets():
    """
    Load and cache the standalone Human Mimicry Detection V2 model.

    V2 labels:
        0 = HUMAN MIMIC
        1 = REAL HUMAN
    """
    global _mimicry_model
    global _mimicry_feature_columns

    if (
        _mimicry_model is not None
        and _mimicry_feature_columns is not None
    ):
        return _mimicry_model, _mimicry_feature_columns

    if not os.path.exists(MIMICRY_MODEL_PATH):
        raise FileNotFoundError(
            "Human mimicry model not found: "
            + MIMICRY_MODEL_PATH
        )

    if not os.path.exists(MIMICRY_FEATURE_PATH):
        raise FileNotFoundError(
            "Human mimicry feature schema not found: "
            + MIMICRY_FEATURE_PATH
        )

    _mimicry_model = joblib.load(MIMICRY_MODEL_PATH)
    _mimicry_feature_columns = joblib.load(MIMICRY_FEATURE_PATH)

    return _mimicry_model, _mimicry_feature_columns


MIMICRY_SAMPLE_RATE = 22050
MIMICRY_SEGMENT_DURATION = 3.0
MIMICRY_MIN_FINAL_SEGMENT_SECONDS = 1.0
MIMICRY_MIN_TRIMMED_SEGMENT_SECONDS = 0.40

# Safety-oriented decision bands used after multi-segment aggregation.
# These are post-processing thresholds, not retrained model thresholds.
MIMICRY_REAL_THRESHOLD = 0.45
MIMICRY_POSSIBLE_THRESHOLD = 0.65


def extract_mimicry_features_from_audio(
    audio,
    sample_rate
):
    """
    Reproduce the validated V2 feature pipeline for one audio segment.

    V2 preprocessing:
        - mono
        - 22050 Hz librosa analysis rate
        - silence trimming
        - RMS
        - spectral centroid
        - spectral bandwidth
        - spectral rolloff
        - zero crossing rate
        - MFCC 1-20
    """

    audio, _ = librosa.effects.trim(
        audio
    )

    if len(audio) == 0:
        return None

    if len(audio) < int(
        MIMICRY_MIN_TRIMMED_SEGMENT_SECONDS
        * sample_rate
    ):
        return None

    features = {}

    features["rmse"] = float(
        np.mean(
            librosa.feature.rms(
                y=audio
            )
        )
    )

    features["spectral_centroid"] = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=audio,
                sr=sample_rate
            )
        )
    )

    features["spectral_bandwidth"] = float(
        np.mean(
            librosa.feature.spectral_bandwidth(
                y=audio,
                sr=sample_rate
            )
        )
    )

    features["rolloff"] = float(
        np.mean(
            librosa.feature.spectral_rolloff(
                y=audio,
                sr=sample_rate
            )
        )
    )

    features["zero_crossing_rate"] = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                audio
            )
        )
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=20
    )

    for index in range(20):
        features[f"mfcc{index + 1}"] = float(
            np.mean(
                mfcc[index]
            )
        )

    return features


def predict_mimicry_segment(
    audio,
    sample_rate,
    model,
    feature_columns
):
    """
    Run the Human Mimicry V2 model on one already-loaded segment.

    V2 labels:
        0 = HUMAN MIMIC
        1 = REAL HUMAN
    """

    extracted_features = extract_mimicry_features_from_audio(
        audio,
        sample_rate
    )

    if extracted_features is None:
        return None

    missing_features = [
        column
        for column in feature_columns
        if column not in extracted_features
    ]

    if missing_features:
        raise ValueError(
            "Missing mimicry features: "
            + ", ".join(missing_features)
        )

    input_frame = pd.DataFrame(
        [
            {
                column:
                    extracted_features[column]
                for column in feature_columns
            }
        ],
        columns=feature_columns
    )

    probabilities = model.predict_proba(
        input_frame
    )[0]

    probability_map = {
        int(class_label):
            float(probability)
        for class_label, probability
        in zip(
            model.classes_,
            probabilities
        )
    }

    return {
        "human_mimic":
            float(
                probability_map.get(
                    0,
                    0.0
                )
            ),

        "real_human":
            float(
                probability_map.get(
                    1,
                    0.0
                )
            )
    }


def analyze_human_mimicry(audio_path):
    """
    Analyze the complete recording using non-overlapping
    3-second Human Mimicry V2 segments.

    Final decision:
        - each usable segment is analyzed independently
        - final mimic probability = median segment mimic probability
        - median is used to reduce the effect of isolated noisy outliers

    Decision bands:
        < 0.45       -> REAL_HUMAN
        0.45 - 0.65  -> UNCERTAIN
        >= 0.65      -> POSSIBLE_HUMAN_MIMIC
    """

    model, feature_columns = load_mimicry_assets()

    audio, sample_rate = librosa.load(
        audio_path,
        sr=MIMICRY_SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError(
            "Audio file is empty."
        )

    total_duration = (
        len(audio)
        / sample_rate
    )

    segment_samples = int(
        MIMICRY_SEGMENT_DURATION
        * sample_rate
    )

    segment_results = []
    segment_number = 1

    for start_sample in range(
        0,
        len(audio),
        segment_samples
    ):

        end_sample = min(
            start_sample
            + segment_samples,
            len(audio)
        )

        segment_audio = audio[
            start_sample:
            end_sample
        ]

        if len(segment_audio) < int(
            MIMICRY_MIN_FINAL_SEGMENT_SECONDS
            * sample_rate
        ):
            continue

        prediction = predict_mimicry_segment(
            segment_audio,
            sample_rate,
            model,
            feature_columns
        )

        if prediction is None:
            continue

        start_time = (
            start_sample
            / sample_rate
        )

        end_time = (
            end_sample
            / sample_rate
        )

        segment_results.append(
            {
                "segment":
                    segment_number,

                "start":
                    round(
                        start_time,
                        2
                    ),

                "end":
                    round(
                        end_time,
                        2
                    ),

                "human_mimic":
                    round(
                        prediction[
                            "human_mimic"
                        ],
                        4
                    ),

                "real_human":
                    round(
                        prediction[
                            "real_human"
                        ],
                        4
                    )
            }
        )

        segment_number += 1

    if not segment_results:
        raise ValueError(
            "No usable speech segments were found "
            "for human mimicry analysis."
        )

    mimic_probabilities = np.array(
        [
            segment[
                "human_mimic"
            ]
            for segment
            in segment_results
        ],
        dtype=np.float64
    )

    real_probabilities = np.array(
        [
            segment[
                "real_human"
            ]
            for segment
            in segment_results
        ],
        dtype=np.float64
    )

    mean_mimic_probability = float(
        np.mean(
            mimic_probabilities
        )
    )

    median_mimic_probability = float(
        np.median(
            mimic_probabilities
        )
    )

    mimic_std_deviation = float(
        np.std(
            mimic_probabilities
        )
    )

    mean_real_probability = float(
        np.mean(
            real_probabilities
        )
    )

    mimic_segments = int(
        np.sum(
            mimic_probabilities
            >= 0.50
        )
    )

    real_segments = int(
        np.sum(
            mimic_probabilities
            < 0.50
        )
    )

    final_mimic_probability = (
        median_mimic_probability
    )

    final_real_probability = (
        1.0
        - final_mimic_probability
    )

    if mimic_std_deviation < 0.10:
        segment_consistency = "HIGH"

    elif mimic_std_deviation < 0.20:
        segment_consistency = "MEDIUM"

    else:
        segment_consistency = "LOW"

    if (
        final_mimic_probability
        < MIMICRY_REAL_THRESHOLD
    ):
        classification = "REAL_HUMAN"
        confidence = final_real_probability

        security_alert = {
            "alert":
                False,

            "message":
                (
                    "No strong human mimicry evidence "
                    "was detected across the recording."
                ),

            "action":
                (
                    "Continue normal verification. "
                    "Use speaker verification for "
                    "identity-sensitive actions."
                )
        }

    elif (
        final_mimic_probability
        < MIMICRY_POSSIBLE_THRESHOLD
    ):
        classification = "UNCERTAIN"
        confidence = max(
            final_mimic_probability,
            final_real_probability
        )

        security_alert = {
            "alert":
                True,

            "message":
                (
                    "Human mimicry analysis is uncertain. "
                    "Background noise, call quality or "
                    "recording conditions may be affecting "
                    "the result."
                ),

            "action":
                (
                    "Verify the speaker using SonicT speaker "
                    "verification or another trusted "
                    "secondary identity check."
                )
        }

    else:
        classification = "POSSIBLE_HUMAN_MIMIC"
        confidence = final_mimic_probability

        security_alert = {
            "alert":
                True,

            "message":
                (
                    "Possible human voice mimicry or "
                    "impersonation detected across "
                    "the recording."
                ),

            "action":
                (
                    "Verify the speaker through a trusted "
                    "secondary channel before sensitive actions."
                )
        }

    return {
        "classification":
            classification,

        "confidence":
            round(
                confidence,
                4
            ),

        "probabilities": {
            "human_mimic":
                round(
                    final_mimic_probability,
                    4
                ),

            "real_human":
                round(
                    final_real_probability,
                    4
                )
        },

        "mimicry_score":
            round(
                final_mimic_probability
                * 100,
                2
            ),

        "analysis_method":
            "MULTI_SEGMENT_MEDIAN",

        "aggregation":
            "MEDIAN",

        "decision_thresholds": {
            "real_human_below":
                MIMICRY_REAL_THRESHOLD,

            "uncertain_from":
                MIMICRY_REAL_THRESHOLD,

            "possible_human_mimic_from":
                MIMICRY_POSSIBLE_THRESHOLD
        },

        "segment_analysis": {
            "segment_duration_seconds":
                MIMICRY_SEGMENT_DURATION,

            "total_duration_seconds":
                round(
                    total_duration,
                    2
                ),

            "segments_analyzed":
                len(
                    segment_results
                ),

            "mimic_segments":
                mimic_segments,

            "real_segments":
                real_segments,

            "mean_mimic_probability":
                round(
                    mean_mimic_probability,
                    4
                ),

            "median_mimic_probability":
                round(
                    median_mimic_probability,
                    4
                ),

            "mean_real_probability":
                round(
                    mean_real_probability,
                    4
                ),

            "mimic_std_deviation":
                round(
                    mimic_std_deviation,
                    4
                ),

            "segment_consistency":
                segment_consistency,

            "segments":
                segment_results
        },

        "security_alert":
            security_alert
    }


def convert_to_mimicry_wav(
    input_path,
    output_path
):
    """
    Convert uploaded audio/video to mono PCM WAV while
    preserving the source sampling rate. The mimicry extractor
    then performs the validated 22050 Hz librosa resampling.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Mimicry audio conversion failed.\n"
            + result.stderr
        )

    if not os.path.exists(output_path):
        raise RuntimeError(
            "Mimicry WAV file was not created."
        )

    return output_path


# =========================================================
# EVIDENCE INTEGRITY - SHA-256
# =========================================================

def calculate_sha256(
    file_path
):
    """
    Generate a SHA-256 digital fingerprint
    from the original uploaded evidence file.

    The hash is calculated before audio conversion,
    so it represents the exact file submitted to SonicT.
    """

    sha256_hash = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as evidence_file:

        for chunk in iter(
            lambda: evidence_file.read(
                1024 * 1024
            ),
            b""
        ):

            sha256_hash.update(
                chunk
            )

    return sha256_hash.hexdigest()


# =========================================================
# AUDIO CONVERSION
# =========================================================

def convert_to_wav(
    input_path,
    output_path
):
    """
    Convert uploaded audio/video into
    SonicT standard audio format.

    Output:
        WAV
        16 kHz
        Mono
        PCM 16-bit
    """

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Audio conversion failed.\n"
            + result.stderr
        )

    if not os.path.exists(
        output_path
    ):

        raise RuntimeError(
            "Converted WAV file was not created."
        )

    return output_path


# =========================================================
# VOICE INTEGRITY RISK
# =========================================================

def calculate_risk(
    class_probabilities
):

    genuine_prob = float(
        class_probabilities.get(
            "genuine",
            0.0
        )
    )

    risk_score = (
        1.0 - genuine_prob
    ) * 100


    if risk_score < 30:

        risk_level = "LOW"

        recommendation = (
            "No major threat detected. "
            "Continue normal verification."
        )


    elif risk_score < 60:

        risk_level = "MEDIUM"

        recommendation = (
            "Exercise caution and verify caller identity "
            "if sensitive information is requested."
        )


    elif risk_score < 80:

        risk_level = "HIGH"

        recommendation = (
            "Suspicious voice activity detected. "
            "Perform secondary verification before "
            "any sensitive action."
        )


    else:

        risk_level = "CRITICAL"

        recommendation = (
            "High voice integrity risk. "
            "Do not authorize sensitive actions until "
            "the caller is independently verified."
        )


    return {

        "risk_score":
            round(
                risk_score,
                2
            ),

        "risk_level":
            risk_level,

        "recommendation":
            recommendation
    }


# =========================================================
# EXPLAINABLE VOICE INTEGRITY RISK
# =========================================================

def generate_risk_explanation(analysis_result, risk):
    """Explain the existing risk using available F1-F5 evidence without changing model outputs."""
    features = analysis_result.get("features", {}) or {}
    tampering = analysis_result.get("tampering", {}) or {}
    classification = str(analysis_result.get("classification", "unknown")).lower()
    f1 = float(features.get("f1_deepfake", features.get("deepfake_probability", 0.0)) or 0.0)
    f2 = float(features.get("f2_spectrogram", features.get("spectrogram_probability", 0.0)) or 0.0)
    f3 = float(features.get("f3_voice", features.get("voice_probability", 0.0)) or 0.0)
    f4 = float(features.get("f4_tamper", tampering.get("f4_max", 0.0)) or 0.0)
    f5 = float(features.get("f5_replay", features.get("replay_probability", 0.0)) or 0.0)
    reasons = []
    if f1 >= 0.70: reasons.append("Strong AI voice-clone evidence was detected.")
    elif f1 >= 0.50: reasons.append("Moderate AI voice-clone evidence was detected.")
    if f2 >= 0.70: reasons.append("Strong spectrogram artifacts were detected.")
    elif f2 >= 0.50: reasons.append("Some suspicious spectrogram patterns were detected.")
    if f3 >= 0.70: reasons.append("Voice characteristics show strong acoustic inconsistency.")
    elif f3 >= 0.50: reasons.append("Voice characteristics show moderate acoustic inconsistency.")
    if f4 >= 0.70: reasons.append("Strong audio tampering or splicing evidence was detected.")
    elif f4 >= 0.50: reasons.append("Possible audio manipulation was detected.")
    if f5 >= 0.70: reasons.append("Strong replay or re-recording evidence was detected.")
    elif f5 >= 0.50: reasons.append("Possible replay characteristics were detected.")
    scores=[f1,f2,f3,f4,f5]
    suspicious_branches=sum(x>=0.50 for x in scores); strong_branches=sum(x>=0.70 for x in scores)
    if strong_branches >= 2: reasons.append("Multiple forensic branches produced strong suspicious evidence.")
    elif suspicious_branches >= 2: reasons.append("Multiple forensic branches produced suspicious evidence.")
    if not reasons:
        reasons.append("The available forensic branches did not produce strong threat evidence." if classification == "genuine" else "The final fusion model detected suspicious audio evidence.")
    return {"risk_score": risk.get("risk_score",0.0), "risk_level": risk.get("risk_level","LOW"), "classification": classification.upper(), "reasons": reasons[:4], "evidence_summary": {"suspicious_branches": suspicious_branches, "strong_branches": strong_branches}, "recommended_action": risk.get("recommendation",""), "classification_overridden": False, "risk_score_modified": False}


# =========================================================
# SECURITY ALERT
# =========================================================

def generate_alert(
    risk_level
):

    if risk_level == "LOW":

        return {

            "alert":
                False,

            "message":
                "Voice integrity appears normal.",

            "action":
                "Continue normal verification."
        }


    elif risk_level == "MEDIUM":

        return {

            "alert":
                True,

            "message":
                "Moderate voice integrity risk detected.",

            "action":
                (
                    "Verify caller identity before "
                    "sensitive actions."
                )
        }


    elif risk_level == "HIGH":

        return {

            "alert":
                True,

            "message":
                "High voice impersonation risk detected.",

            "action":
                (
                    "Perform secondary verification "
                    "such as callback or MFA."
                )
        }


    else:

        return {

            "alert":
                True,

            "message":
                "Critical voice integrity risk detected.",

            "action":
                (
                    "Do not authorize sensitive actions "
                    "until independent verification "
                    "is completed."
                )
        }




# =========================================================
# EVIDENCE DISAGREEMENT WARNING
# =========================================================

def find_f4_evidence_cluster(
    suspicious_windows,
    minimum_windows=2,
    minimum_cluster_max=0.75,
    expected_hop=0.5
):
    """
    Find the strongest consecutive F4 suspicious-window cluster.

    Experimental operational rule from the small controlled
    SonicT validation set:

        >= 2 consecutive suspicious windows
        AND
        cluster maximum probability >= 0.75

    F4 already defines suspicious windows as probability >= 0.50.

    This helper does not modify the F4 model, Extra Trees fusion,
    class probabilities, or final classification.
    """

    if not suspicious_windows:
        return {
            "strong": False,
            "cluster_count": 0,
            "largest_cluster_windows": 0,
            "cluster_start": None,
            "cluster_end": None,
            "cluster_duration_seconds": 0.0,
            "cluster_max_probability": 0.0,
            "cluster_mean_probability": 0.0
        }

    windows = sorted(
        suspicious_windows,
        key=lambda item: float(
            item.get("start", 0.0)
        )
    )

    clusters = []
    current_cluster = [windows[0]]

    for window in windows[1:]:
        previous = current_cluster[-1]

        current_start = float(
            window.get("start", 0.0)
        )
        previous_start = float(
            previous.get("start", 0.0)
        )

        start_gap = current_start - previous_start

        # F4 uses a 0.5-second hop.
        if start_gap <= expected_hop + 1e-6:
            current_cluster.append(window)
        else:
            clusters.append(current_cluster)
            current_cluster = [window]

    clusters.append(current_cluster)

    # Prefer the largest cluster. If two clusters have the same
    # number of windows, prefer the one with the higher peak.
    strongest_cluster = max(
        clusters,
        key=lambda cluster: (
            len(cluster),
            max(
                float(
                    item.get("probability", 0.0)
                )
                for item in cluster
            )
        )
    )

    cluster_probabilities = [
        float(
            item.get("probability", 0.0)
        )
        for item in strongest_cluster
    ]

    cluster_windows = len(strongest_cluster)

    cluster_start = float(
        strongest_cluster[0].get(
            "start",
            0.0
        )
    )

    cluster_end = float(
        strongest_cluster[-1].get(
            "end",
            cluster_start
        )
    )

    cluster_duration = max(
        0.0,
        cluster_end - cluster_start
    )

    cluster_max = max(cluster_probabilities)

    cluster_mean = (
        sum(cluster_probabilities)
        / len(cluster_probabilities)
    )

    strong = (
        cluster_windows >= minimum_windows
        and cluster_max >= minimum_cluster_max
    )

    return {
        "strong": strong,
        "cluster_count": len(clusters),
        "largest_cluster_windows": cluster_windows,
        "cluster_start": round(cluster_start, 3),
        "cluster_end": round(cluster_end, 3),
        "cluster_duration_seconds": round(
            cluster_duration,
            3
        ),
        "cluster_max_probability": round(
            cluster_max,
            4
        ),
        "cluster_mean_probability": round(
            cluster_mean,
            4
        )
    }


def generate_evidence_warning(analysis_result):
    """
    Surface strong disagreement between the final Extra Trees
    classification and independent forensic branches.

    F4 uses localized consecutive-window evidence rather than
    the previous whole-recording mean/ratio rule.

    This does NOT override the final classification, class
    probabilities, risk score, F1-F5 fusion, or F1B mode.
    """

    classification = str(
        analysis_result.get("classification", "")
    ).lower()

    features = analysis_result.get("features", {}) or {}
    tampering = analysis_result.get("tampering", {}) or {}
    f1b = analysis_result.get(
        "f1b_robust_deepfake", {}
    ) or {}

    replay_probability = float(
        features.get("replay_probability", 0.0) or 0.0
    )

    tamper_max = float(
        tampering.get("f4_max", 0.0) or 0.0
    )
    tamper_mean = float(
        tampering.get("f4_mean", 0.0) or 0.0
    )
    tamper_suspicious_ratio = float(
        tampering.get(
            "f4_suspicious_ratio", 0.0
        ) or 0.0
    )

    suspicious_windows = (
        tampering.get("suspicious_windows", [])
        or []
    )

    f1b_deepfake_probability = float(
        f1b.get(
            "deepfake_probability", 0.0
        ) or 0.0
    )
    f1b_prediction = str(
        f1b.get("prediction", "UNAVAILABLE")
    ).upper()

    # Existing exploratory replay boundary.
    strong_replay = replay_probability >= 0.90

    # Experimental localized F4 candidate:
    # >= 2 consecutive suspicious windows
    # AND cluster max >= 0.75.
    f4_cluster = find_f4_evidence_cluster(
        suspicious_windows=suspicious_windows,
        minimum_windows=2,
        minimum_cluster_max=0.75,
        expected_hop=0.5
    )

    strong_tampering = bool(
        f4_cluster["strong"]
    )

    # Existing separate F1B evidence rule.
    strong_f1b = (
        f1b_prediction == "DEEPFAKE"
        and f1b_deepfake_probability >= 0.50
    )

    evidence_flags = []

    if strong_replay:
        evidence_flags.append(
            "STRONG_REPLAY_EVIDENCE"
        )

    if strong_tampering:
        evidence_flags.append(
            "STRONG_TEMPORAL_TAMPERING_EVIDENCE"
        )

    if strong_f1b:
        evidence_flags.append(
            "F1B_DEEPFAKE_EVIDENCE"
        )

    warning = (
        classification == "genuine"
        and len(evidence_flags) > 0
    )

    if warning:
        warning_type = "MODEL_EVIDENCE_DISAGREEMENT"
        level = (
            "HIGH"
            if len(evidence_flags) >= 2 or strong_f1b
            else "MEDIUM"
        )
        message = (
            "The final SonicT fusion classified the recording "
            "as genuine, but one or more independent forensic "
            "branches produced strong conflicting evidence."
        )
        recommendation = (
            "Do not treat the Genuine classification alone as "
            "conclusive. Perform independent speaker or caller "
            "verification before sensitive actions."
        )
    else:
        warning_type = "NO_SIGNIFICANT_DISAGREEMENT"
        level = "NONE"
        message = (
            "No strong model/evidence disagreement was detected "
            "under the current evidence-warning rules."
        )
        recommendation = (
            "Interpret the final classification together with "
            "the individual forensic evidence."
        )

    return {
        "warning": warning,
        "level": level,
        "type": warning_type,
        "final_classification": (
            classification.upper()
            if classification
            else "UNKNOWN"
        ),
        "evidence_flags": evidence_flags,
        "evidence": {
            "replay_probability": round(
                replay_probability, 4
            ),
            "f4_max_tamper_probability": round(
                tamper_max, 4
            ),
            "f4_mean_tamper_probability": round(
                tamper_mean, 4
            ),
            "f4_suspicious_ratio": round(
                tamper_suspicious_ratio, 4
            ),
            "f4_suspicious_window_count": len(
                suspicious_windows
            ),
            "f4_cluster_count": f4_cluster[
                "cluster_count"
            ],
            "f4_largest_cluster_windows": f4_cluster[
                "largest_cluster_windows"
            ],
            "f4_cluster_start": f4_cluster[
                "cluster_start"
            ],
            "f4_cluster_end": f4_cluster[
                "cluster_end"
            ],
            "f4_cluster_duration_seconds": f4_cluster[
                "cluster_duration_seconds"
            ],
            "f4_cluster_max_probability": f4_cluster[
                "cluster_max_probability"
            ],
            "f4_cluster_mean_probability": f4_cluster[
                "cluster_mean_probability"
            ],
            "f4_localized_evidence": strong_tampering,
            "f4_candidate_rule": {
                "minimum_consecutive_windows": 2,
                "minimum_cluster_max_probability": 0.75,
                "window_suspicious_threshold": 0.50,
                "expected_window_hop_seconds": 0.5,
                "status": "EXPERIMENTAL"
            },
            "f1b_prediction": f1b_prediction,
            "f1b_deepfake_probability": round(
                f1b_deepfake_probability, 4
            )
        },
        "message": message,
        "recommendation": recommendation,
        "final_classification_overridden": False,
        "note": (
            "This is an evidence-disagreement warning layer, "
            "not a replacement classifier. The F4 localized "
            "evidence boundary is experimental and was selected "
            "from a small controlled validation set. Broader "
            "independent validation is required before "
            "production use."
        )
    }


# =========================================================
# OPERATIONAL ASSESSMENT
# =========================================================

def generate_operational_assessment(
    analysis_result,
    risk,
    evidence_warning
):
    """
    Build the operational decision layer without overriding
    the trained SonicT classification.

    Evidence disagreement can escalate the recommended action
    even when the fusion-model risk score itself is LOW.
    """

    classification = str(
        analysis_result.get(
            "classification",
            "unknown"
        )
    ).upper()

    confidence = float(
        analysis_result.get(
            "confidence",
            0.0
        ) or 0.0
    )

    if evidence_warning.get(
        "warning",
        False
    ):
        return {
            "status":
                "VERIFICATION_REQUIRED",

            "evidence_status":
                "CONFLICTING_EVIDENCE",

            "model_classification":
                classification,

            "model_confidence":
                round(
                    confidence,
                    4
                ),

            "fusion_risk_level":
                risk.get(
                    "risk_level",
                    "UNKNOWN"
                ),

            "classification_overridden":
                False,

            "message":
                (
                    "The final model classification is retained, "
                    "but conflicting independent forensic evidence "
                    "requires secondary verification."
                ),

            "recommended_action":
                (
                    "Verify the caller or speaker through an "
                    "independent trusted channel before sensitive "
                    "actions."
                )
        }

    return {
        "status":
            (
                "MONITOR"
                if risk.get(
                    "risk_level"
                ) in [
                    "LOW",
                    "MEDIUM"
                ]
                else "ACTION_REQUIRED"
            ),

        "evidence_status":
            "NO_STRONG_DISAGREEMENT",

        "model_classification":
            classification,

        "model_confidence":
            round(
                confidence,
                4
            ),

        "fusion_risk_level":
            risk.get(
                "risk_level",
                "UNKNOWN"
            ),

        "classification_overridden":
            False,

        "message":
            (
                "No strong independent model disagreement "
                "was detected."
            ),

        "recommended_action":
            risk.get(
                "recommendation",
                "Continue normal verification."
            )
    }


def apply_evidence_aware_alert(
    base_alert,
    evidence_warning
):
    """
    Escalate only the operational alert when strong conflicting
    evidence exists. The trained classification and risk score
    remain unchanged.
    """

    if not evidence_warning.get(
        "warning",
        False
    ):
        return base_alert

    return {
        "alert":
            True,

        "message":
            (
                "Conflicting forensic evidence detected despite "
                "the final model classification."
            ),

        "action":
            (
                "Perform independent speaker or caller verification "
                "before sensitive actions."
            ),

        "trigger":
            "MODEL_EVIDENCE_DISAGREEMENT",

        "warning_level":
            evidence_warning.get(
                "level",
                "UNKNOWN"
            )
    }



# =========================================================
# INCIDENT CREATION / TRUSTED-CHANNEL VERIFICATION
# =========================================================

def create_incident_if_required(
    filename,
    analysis_result,
    risk,
    security_alert,
    evidence_warning,
    operational_assessment,
    analysis_id=None
):
    """
    Create an operational incident only when SonicT requires
    secondary verification.

    This layer never changes the trained forensic classification.
    """
    risk_level = str(
        risk.get("risk_level", "LOW")
    ).upper()

    conflicting_evidence = bool(
        evidence_warning.get("warning", False)
    )

    verification_required = (
        risk_level in ["HIGH", "CRITICAL"]
        or conflicting_evidence
        or operational_assessment.get("status")
        == "VERIFICATION_REQUIRED"
    )

    if not verification_required:
        return None

    incident_uuid = str(uuid.uuid4())

    return create_incident(
        incident_uuid=incident_uuid,
        filename=filename,
        classification=analysis_result.get(
            "classification",
            "unknown"
        ),
        confidence=float(
            analysis_result.get("confidence", 0.0) or 0.0
        ),
        risk_score=float(
            risk.get("risk_score", 0.0) or 0.0
        ),
        risk_level=risk_level,
        alert_message=security_alert.get(
            "message",
            "Verification required."
        ),
        recommended_action=security_alert.get(
            "action",
            operational_assessment.get(
                "recommended_action",
                "Verify through an independent trusted channel."
            )
        ),
        evidence_warning=conflicting_evidence,
        evidence_flags=evidence_warning.get(
            "evidence_flags",
            []
        ),
        analysis_id=analysis_id,
        verification_required=True
    )


# =========================================================
# PRIVACY AND EVIDENCE RETENTION POLICY
# =========================================================

def get_privacy_policy():
    """
    Return the evidence-handling policy used by SonicT.

    Raw uploaded and converted audio files are used only
    during request processing and are removed in the
    endpoint finally blocks.
    """

    return {

        "raw_audio_retained":
            False,

        "temporary_storage":
            True,

        "automatic_cleanup":
            True,

        "retention_policy":
            (
                "Temporary uploaded and converted audio files "
                "are deleted after request processing."
            ),

        "raw_audio_persisted_to_database":
            False,

        "stored_record_type":
            "Analysis metadata and forensic result only",

        "processing_scope":
            "Current SonicT backend",

        "privacy_status":
            "TEMPORARY PROCESSING / AUTO CLEANUP"
    }


# =========================================================
# USER AUTHENTICATION
# =========================================================

def _public_user(user):
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user.get("role", "INVESTIGATOR"),
    }


@app.post("/auth/signup", status_code=201)
def signup(payload: dict = Body(...)):
    name = str(payload.get("name", "")).strip()
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))

    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Full name is required.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="A valid email is required.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must contain at least 8 characters.")
    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="An account already exists for this email.")

    try:
        user = create_user(name, email, hash_password(password))
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(status_code=409, detail="An account already exists for this email.") from exc
        raise

    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "expires_in": 28800,
        "user": _public_user(user),
    }


@app.post("/auth/login")
def login(payload: dict = Body(...)):
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    user = get_user_by_email(email)

    if not user or not user.get("is_active") or not verify_password(password, user.get("password_hash")):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    mark_user_login(user["id"])
    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "expires_in": 28800,
        "user": _public_user(user),
    }


@app.get("/auth/me")
def current_user(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing access token.")
    payload = verify_access_token(authorization[7:].strip())
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired access token.")
    user = get_user_by_email(payload.get("email", ""))
    if not user or not user.get("is_active"):
        raise HTTPException(status_code=401, detail="User account is unavailable.")
    return {"user": _public_user(user)}


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {

        "message":
            "SonicT API is running",

        "system":
            "SonicT Audio Forensics",

        "version":
            "1.4",

        "live_analysis":
            True,

        "api_security":
            "X-API-Key enabled",

        "human_mimicry_detection":
            True,

        "mimicry_analysis_method":
            "Multi-segment median aggregation"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {

        "status":
            "online",

        "models":
            5,

        "fusion":
            "Extra Trees",

        "human_mimicry_detection": {
            "enabled":
                True,

            "model":
                "Random Forest V2",

            "analysis_method":
                "3-second multi-segment median aggregation",

            "available":
                mimicry_model_available(),

            "test_accuracy":
                86.57,

            "macro_f1":
                86.56,

            "mimic_recall":
                95.0
        },

        "live_analysis":
            True,

        "chunk_analysis":
            True,

        "evidence_integrity":
            "SHA-256",

        "privacy_and_retention":
            "Temporary processing with automatic cleanup",

        "api_security":
            "X-API-Key enabled",

        "supported_formats": [
            "WAV",
            "MP3",
            "MP4",
            "M4A",
            "FLAC",
            "AAC",
            "OGG",
            "WMA",
            "OPUS",
            "WEBM"
        ]
    }


# =========================================================
# ANDROID CONNECTION CHECK
# =========================================================

@app.get(
    "/mobile/status",
    dependencies=[Depends(verify_api_key)]
)
def mobile_status():
    """Verify the Android app's server address and API key."""

    return {
        "status": "ready",
        "service": "SonicT Call Guard",
        "models": 5,
        "live_endpoint": "/analyze-live-chunk"
    }


# =========================================================
# REPORTS
# =========================================================

@app.get(
    "/reports",
    dependencies=[Depends(verify_api_key)]
)
def reports():

    try:

        data = get_reports()

        return {

            "count":
                len(data),

            "reports":
                data
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )




# =========================================================
# DOWNLOAD FORENSIC REPORT AS PDF
# =========================================================

@app.post(
    "/download-forensic-report",
    dependencies=[Depends(verify_api_key)]
)
async def download_forensic_report(
    forensic_report: dict = Body(...)
):
    try:
        pdf_buffer = generate_forensic_pdf(
            forensic_report
        )

        report_info = forensic_report.get(
            "report_information",
            {}
        )

        original_filename = str(
            report_info.get(
                "filename",
                "sonict_evidence"
            )
        )

        base_name = os.path.splitext(
            os.path.basename(
                original_filename
            )
        )[0]

        safe_name = "".join(
            character
            if (
                character.isalnum()
                or character in [
                    "-",
                    "_"
                ]
            )
            else "_"
            for character in base_name
        )

        if not safe_name:
            safe_name = "sonict_evidence"

        download_name = (
            safe_name
            + "_sonict_forensic_report.pdf"
        )

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    f'attachment; filename="{download_name}"'
            }
        )

    except Exception as e:
        print(
            "Forensic PDF generation error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate forensic PDF report: "
                + str(e)
            )
        )


# =========================================================
# NORMAL FULL AUDIO ANALYSIS
# =========================================================

@app.post(
    "/analyze",
    dependencies=[Depends(verify_api_key)]
)
async def analyze(
    file: UploadFile = File(...)
):

    original_temp_path = None
    converted_wav_path = None
    evidence_hash = None


    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )


    original_extension = os.path.splitext(
        file.filename
    )[1].lower()


    if original_extension not in SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Supported formats are "
                "WAV, MP3, MP4, M4A, FLAC, "
                "AAC, OGG, WMA, OPUS and WEBM."
            )
        )


    unique_id = str(
        uuid.uuid4()
    )


    original_temp_path = os.path.join(
        TEMP_DIR,
        unique_id + original_extension
    )


    converted_wav_path = os.path.join(
        TEMP_DIR,
        unique_id + "_converted.wav"
    )


    try:

        # =================================================
        # STEP 1
        # SAVE ORIGINAL FILE
        # =================================================

        with open(
            original_temp_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # =================================================
        # STEP 2
        # VERIFY ORIGINAL EVIDENCE FILE
        # =================================================

        if (
            not os.path.exists(
                original_temp_path
            )
            or os.path.getsize(
                original_temp_path
            ) == 0
        ):

            raise HTTPException(
                status_code=400,
                detail="Received empty audio evidence file."
            )


        # =================================================
        # STEP 3
        # GENERATE SHA-256 EVIDENCE HASH
        # =================================================

        evidence_hash = calculate_sha256(
            original_temp_path
        )


        # =================================================
        # STEP 4
        # CONVERT TO WAV
        # =================================================

        convert_to_wav(
            original_temp_path,
            converted_wav_path
        )


        # =================================================
        # STEP 5
        # RUN SONICT
        # =================================================

        result = analyze_audio(
            converted_wav_path
        )


        # =================================================
        # STEP 6
        # VOICE INTEGRITY RISK
        # =================================================

        risk = calculate_risk(
            result[
                "class_probabilities"
            ]
        )


        result[
            "voice_integrity_risk"
        ] = risk

        risk_explanation = generate_risk_explanation(result, risk)
        result["risk_explanation"] = risk_explanation


        # =================================================
        # STEP 7
        # SECURITY ALERT
        # =================================================

        alert = generate_alert(
            risk[
                "risk_level"
            ]
        )


        result[
            "security_alert"
        ] = alert

        evidence_warning = generate_evidence_warning(
            result
        )

        result[
            "evidence_warning"
        ] = evidence_warning

        alert = apply_evidence_aware_alert(
            alert,
            evidence_warning
        )

        result[
            "security_alert"
        ] = alert

        operational_assessment = generate_operational_assessment(
            result,
            risk,
            evidence_warning
        )

        result[
            "operational_assessment"
        ] = operational_assessment


        # =================================================
        # STEP 8
        # INPUT INFORMATION
        # =================================================

        result[
            "input_file"
        ] = {

            "original_filename":
                file.filename,

            "original_format":
                original_extension
                .replace(
                    ".",
                    ""
                )
                .upper(),

            "analysis_format":
                "WAV",

            "sample_rate":
                16000,

            "channels":
                1,

            "encoding":
                "PCM 16-bit",

            "converted":
                original_extension != ".wav",

            "evidence_hash_algorithm":
                "SHA-256",

            "evidence_sha256":
                evidence_hash
        }


        # =================================================
        # STEP 9
        # BUILD STRUCTURED FORENSIC REPORT
        # =================================================

        privacy_info = get_privacy_policy()


        forensic_report = build_forensic_report(
            filename=file.filename,
            analysis_result=result,
            risk_info=risk,
            security_alert=alert,
            speaker_verification=None,
            evidence_hash=evidence_hash,
            privacy_info=privacy_info
        )

        forensic_report[
            "evidence_warning"
        ] = evidence_warning

        forensic_report[
            "operational_assessment"
        ] = operational_assessment

        forensic_report["risk_explanation"] = risk_explanation

        if evidence_warning.get(
            "warning",
            False
        ):
            forensic_report[
                "security_alert"
            ] = alert

            forensic_report[
                "forensic_conclusion"
            ] = (
                "The SonicT fusion model classified the submitted "
                "recording as "
                + str(
                    result.get(
                        "classification",
                        "unknown"
                    )
                ).upper()
                + ", but independent forensic evidence produced "
                "a significant model/evidence disagreement. "
                "The model classification has not been overridden."
            )

            forensic_report[
                "recommended_action"
            ] = (
                "Independent speaker or caller verification is "
                "required before sensitive actions."
            )

        result[
            "forensic_report"
        ] = forensic_report


        result[
            "evidence_integrity"
        ] = forensic_report.get(
            "evidence_integrity",
            {
                "algorithm": "SHA-256",
                "sha256": evidence_hash,
                "status": "HASH GENERATED"
            }
        )


        result[
            "privacy_and_retention"
        ] = forensic_report.get(
            "privacy_and_retention",
            privacy_info
        )


        # =================================================
        # STEP 10
        # SAVE DATABASE REPORT
        # =================================================

        analysis_id = save_analysis(
            file.filename,
            result
        )

        incident = create_incident_if_required(
            filename=file.filename,
            analysis_result=result,
            risk=risk,
            security_alert=alert,
            evidence_warning=evidence_warning,
            operational_assessment=operational_assessment,
            analysis_id=analysis_id
        )

        result["incident"] = incident


        # =================================================
        # STEP 11
        # RETURN RESULT
        # =================================================

        return result


    except HTTPException:

        raise


    except Exception as e:

        print(
            "Analysis error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        if (
            original_temp_path
            and os.path.exists(
                original_temp_path
            )
        ):

            try:

                os.remove(
                    original_temp_path
                )

            except Exception:

                pass


        if (
            converted_wav_path
            and os.path.exists(
                converted_wav_path
            )
        ):

            try:

                os.remove(
                    converted_wav_path
                )

            except Exception:

                pass


# =========================================================
# CHUNK-BASED NEAR REAL-TIME ANALYSIS
# =========================================================

@app.post(
    "/analyze-chunks",
    dependencies=[Depends(verify_api_key)]
)
async def analyze_chunks(
    file: UploadFile = File(...)
):

    original_temp_path = None
    converted_wav_path = None


    # -----------------------------------------------------
    # VALIDATE FILE
    # -----------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )


    original_extension = os.path.splitext(
        file.filename
    )[1].lower()


    if original_extension not in SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Supported formats are "
                "WAV, MP3, MP4, M4A, FLAC, "
                "AAC, OGG, WMA, OPUS and WEBM."
            )
        )


    unique_id = str(
        uuid.uuid4()
    )


    original_temp_path = os.path.join(
        TEMP_DIR,
        unique_id
        + "_chunk_input"
        + original_extension
    )


    converted_wav_path = os.path.join(
        TEMP_DIR,
        unique_id
        + "_chunk_converted.wav"
    )


    try:

        # =================================================
        # STEP 1
        # SAVE ORIGINAL FILE
        # =================================================

        with open(
            original_temp_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # =================================================
        # STEP 2
        # CONVERT TO SONICT FORMAT
        # =================================================

        convert_to_wav(
            original_temp_path,
            converted_wav_path
        )


        # =================================================
        # STEP 3
        # LOAD AUDIO
        # =================================================

        import librosa
        import soundfile as sf


        audio, sr = librosa.load(
            converted_wav_path,
            sr=16000,
            mono=True
        )


        total_duration = (
            len(audio) / sr
        )


        # =================================================
        # STEP 4
        # CHUNK CONFIGURATION
        # =================================================

        chunk_duration = 5.0

        chunk_samples = int(
            chunk_duration * sr
        )

        chunk_results = []


        # =================================================
        # STEP 5
        # ANALYZE EVERY 5-SECOND CHUNK
        # =================================================

        for start_sample in range(
            0,
            len(audio),
            chunk_samples
        ):

            end_sample = min(
                start_sample
                + chunk_samples,
                len(audio)
            )


            chunk_audio = audio[
                start_sample:
                end_sample
            ]


            # Skip final chunk below 1 second

            if len(
                chunk_audio
            ) < sr:

                continue


            start_time = (
                start_sample / sr
            )

            end_time = (
                end_sample / sr
            )


            chunk_id = str(
                uuid.uuid4()
            )


            chunk_path = os.path.join(
                TEMP_DIR,
                chunk_id
                + "_5sec.wav"
            )


            sf.write(
                chunk_path,
                chunk_audio,
                sr,
                subtype="PCM_16"
            )


            try:

                # =========================================
                # RUN F1-F5 + EXTRA TREES
                # =========================================

                chunk_result = analyze_audio(
                    chunk_path
                )


                # =========================================
                # CHUNK RISK
                # =========================================

                risk = calculate_risk(
                    chunk_result[
                        "class_probabilities"
                    ]
                )

                risk_explanation = generate_risk_explanation(chunk_result, risk)


                # =========================================
                # CHUNK ALERT
                # =========================================

                alert = generate_alert(
                    risk[
                        "risk_level"
                    ]
                )

                evidence_warning = generate_evidence_warning(
                    chunk_result
                )

                alert = apply_evidence_aware_alert(
                    alert,
                    evidence_warning
                )

                operational_assessment = generate_operational_assessment(
                    chunk_result,
                    risk,
                    evidence_warning
                )


                # =========================================
                # STORE RESULT
                # =========================================

                chunk_results.append(
                    {

                        "start":
                            round(
                                start_time,
                                2
                            ),

                        "end":
                            round(
                                end_time,
                                2
                            ),

                        "classification":
                            chunk_result[
                                "classification"
                            ],

                        "confidence":
                            round(
                                float(
                                    chunk_result[
                                        "confidence"
                                    ]
                                ),
                                4
                            ),

                        "class_probabilities":
                            chunk_result[
                                "class_probabilities"
                            ],

                        "risk_score":
                            risk[
                                "risk_score"
                            ],

                        "risk_level":
                            risk[
                                "risk_level"
                            ],

                        "recommendation":
                            risk[
                                "recommendation"
                            ],

                        "risk_explanation":
                            risk_explanation,

                        "security_alert":
                            alert,

                        "evidence_warning":
                            evidence_warning,

                        "operational_assessment":
                            operational_assessment

                    }
                )


            finally:

                if os.path.exists(
                    chunk_path
                ):

                    try:

                        os.remove(
                            chunk_path
                        )

                    except Exception:

                        pass


        # =================================================
        # STEP 6
        # COUNT HIGH / CRITICAL CHUNKS
        # =================================================

        high_risk_chunks = [

            chunk

            for chunk
            in chunk_results

            if chunk[
                "risk_level"
            ] in [
                "HIGH",
                "CRITICAL"
            ]
        ]


        # =================================================
        # STEP 7
        # FIND HIGHEST RISK CHUNK
        # =================================================

        highest_risk = (

            max(
                chunk_results,

                key=lambda x:
                    x[
                        "risk_score"
                    ]
            )

            if chunk_results

            else None
        )


        # =================================================
        # STEP 8
        # AVERAGE RISK
        # =================================================

        if chunk_results:

            average_risk = sum(

                chunk[
                    "risk_score"
                ]

                for chunk
                in chunk_results

            ) / len(
                chunk_results
            )

        else:

            average_risk = 0.0


        # =================================================
        # STEP 9
        # CONTINUOUS SUSPICIOUS SEGMENTS
        # =================================================

        continuous_suspicious_segments = []

        current_segment = None


        for chunk in chunk_results:

            is_suspicious = (
                chunk[
                    "risk_level"
                ]
                in [
                    "HIGH",
                    "CRITICAL"
                ]
            )


            if is_suspicious:

                if current_segment is None:

                    current_segment = {

                        "start":
                            chunk[
                                "start"
                            ],

                        "end":
                            chunk[
                                "end"
                            ],

                        "chunks":
                            1,

                        "max_risk":
                            chunk[
                                "risk_score"
                            ]
                    }


                else:

                    current_segment[
                        "end"
                    ] = chunk[
                        "end"
                    ]


                    current_segment[
                        "chunks"
                    ] += 1


                    current_segment[
                        "max_risk"
                    ] = max(

                        current_segment[
                            "max_risk"
                        ],

                        chunk[
                            "risk_score"
                        ]
                    )


            else:

                if current_segment is not None:

                    if (
                        current_segment[
                            "chunks"
                        ] >= 2
                    ):

                        continuous_suspicious_segments.append(
                            current_segment
                        )


                    current_segment = None


        # =================================================
        # STEP 10
        # HANDLE ENDING SEQUENCE
        # =================================================

        if current_segment is not None:

            if (
                current_segment[
                    "chunks"
                ] >= 2
            ):

                continuous_suspicious_segments.append(
                    current_segment
                )


        # =================================================
        # STEP 11
        # CONTINUOUS ALERT
        # =================================================

        continuous_alert = (
            len(
                continuous_suspicious_segments
            ) > 0
        )


        if continuous_alert:

            continuous_alert_info = {

                "alert":
                    True,

                "message":
                    (
                        "Continuous suspicious voice "
                        "activity detected."
                    ),

                "action":
                    (
                        "Perform secondary verification "
                        "immediately using callback, MFA "
                        "or another trusted channel."
                    )
            }


        else:

            continuous_alert_info = {

                "alert":
                    False,

                "message":
                    (
                        "No continuous high-risk "
                        "voice activity detected."
                    ),

                "action":
                    (
                        "Continue normal verification "
                        "and monitor isolated risk events."
                    )
            }


        # =================================================
        # STEP 12
        # RETURN CHUNK ANALYSIS
        # =================================================

        return {

            "filename":
                file.filename,

            "analysis_type":
                "chunk_based",

            "sample_rate":
                16000,

            "channels":
                1,

            "chunk_duration":
                chunk_duration,

            "total_duration":
                round(
                    total_duration,
                    2
                ),

            "total_chunks":
                len(
                    chunk_results
                ),

            "average_risk":
                round(
                    average_risk,
                    2
                ),

            "high_risk_chunks":
                len(
                    high_risk_chunks
                ),

            "highest_risk":
                highest_risk,

            "continuous_alert":
                continuous_alert,

            "continuous_alert_info":
                continuous_alert_info,

            "continuous_suspicious_segments":
                continuous_suspicious_segments,

            "chunks":
                chunk_results
        }


    except HTTPException:

        raise


    except Exception as e:

        print(
            "Chunk analysis error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        if (
            original_temp_path
            and os.path.exists(
                original_temp_path
            )
        ):

            try:

                os.remove(
                    original_temp_path
                )

            except Exception:

                pass


        if (
            converted_wav_path
            and os.path.exists(
                converted_wav_path
            )
        ):

            try:

                os.remove(
                    converted_wav_path
                )

            except Exception:

                pass


# =========================================================
# LIVE MICROPHONE / LIVE CALL CHUNK ANALYSIS
# =========================================================

@app.post(
    "/analyze-live-chunk",
    dependencies=[Depends(verify_api_key)]
)
async def analyze_live_chunk(
    file: UploadFile = File(...)
):

    original_temp_path = None
    converted_wav_path = None


    # =====================================================
    # STEP 1
    # VALIDATE LIVE CHUNK
    # =====================================================

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No live audio filename provided."
        )


    original_extension = os.path.splitext(
        file.filename
    )[1].lower()


    if original_extension not in SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported live audio format. "
                "Supported formats are WAV, MP3, MP4, "
                "M4A, FLAC, AAC, OGG, WMA, OPUS "
                "and WEBM."
            )
        )


    unique_id = str(
        uuid.uuid4()
    )


    original_temp_path = os.path.join(
        TEMP_DIR,
        unique_id
        + "_live_input"
        + original_extension
    )


    converted_wav_path = os.path.join(
        TEMP_DIR,
        unique_id
        + "_live_converted.wav"
    )


    try:

        # =================================================
        # STEP 2
        # SAVE LIVE AUDIO CHUNK
        # =================================================

        with open(
            original_temp_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # =================================================
        # STEP 3
        # CHECK FILE SIZE
        # =================================================

        file_size = os.path.getsize(
            original_temp_path
        )


        if file_size == 0:

            raise HTTPException(
                status_code=400,
                detail="Received empty live audio chunk."
            )


        # =================================================
        # STEP 4
        # CONVERT LIVE CHUNK TO 16 KHZ WAV
        # =================================================

        convert_to_wav(
            original_temp_path,
            converted_wav_path
        )


        # =================================================
        # STEP 5
        # RUN SONICT F1-F5 + EXTRA TREES
        # =================================================

        result = analyze_audio(
            converted_wav_path
        )


        # =================================================
        # STEP 6
        # VOICE INTEGRITY RISK
        # =================================================

        risk = calculate_risk(
            result[
                "class_probabilities"
            ]
        )

        risk_explanation = generate_risk_explanation(result, risk)


        # =================================================
        # STEP 7
        # SECURITY ALERT
        # =================================================

        alert = generate_alert(
            risk[
                "risk_level"
            ]
        )

        evidence_warning = generate_evidence_warning(
            result
        )

        alert = apply_evidence_aware_alert(
            alert,
            evidence_warning
        )

        operational_assessment = generate_operational_assessment(
            result,
            risk,
            evidence_warning
        )


        # =================================================
        # STEP 8
        # RETURN LIVE ANALYSIS
        # =================================================

        return {

            "analysis_type":
                "live_chunk",

            "filename":
                file.filename,

            "sample_rate":
                16000,

            "channels":
                1,

            "classification":
                result[
                    "classification"
                ],

            "confidence":
                round(
                    float(
                        result[
                            "confidence"
                        ]
                    ),
                    4
                ),

            "class_probabilities":
                result[
                    "class_probabilities"
                ],

            # Individual SonicT forensic branches. These values are
            # probabilities in the range 0.0-1.0 and are returned so mobile
            # clients can explain the final fusion decision.
            "feature_analysis": {
                "f1_voice_clone": round(
                    float(result.get("features", {}).get(
                        "voice_clone_probability", 0.0
                    )), 4
                ),
                "f2_spectrogram_artifacts": round(
                    float(result.get("features", {}).get(
                        "spectrogram_probability", 0.0
                    )), 4
                ),
                "f3_voice_features": round(
                    float(result.get("features", {}).get(
                        "voice_feature_probability", 0.0
                    )), 4
                ),
                "f4_tampering": round(
                    float(result.get("tampering", {}).get(
                        "f4_max", 0.0
                    )), 4
                ),
                "f4_mean_tampering": round(
                    float(result.get("tampering", {}).get(
                        "f4_mean", 0.0
                    )), 4
                ),
                "f4_suspicious_ratio": round(
                    float(result.get("tampering", {}).get(
                        "f4_suspicious_ratio", 0.0
                    )), 4
                ),
                "f4_suspicious_windows": result.get(
                    "tampering", {}
                ).get("suspicious_windows", []),
                "f5_replay_attack": round(
                    float(result.get("features", {}).get(
                        "replay_probability", 0.0
                    )), 4
                )
            },

            "risk_score":
                risk[
                    "risk_score"
                ],

            "risk_level":
                risk[
                    "risk_level"
                ],

            "recommendation":
                risk[
                    "recommendation"
                ],

            "risk_explanation":
                risk_explanation,

            "security_alert":
                alert,

            "evidence_warning":
                evidence_warning,

            "operational_assessment":
                operational_assessment,

            "live_status":
                (
                    "VERIFICATION_REQUIRED"
                    if evidence_warning.get(
                        "warning",
                        False
                    )
                    else (
                        "THREAT"
                        if risk[
                            "risk_level"
                        ] in [
                            "HIGH",
                            "CRITICAL"
                        ]
                        else "MONITOR"
                    )
                )
        }


    except HTTPException:

        raise


    except Exception as e:

        print(
            "Live chunk analysis error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        # =================================================
        # DELETE LIVE INPUT
        # =================================================

        if (
            original_temp_path
            and os.path.exists(
                original_temp_path
            )
        ):

            try:

                os.remove(
                    original_temp_path
                )

            except Exception:

                pass


        # =================================================
        # DELETE LIVE CONVERTED WAV
        # =================================================

        if (
            converted_wav_path
            and os.path.exists(
                converted_wav_path
            )
        ):

            try:

                os.remove(
                    converted_wav_path
                )

            except Exception:

                pass

# =========================================================
# HUMAN MIMICRY / HUMAN IMPERSONATION ANALYSIS
# =========================================================

@app.post(
    "/analyze-mimicry",
    dependencies=[Depends(verify_api_key)]
)
async def analyze_mimicry(
    file: UploadFile = File(...)
):
    """
    Human mimicry / human impersonation analysis using
    multi-segment median aggregation.
    """

    original_temp_path = None
    mimicry_wav_path = None
    evidence_hash = None

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No mimicry audio filename provided."
        )

    original_extension = os.path.splitext(
        file.filename
    )[1].lower()

    if original_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported mimicry audio format. "
                "Supported formats are WAV, MP3, MP4, "
                "M4A, FLAC, AAC, OGG, WMA, OPUS and WEBM."
            )
        )

    unique_id = str(
        uuid.uuid4()
    )

    original_temp_path = os.path.join(
        TEMP_DIR,
        unique_id
        + "_mimicry_input"
        + original_extension
    )

    mimicry_wav_path = os.path.join(
        TEMP_DIR,
        unique_id
        + "_mimicry.wav"
    )

    try:

        with open(
            original_temp_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        if (
            not os.path.exists(
                original_temp_path
            )
            or os.path.getsize(
                original_temp_path
            ) == 0
        ):
            raise HTTPException(
                status_code=400,
                detail="Received empty mimicry audio file."
            )

        evidence_hash = calculate_sha256(
            original_temp_path
        )

        convert_to_mimicry_wav(
            original_temp_path,
            mimicry_wav_path
        )

        result = analyze_human_mimicry(
            mimicry_wav_path
        )

        result[
            "analysis_type"
        ] = (
            "human_mimicry_detection_multisegment"
        )

        result[
            "input_file"
        ] = {
            "original_filename":
                file.filename,

            "original_format":
                original_extension
                .replace(
                    ".",
                    ""
                )
                .upper(),

            "analysis_format":
                "WAV",

            "feature_analysis_sample_rate":
                MIMICRY_SAMPLE_RATE,

            "segment_duration_seconds":
                MIMICRY_SEGMENT_DURATION,

            "full_recording_analyzed":
                True,

            "mono":
                True,

            "evidence_hash_algorithm":
                "SHA-256",

            "evidence_sha256":
                evidence_hash
        }

        result[
            "model_info"
        ] = {
            "module":
                "SonicT Human Mimicry Detection",

            "version":
                "V2 + Multi-Segment Aggregation",

            "algorithm":
                "Random Forest",

            "aggregation_method":
                "Median of 3-second segment mimic probabilities",

            "core_f1_f5_modified":
                False,

            "fusion_model_modified":
                False,

            "validated_test_files":
                134,

            "dataset_test_accuracy_percent":
                86.57,

            "dataset_macro_f1_percent":
                86.56,

            "dataset_human_mimic_recall_percent":
                95.0,

            "benchmark_note":
                (
                    "The 86.57% dataset accuracy is the validated "
                    "V2 raw-WAV benchmark. Multi-segment aggregation "
                    "is a deployment post-processing method and does "
                    "not represent a separately measured accuracy."
                ),

            "labels": {
                "0":
                    "HUMAN_MIMIC",

                "1":
                    "REAL_HUMAN"
            },

            "features":
                25,

            "preprocessing": [
                "Entire recording loaded",
                "Non-overlapping 3-second segments",
                "Mono audio",
                "22050 Hz librosa analysis rate",
                "Silence trimming per segment",
                "RMS",
                "Spectral centroid",
                "Spectral bandwidth",
                "Spectral rolloff",
                "Zero crossing rate",
                "MFCC 1-20",
                "Median probability aggregation"
            ],

            "decision_bands": {
                "REAL_HUMAN":
                    "mimic probability < 45%",

                "UNCERTAIN":
                    "45% <= mimic probability < 65%",

                "POSSIBLE_HUMAN_MIMIC":
                    "mimic probability >= 65%"
            }
        }

        result[
            "privacy_and_retention"
        ] = get_privacy_policy()

        result[
            "evidence_integrity"
        ] = {
            "algorithm":
                "SHA-256",

            "sha256":
                evidence_hash,

            "status":
                "HASH GENERATED"
        }

        return result

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except HTTPException:

        raise

    except Exception as e:

        print(
            "Human mimicry analysis error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        for path in [
            original_temp_path,
            mimicry_wav_path
        ]:

            if (
                path
                and os.path.exists(
                    path
                )
            ):

                try:
                    os.remove(
                        path
                    )

                except Exception:
                    pass


# =========================================================
# SPEAKER REGISTRATION
# =========================================================

@app.post(
    "/register-speaker",
    dependencies=[Depends(verify_api_key)]
)
async def register_speaker_endpoint(
    speaker_id: str,
    file: UploadFile = File(...)
):

    original_temp_path = None
    converted_wav_path = None

    speaker_id = str(speaker_id).strip()

    if not speaker_id:
        raise HTTPException(
            status_code=400,
            detail="Speaker ID cannot be empty."
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No speaker audio filename provided."
        )

    original_extension = os.path.splitext(
        file.filename
    )[1].lower()

    if original_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported speaker audio format. "
                "Supported formats are WAV, MP3, MP4, "
                "M4A, FLAC, AAC, OGG, WMA, OPUS and WEBM."
            )
        )

    unique_id = str(uuid.uuid4())

    original_temp_path = os.path.join(
        TEMP_DIR,
        unique_id + "_speaker_input" + original_extension
    )

    converted_wav_path = os.path.join(
        TEMP_DIR,
        unique_id + "_speaker_converted.wav"
    )

    try:
        with open(original_temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if (
            not os.path.exists(original_temp_path)
            or os.path.getsize(original_temp_path) == 0
        ):
            raise HTTPException(
                status_code=400,
                detail="Received empty speaker audio file."
            )

        convert_to_wav(
            original_temp_path,
            converted_wav_path
        )

        speaker_result = register_speaker(
            speaker_id=speaker_id,
            audio_path=converted_wav_path
        )

        return {
            "message": "Speaker registered successfully.",
            "speaker": speaker_result
        }

    except HTTPException:
        raise

    except Exception as e:
        print("Speaker registration error:", str(e))
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        for path in [original_temp_path, converted_wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass



# =========================================================
# SPEAKER VERIFICATION
# =========================================================

@app.post(
    "/verify-speaker",
    dependencies=[Depends(verify_api_key)]
)
async def verify_speaker_endpoint(
    speaker_id: str,
    file: UploadFile = File(...),
    threshold: float = 0.80
):
    original_temp_path = None
    converted_wav_path = None

    speaker_id = str(speaker_id).strip()

    if not speaker_id:
        raise HTTPException(
            status_code=400,
            detail="Speaker ID cannot be empty."
        )

    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(
            status_code=400,
            detail="Threshold must be between 0.0 and 1.0."
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No verification audio filename provided."
        )

    original_extension = os.path.splitext(
        file.filename
    )[1].lower()

    if original_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported verification audio format. "
                "Supported formats are WAV, MP3, MP4, M4A, "
                "FLAC, AAC, OGG, WMA, OPUS and WEBM."
            )
        )

    unique_id = str(uuid.uuid4())

    original_temp_path = os.path.join(
        TEMP_DIR,
        unique_id + "_verify_input" + original_extension
    )

    converted_wav_path = os.path.join(
        TEMP_DIR,
        unique_id + "_verify_converted.wav"
    )

    try:
        with open(original_temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if (
            not os.path.exists(original_temp_path)
            or os.path.getsize(original_temp_path) == 0
        ):
            raise HTTPException(
                status_code=400,
                detail="Received empty verification audio file."
            )

        convert_to_wav(
            original_temp_path,
            converted_wav_path
        )

        verification_result = verify_speaker(
            speaker_id=speaker_id,
            audio_path=converted_wav_path,
            threshold=threshold
        )

        return {
            "message": "Speaker verification completed.",
            "speaker_verification": verification_result
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except HTTPException:
        raise

    except Exception as e:
        print("Speaker verification error:", str(e))
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        for path in [original_temp_path, converted_wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


# =========================================================
# ALERTS & INCIDENTS
# =========================================================

@app.get(
    "/incidents",
    dependencies=[Depends(verify_api_key)]
)
def incidents(limit: int = 100):
    """
    Return recent SonicT verification incidents.
    """
    try:
        data = get_incidents(limit=limit)

        return {
            "count": len(data),
            "incidents": data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get(
    "/incidents/{incident_uuid}",
    dependencies=[Depends(verify_api_key)]
)
def incident_details(incident_uuid: str):
    incident = get_incident_by_uuid(
        str(incident_uuid).strip()
    )

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found."
        )

    return incident


@app.get(
    "/incidents/{incident_uuid}/audit",
    dependencies=[Depends(verify_api_key)]
)
def incident_audit_history(incident_uuid: str):
    incident_uuid = str(incident_uuid).strip()
    incident = get_incident_by_uuid(incident_uuid)

    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    events = get_incident_audit_events(incident_uuid)
    return {
        "incident_uuid": incident_uuid,
        "count": len(events),
        "events": events,
    }


@app.get(
    "/incidents/{incident_uuid}/report",
    dependencies=[Depends(verify_api_key)]
)
def download_incident_report(incident_uuid: str):
    incident_uuid = str(incident_uuid).strip()
    incident = get_incident_by_uuid(incident_uuid)

    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    report = {
        "report_type": "SONICT_INCIDENT_EVIDENCE_REPORT",
        "report_version": "1.0",
        "incident": incident,
        "audit_trail": get_incident_audit_events(incident_uuid),
        "integrity_note": (
            "Trusted-channel actions are operational records and do not "
            "modify SonicT's original forensic classification."
        ),
    }

    filename = f"sonict_incident_{incident_uuid}.json"
    return Response(
        content=json.dumps(report, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


@app.post(
    "/incidents/{incident_uuid}/lifecycle",
    dependencies=[Depends(verify_api_key)]
)
def manage_incident_lifecycle(
    incident_uuid: str,
    payload: dict = Body(...)
):
    incident_uuid = str(incident_uuid).strip()
    action = str(payload.get("action", "")).strip().upper()
    allowed_actions = {
        "ACKNOWLEDGE",
        "START_INVESTIGATION",
        "ADD_NOTE",
        "RESOLVE",
        "REOPEN",
    }

    if action not in allowed_actions:
        raise HTTPException(
            status_code=400,
            detail="A valid lifecycle action is required."
        )

    resolution = payload.get("resolution")
    allowed_resolutions = {
        "CONFIRMED_THREAT",
        "FALSE_POSITIVE",
        "VERIFIED_LEGITIMATE",
        "INCONCLUSIVE",
    }
    if action == "RESOLVE" and str(resolution or "").upper() not in allowed_resolutions:
        raise HTTPException(
            status_code=400,
            detail="Choose a valid resolution before resolving the incident."
        )

    try:
        incident = update_incident_lifecycle(
            incident_uuid=incident_uuid,
            action=action,
            investigator_notes=payload.get("investigator_notes"),
            resolution=resolution,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    return {
        "message": "Incident lifecycle updated.",
        "incident": incident,
        "forensic_classification_changed": False,
    }


@app.post(
    "/verification/request",
    dependencies=[Depends(verify_api_key)]
)
def request_trusted_verification(
    payload: dict = Body(...)
):
    """
    Start a trusted-channel OTP challenge.

    DEVELOPMENT MODE:
    The OTP is returned in the API response so the local SIH
    prototype can demonstrate the workflow without an SMS/email
    provider. Do not expose development_otp in production.
    """
    incident_uuid = str(
        payload.get("incident_uuid", "")
    ).strip()

    if not incident_uuid:
        raise HTTPException(
            status_code=400,
            detail="incident_uuid is required."
        )

    incident = get_incident_by_uuid(
        incident_uuid
    )

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found."
        )

    if not incident.get(
        "verification_required",
        False
    ):
        raise HTTPException(
            status_code=400,
            detail="This incident does not require verification."
        )

    otp_record = create_otp_record(
        incident_uuid
    )

    saved = save_otp_challenge(
        incident_uuid=incident_uuid,
        otp_hash=otp_record["otp_hash"],
        otp_expires_at=otp_record[
            "otp_expires_at"
        ],
        otp_attempts=0
    )

    if not saved:
        raise HTTPException(
            status_code=500,
            detail="Unable to create OTP challenge."
        )

    return {
        "message":
            "Trusted-channel verification started.",

        "incident_uuid":
            incident_uuid,

        "verification_status":
            "PENDING",

        "verification_method":
            "TRUSTED_CHANNEL_OTP",

        "expires_at":
            otp_record["otp_expires_at"],

        "development_mode":
            True,

        "development_otp":
            otp_record["development_otp"],

        "note":
            (
                "For the local prototype only. In production, "
                "send this OTP through a pre-registered trusted "
                "SMS/email channel and do not return it to the client."
            )
    }


@app.post(
    "/verification/verify",
    dependencies=[Depends(verify_api_key)]
)
def verify_trusted_channel(
    payload: dict = Body(...)
):
    incident_uuid = str(
        payload.get("incident_uuid", "")
    ).strip()

    otp = str(
        payload.get("otp", "")
    ).strip()

    if not incident_uuid:
        raise HTTPException(
            status_code=400,
            detail="incident_uuid is required."
        )

    if not otp:
        raise HTTPException(
            status_code=400,
            detail="otp is required."
        )

    secret_record = get_incident_verification_secret(
        incident_uuid
    )

    if secret_record is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found."
        )

    if not secret_record.get("otp_hash"):
        raise HTTPException(
            status_code=400,
            detail=(
                "No OTP challenge is active. "
                "Start verification first."
            )
        )

    success, status, attempts, message = verify_otp_value(
        incident_id=incident_uuid,
        supplied_otp=otp,
        stored_hash=secret_record.get("otp_hash"),
        expires_at=secret_record.get("otp_expires_at"),
        attempts=secret_record.get("otp_attempts", 0)
    )

    update_verification_result(
        incident_uuid=incident_uuid,
        verification_status=status,
        otp_attempts=attempts,
        verified=success
    )

    incident = get_incident_by_uuid(
        incident_uuid
    )

    return {
        "verified":
            success,

        "message":
            message,

        "verification_status":
            status,

        "incident":
            incident,

        "forensic_classification_changed":
            False,

        "note":
            (
                "Trusted-channel verification authenticates a "
                "separate verification step. It does not change "
                "SonicT's forensic classification."
            )
    }


# =========================================================
# OFFLINE FRONTEND
# =========================================================
# Serve the Vite build from a static folder beside this app.py file.
# Keep this block after every API endpoint.
STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(STATIC_DIR), html=True),
        name="frontend",
    )
