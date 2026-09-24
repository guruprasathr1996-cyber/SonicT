import numpy as np
import pandas as pd
import librosa
import torch

from model_loader import (
    DEVICE,
    wavlm_processor,
    wavlm,
    model_f1,
    model_f2,
    model_f3,
    f3_columns,
    model_f4,
    model_f5,
    fusion_model,
    fusion_features,
    model_f1b,
    f1b_feature_mean,
    f1b_feature_std
)

SR = 16000


def load_audio_fixed(path, duration=5.0):
    y, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    target = int(
        SR * duration
    )

    y = y[:target]

    if len(y) < target:
        y = np.pad(
            y,
            (0, target - len(y))
        )

    return y


def create_logmel_5sec(path):
    y = load_audio_fixed(
        path,
        duration=5.0
    )

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=1024,
        hop_length=320,
        n_mels=128,
        fmin=20,
        fmax=8000
    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max
    ).astype(np.float32)

    logmel = (
        logmel - logmel.mean()
    ) / (
        logmel.std() + 1e-6
    )

    return logmel


def predict_feature1(path):
    y = load_audio_fixed(
        path,
        duration=5.0
    )

    inputs = wavlm_processor(
        y,
        sampling_rate=SR,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = wavlm(
            inputs.input_values.to(DEVICE)
        )

        embedding = (
            output.last_hidden_state
        )

        logits = model_f1(
            embedding
        )

        prob = torch.softmax(
            logits,
            dim=1
        )[0, 1].item()

    return prob


def predict_feature2(path):
    spec = create_logmel_5sec(
        path
    )

    x = torch.from_numpy(
        spec
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model_f2(x)

        prob = torch.softmax(
            logits,
            dim=1
        )[0, 1].item()

    return prob


def extract_voice_features(path):
    y, sr = librosa.load(
        path,
        sr=SR,
        mono=True,
        duration=5.0
    )

    target_len = SR * 5

    if len(y) < target_len:
        y = np.pad(
            y,
            (0, target_len - len(y))
        )

    y = y[:target_len]

    features = {}

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    for i in range(13):
        features[
            f"mfcc_{i+1}_mean"
        ] = np.mean(
            mfcc[i]
        )

        features[
            f"mfcc_{i+1}_std"
        ] = np.std(
            mfcc[i]
        )

    zcr = librosa.feature.zero_crossing_rate(
        y
    )

    features["zcr_mean"] = np.mean(
        zcr
    )

    features["zcr_std"] = np.std(
        zcr
    )

    rms = librosa.feature.rms(
        y=y
    )

    features["rms_mean"] = np.mean(
        rms
    )

    features["rms_std"] = np.std(
        rms
    )

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features["centroid_mean"] = np.mean(
        centroid
    )

    features["centroid_std"] = np.std(
        centroid
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features["bandwidth_mean"] = np.mean(
        bandwidth
    )

    features["bandwidth_std"] = np.std(
        bandwidth
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    features["rolloff_mean"] = np.mean(
        rolloff
    )

    features["rolloff_std"] = np.std(
        rolloff
    )

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    valid_f0 = f0[
        ~np.isnan(f0)
    ]

    if len(valid_f0) > 0:

        features["f0_mean"] = np.mean(
            valid_f0
        )

        features["f0_std"] = np.std(
            valid_f0
        )

        features["f0_min"] = np.min(
            valid_f0
        )

        features["f0_max"] = np.max(
            valid_f0
        )

    else:

        features["f0_mean"] = 0.0
        features["f0_std"] = 0.0
        features["f0_min"] = 0.0
        features["f0_max"] = 0.0

    return features


def predict_feature3(path):
    features = extract_voice_features(
        path
    )

    row = pd.DataFrame(
        [features]
    )

    row = row[
        f3_columns
    ]

    prob = model_f3.predict_proba(
        row
    )[0, 1]

    return float(prob)


def predict_feature4_windows(path):
    y, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    window_sec = 2.0
    hop_sec = 0.5

    window_samples = int(
        window_sec * SR
    )

    results = []

    start_sec = 0.0

    while True:

        start = int(
            start_sec * SR
        )

        end = (
            start + window_samples
        )

        if end > len(y):
            break

        segment = y[start:end]

        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=SR,
            n_fft=512,
            hop_length=160,
            n_mels=64,
            fmin=20,
            fmax=8000
        )

        logmel = librosa.power_to_db(
            mel,
            ref=np.max
        ).astype(np.float32)

        logmel = (
            logmel - logmel.mean()
        ) / (
            logmel.std() + 1e-6
        )

        x = torch.from_numpy(
            logmel
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            logits = model_f4(x)

            prob = torch.softmax(
                logits,
                dim=1
            )[0, 1].item()

        results.append({
            "start": start_sec,
            "end": start_sec + window_sec,
            "probability": prob
        })

        start_sec += hop_sec

    return results


def extract_f4_temporal_features(path):
    windows = predict_feature4_windows(
        path
    )

    if len(windows) == 0:
        return {
            "f4_max": 0.0,
            "f4_mean": 0.0,
            "f4_median": 0.0,
            "f4_std": 0.0,
            "f4_suspicious_ratio": 0.0,
            "f4_high_ratio": 0.0
        }

    probs = np.array([
        w["probability"]
        for w in windows
    ])

    return {
        "f4_max":
            float(np.max(probs)),

        "f4_mean":
            float(np.mean(probs)),

        "f4_median":
            float(np.median(probs)),

        "f4_std":
            float(np.std(probs)),

        "f4_suspicious_ratio":
            float(np.mean(
                probs >= 0.50
            )),

        "f4_high_ratio":
            float(np.mean(
                probs >= 0.70
            ))
    }


def predict_feature5(path):
    spec = create_logmel_5sec(
        path
    )

    x = torch.from_numpy(
        spec
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model_f5(x)

        prob = torch.softmax(
            logits,
            dim=1
        )[0, 1].item()

    return prob



# =========================================================
# F1B - V7.2 ROBUST DEEPFAKE DETECTOR
# Separate experimental layer; does not change F1-F5 fusion.
# =========================================================

def _make_f1b_windows(path):

    y, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    peak = (
        np.max(np.abs(y))
        if len(y)
        else 0.0
    )

    if peak > 0:
        y = y / peak

    window_samples = 6 * SR
    minimum_samples = 1 * SR

    if len(y) < window_samples:

        if len(y) >= minimum_samples:
            return [y]

        return []

    windows = []

    for start in range(
        0,
        len(y),
        window_samples
    ):

        segment = y[
            start:start + window_samples
        ]

        if len(segment) >= minimum_samples:
            windows.append(segment)

    return windows


def _extract_f1b_embeddings(windows):

    embeddings = []

    batch_size = (
        6
        if DEVICE.type == "cuda"
        else 2
    )

    for start in range(
        0,
        len(windows),
        batch_size
    ):

        batch = windows[
            start:start + batch_size
        ]

        inputs = wavlm_processor(
            batch,
            sampling_rate=SR,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_values = (
            inputs.input_values.to(
                DEVICE
            )
        )

        attention_mask = getattr(
            inputs,
            "attention_mask",
            None
        )

        if attention_mask is not None:
            attention_mask = attention_mask.to(
                DEVICE
            )

        with torch.no_grad():

            hidden = wavlm(
                input_values=input_values,
                attention_mask=attention_mask
            ).last_hidden_state

            try:

                if attention_mask is not None:

                    feature_mask = (
                        wavlm
                        ._get_feature_vector_attention_mask(
                            hidden.shape[1],
                            attention_mask
                        )
                    )

                else:

                    feature_mask = torch.ones(
                        hidden.shape[:2],
                        dtype=torch.bool,
                        device=DEVICE
                    )

            except Exception:

                feature_mask = torch.ones(
                    hidden.shape[:2],
                    dtype=torch.bool,
                    device=DEVICE
                )

            mask = (
                feature_mask
                .unsqueeze(-1)
                .float()
            )

            count = (
                mask.sum(1)
                .clamp(min=1)
            )

            mean = (
                hidden * mask
            ).sum(1) / count

            variance = (
                (
                    (
                        hidden
                        - mean[:, None]
                    ) ** 2
                )
                * mask
            ).sum(1) / count

            std = torch.sqrt(
                variance.clamp(
                    min=1e-8
                )
            )

            embedding = torch.cat(
                [mean, std],
                dim=1
            )

            embeddings.append(
                embedding
                .cpu()
                .numpy()
            )

    return np.concatenate(
        embeddings,
        axis=0
    ).astype(np.float32)


def predict_feature1b(path):

    windows = _make_f1b_windows(
        path
    )

    if len(windows) == 0:

        return {
            "prediction": "UNAVAILABLE",
            "deepfake_probability": 0.0,
            "genuine_probability": 0.0,
            "mean_deepfake_probability": 0.0,
            "p90_deepfake_probability": 0.0,
            "max_deepfake_probability": 0.0,
            "suspicious_ratio": 0.0,
            "window_count": 0,
            "status":
                "Audio is shorter than 1 second."
        }

    embeddings = (
        _extract_f1b_embeddings(
            windows
        )
    )

    normalized = (
        embeddings
        - f1b_feature_mean
    ) / f1b_feature_std

    x = torch.tensor(
        normalized,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        logits = model_f1b(
            x
        )

        deepfake_probs = (
            torch.softmax(
                logits,
                dim=1
            )[:, 1]
            .cpu()
            .numpy()
        )

    median_prob = float(
        np.median(
            deepfake_probs
        )
    )

    mean_prob = float(
        np.mean(
            deepfake_probs
        )
    )

    p90_prob = float(
        np.percentile(
            deepfake_probs,
            90
        )
    )

    max_prob = float(
        np.max(
            deepfake_probs
        )
    )

    suspicious_ratio = float(
        np.mean(
            deepfake_probs >= 0.50
        )
    )

    prediction = (
        "DEEPFAKE"
        if median_prob >= 0.50
        else "GENUINE"
    )

    return {
        "prediction":
            prediction,

        "deepfake_probability":
            median_prob,

        "genuine_probability":
            1.0 - median_prob,

        "mean_deepfake_probability":
            mean_prob,

        "p90_deepfake_probability":
            p90_prob,

        "max_deepfake_probability":
            max_prob,

        "suspicious_ratio":
            suspicious_ratio,

        "window_count":
            int(len(deepfake_probs)),

        "status":
            "Experimental robustness layer; "
            "not part of the existing F1-F5 fusion."
    }



def analyze_audio(path):

    f1 = predict_feature1(path)
    f2 = predict_feature2(path)
    f3 = predict_feature3(path)
    f4 = extract_f4_temporal_features(
        path
    )
    f5 = predict_feature5(path)

    # F1B is evaluated separately. It does not change the
    # existing Extra Trees fusion input or final classification.
    f1b = predict_feature1b(path)

    deepfake_values = np.array(
        [f1, f2, f3]
    )

    deepfake_mean = float(
        np.mean(
            deepfake_values
        )
    )

    deepfake_std = float(
        np.std(
            deepfake_values,
            ddof=1
        )
    )

    deepfake_max = float(
        np.max(
            deepfake_values
        )
    )

    deepfake_min = float(
        np.min(
            deepfake_values
        )
    )

    row = {
        "f1_deepfake": f1,
        "f2_spectrogram": f2,
        "f3_voice": f3,

        "f4_max":
            f4["f4_max"],

        "f4_mean":
            f4["f4_mean"],

        "f4_median":
            f4["f4_median"],

        "f4_std":
            f4["f4_std"],

        "f4_suspicious_ratio":
            f4[
                "f4_suspicious_ratio"
            ],

        "f4_high_ratio":
            f4[
                "f4_high_ratio"
            ],

        "f5_replay": f5,

        "deepfake_mean":
            deepfake_mean,

        "deepfake_std":
            deepfake_std,

        "deepfake_max":
            deepfake_max,

        "deepfake_min":
            deepfake_min,

        "deepfake_range":
            deepfake_max
            - deepfake_min,

        "replay_margin":
            f5 - deepfake_mean,

        "tamper_mean_margin":
            f4["f4_mean"]
            - deepfake_mean
    }

    X = pd.DataFrame(
        [row],
        columns=fusion_features
    )

    prediction = (
        fusion_model.predict(
            X
        )[0]
    )

    probabilities = (
        fusion_model.predict_proba(
            X
        )[0]
    )

    class_probs = {
        cls: float(prob)
        for cls, prob
        in zip(
            fusion_model.classes_,
            probabilities
        )
    }

    suspicious_windows = [
        w
        for w
        in predict_feature4_windows(
            path
        )
        if w["probability"] >= 0.50
    ]

    return {
        "classification":
            prediction,

        "confidence":
            float(
                max(probabilities)
            ),

        "class_probabilities":
            class_probs,

        "features": {
            "voice_clone_probability":
                f1,

            "spectrogram_probability":
                f2,

            "voice_feature_probability":
                f3,

            "replay_probability":
                f5
        },

        "tampering": {
            **f4,
            "suspicious_windows":
                suspicious_windows
        },

        "f1b_robust_deepfake": {
            **f1b,

            "model":
                "WavLM + MLP V7.2",

            "integration_mode":
                "separate_experimental_layer"
        }
    }