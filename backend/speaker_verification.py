import os
import numpy as np
import librosa
import torch

from model_loader import wavlm_processor, wavlm


# =========================================================
# SETTINGS
# =========================================================

SAMPLE_RATE = 16000

SPEAKER_PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "speaker_profiles"
)

os.makedirs(
    SPEAKER_PROFILE_DIR,
    exist_ok=True
)


# =========================================================
# DEVICE
# =========================================================

try:
    DEVICE = next(wavlm.parameters()).device
except Exception:
    DEVICE = torch.device("cpu")


# =========================================================
# LOAD AUDIO
# =========================================================

def load_speaker_audio(audio_path):
    """
    Loads audio as:
    - 16 kHz
    - mono
    - float32
    """

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio is None or len(audio) == 0:
        raise ValueError(
            "Audio file is empty."
        )

    duration = len(audio) / SAMPLE_RATE

    if duration < 1.0:
        raise ValueError(
            "Speaker verification requires at least 1 second of speech."
        )

    return audio.astype(np.float32)


# =========================================================
# SPEAKER EMBEDDING
# =========================================================

def extract_speaker_embedding(audio_path):
    """
    Extract a fixed-size voice representation
    using the existing SonicT WavLM model.
    """

    audio = load_speaker_audio(
        audio_path
    )

    inputs = wavlm_processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(
        DEVICE
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

    wavlm.eval()

    with torch.no_grad():

        if attention_mask is not None:

            outputs = wavlm(
                input_values=input_values,
                attention_mask=attention_mask
            )

        else:

            outputs = wavlm(
                input_values=input_values
            )

    hidden_states = outputs.last_hidden_state

    # Mean pooling across time
    embedding = hidden_states.mean(
        dim=1
    )

    # L2 normalization
    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1
    )

    embedding = (
        embedding
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return embedding


# =========================================================
# SAFE SPEAKER ID
# =========================================================

def sanitize_speaker_id(speaker_id):

    speaker_id = str(
        speaker_id
    ).strip()

    if not speaker_id:
        raise ValueError(
            "Speaker ID cannot be empty."
        )

    safe_id = "".join(
        char
        for char in speaker_id
        if char.isalnum()
        or char in ["_", "-"]
    )

    if not safe_id:
        raise ValueError(
            "Invalid speaker ID."
        )

    return safe_id


# =========================================================
# SPEAKER PROFILE PATH
# =========================================================

def get_profile_path(speaker_id):

    safe_id = sanitize_speaker_id(
        speaker_id
    )

    return os.path.join(
        SPEAKER_PROFILE_DIR,
        f"{safe_id}.npy"
    )


# =========================================================
# REGISTER SPEAKER
# =========================================================

def register_speaker(
    speaker_id,
    audio_path
):

    safe_id = sanitize_speaker_id(
        speaker_id
    )

    embedding = (
        extract_speaker_embedding(
            audio_path
        )
    )

    profile_path = get_profile_path(
        safe_id
    )

    np.save(
        profile_path,
        embedding
    )

    return {
        "speaker_id": safe_id,
        "registered": True,
        "embedding_dimension": int(
            embedding.shape[0]
        ),
        "message": (
            "Trusted speaker profile "
            "registered successfully."
        )
    }


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(
    embedding_a,
    embedding_b
):

    embedding_a = np.asarray(
        embedding_a,
        dtype=np.float32
    )

    embedding_b = np.asarray(
        embedding_b,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(
            embedding_a
        )
        *
        np.linalg.norm(
            embedding_b
        )
    )

    if denominator == 0:
        return 0.0

    similarity = (
        np.dot(
            embedding_a,
            embedding_b
        )
        / denominator
    )

    return float(
        similarity
    )


# =========================================================
# VERIFY SPEAKER
# =========================================================

def verify_speaker(
    speaker_id,
    audio_path,
    threshold=0.80
):

    safe_id = sanitize_speaker_id(
        speaker_id
    )

    profile_path = get_profile_path(
        safe_id
    )

    if not os.path.exists(
        profile_path
    ):
        raise FileNotFoundError(
            f"No registered speaker profile found for '{safe_id}'."
        )

    trusted_embedding = np.load(
        profile_path
    )

    current_embedding = (
        extract_speaker_embedding(
            audio_path
        )
    )

    similarity = cosine_similarity(
        trusted_embedding,
        current_embedding
    )

    speaker_match = (
        similarity >= threshold
    )

    if speaker_match:
        status = "VERIFIED"
        recommendation = (
            "Current voice is consistent "
            "with the registered speaker profile."
        )

    else:
        status = "MISMATCH"
        recommendation = (
            "Voice does not sufficiently match "
            "the registered speaker. "
            "Perform secondary identity verification."
        )

    return {
        "speaker_id": safe_id,

        "speaker_similarity": round(
            similarity,
            4
        ),

        "speaker_similarity_percentage": round(
            similarity * 100,
            2
        ),

        "threshold": threshold,

        "speaker_match": speaker_match,

        "status": status,

        "recommendation": recommendation
    }


# =========================================================
# CHECK SPEAKER EXISTS
# =========================================================

def speaker_exists(
    speaker_id
):

    try:

        profile_path = get_profile_path(
            speaker_id
        )

        return os.path.exists(
            profile_path
        )

    except Exception:

        return False