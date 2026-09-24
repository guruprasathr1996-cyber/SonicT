import os
import torch
import joblib
import numpy as np

from torch import nn

from transformers import (
    WavLMModel,
    Wav2Vec2FeatureExtractor
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"SonicT device: {DEVICE}")


# =========================================================
# PATHS
# =========================================================

WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"

# Local copy of WavLM.
# Once downloaded, SonicT can work without internet.
WAVLM_LOCAL_PATH =  r"C:\Users\Guruprasath\.cache\huggingface\hub\models--microsoft--wavlm-base-plus\snapshots\4c66d4806a428f2e922ccfa1a962776e232d487b"

F1_MODEL_PATH = (
    r"H:\AudioProject\attention_classifier_v2.pt"
)

F2_MODEL_PATH = (
    r"H:\AudioProject\spectrogram_cnn_v1.pt"
)

F3_MODEL_PATH = (
    r"H:\AudioProject\voice_feature_rf_v1.joblib"
)

F4_MODEL_PATH = (
    r"H:\AudioProject\tamper_boundary_cnn_v2.pt"
)

F5_MODEL_PATH = (
    r"E:\ASVspoof2017\replay_cnn_v1.pt"
)

FUSION_MODEL_PATH = (
    r"H:\AudioProject\sonict_final_fusion_model.pkl"
)

FUSION_SCHEMA_PATH = (
    r"H:\AudioProject\sonict_final_feature_schema.pkl"
)



# =========================================================
# EXPERIMENTAL F1B - ROBUST DEEPFAKE DETECTOR V7.2
# Kept separate from the existing F1-F5 fusion.
# =========================================================

F1B_MODEL_PATH = (
    r"H:\SonicT_Compression_Test\experiment_v7_2_symmetric_codec"
    r"\wavlm_cross_generator_v7_2_symmetric_codec.pt"
)

# =========================================================
# FEATURE 1
# ATTENTION CLASSIFIER
# =========================================================

class AttentionClassifier(nn.Module):

    def __init__(
        self,
        input_dim=768
    ):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(
                input_dim,
                128
            ),

            nn.Tanh(),

            nn.Linear(
                128,
                1
            )
        )

        self.dropout = nn.Dropout(
            0.3
        )

        self.classifier = nn.Linear(
            input_dim,
            2
        )


    def forward(
        self,
        x
    ):

        scores = self.attention(
            x
        )

        weights = torch.softmax(
            scores,
            dim=1
        )

        pooled = torch.sum(
            weights * x,
            dim=1
        )

        pooled = self.dropout(
            pooled
        )

        return self.classifier(
            pooled
        )


# =========================================================
# FEATURE 2 / 4 / 5 CNN
# =========================================================

class BasicCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                16
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                16,
                32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),

            nn.AdaptiveAvgPool2d(
                (4, 4)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 4 * 4,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                2
            )
        )


    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        return self.classifier(
            x
        )



# =========================================================
# F1B - V7.2 WAVLM MLP
# 1536 -> 256 -> 64 -> 2
# =========================================================

class F1BClassifier(nn.Module):

    def __init__(self, input_dim=1536):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 2)
        )


    def forward(self, x):

        return self.net(x)


# =========================================================
# CHECK MODEL FILE
# =========================================================

def check_file(
    path,
    model_name
):

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"{model_name} not found:\n{path}"
        )


# =========================================================
# WAVLM LOADER
# =========================================================

def load_wavlm():

    print(
        "\nLoading WavLM backbone..."
    )


    # -----------------------------------------------------
    # OPTION 1
    # Load local copy
    # -----------------------------------------------------

    if os.path.isdir(
        WAVLM_LOCAL_PATH
    ):

        print(
            "Local WavLM model found."
        )

        print(
            f"Loading from: {WAVLM_LOCAL_PATH}"
        )

        try:

            processor = (
                Wav2Vec2FeatureExtractor
                .from_pretrained(
                    WAVLM_LOCAL_PATH,
                    local_files_only=True
                )
            )

            model = (
                WavLMModel
                .from_pretrained(
                    WAVLM_LOCAL_PATH,
                    local_files_only=True
                )
            )

            model.to(
                DEVICE
            )

            model.eval()

            print(
                "WavLM loaded locally."
            )

            return (
                processor,
                model
            )

        except Exception as error:

            print(
                "Local WavLM folder exists "
                "but could not be loaded."
            )

            print(
                error
            )


    # -----------------------------------------------------
    # OPTION 2
    # Try Hugging Face
    # -----------------------------------------------------

    print(
        "Local WavLM model not available."
    )

    print(
        "Trying Hugging Face..."
    )

    try:

        processor = (
            Wav2Vec2FeatureExtractor
            .from_pretrained(
                WAVLM_MODEL_NAME
            )
        )

        model = (
            WavLMModel
            .from_pretrained(
                WAVLM_MODEL_NAME
            )
        )

        model.to(
            DEVICE
        )

        model.eval()

        print(
            "WavLM downloaded/loaded successfully."
        )

        print(
            "\nIMPORTANT:"
        )

        print(
            "For fully offline SonicT operation, "
            "save WavLM to:"
        )

        print(
            WAVLM_LOCAL_PATH
        )

        return (
            processor,
            model
        )

    except Exception as error:

        raise RuntimeError(

            "\n\n"
            "========================================\n"
            "SONICT WAVLM LOADING FAILED\n"
            "========================================\n"
            "\n"
            "SonicT could not load "
            "'microsoft/wavlm-base-plus'.\n"
            "\n"
            "Possible reason:\n"
            "Internet / DNS connection to "
            "huggingface.co is unavailable.\n"
            "\n"
            "Your trained SonicT models are NOT "
            "damaged.\n"
            "\n"
            "Connect to the internet once and "
            "download WavLM, or place the model "
            "locally at:\n"
            f"{WAVLM_LOCAL_PATH}\n"
            "\n"
            "Original error:\n"
            f"{error}\n"
        )


# =========================================================
# START MODEL LOADING
# =========================================================

print(
    "\n========================================="
)

print(
    "Loading SonicT models..."
)

print(
    "========================================="
)


# =========================================================
# WAVLM BACKBONE
# =========================================================

wavlm_processor, wavlm = (
    load_wavlm()
)


# =========================================================
# FEATURE 1
# =========================================================

print(
    "\nLoading F1 - Voice Clone Detection..."
)

check_file(
    F1_MODEL_PATH,
    "F1 model"
)

model_f1 = AttentionClassifier()

model_f1.load_state_dict(

    torch.load(
        F1_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

)

model_f1.to(
    DEVICE
)

model_f1.eval()

print(
    "F1 loaded."
)


# =========================================================
# FEATURE 2
# =========================================================

print(
    "Loading F2 - Spectrogram CNN..."
)

check_file(
    F2_MODEL_PATH,
    "F2 model"
)

model_f2 = BasicCNN()

model_f2.load_state_dict(

    torch.load(
        F2_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

)

model_f2.to(
    DEVICE
)

model_f2.eval()

print(
    "F2 loaded."
)


# =========================================================
# FEATURE 3
# =========================================================

print(
    "Loading F3 - Voice Feature RF..."
)

check_file(
    F3_MODEL_PATH,
    "F3 model"
)

f3_saved = joblib.load(
    F3_MODEL_PATH
)

model_f3 = f3_saved[
    "model"
]

f3_columns = f3_saved[
    "feature_columns"
]

print(
    "F3 loaded."
)


# =========================================================
# FEATURE 4
# =========================================================

print(
    "Loading F4 - Tampering Detection..."
)

check_file(
    F4_MODEL_PATH,
    "F4 model"
)

model_f4 = BasicCNN()

model_f4.load_state_dict(

    torch.load(
        F4_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

)

model_f4.to(
    DEVICE
)

model_f4.eval()

print(
    "F4 loaded."
)


# =========================================================
# FEATURE 5
# =========================================================

print(
    "Loading F5 - Replay Detection..."
)

check_file(
    F5_MODEL_PATH,
    "F5 model"
)

model_f5 = BasicCNN()

model_f5.load_state_dict(

    torch.load(
        F5_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

)

model_f5.to(
    DEVICE
)

model_f5.eval()

print(
    "F5 loaded."
)


# =========================================================
# FINAL FUSION MODEL
# =========================================================

print(
    "Loading Final Fusion Model..."
)

check_file(
    FUSION_MODEL_PATH,
    "Fusion model"
)

fusion_model = joblib.load(
    FUSION_MODEL_PATH
)


# =========================================================
# FINAL FEATURE SCHEMA
# =========================================================

check_file(
    FUSION_SCHEMA_PATH,
    "Fusion feature schema"
)

fusion_features = joblib.load(
    FUSION_SCHEMA_PATH
)



# =========================================================
# EXPERIMENTAL F1B - V7.2
# =========================================================

print(
    "Loading F1B - Robust Deepfake Detector V7.2..."
)

check_file(
    F1B_MODEL_PATH,
    "F1B V7.2 model"
)

# This is the user's own trusted checkpoint. It contains
# NumPy normalization arrays as well as the model state dict.
f1b_saved = torch.load(
    F1B_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

f1b_input_dim = int(
    f1b_saved.get(
        "embedding_size",
        1536
    )
)

model_f1b = F1BClassifier(
    input_dim=f1b_input_dim
)

model_f1b.load_state_dict(
    f1b_saved["model_state_dict"]
)

model_f1b.to(
    DEVICE
)

model_f1b.eval()

f1b_feature_mean = np.asarray(
    f1b_saved["feature_mean"],
    dtype=np.float32
)

f1b_feature_std = np.asarray(
    f1b_saved["feature_std"],
    dtype=np.float32
)

f1b_feature_std[
    f1b_feature_std < 1e-6
] = 1.0

print(
    "F1B V7.2 loaded."
)

print(
    "F1B remains separate from "
    "the existing F1-F5 fusion."
)


# =========================================================
# COMPLETE
# =========================================================

print(
    "\n========================================="
)

print(
    "All SonicT models loaded successfully."
)

print(
    "=========================================\n"
)