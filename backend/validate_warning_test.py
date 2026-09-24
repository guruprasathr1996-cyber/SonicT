from pathlib import Path
import csv
import numpy as np

from inference import analyze_audio


# =========================================================
# CONFIGURATION
# =========================================================

TEST_ROOT = Path(
    r"H:\SonicT_Warning_Test"
)

OUTPUT_CSV = TEST_ROOT / "sonict_warning_validation.csv"

CLASSES = [
    "genuine",
    "deepfake",
    "replay",
    "tampered"
]

SUPPORTED = {
    ".wav",
    ".mp3",
    ".mpeg",
    ".m4a",
    ".flac",
    ".aac"
}


# =========================================================
# SAME WARNING LOGIC AS CURRENT app.py
# =========================================================

def generate_evidence_warning(result):

    classification = str(
        result.get("classification", "")
    ).lower()

    features = result.get(
        "features",
        {}
    )

    tampering = result.get(
        "tampering",
        {}
    )

    f1b = result.get(
        "f1b_robust_deepfake",
        {}
    )

    replay_probability = float(
        features.get(
            "replay_probability",
            0.0
        ) or 0.0
    )

    f4_max = float(
        tampering.get(
            "f4_max",
            0.0
        ) or 0.0
    )

    f4_mean = float(
        tampering.get(
            "f4_mean",
            0.0
        ) or 0.0
    )

    f4_suspicious_ratio = float(
        tampering.get(
            "f4_suspicious_ratio",
            0.0
        ) or 0.0
    )

    f1b_prediction = str(
        f1b.get(
            "prediction",
            ""
        )
    ).upper()

    f1b_probability = float(
        f1b.get(
            "deepfake_probability",
            0.0
        ) or 0.0
    )


    # -----------------------------------------------------
    # CURRENT EXPLORATORY WARNING BOUNDARIES
    # -----------------------------------------------------

    strong_replay = (
        replay_probability >= 0.90
    )

    strong_tampering = (
        f4_max >= 0.90
        and
        (
            f4_mean >= 0.30
            or
            f4_suspicious_ratio >= 0.20
        )
    )

    strong_f1b = (
        f1b_prediction == "DEEPFAKE"
        and
        f1b_probability >= 0.50
    )


    flags = []

    if strong_replay:
        flags.append(
            "STRONG_REPLAY_EVIDENCE"
        )

    if strong_tampering:
        flags.append(
            "STRONG_TEMPORAL_TAMPERING_EVIDENCE"
        )

    if strong_f1b:
        flags.append(
            "F1B_DEEPFAKE_EVIDENCE"
        )


    warning = (
        classification == "genuine"
        and
        len(flags) > 0
    )


    if warning:

        if (
            len(flags) >= 2
            or
            strong_f1b
        ):
            level = "HIGH"

        else:
            level = "MEDIUM"

    else:
        level = "NONE"


    return {
        "warning": warning,
        "level": level,
        "flags": flags
    }


# =========================================================
# FIND TEST FILES
# =========================================================

test_files = []

for true_label in CLASSES:

    folder = TEST_ROOT / true_label

    if not folder.exists():

        raise RuntimeError(
            f"Missing folder: {folder}"
        )

    files = sorted(
        [
            p
            for p in folder.iterdir()
            if (
                p.is_file()
                and
                p.suffix.lower() in SUPPORTED
            )
        ]
    )

    print(
        true_label,
        "=",
        len(files),
        "files"
    )

    for path in files:

        test_files.append(
            (
                true_label,
                path
            )
        )


print()
print(
    "Total files:",
    len(test_files)
)
print()


# =========================================================
# RESULTS
# =========================================================

rows = []

confusion = {
    true: {
        pred: 0
        for pred in CLASSES
    }
    for true in CLASSES
}


# =========================================================
# RUN SONICT
# =========================================================

for index, (
    true_label,
    audio_path
) in enumerate(
    test_files,
    start=1
):

    print(
        "=" * 70
    )

    print(
        f"[{index}/{len(test_files)}]"
    )

    print(
        "True label:",
        true_label.upper()
    )

    print(
        "File:",
        audio_path.name
    )


    try:

        result = analyze_audio(
            str(audio_path)
        )


        prediction = str(
            result.get(
                "classification",
                "unknown"
            )
        ).lower()


        confidence = float(
            result.get(
                "confidence",
                0.0
            ) or 0.0
        )


        features = result.get(
            "features",
            {}
        )

        tampering = result.get(
            "tampering",
            {}
        )

        f1b = result.get(
            "f1b_robust_deepfake",
            {}
        )


        f1b_df = float(
            f1b.get(
                "deepfake_probability",
                0.0
            ) or 0.0
        )


        f5_replay = float(
            features.get(
                "replay_probability",
                0.0
            ) or 0.0
        )


        f4_max = float(
            tampering.get(
                "f4_max",
                0.0
            ) or 0.0
        )


        f4_mean = float(
            tampering.get(
                "f4_mean",
                0.0
            ) or 0.0
        )


        f4_ratio = float(
            tampering.get(
                "f4_suspicious_ratio",
                0.0
            ) or 0.0
        )


        warning = generate_evidence_warning(
            result
        )


        correct = (
            prediction == true_label
        )


        if (
            true_label in confusion
            and
            prediction in confusion[true_label]
        ):

            confusion[
                true_label
            ][
                prediction
            ] += 1


        operational_status = (
            "VERIFICATION_REQUIRED"
            if warning["warning"]
            else "NORMAL_MODEL_FLOW"
        )


        row = {

            "file":
                audio_path.name,

            "true_label":
                true_label,

            "prediction":
                prediction,

            "correct":
                correct,

            "confidence":
                round(
                    confidence,
                    6
                ),

            "confidence_pct":
                round(
                    confidence * 100,
                    2
                ),

            "f1b_deepfake_probability":
                round(
                    f1b_df,
                    6
                ),

            "f1b_deepfake_pct":
                round(
                    f1b_df * 100,
                    2
                ),

            "f5_replay_probability":
                round(
                    f5_replay,
                    6
                ),

            "f5_replay_pct":
                round(
                    f5_replay * 100,
                    2
                ),

            "f4_max":
                round(
                    f4_max,
                    6
                ),

            "f4_max_pct":
                round(
                    f4_max * 100,
                    2
                ),

            "f4_mean":
                round(
                    f4_mean,
                    6
                ),

            "f4_mean_pct":
                round(
                    f4_mean * 100,
                    2
                ),

            "f4_suspicious_ratio":
                round(
                    f4_ratio,
                    6
                ),

            "warning":
                warning[
                    "warning"
                ],

            "warning_level":
                warning[
                    "level"
                ],

            "warning_flags":
                "|".join(
                    warning[
                        "flags"
                    ]
                ),

            "operational_status":
                operational_status
        }


        rows.append(
            row
        )


        print(
            "Prediction:",
            prediction.upper()
        )

        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )

        print(
            "F1B Deepfake:",
            f"{f1b_df * 100:.2f}%"
        )

        print(
            "F5 Replay:",
            f"{f5_replay * 100:.2f}%"
        )

        print(
            "F4 Max:",
            f"{f4_max * 100:.2f}%"
        )

        print(
            "F4 Mean:",
            f"{f4_mean * 100:.2f}%"
        )

        print(
            "Warning:",
            warning["warning"]
        )

        print(
            "Flags:",
            warning["flags"]
        )


    except Exception as e:

        print(
            "ERROR:",
            e
        )

        rows.append(
            {
                "file":
                    audio_path.name,

                "true_label":
                    true_label,

                "prediction":
                    "ERROR",

                "correct":
                    False,

                "error":
                    str(e)
            }
        )


# =========================================================
# SAVE CSV
# =========================================================

fieldnames = [
    "file",
    "true_label",
    "prediction",
    "correct",
    "confidence",
    "confidence_pct",
    "f1b_deepfake_probability",
    "f1b_deepfake_pct",
    "f5_replay_probability",
    "f5_replay_pct",
    "f4_max",
    "f4_max_pct",
    "f4_mean",
    "f4_mean_pct",
    "f4_suspicious_ratio",
    "warning",
    "warning_level",
    "warning_flags",
    "operational_status",
    "error"
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore"
    )

    writer.writeheader()

    for row in rows:

        writer.writerow(
            row
        )


# =========================================================
# BASIC ACCURACY
# =========================================================

valid_rows = [
    row
    for row in rows
    if row.get(
        "prediction"
    ) != "ERROR"
]


correct_count = sum(
    1
    for row in valid_rows
    if row.get(
        "correct"
    )
)


accuracy = (
    correct_count /
    len(valid_rows)
    if valid_rows
    else 0
)


# =========================================================
# WARNING STATISTICS
# =========================================================

genuine_rows = [
    row
    for row in valid_rows
    if row[
        "true_label"
    ] == "genuine"
]


genuine_warnings = sum(
    1
    for row in genuine_rows
    if row.get(
        "warning"
    )
)


false_warning_rate = (
    genuine_warnings /
    len(genuine_rows)
    if genuine_rows
    else 0
)


non_genuine_rows = [
    row
    for row in valid_rows
    if row[
        "true_label"
    ] != "genuine"
]


non_genuine_warnings = sum(
    1
    for row in non_genuine_rows
    if row.get(
        "warning"
    )
)


warning_detection_rate = (
    non_genuine_warnings /
    len(non_genuine_rows)
    if non_genuine_rows
    else 0
)


# =========================================================
# PER-CLASS ACCURACY
# =========================================================

print()
print()
print("=" * 70)
print("SONICT WARNING VALIDATION SUMMARY")
print("=" * 70)

print(
    "Valid files:",
    len(valid_rows)
)

print(
    "Correct predictions:",
    correct_count
)

print(
    "Overall accuracy:",
    f"{accuracy * 100:.2f}%"
)


print()
print("PER-CLASS RESULTS")
print("-" * 70)


for label in CLASSES:

    class_rows = [
        row
        for row in valid_rows
        if row[
            "true_label"
        ] == label
    ]

    class_correct = sum(
        1
        for row in class_rows
        if row.get(
            "correct"
        )
    )

    class_accuracy = (
        class_correct /
        len(class_rows)
        if class_rows
        else 0
    )

    print(
        f"{label.upper():10s}",
        f"{class_correct}/{len(class_rows)}",
        f"= {class_accuracy * 100:.2f}%"
    )


# =========================================================
# CONFUSION MATRIX
# =========================================================

print()
print("CONFUSION MATRIX")
print(
    "Rows = True label, "
    "Columns = Prediction"
)

print()

print(
    " " * 12,
    "GEN",
    "DF",
    "REP",
    "TAMP"
)


for true_label in CLASSES:

    values = [
        confusion[
            true_label
        ][pred]
        for pred in CLASSES
    ]

    print(
        f"{true_label:12s}",
        *[
            f"{value:4d}"
            for value in values
        ]
    )


# =========================================================
# WARNING RESULTS
# =========================================================

print()
print("WARNING-LAYER RESULTS")
print("-" * 70)

print(
    "Genuine warning count:",
    f"{genuine_warnings}/"
    f"{len(genuine_rows)}"
)

print(
    "Genuine false-warning rate:",
    f"{false_warning_rate * 100:.2f}%"
)

print(
    "Non-genuine warnings:",
    f"{non_genuine_warnings}/"
    f"{len(non_genuine_rows)}"
)

print(
    "Non-genuine warning rate:",
    f"{warning_detection_rate * 100:.2f}%"
)


# =========================================================
# FLAG COUNTS
# =========================================================

flag_counts = {}

for row in valid_rows:

    flags = row.get(
        "warning_flags",
        ""
    )

    if not flags:
        continue

    for flag in flags.split("|"):

        flag_counts[flag] = (
            flag_counts.get(
                flag,
                0
            )
            +
            1
        )


print()
print("EVIDENCE FLAG COUNTS")
print("-" * 70)

if flag_counts:

    for flag, count in sorted(
        flag_counts.items()
    ):

        print(
            flag,
            "=",
            count
        )

else:

    print(
        "No evidence flags generated."
    )


print()
print(
    "CSV saved to:"
)

print(
    OUTPUT_CSV
)

print("=" * 70)