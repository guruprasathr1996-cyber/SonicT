from pathlib import Path
import csv
import pandas as pd

from inference import predict_feature4_windows


# =========================================================
# PATHS
# =========================================================

TEST_ROOT = Path(r"H:\SonicT_Warning_Test")

TAMPERED_DIR = TEST_ROOT / "tampered"

GROUND_TRUTH = (
    TEST_ROOT / "tampered_ground_truth.csv"
)

OUTPUT_CSV = (
    TEST_ROOT / "f4_localization_validation.csv"
)


# =========================================================
# SETTINGS
# =========================================================

# Same suspicious threshold currently used by SonicT
THRESHOLD = 0.50


# =========================================================
# INTERVAL OVERLAP
# =========================================================

def interval_overlap(
    pred_start,
    pred_end,
    true_start,
    true_end
):
    """
    Return overlap duration in seconds.
    """

    start = max(
        pred_start,
        true_start
    )

    end = min(
        pred_end,
        true_end
    )

    return max(
        0.0,
        end - start
    )


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

if not GROUND_TRUTH.exists():

    raise FileNotFoundError(
        f"Ground truth not found:\n{GROUND_TRUTH}"
    )


df = pd.read_csv(
    GROUND_TRUTH
)


print("=" * 75)
print("SONICT F4 TAMPERING LOCALIZATION VALIDATION")
print("=" * 75)

print(
    "Ground truth:",
    GROUND_TRUTH
)

print(
    "Tampered folder:",
    TAMPERED_DIR
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Files:",
    len(df)
)

print()


# =========================================================
# VALIDATION
# =========================================================

results = []


for index, row in df.iterrows():

    # -----------------------------------------------------
    # Read filename
    # -----------------------------------------------------

    filename = str(
        row["tampered_file"]
    )


    audio_path = (
        TAMPERED_DIR / filename
    )


    if not audio_path.exists():

        print(
            f"[{index + 1}/{len(df)}] "
            f"MISSING: {filename}"
        )

        continue


    # -----------------------------------------------------
    # Ground-truth interval
    # -----------------------------------------------------

    true_start = float(
        row["tamper_start_sec"]
    )

    true_end = float(
        row["tamper_end_sec"]
    )

    true_duration = (
        true_end - true_start
    )


    print("=" * 75)

    print(
        f"[{index + 1}/{len(df)}] "
        f"{filename}"
    )

    print(
        f"Actual splice: "
        f"{true_start:.3f}s "
        f"-> {true_end:.3f}s"
    )


    # -----------------------------------------------------
    # Run F4
    # -----------------------------------------------------

    windows = predict_feature4_windows(
        str(audio_path)
    )


    suspicious = [
        w
        for w in windows
        if w["probability"] >= THRESHOLD
    ]


    # -----------------------------------------------------
    # Find suspicious windows overlapping true splice
    # -----------------------------------------------------

    overlapping_windows = []

    total_overlap = 0.0

    best_overlap = 0.0

    best_window = None


    for w in suspicious:

        overlap = interval_overlap(
            w["start"],
            w["end"],
            true_start,
            true_end
        )


        if overlap > 0:

            overlapping_windows.append(
                {
                    **w,
                    "overlap":
                        overlap
                }
            )


            if overlap > best_overlap:

                best_overlap = overlap

                best_window = w


    # -----------------------------------------------------
    # Merge overlapping detected intervals
    # -----------------------------------------------------

    intervals = []

    for w in overlapping_windows:

        intervals.append(
            (
                max(
                    w["start"],
                    true_start
                ),
                min(
                    w["end"],
                    true_end
                )
            )
        )


    intervals.sort()


    merged = []

    for start, end in intervals:

        if not merged:

            merged.append(
                [start, end]
            )

        else:

            last = merged[-1]

            if start <= last[1]:

                last[1] = max(
                    last[1],
                    end
                )

            else:

                merged.append(
                    [start, end]
                )


    total_overlap = sum(
        end - start
        for start, end
        in merged
    )


    # -----------------------------------------------------
    # Localization coverage
    # -----------------------------------------------------

    if true_duration > 0:

        coverage = (
            total_overlap
            /
            true_duration
        )

    else:

        coverage = 0.0


    coverage = min(
        coverage,
        1.0
    )


    # -----------------------------------------------------
    # Did F4 localize the true splice at all?
    # -----------------------------------------------------

    localized = (
        len(overlapping_windows) > 0
    )


    # -----------------------------------------------------
    # Maximum F4 probability inside overlapping windows
    # -----------------------------------------------------

    if overlapping_windows:

        max_overlap_probability = max(
            w["probability"]
            for w
            in overlapping_windows
        )

    else:

        max_overlap_probability = 0.0


    # -----------------------------------------------------
    # Maximum probability anywhere in file
    # -----------------------------------------------------

    if windows:

        global_max_window = max(
            windows,
            key=lambda x:
                x["probability"]
        )

        global_max_probability = (
            global_max_window[
                "probability"
            ]
        )

        global_max_start = (
            global_max_window[
                "start"
            ]
        )

        global_max_end = (
            global_max_window[
                "end"
            ]
        )

    else:

        global_max_probability = 0.0
        global_max_start = 0.0
        global_max_end = 0.0


    # -----------------------------------------------------
    # False suspicious windows outside true splice
    # -----------------------------------------------------

    outside_suspicious = []

    for w in suspicious:

        overlap = interval_overlap(
            w["start"],
            w["end"],
            true_start,
            true_end
        )

        if overlap <= 0:

            outside_suspicious.append(
                w
            )


    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print(
        "Total F4 windows:",
        len(windows)
    )

    print(
        "Suspicious windows:",
        len(suspicious)
    )

    print(
        "Suspicious windows overlapping splice:",
        len(overlapping_windows)
    )

    print(
        "Suspicious windows outside splice:",
        len(outside_suspicious)
    )

    print(
        "Localization:",
        "YES"
        if localized
        else "NO"
    )

    print(
        "Splice coverage:",
        f"{coverage * 100:.2f}%"
    )

    print(
        "Maximum probability in splice:",
        f"{max_overlap_probability * 100:.2f}%"
    )

    print(
        "Global maximum probability:",
        f"{global_max_probability * 100:.2f}%"
    )


    if best_window is not None:

        print(
            "Best overlapping window:",
            f"{best_window['start']:.2f}s "
            f"-> "
            f"{best_window['end']:.2f}s"
        )


    print()


    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    results.append(
        {
            "filename":
                filename,

            "true_start":
                true_start,

            "true_end":
                true_end,

            "true_duration":
                true_duration,

            "total_windows":
                len(windows),

            "suspicious_windows":
                len(suspicious),

            "overlapping_suspicious_windows":
                len(overlapping_windows),

            "outside_suspicious_windows":
                len(outside_suspicious),

            "localized":
                localized,

            "splice_coverage":
                coverage,

            "splice_coverage_pct":
                coverage * 100,

            "max_probability_in_splice":
                max_overlap_probability,

            "max_probability_in_splice_pct":
                max_overlap_probability * 100,

            "global_max_probability":
                global_max_probability,

            "global_max_probability_pct":
                global_max_probability * 100,

            "global_max_start":
                global_max_start,

            "global_max_end":
                global_max_end
        }
    )


# =========================================================
# SAVE CSV
# =========================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 75)
print("F4 LOCALIZATION SUMMARY")
print("=" * 75)


if len(result_df) == 0:

    print(
        "No files were evaluated."
    )

else:

    localized_count = int(
        result_df[
            "localized"
        ].sum()
    )


    localization_rate = (
        localized_count
        /
        len(result_df)
    )


    mean_coverage = float(
        result_df[
            "splice_coverage_pct"
        ].mean()
    )


    median_coverage = float(
        result_df[
            "splice_coverage_pct"
        ].median()
    )


    mean_max_probability = float(
        result_df[
            "max_probability_in_splice_pct"
        ].mean()
    )


    total_suspicious = int(
        result_df[
            "suspicious_windows"
        ].sum()
    )


    total_overlapping = int(
        result_df[
            "overlapping_suspicious_windows"
        ].sum()
    )


    total_outside = int(
        result_df[
            "outside_suspicious_windows"
        ].sum()
    )


    print(
        "Files evaluated:",
        len(result_df)
    )

    print(
        "Files localized:",
        f"{localized_count}/"
        f"{len(result_df)}"
    )

    print(
        "Localization hit rate:",
        f"{localization_rate * 100:.2f}%"
    )

    print(
        "Mean splice coverage:",
        f"{mean_coverage:.2f}%"
    )

    print(
        "Median splice coverage:",
        f"{median_coverage:.2f}%"
    )

    print(
        "Mean maximum F4 probability "
        "inside splice:",
        f"{mean_max_probability:.2f}%"
    )

    print()

    print(
        "Total suspicious windows:",
        total_suspicious
    )

    print(
        "Overlapping suspicious windows:",
        total_overlapping
    )

    print(
        "Suspicious windows outside splice:",
        total_outside
    )


    if total_suspicious > 0:

        precision_like = (
            total_overlapping
            /
            total_suspicious
        )

        print(
            "Window overlap proportion:",
            f"{precision_like * 100:.2f}%"
        )


print()
print(
    "Results saved:"
)

print(
    OUTPUT_CSV
)

print("=" * 75)