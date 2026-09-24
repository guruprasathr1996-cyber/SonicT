from pathlib import Path
import pandas as pd

# =========================================================
# PATH
# =========================================================

CSV_PATH = Path(
    r"H:\SonicT_Warning_Test\f4_cluster_comparison.csv"
)

OUTPUT_PATH = Path(
    r"H:\SonicT_Warning_Test\f4_warning_rule_evaluation.csv"
)


# =========================================================
# LOAD RESULTS
# =========================================================

df = pd.read_csv(CSV_PATH)

required_columns = [
    "true_label",
    "largest_cluster_windows",
    "largest_cluster_max_probability"
]

for col in required_columns:
    if col not in df.columns:
        raise RuntimeError(
            f"Missing required column: {col}"
        )


# Only genuine and tampered
df = df[
    df["true_label"].isin(
        ["genuine", "tampered"]
    )
].copy()


print("=" * 78)
print("SONICT F4 CANDIDATE WARNING-RULE EVALUATION")
print("=" * 78)

print("Files:", len(df))
print(
    "Genuine:",
    len(df[df["true_label"] == "genuine"])
)
print(
    "Tampered:",
    len(df[df["true_label"] == "tampered"])
)
print()


# =========================================================
# CANDIDATE RULES
# =========================================================

cluster_sizes = [
    1,
    2,
    3
]

probability_thresholds = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

results = []


# =========================================================
# EVALUATE
# =========================================================

for minimum_windows in cluster_sizes:

    for probability_threshold in probability_thresholds:

        triggered = (
            (
                df["largest_cluster_windows"]
                >= minimum_windows
            )
            &
            (
                df[
                    "largest_cluster_max_probability"
                ]
                >= probability_threshold
            )
        )

        true_tampered = (
            df["true_label"] == "tampered"
        )

        true_genuine = (
            df["true_label"] == "genuine"
        )


        # ---------------------------------------------
        # Confusion values for this warning rule
        # ---------------------------------------------

        tp = int(
            (
                triggered
                &
                true_tampered
            ).sum()
        )

        fn = int(
            (
                (~triggered)
                &
                true_tampered
            ).sum()
        )

        fp = int(
            (
                triggered
                &
                true_genuine
            ).sum()
        )

        tn = int(
            (
                (~triggered)
                &
                true_genuine
            ).sum()
        )


        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        false_warning_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        balanced_accuracy = (
            sensitivity + specificity
        ) / 2


        results.append({
            "minimum_consecutive_windows":
                minimum_windows,

            "cluster_max_threshold":
                probability_threshold,

            "tampered_triggered":
                tp,

            "tampered_missed":
                fn,

            "genuine_false_warnings":
                fp,

            "genuine_correct_no_warning":
                tn,

            "sensitivity":
                sensitivity,

            "specificity":
                specificity,

            "false_warning_rate":
                false_warning_rate,

            "precision":
                precision,

            "balanced_accuracy":
                balanced_accuracy
        })


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by=[
        "balanced_accuracy",
        "false_warning_rate",
        "sensitivity"
    ],
    ascending=[
        False,
        True,
        False
    ]
)


# =========================================================
# DISPLAY
# =========================================================

print(
    f"{'WIN':>4} "
    f"{'PROB':>7} "
    f"{'TAMP':>7} "
    f"{'GEN-FP':>8} "
    f"{'SENS':>8} "
    f"{'SPEC':>8} "
    f"{'FWR':>8} "
    f"{'BAL-ACC':>9}"
)

print("-" * 78)


for _, row in results_df.iterrows():

    print(
        f"{int(row['minimum_consecutive_windows']):>4} "
        f"{row['cluster_max_threshold'] * 100:>6.0f}% "
        f"{int(row['tampered_triggered']):>3}/10 "
        f"{int(row['genuine_false_warnings']):>4}/10 "
        f"{row['sensitivity'] * 100:>7.1f}% "
        f"{row['specificity'] * 100:>7.1f}% "
        f"{row['false_warning_rate'] * 100:>7.1f}% "
        f"{row['balanced_accuracy'] * 100:>8.1f}%"
    )


# =========================================================
# SAVE
# =========================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# SHOW HIGH-SENSITIVITY CANDIDATES
# =========================================================

print()
print("=" * 78)
print("HIGH-SENSITIVITY CANDIDATES")
print("=" * 78)

high_sensitivity = results_df[
    results_df["sensitivity"] >= 0.80
]


if len(high_sensitivity) == 0:

    print(
        "No candidate reached >= 80% "
        "tampered sensitivity."
    )

else:

    for _, row in high_sensitivity.iterrows():

        print(
            f">= {int(row['minimum_consecutive_windows'])} "
            f"window(s) AND "
            f"cluster max >= "
            f"{row['cluster_max_threshold'] * 100:.0f}%"
        )

        print(
            f"   Tampered triggered: "
            f"{int(row['tampered_triggered'])}/10"
        )

        print(
            f"   Genuine false warnings: "
            f"{int(row['genuine_false_warnings'])}/10"
        )

        print(
            f"   Sensitivity: "
            f"{row['sensitivity'] * 100:.1f}%"
        )

        print(
            f"   Specificity: "
            f"{row['specificity'] * 100:.1f}%"
        )

        print()


print("=" * 78)
print("Saved:")
print(OUTPUT_PATH)
print("=" * 78)