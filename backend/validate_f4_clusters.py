from pathlib import Path
import pandas as pd

from inference import predict_feature4_windows


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(r"H:\SonicT_Warning_Test")

GENUINE_DIR = ROOT / "genuine"
TAMPERED_DIR = ROOT / "tampered"

OUTPUT_CSV = ROOT / "f4_cluster_comparison.csv"

THRESHOLD = 0.50

# F4 uses 2-second windows with 0.5-second hop
HOP_SEC = 0.5


# =========================================================
# FIND SUSPICIOUS CLUSTERS
# =========================================================

def find_clusters(windows):

    suspicious = [
        w for w in windows
        if w["probability"] >= THRESHOLD
    ]

    if not suspicious:
        return []

    clusters = []
    current = [suspicious[0]]

    for window in suspicious[1:]:

        previous = current[-1]

        # Consecutive F4 windows should start
        # approximately 0.5 seconds apart.
        gap = (
            window["start"]
            - previous["start"]
        )

        if gap <= HOP_SEC + 1e-6:

            current.append(window)

        else:

            clusters.append(current)
            current = [window]

    clusters.append(current)

    return clusters


# =========================================================
# ANALYZE ONE FILE
# =========================================================

def analyze_file(path, true_label):

    windows = predict_feature4_windows(
        str(path)
    )

    suspicious = [
        w for w in windows
        if w["probability"] >= THRESHOLD
    ]

    high_windows = [
        w for w in windows
        if w["probability"] >= 0.70
    ]

    very_high_windows = [
        w for w in windows
        if w["probability"] >= 0.90
    ]

    clusters = find_clusters(windows)

    if clusters:

        largest_cluster = max(
            clusters,
            key=len
        )

        largest_count = len(
            largest_cluster
        )

        largest_start = (
            largest_cluster[0]["start"]
        )

        largest_end = (
            largest_cluster[-1]["end"]
        )

        largest_duration = (
            largest_end
            - largest_start
        )

        largest_max_probability = max(
            w["probability"]
            for w in largest_cluster
        )

        largest_mean_probability = sum(
            w["probability"]
            for w in largest_cluster
        ) / largest_count

    else:

        largest_count = 0
        largest_start = None
        largest_end = None
        largest_duration = 0.0
        largest_max_probability = 0.0
        largest_mean_probability = 0.0


    if windows:

        global_max = max(
            w["probability"]
            for w in windows
        )

    else:

        global_max = 0.0


    return {

        "filename":
            path.name,

        "true_label":
            true_label,

        "total_windows":
            len(windows),

        "suspicious_windows_50":
            len(suspicious),

        "high_windows_70":
            len(high_windows),

        "very_high_windows_90":
            len(very_high_windows),

        "cluster_count":
            len(clusters),

        "largest_cluster_windows":
            largest_count,

        "largest_cluster_start":
            largest_start,

        "largest_cluster_end":
            largest_end,

        "largest_cluster_duration_sec":
            largest_duration,

        "largest_cluster_max_probability":
            largest_max_probability,

        "largest_cluster_mean_probability":
            largest_mean_probability,

        "global_max_probability":
            global_max
    }


# =========================================================
# RUN TEST
# =========================================================

rows = []

folders = [
    ("genuine", GENUINE_DIR),
    ("tampered", TAMPERED_DIR)
]


for label, folder in folders:

    files = sorted(
        folder.glob("*.wav")
    )

    print()
    print("=" * 70)
    print(
        label.upper(),
        "-",
        len(files),
        "files"
    )
    print("=" * 70)


    for i, path in enumerate(
        files,
        start=1
    ):

        print(
            f"[{i}/{len(files)}] "
            f"{path.name}"
        )

        try:

            result = analyze_file(
                path,
                label
            )

            rows.append(result)

            print(
                "  Suspicious windows:",
                result[
                    "suspicious_windows_50"
                ]
            )

            print(
                "  Clusters:",
                result[
                    "cluster_count"
                ]
            )

            print(
                "  Largest cluster:",
                result[
                    "largest_cluster_windows"
                ],
                "windows /",
                f"{result['largest_cluster_duration_sec']:.2f}s"
            )

            print(
                "  Cluster max:",
                f"{result['largest_cluster_max_probability'] * 100:.2f}%"
            )

        except Exception as e:

            print(
                "  ERROR:",
                e
            )


# =========================================================
# SAVE
# =========================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 70)
print("F4 CLUSTER COMPARISON SUMMARY")
print("=" * 70)


for label in [
    "genuine",
    "tampered"
]:

    subset = df[
        df["true_label"] == label
    ]

    if len(subset) == 0:
        continue

    print()
    print(label.upper())

    print(
        "Files:",
        len(subset)
    )

    print(
        "Mean suspicious windows:",
        f"{subset['suspicious_windows_50'].mean():.2f}"
    )

    print(
        "Median suspicious windows:",
        f"{subset['suspicious_windows_50'].median():.2f}"
    )

    print(
        "Mean largest cluster:",
        f"{subset['largest_cluster_windows'].mean():.2f}",
        "windows"
    )

    print(
        "Median largest cluster:",
        f"{subset['largest_cluster_windows'].median():.2f}",
        "windows"
    )

    print(
        "Mean largest cluster duration:",
        f"{subset['largest_cluster_duration_sec'].mean():.2f}s"
    )

    print(
        "Median largest cluster duration:",
        f"{subset['largest_cluster_duration_sec'].median():.2f}s"
    )

    print(
        "Mean cluster max probability:",
        f"{subset['largest_cluster_max_probability'].mean() * 100:.2f}%"
    )

    print(
        "Files with >= 2 consecutive suspicious windows:",
        int(
            (
                subset[
                    "largest_cluster_windows"
                ] >= 2
            ).sum()
        ),
        "/",
        len(subset)
    )

    print(
        "Files with >= 3 consecutive suspicious windows:",
        int(
            (
                subset[
                    "largest_cluster_windows"
                ] >= 3
            ).sum()
        ),
        "/",
        len(subset)
    )

    print(
        "Files with >= 4 consecutive suspicious windows:",
        int(
            (
                subset[
                    "largest_cluster_windows"
                ] >= 4
            ).sum()
        ),
        "/",
        len(subset)
    )


print()
print(
    "CSV saved to:"
)

print(
    OUTPUT_CSV
)

print("=" * 70)