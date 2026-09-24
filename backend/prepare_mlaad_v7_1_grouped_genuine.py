from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# SONICT F1B V7.1 - GROUP-DISJOINT GENUINE SPLIT FIX
#
# Keeps the existing generator-disjoint fake split unchanged.
# Rebuilds ONLY the genuine split so the same source/book group
# cannot appear in train, validation, and test.
#
# No audio files are copied or modified.
# ============================================================

SEED = 42

FAKE_MANIFEST = Path(
    r"H:\SonicT_Compression_Test\v7_mlaad_split\fake_manifest_v7.csv"
)

GENUINE_ROOT = Path(
    r"H:\MLAAD-tiny\original\en"
)

OUTPUT_DIR = Path(
    r"H:\SonicT_Compression_Test\v7_mlaad_split"
)

OUTPUT_GENUINE = OUTPUT_DIR / "genuine_manifest_v7_1_grouped.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "split_summary_v7_1_grouped.csv"
OUTPUT_GROUPS = OUTPUT_DIR / "genuine_group_stats_v7_1.csv"

AUDIO_EXTS = {
    ".wav", ".flac", ".mp3", ".m4a",
    ".ogg", ".opus", ".aac", ".wma",
    ".mpeg", ".mpg"
}

if not FAKE_MANIFEST.exists():
    raise FileNotFoundError(
        f"Fake manifest not found: {FAKE_MANIFEST}"
    )

if not GENUINE_ROOT.exists():
    raise FileNotFoundError(
        f"Genuine root not found: {GENUINE_ROOT}"
    )

fake_df = pd.read_csv(FAKE_MANIFEST)

required = {"split", "generator", "label"}
missing = required - set(fake_df.columns)
if missing:
    raise RuntimeError(
        f"Fake manifest missing columns: {sorted(missing)}"
    )

# Exact genuine counts needed to match the already-created fake split.
targets = {
    split_name: int((fake_df["split"] == split_name).sum())
    for split_name in ["train", "validation", "test"]
}

print("=" * 78)
print("SONICT F1B V7.1 - GROUP-DISJOINT GENUINE SPLIT")
print("=" * 78)
print("Target genuine counts:")
for k, v in targets.items():
    print(f"  {k:10s}: {v}")


def genuine_group_from_filename(path: Path) -> str:
    """
    MLAAD-tiny original examples look like:
        wives_and_daughters_60_f000070.wav

    We remove the final _fNNNNNN portion and use the remaining
    prefix as a conservative source/book group.
    """
    stem = path.stem

    m = re.match(r"^(.*)_f\d+$", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Fallback: remove a trailing numeric token if present.
    m = re.match(r"^(.*?)[_-]\d+$", stem)
    if m:
        return m.group(1).lower()

    return stem.lower()


genuine_files = sorted([
    p for p in GENUINE_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in AUDIO_EXTS
])

if not genuine_files:
    raise RuntimeError("No genuine English audio files found.")

genuine_df = pd.DataFrame([
    {
        "path": str(p),
        "filename": p.name,
        "group": genuine_group_from_filename(p),
        "generator": "genuine_original",
        "label": 0,
        "label_name": "genuine",
    }
    for p in genuine_files
])

group_stats = (
    genuine_df.groupby("group")
    .size()
    .reset_index(name="n_files")
    .sort_values(["n_files", "group"], ascending=[False, True])
    .reset_index(drop=True)
)

group_stats.to_csv(OUTPUT_GROUPS, index=False)

print(f"\nTotal genuine files : {len(genuine_df)}")
print(f"Unique source groups: {genuine_df['group'].nunique()}")

print("\nLargest genuine groups:")
print(group_stats.head(20).to_string(index=False))

# ------------------------------------------------------------------
# Search for a group assignment that gives every split enough files
# while staying close to requested train/validation/test proportions.
# ------------------------------------------------------------------

group_sizes = dict(
    zip(group_stats["group"], group_stats["n_files"])
)

groups = list(group_sizes.keys())
target_total = sum(targets.values())

target_ratios = {
    k: v / target_total
    for k, v in targets.items()
}

best = None

for attempt in range(3000):
    rng = np.random.default_rng(SEED + attempt)
    shuffled = groups.copy()
    rng.shuffle(shuffled)

    assigned = {
        "train": [],
        "validation": [],
        "test": [],
    }

    counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    # Greedy proportional allocation with randomized group order.
    for group in shuffled:
        size = int(group_sizes[group])

        def score(split_name):
            target = targets[split_name]
            current = counts[split_name]

            # Prefer the split with the largest normalized deficit.
            deficit_ratio = (target - current) / max(target, 1)

            # Slight penalty for overshooting.
            projected = current + size
            overshoot = max(0, projected - target) / max(target, 1)

            return deficit_ratio - 0.35 * overshoot

        split_name = max(
            ["train", "validation", "test"],
            key=score
        )

        assigned[split_name].append(group)
        counts[split_name] += size

    deficits = {
        k: max(0, targets[k] - counts[k])
        for k in targets
    }

    total_deficit = sum(deficits.values())

    # Among assignments with no deficit, prefer less overshoot.
    total_overshoot = sum(
        max(0, counts[k] - targets[k])
        for k in targets
    )

    objective = (total_deficit, total_overshoot)

    if best is None or objective < best[0]:
        best = (
            objective,
            assigned,
            counts,
            attempt,
        )

    if total_deficit == 0 and total_overshoot <= 150:
        break

if best is None:
    raise RuntimeError("Could not build grouped genuine split.")

objective, assigned, available_counts, best_attempt = best

print("\nBest grouped allocation:")
print(f"  Search attempt: {best_attempt}")
for split_name in ["train", "validation", "test"]:
    print(
        f"  {split_name:10s}: "
        f"{available_counts[split_name]} files across "
        f"{len(assigned[split_name])} groups "
        f"(target {targets[split_name]})"
    )

if objective[0] > 0:
    raise RuntimeError(
        "Unable to allocate enough group-disjoint genuine files "
        "to match all fake split counts."
    )

# ------------------------------------------------------------------
# Select exactly the target number of genuine files from each
# group-disjoint pool.
# ------------------------------------------------------------------

selected_parts = []

for split_name in ["train", "validation", "test"]:
    split_groups = set(assigned[split_name])

    pool = genuine_df[
        genuine_df["group"].isin(split_groups)
    ].copy()

    pool = pool.sample(
        frac=1.0,
        random_state=SEED + {
            "train": 1,
            "validation": 2,
            "test": 3,
        }[split_name]
    )

    selected = pool.iloc[:targets[split_name]].copy()
    selected["split"] = split_name

    selected_parts.append(selected)

selected_genuine = pd.concat(
    selected_parts,
    ignore_index=True
)

# ------------------------------------------------------------------
# Leakage checks.
# ------------------------------------------------------------------

train_groups = set(
    selected_genuine.loc[
        selected_genuine["split"] == "train",
        "group"
    ]
)

val_groups = set(
    selected_genuine.loc[
        selected_genuine["split"] == "validation",
        "group"
    ]
)

test_groups = set(
    selected_genuine.loc[
        selected_genuine["split"] == "test",
        "group"
    ]
)

if train_groups & val_groups:
    raise RuntimeError("Genuine group leakage: train/validation overlap.")

if train_groups & test_groups:
    raise RuntimeError("Genuine group leakage: train/test overlap.")

if val_groups & test_groups:
    raise RuntimeError("Genuine group leakage: validation/test overlap.")

selected_genuine.to_csv(
    OUTPUT_GENUINE,
    index=False
)

summary_rows = []

for split_name in ["train", "validation", "test"]:
    fake_n = int((fake_df["split"] == split_name).sum())

    genuine_part = selected_genuine[
        selected_genuine["split"] == split_name
    ]

    genuine_n = len(genuine_part)
    genuine_groups_n = genuine_part["group"].nunique()

    fake_generators_n = (
        fake_df.loc[
            fake_df["split"] == split_name,
            "generator"
        ].nunique()
    )

    summary_rows.append({
        "split": split_name,
        "fake_files": fake_n,
        "genuine_files": genuine_n,
        "fake_generators": int(fake_generators_n),
        "genuine_source_groups": int(genuine_groups_n),
        "total_files": fake_n + genuine_n,
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

print("\n" + "=" * 78)
print("V7.1 GROUP-DISJOINT SUMMARY")
print("=" * 78)
print(summary_df.to_string(index=False))

print("\nLeakage checks:")
print("  Fake generators: already generator-disjoint from V7.")
print("  Genuine train/validation overlap:", len(train_groups & val_groups))
print("  Genuine train/test overlap      :", len(train_groups & test_groups))
print("  Genuine validation/test overlap :", len(val_groups & test_groups))

print("\nSaved:")
print(OUTPUT_GENUINE)
print(OUTPUT_SUMMARY)
print(OUTPUT_GROUPS)

print("\nIMPORTANT:")
print("No audio files were copied or modified.")
print("Use genuine_manifest_v7_1_grouped.csv for V7 training.")
print("Do NOT train yet until this grouped split is reviewed.")
