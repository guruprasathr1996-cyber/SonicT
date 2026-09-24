from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# SONICT F1B V6 - LARGE BALANCED ASVSPOOF SUBSET SELECTOR
#
# Purpose:
# - Stop training on only 100 genuine + 100 spoof originals.
# - Build a larger, more diverse 1000 + 1000 subset.
# - Keep source/codec/attack/speaker diversity.
# - No model training in this step.
# ============================================================

SEED = 42
TARGET_PER_CLASS = 1000
MAX_PER_SPEAKER = 35

AUDIO_DIR = Path(
    r"H:\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\flac"
)

KEY_FILE = Path(
    r"H:\ASVspoof2021_DF_keys\keys\DF\CM\trial_metadata.txt"
)

OUTPUT_DIR = Path(
    r"H:\SonicT_Compression_Test\asvspoof_v6_balanced"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MANIFEST = OUTPUT_DIR / "manifest_v6.csv"

rng = np.random.default_rng(SEED)

if not AUDIO_DIR.exists():
    raise FileNotFoundError(f"Audio folder not found: {AUDIO_DIR}")

if not KEY_FILE.exists():
    raise FileNotFoundError(f"Metadata file not found: {KEY_FILE}")

# Official ASVspoof 2021 DF metadata layout used here:
# 0 speaker
# 1 file_id
# 2 codec
# 3 source
# 4 attack
# 5 label

rows = []

with open(KEY_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) < 6:
            continue

        speaker = parts[0]
        file_id = parts[1]
        codec = parts[2]
        source = parts[3]
        attack = parts[4]
        label = parts[5].lower()

        if label not in {"bonafide", "spoof"}:
            continue

        audio_path = AUDIO_DIR / f"{file_id}.flac"

        if not audio_path.exists():
            continue

        rows.append({
            "speaker": speaker,
            "file_id": file_id,
            "codec": codec,
            "source": source,
            "attack": attack,
            "label": label,
            "source_path": str(audio_path),
        })

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No locally available ASVspoof metadata/audio matches found.")

print("=" * 78)
print("SONICT F1B V6 - LARGE BALANCED SUBSET SELECTOR")
print("=" * 78)
print(f"Locally available rows : {len(df)}")
print(f"Bonafide available     : {(df['label'] == 'bonafide').sum()}")
print(f"Spoof available        : {(df['label'] == 'spoof').sum()}")
print(f"Unique speakers        : {df['speaker'].nunique()}")


def round_robin_diverse_select(class_df, target, class_name):
    """
    Selects examples in repeated passes across diversity strata while
    limiting dominance by any one speaker.

    bonafide strata: source + codec
    spoof strata   : source + codec + attack
    """

    class_df = class_df.copy().reset_index(drop=True)

    if class_name == "bonafide":
        strata_cols = ["source", "codec"]
    else:
        strata_cols = ["source", "codec", "attack"]

    # Shuffle once deterministically.
    class_df = class_df.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    grouped = {}

    for key, group in class_df.groupby(strata_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        grouped[key] = group.sample(
            frac=1.0,
            random_state=SEED,
        ).to_dict("records")

    selected = []
    speaker_counts = {}
    positions = {key: 0 for key in grouped}

    keys = list(grouped.keys())
    rng.shuffle(keys)

    # Round robin through strata until target reached or exhausted.
    progress = True

    while len(selected) < target and progress:
        progress = False

        for key in keys:
            records = grouped[key]
            pos = positions[key]

            while pos < len(records):
                candidate = records[pos]
                pos += 1
                positions[key] = pos

                spk = candidate["speaker"]

                if speaker_counts.get(spk, 0) >= MAX_PER_SPEAKER:
                    continue

                selected.append(candidate)
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
                progress = True
                break

            if len(selected) >= target:
                break

    # If speaker cap prevented reaching target, fill remaining slots
    # from still-unused rows, but keep deterministic ordering.
    if len(selected) < target:
        used_ids = {r["file_id"] for r in selected}

        remaining = class_df[
            ~class_df["file_id"].isin(used_ids)
        ].sample(
            frac=1.0,
            random_state=SEED + 1,
        )

        for _, row in remaining.iterrows():
            selected.append(row.to_dict())

            if len(selected) >= target:
                break

    return pd.DataFrame(selected[:target])


bonafide_df = df[df["label"] == "bonafide"].copy()
spoof_df = df[df["label"] == "spoof"].copy()

if len(bonafide_df) < TARGET_PER_CLASS:
    raise RuntimeError(
        f"Only {len(bonafide_df)} bonafide files available; "
        f"cannot select {TARGET_PER_CLASS}."
    )

if len(spoof_df) < TARGET_PER_CLASS:
    raise RuntimeError(
        f"Only {len(spoof_df)} spoof files available; "
        f"cannot select {TARGET_PER_CLASS}."
    )

selected_bona = round_robin_diverse_select(
    bonafide_df,
    TARGET_PER_CLASS,
    "bonafide",
)

selected_spoof = round_robin_diverse_select(
    spoof_df,
    TARGET_PER_CLASS,
    "spoof",
)

selected = pd.concat(
    [selected_bona, selected_spoof],
    ignore_index=True,
)

selected = selected.sample(
    frac=1.0,
    random_state=SEED,
).reset_index(drop=True)

selected.to_csv(
    OUTPUT_MANIFEST,
    index=False,
)

print("\n" + "=" * 78)
print("SELECTED SUBSET")
print("=" * 78)

print("\nClass counts:")
print(selected["label"].value_counts())

for label in ["bonafide", "spoof"]:
    part = selected[selected["label"] == label]

    print("\n" + "-" * 70)
    print(label.upper())
    print("-" * 70)

    print(f"Files           : {len(part)}")
    print(f"Unique speakers : {part['speaker'].nunique()}")
    print(f"Unique codecs   : {part['codec'].nunique()}")
    print(f"Unique sources  : {part['source'].nunique()}")
    print(f"Unique attacks  : {part['attack'].nunique()}")

    print("\nCodec distribution:")
    print(part["codec"].value_counts().sort_index())

    print("\nSource distribution:")
    print(part["source"].value_counts().sort_index())

    if label == "spoof":
        print("\nTop attack distribution:")
        print(part["attack"].value_counts().head(25))

    print("\nTop speaker counts:")
    print(part["speaker"].value_counts().head(10))

print("\n" + "=" * 78)
print("SPEAKER OVERLAP CHECK")
print("=" * 78)

bona_speakers = set(selected_bona["speaker"])
spoof_speakers = set(selected_spoof["speaker"])
overlap = sorted(bona_speakers & spoof_speakers)

print(f"Bonafide speakers : {len(bona_speakers)}")
print(f"Spoof speakers    : {len(spoof_speakers)}")
print(f"Speaker overlap   : {len(overlap)}")

if overlap:
    print("Example overlapping speakers:")
    for x in overlap[:20]:
        print(" ", x)

print("\n" + "=" * 78)
print("V6 SUBSET PREPARATION COMPLETE")
print("=" * 78)
print(f"Manifest saved to: {OUTPUT_MANIFEST}")
print("\nIMPORTANT:")
print("Do not train V6 yet until these subset statistics are reviewed.")
