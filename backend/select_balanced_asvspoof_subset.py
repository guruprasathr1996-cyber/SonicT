from pathlib import Path
import random
import shutil
import csv
from collections import defaultdict

AUDIO_DIR = Path(
    r"H:\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\flac"
)

METADATA = Path(
    r"H:\ASVspoof2021_DF_keys\keys\DF\CM\trial_metadata.txt"
)

OUT_ROOT = Path(
    r"H:\SonicT_Compression_Test\asvspoof_balanced"
)

BONAFIDE_DIR = OUT_ROOT / "bonafide"
SPOOF_DIR = OUT_ROOT / "spoof"

MANIFEST = OUT_ROOT / "manifest.csv"

N_PER_CLASS = 100
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

BONAFIDE_DIR.mkdir(parents=True, exist_ok=True)
SPOOF_DIR.mkdir(parents=True, exist_ok=True)

available = {p.stem for p in AUDIO_DIR.glob("*.flac")}

rows = []

with open(METADATA, "r", encoding="utf-8") as f:
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

        if file_id not in available:
            continue

        if label not in {"bonafide", "spoof"}:
            continue

        rows.append({
            "speaker": speaker,
            "file_id": file_id,
            "codec": codec,
            "source": source,
            "attack": attack,
            "label": label
        })


bonafide_rows = [r for r in rows if r["label"] == "bonafide"]
spoof_rows = [r for r in rows if r["label"] == "spoof"]


def diverse_sample(data, n):
    """
    Try to keep codec/source/attack diversity.
    """

    groups = defaultdict(list)

    for row in data:
        key = (
            row["codec"],
            row["source"],
            row["attack"]
        )
        groups[key].append(row)

    for group in groups.values():
        random.shuffle(group)

    selected = []
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    while len(selected) < n:

        made_progress = False

        for key in group_keys:

            if groups[key]:
                selected.append(groups[key].pop())
                made_progress = True

                if len(selected) >= n:
                    break

        if not made_progress:
            break

    return selected


selected_bonafide = diverse_sample(
    bonafide_rows,
    N_PER_CLASS
)

selected_spoof = diverse_sample(
    spoof_rows,
    N_PER_CLASS
)


print("=" * 70)
print("ASVSPOOF 2021 DF BALANCED SUBSET")
print("=" * 70)

print(f"\nBonafide selected : {len(selected_bonafide)}")
print(f"Spoof selected    : {len(selected_spoof)}")


all_selected = []

for row in selected_bonafide:

    src = AUDIO_DIR / f'{row["file_id"]}.flac'
    dst = BONAFIDE_DIR / src.name

    shutil.copy2(src, dst)

    all_selected.append(row)


for row in selected_spoof:

    src = AUDIO_DIR / f'{row["file_id"]}.flac'
    dst = SPOOF_DIR / src.name

    shutil.copy2(src, dst)

    all_selected.append(row)


with open(
    MANIFEST,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "speaker",
            "file_id",
            "codec",
            "source",
            "attack",
            "label"
        ]
    )

    writer.writeheader()
    writer.writerows(all_selected)


def show_stats(name, selected):

    codecs = defaultdict(int)
    sources = defaultdict(int)
    attacks = defaultdict(int)
    speakers = set()

    for row in selected:

        codecs[row["codec"]] += 1
        sources[row["source"]] += 1
        attacks[row["attack"]] += 1
        speakers.add(row["speaker"])

    print(f"\n{name}")

    print(f"Unique speakers: {len(speakers)}")

    print("\nCodecs:")
    for k, v in sorted(codecs.items()):
        print(f"  {k}: {v}")

    print("\nSources:")
    for k, v in sorted(sources.items()):
        print(f"  {k}: {v}")

    print("\nAttacks:")
    for k, v in sorted(attacks.items()):
        print(f"  {k}: {v}")


show_stats(
    "BONAFIDE",
    selected_bonafide
)

show_stats(
    "SPOOF",
    selected_spoof
)


print("\n" + "=" * 70)

print("Saved folders:")

print(BONAFIDE_DIR)
print(SPOOF_DIR)

print("\nManifest:")
print(MANIFEST)

print("=" * 70)