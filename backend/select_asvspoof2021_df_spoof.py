"""
Select confirmed ASVspoof 2021 DF spoof files and copy them into
H:\SonicT_Compression_Test\deepfake

This script reads:
H:\ASVspoof2021_DF_keys\keys\DF\CM\trial_metadata.txt

Metadata columns used:
0 = speaker_id
1 = file_id
2 = codec_condition
3 = source_dataset
4 = attack_id
5 = label  -> spoof / bonafide

It tries to choose a diverse set across codec/source/attack conditions.
"""

from pathlib import Path
import random
import shutil
from collections import defaultdict, Counter

METADATA = Path(r"H:\ASVspoof2021_DF_keys\keys\DF\CM\trial_metadata.txt")
AUDIO_DIR = Path(r"H:\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\flac")
DEST_DIR = Path(r"H:\SonicT_Compression_Test\deepfake")

N_TO_COPY = 40
RANDOM_SEED = 42

AUDIO_EXTENSIONS = [".flac", ".wav", ".mp3", ".mpeg", ".m4a", ".ogg", ".opus"]


def parse_metadata():
    rows = []

    with METADATA.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) < 6:
                continue

            speaker_id = parts[0]
            file_id = parts[1]
            codec = parts[2]
            source = parts[3]
            attack = parts[4]
            label = parts[5].lower()

            rows.append({
                "speaker_id": speaker_id,
                "file_id": file_id,
                "codec": codec,
                "source": source,
                "attack": attack,
                "label": label,
            })

    return rows


def find_audio(file_id):
    for ext in AUDIO_EXTENSIONS:
        p = AUDIO_DIR / f"{file_id}{ext}"
        if p.exists():
            return p

    # fallback in case Windows hides extension or different case
    matches = list(AUDIO_DIR.glob(f"{file_id}.*"))
    return matches[0] if matches else None


def select_diverse(spoof_rows, n):
    random.seed(RANDOM_SEED)

    # Group by a diversity key so we do not take only one attack/codec type.
    groups = defaultdict(list)

    for row in spoof_rows:
        key = (row["source"], row["codec"], row["attack"])
        groups[key].append(row)

    for values in groups.values():
        random.shuffle(values)

    selected = []
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    # Round-robin over groups.
    while len(selected) < n:
        added = False

        for key in list(group_keys):
            if groups[key]:
                selected.append(groups[key].pop())
                added = True

                if len(selected) >= n:
                    break

        if not added:
            break

    return selected


def main():
    print("=" * 72)
    print("ASVSPOOF 2021 DF - CONFIRMED SPOOF SELECTOR")
    print("=" * 72)

    if not METADATA.exists():
        raise FileNotFoundError(f"Metadata file not found:\n{METADATA}")

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio directory not found:\n{AUDIO_DIR}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    rows = parse_metadata()

    spoof_rows = [r for r in rows if r["label"] == "spoof"]
    bonafide_rows = [r for r in rows if r["label"] == "bonafide"]

    print(f"Metadata rows : {len(rows)}")
    print(f"Spoof rows    : {len(spoof_rows)}")
    print(f"Bonafide rows : {len(bonafide_rows)}")

    selected = select_diverse(spoof_rows, N_TO_COPY)

    copied = []
    missing = []

    for row in selected:
        src = find_audio(row["file_id"])

        if src is None:
            missing.append(row)
            continue

        dst = DEST_DIR / src.name

        if not dst.exists():
            shutil.copy2(src, dst)

        copied.append(row)

    print("\nCopied confirmed spoof files:")
    print("-" * 72)

    for i, row in enumerate(copied, start=1):
        print(
            f"{i:02d}. {row['file_id']} | "
            f"codec={row['codec']} | "
            f"source={row['source']} | "
            f"attack={row['attack']}"
        )

    print("\nSummary")
    print("-" * 72)
    print(f"Requested : {N_TO_COPY}")
    print(f"Copied    : {len(copied)}")
    print(f"Missing   : {len(missing)}")
    print(f"Destination: {DEST_DIR}")

    print("\nCodec distribution:")
    for k, v in Counter(r["codec"] for r in copied).most_common():
        print(f"  {k}: {v}")

    print("\nSource distribution:")
    for k, v in Counter(r["source"] for r in copied).most_common():
        print(f"  {k}: {v}")

    print("\nAttack distribution:")
    for k, v in Counter(r["attack"] for r in copied).most_common():
        print(f"  {k}: {v}")

    # Save manifest
    manifest = DEST_DIR / "selected_spoof_manifest.csv"

    with manifest.open("w", encoding="utf-8") as f:
        f.write("file_id,speaker_id,codec,source,attack,label\n")
        for row in copied:
            f.write(
                f"{row['file_id']},"
                f"{row['speaker_id']},"
                f"{row['codec']},"
                f"{row['source']},"
                f"{row['attack']},"
                f"{row['label']}\n"
            )

    print(f"\nManifest saved: {manifest}")

    if len(copied) < 5:
        print("\nWARNING: fewer than 5 spoof files were found/copied.")
    else:
        print("\nReady for the codec-robust SonicT experiment.")


if __name__ == "__main__":
    main()
