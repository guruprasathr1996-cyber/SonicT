from pathlib import Path

AUDIO_DIR = Path(
    r"H:\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\flac"
)

METADATA = Path(
    r"H:\ASVspoof2021_DF_keys\keys\DF\CM\trial_metadata.txt"
)

available = {p.stem for p in AUDIO_DIR.glob("*.flac")}

bonafide = []
spoof = []

with open(METADATA, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) < 6:
            continue

        file_id = parts[1]
        codec = parts[2]
        source = parts[3]
        attack = parts[4]
        label = parts[5].lower()

        if file_id not in available:
            continue

        row = {
            "file_id": file_id,
            "codec": codec,
            "source": source,
            "attack": attack,
            "label": label,
        }

        if label == "bonafide":
            bonafide.append(row)

        elif label == "spoof":
            spoof.append(row)


print("=" * 60)
print("LOCAL ASVSPOOF 2021 DF PART00")
print("=" * 60)

print(f"\nFLAC files available : {len(available)}")
print(f"Bonafide available   : {len(bonafide)}")
print(f"Spoof available      : {len(spoof)}")

print("\nBonafide examples:")
for row in bonafide[:10]:
    print(
        row["file_id"],
        row["codec"],
        row["source"],
        row["attack"]
    )

print("\nSpoof examples:")
for row in spoof[:10]:
    print(
        row["file_id"],
        row["codec"],
        row["source"],
        row["attack"]
    )