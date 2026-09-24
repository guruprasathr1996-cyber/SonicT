from pathlib import Path
import shutil

# =========================================================
# PATHS
# =========================================================

PROTOCOL = Path(
    r"E:\ASVspoof2017\protocol_V2\protocol_V2\ASVspoof2017_V2_dev.trl.txt"
)

AUDIO_ROOT = Path(
    r"E:\ASVspoof2017\ASVspoof2017_V2_dev\ASVspoof2017_V2_dev"
)

OUTPUT = Path(
    r"H:\SonicT_Warning_Test\replay"
)
NUMBER_OF_FILES = 10


# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# READ PROTOCOL
# =========================================================

with open(
    PROTOCOL,
    "r",
    encoding="utf-8"
) as f:
    lines = f.readlines()


print("Protocol entries:", len(lines))


# =========================================================
# FIND SPOOF / REPLAY FILE IDs
# =========================================================

spoof_ids = []

for line in lines:

    parts = line.strip().split()

    if not parts:
        continue

    # ASVspoof protocol contains the label
    # "spoof" for replay attack recordings.

    if len(parts) >= 2 and parts[1].lower() == "spoof":
        audio_id = Path(parts[0]).stem
        spoof_ids.append(audio_id)
        for item in parts:

            if item.startswith("D_"):
                audio_id = item
                break

        if audio_id:
            spoof_ids.append(audio_id)


print("Replay/spoof entries found:", len(spoof_ids))


# =========================================================
# FIND AUDIO FILES
# =========================================================

copied = 0

for audio_id in spoof_ids:

    # Search recursively because the audio may be
    # inside another subfolder.

    matches = list(
        AUDIO_ROOT.rglob(audio_id + ".*")
    )

    if not matches:
        continue

    source = matches[0]

    destination = (
        OUTPUT /
        source.name
    )

    shutil.copy2(
        source,
        destination
    )

    copied += 1

    print(
        f"[{copied}/{NUMBER_OF_FILES}]",
        source.name
    )

    if copied >= NUMBER_OF_FILES:
        break


# =========================================================
# RESULT
# =========================================================

print()
print("=" * 50)

if copied == NUMBER_OF_FILES:

    print("SUCCESS")
    print(
        f"{copied} replay attack files copied."
    )

    print(
        "Output:",
        OUTPUT
    )

else:

    print(
        f"Only {copied} files were copied."
    )

    print(
        "We may need to inspect the "
        "dataset folder structure."
    )

print("=" * 50)