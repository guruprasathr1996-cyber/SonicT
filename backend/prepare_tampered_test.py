from pathlib import Path
import subprocess
import wave
import csv
import random
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

SOURCE_DIR = Path(
    r"H:\SonicT_Warning_Test\tamper_source"
)

CONVERTED_DIR = Path(
    r"H:\SonicT_Warning_Test\tamper_source_wav"
)

OUTPUT_DIR = Path(
    r"H:\SonicT_Warning_Test\tampered"
)

CSV_PATH = Path(
    r"H:\SonicT_Warning_Test\tampered_ground_truth.csv"
)

NUMBER_OF_FILES = 10
TARGET_SR = 16000

random.seed(2026)


# =========================================================
# CREATE FOLDERS
# =========================================================

CONVERTED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# FIND FFMPEG
# =========================================================

def find_ffmpeg():

    # First try FFmpeg from PATH
    try:

        result = subprocess.run(
            [
                "ffmpeg",
                "-version"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if result.returncode == 0:
            return "ffmpeg"

    except FileNotFoundError:
        pass


    # Search common H: locations
    possible_locations = [
        Path(r"H:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"H:\FFmpeg\bin\ffmpeg.exe"),
        Path(r"H:\ffmpeg.exe")
    ]

    for path in possible_locations:

        if path.exists():
            return str(path)


    raise RuntimeError(
        "FFmpeg could not be found. "
        "Add FFmpeg to PATH or update the "
        "possible_locations list."
    )


FFMPEG = find_ffmpeg()

print(
    "FFmpeg:",
    FFMPEG
)


# =========================================================
# CONVERT AUDIO TO STANDARD WAV
# =========================================================

def convert_to_wav(source, destination):

    command = [
        FFMPEG,
        "-y",
        "-i",
        str(source),

        "-vn",

        "-ac",
        "1",

        "-ar",
        str(TARGET_SR),

        "-acodec",
        "pcm_s16le",

        str(destination)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr[-1000:]
        )


# =========================================================
# LOAD WAV
# =========================================================

def load_wav(path):

    with wave.open(
        str(path),
        "rb"
    ) as wf:

        channels = wf.getnchannels()

        sample_width = wf.getsampwidth()

        sample_rate = wf.getframerate()

        frames = wf.readframes(
            wf.getnframes()
        )

    if channels != 1:

        raise ValueError(
            "Expected mono WAV."
        )

    if sample_width != 2:

        raise ValueError(
            "Expected PCM 16-bit WAV."
        )

    if sample_rate != TARGET_SR:

        raise ValueError(
            f"Expected {TARGET_SR} Hz."
        )

    return np.frombuffer(
        frames,
        dtype=np.int16
    ).copy()


# =========================================================
# SAVE WAV
# =========================================================

def save_wav(path, audio):

    with wave.open(
        str(path),
        "wb"
    ) as wf:

        wf.setnchannels(1)

        wf.setsampwidth(2)

        wf.setframerate(
            TARGET_SR
        )

        wf.writeframes(
            audio.astype(
                np.int16
            ).tobytes()
        )


# =========================================================
# FIND SOURCE RECORDINGS
# =========================================================

supported_extensions = {
    ".m4a",
    ".wav",
    ".mp3",
    ".mpeg",
    ".aac",
    ".flac"
}

source_files = sorted(
    [
        path
        for path in SOURCE_DIR.iterdir()

        if (
            path.is_file()
            and
            path.suffix.lower()
            in supported_extensions
        )
    ]
)


print()
print(
    "Source recordings found:",
    len(source_files)
)


if len(source_files) < 2:

    raise RuntimeError(
        "At least 2 source recordings "
        "are required."
    )


# =========================================================
# CONVERT SOURCES
# =========================================================

converted_files = []

print()
print(
    "Converting sources to "
    "16 kHz mono PCM16 WAV..."
)
print()


for index, source in enumerate(
    source_files,
    start=1
):

    output_name = (
        f"source_{index:02d}.wav"
    )

    destination = (
        CONVERTED_DIR /
        output_name
    )

    try:

        convert_to_wav(
            source,
            destination
        )

        audio = load_wav(
            destination
        )

        duration = (
            len(audio) /
            TARGET_SR
        )

        if duration < 8:

            print(
                f"[SKIP] {source.name}"
                f" - only {duration:.2f}s"
            )

            continue


        converted_files.append(
            {
                "original":
                    source,

                "wav":
                    destination,

                "audio":
                    audio,

                "duration":
                    duration
            }
        )


        print(
            f"[{index}/{len(source_files)}]",
            source.name,
            "->",
            output_name,
            f"({duration:.2f}s)"
        )


    except Exception as e:

        print(
            "[ERROR]",
            source.name,
            e
        )


print()
print(
    "Usable converted recordings:",
    len(converted_files)
)


if len(converted_files) < 2:

    raise RuntimeError(
        "Need at least 2 usable "
        "recordings longer than 8 seconds."
    )


# =========================================================
# CREATE CONTROLLED TAMPERED AUDIO
# =========================================================

ground_truth = []

created = 0
attempts = 0


while created < NUMBER_OF_FILES:

    attempts += 1

    if attempts > 200:

        raise RuntimeError(
            "Unable to create enough "
            "tampered samples."
        )


    # Pick two different recordings
    target, donor = random.sample(
        converted_files,
        2
    )


    target_audio = (
        target["audio"]
    )

    donor_audio = (
        donor["audio"]
    )


    # -----------------------------------------------------
    # Choose tampered duration
    # 2.5 - 4 seconds
    # -----------------------------------------------------

    tamper_duration = random.uniform(
        2.5,
        4.0
    )

    tamper_samples = int(
        tamper_duration *
        TARGET_SR
    )


    # -----------------------------------------------------
    # DONOR LOCATION
    # -----------------------------------------------------

    one_second = TARGET_SR

    donor_min = one_second

    donor_max = (
        len(donor_audio)
        -
        tamper_samples
        -
        one_second
    )

    if donor_max <= donor_min:
        continue


    donor_start = random.randint(
        donor_min,
        donor_max
    )

    donor_end = (
        donor_start
        +
        tamper_samples
    )


    donor_segment = donor_audio[
        donor_start:
        donor_end
    ].copy()


    # -----------------------------------------------------
    # TARGET SPLICE LOCATION
    # Keep splice away from beginning/end
    # -----------------------------------------------------

    two_seconds = (
        2 *
        TARGET_SR
    )

    target_min = two_seconds

    target_max = (
        len(target_audio)
        -
        tamper_samples
        -
        two_seconds
    )

    if target_max <= target_min:
        continue


    splice_start = random.randint(
        target_min,
        target_max
    )

    splice_end = (
        splice_start
        +
        tamper_samples
    )


    # -----------------------------------------------------
    # CREATE TAMPERED AUDIO
    # -----------------------------------------------------

    tampered_audio = (
        target_audio.copy()
    )

    tampered_audio[
        splice_start:
        splice_end
    ] = donor_segment


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    created += 1

    output_name = (
        f"tampered_{created:02d}.wav"
    )

    output_path = (
        OUTPUT_DIR /
        output_name
    )

    save_wav(
        output_path,
        tampered_audio
    )


    # -----------------------------------------------------
    # TIME INFORMATION
    # -----------------------------------------------------

    tamper_start_sec = (
        splice_start /
        TARGET_SR
    )

    tamper_end_sec = (
        splice_end /
        TARGET_SR
    )

    donor_start_sec = (
        donor_start /
        TARGET_SR
    )

    donor_end_sec = (
        donor_end /
        TARGET_SR
    )


    # -----------------------------------------------------
    # SAVE GROUND TRUTH
    # -----------------------------------------------------

    ground_truth.append(
        {
            "tampered_file":
                output_name,

            "target_source":
                target["original"].name,

            "donor_source":
                donor["original"].name,

            "tamper_start_sec":
                round(
                    tamper_start_sec,
                    3
                ),

            "tamper_end_sec":
                round(
                    tamper_end_sec,
                    3
                ),

            "tamper_duration_sec":
                round(
                    tamper_end_sec
                    -
                    tamper_start_sec,
                    3
                ),

            "donor_start_sec":
                round(
                    donor_start_sec,
                    3
                ),

            "donor_end_sec":
                round(
                    donor_end_sec,
                    3
                )
        }
    )


    print()
    print(
        f"[{created}/{NUMBER_OF_FILES}]",
        output_name
    )

    print(
        " Target:",
        target["original"].name
    )

    print(
        " Donor :",
        donor["original"].name
    )

    print(
        " Tampered region:",
        f"{tamper_start_sec:.2f}s",
        "-",
        f"{tamper_end_sec:.2f}s"
    )


# =========================================================
# WRITE GROUND-TRUTH CSV
# =========================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "tampered_file",
        "target_source",
        "donor_source",
        "tamper_start_sec",
        "tamper_end_sec",
        "tamper_duration_sec",
        "donor_start_sec",
        "donor_end_sec"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        ground_truth
    )


# =========================================================
# FINISHED
# =========================================================

print()
print("=" * 60)

print("SUCCESS")

print(
    f"{created} controlled tampered "
    "recordings created."
)

print()
print(
    "Converted genuine sources:"
)
print(
    CONVERTED_DIR
)

print()
print(
    "Tampered recordings:"
)
print(
    OUTPUT_DIR
)

print()
print(
    "Ground-truth CSV:"
)
print(
    CSV_PATH
)

print("=" * 60)