from pathlib import Path
import pandas as pd
import shutil

MANIFEST = Path(
    r"H:\SonicT_Compression_Test\v7_mlaad_split\fake_manifest_v7.csv"
)

OUTPUT = Path(
    r"H:\SonicT_F1B_External_Test\compressed_deepfake"
)

FILES_PER_GENERATOR = 5
RANDOM_SEED = 42

OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load manifest
# ---------------------------------------------------------

df = pd.read_csv(MANIFEST)

print("=" * 70)
print("V8 DEEPFAKE SET PREPARATION")
print("=" * 70)

print("\nColumns:")
print(df.columns.tolist())

print("\nTotal manifest rows:", len(df))

# ---------------------------------------------------------
# Detect important columns
# ---------------------------------------------------------

path_col = None
generator_col = None
split_col = None
class_col = None

for col in df.columns:
    name = col.lower().strip()

    if name in ["path", "filepath", "file_path", "audio_path"]:
        path_col = col

    if name in ["generator", "generator_name", "model", "attack"]:
        generator_col = col

    if name == "split":
        split_col = col

    if name in ["class", "label_name", "category"]:
        class_col = col

if path_col is None:
    raise RuntimeError(
        "Could not automatically identify the audio path column."
    )

if generator_col is None:
    raise RuntimeError(
        "Could not automatically identify the generator column."
    )

print("\nDetected:")
print("Path column      :", path_col)
print("Generator column :", generator_col)
print("Split column     :", split_col)
print("Class column     :", class_col)

# ---------------------------------------------------------
# Keep TEST deepfakes only
# ---------------------------------------------------------

work = df.copy()

if split_col is not None:
    work = work[
        work[split_col].astype(str).str.lower().str.strip() == "test"
    ]

if class_col is not None:
    work = work[
        work[class_col]
        .astype(str)
        .str.lower()
        .str.contains("deepfake|fake|spoof", regex=True)
    ]

print("\nEligible test deepfakes:", len(work))

# ---------------------------------------------------------
# Show generators
# ---------------------------------------------------------

generators = sorted(
    work[generator_col]
    .dropna()
    .astype(str)
    .unique()
)

print("\nGenerators found:")
for g in generators:
    count = (
        work[generator_col]
        .astype(str)
        .eq(g)
        .sum()
    )

    print(f"{g:35s} : {count}")

print("\nNumber of generators:", len(generators))

# ---------------------------------------------------------
# Select 5 files from every generator
# ---------------------------------------------------------

selected_parts = []

for generator in generators:

    group = work[
        work[generator_col].astype(str) == generator
    ]

    if len(group) < FILES_PER_GENERATOR:
        print(
            f"WARNING: {generator} only has {len(group)} files."
        )
        sample = group
    else:
        sample = group.sample(
            n=FILES_PER_GENERATOR,
            random_state=RANDOM_SEED
        )

    selected_parts.append(sample)

selected = pd.concat(
    selected_parts,
    ignore_index=True
)

print("\nSelected files:", len(selected))

# ---------------------------------------------------------
# Copy files
# ---------------------------------------------------------

copied = []
failed = []

for index, row in selected.iterrows():

    source = Path(str(row[path_col]))

    generator = str(row[generator_col])

    safe_generator = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in generator
    )

    if not source.exists():
        failed.append({
            "source": str(source),
            "generator": generator,
            "reason": "Source file not found"
        })

        print("MISSING:", source)
        continue

    # Generator name is included so duplicate filenames
    # from different generators cannot overwrite each other.
    destination_name = (
        f"{safe_generator}__{source.name}"
    )

    destination = OUTPUT / destination_name

    shutil.copy2(
        source,
        destination
    )

    copied.append({
        "source": str(source),
        "destination": str(destination),
        "generator": generator,
        "original_filename": source.name
    })

# ---------------------------------------------------------
# Save V8 manifest
# ---------------------------------------------------------

copied_df = pd.DataFrame(copied)

copied_df.to_csv(
    OUTPUT / "v8_deepfake_manifest.csv",
    index=False
)

if failed:

    pd.DataFrame(failed).to_csv(
        OUTPUT / "v8_copy_failures.csv",
        index=False
    )

# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("V8 DEEPFAKE DATASET READY")
print("=" * 70)

print("Files copied :", len(copied))
print("Failed       :", len(failed))

if copied:

    print(
        "Generators   :",
        copied_df["generator"].nunique()
    )

    print("\nFiles per generator:")

    print(
        copied_df["generator"]
        .value_counts()
        .sort_index()
    )

print("\nOutput folder:")
print(OUTPUT)

print("\nOriginal MLAAD files were NOT modified.")