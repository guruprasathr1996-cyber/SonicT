from pathlib import Path
import pandas as pd
import numpy as np
import random

SEED = 42

FAKE_ROOT = Path(r"H:\MLAAD-tiny\fake\en")
GENUINE_ROOT = Path(r"H:\MLAAD-tiny\original\en")
OUTPUT_DIR = Path(r"H:\SonicT_Compression_Test\v7_mlaad_split")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENERATOR_STATS_CSV = OUTPUT_DIR / "generator_stats_v7.csv"
FAKE_MANIFEST_CSV = OUTPUT_DIR / "fake_manifest_v7.csv"
GENUINE_MANIFEST_CSV = OUTPUT_DIR / "genuine_manifest_v7.csv"
SPLIT_SUMMARY_CSV = OUTPUT_DIR / "split_summary_v7.csv"

rng = np.random.default_rng(SEED)
random.seed(SEED)

AUDIO_EXTS = {
    ".wav", ".flac", ".mp3", ".m4a",
    ".ogg", ".opus", ".aac", ".wma",
    ".mpeg", ".mpg"
}

if not FAKE_ROOT.exists():
    raise FileNotFoundError(f"Fake root not found: {FAKE_ROOT}")

if not GENUINE_ROOT.exists():
    raise FileNotFoundError(f"Genuine root not found: {GENUINE_ROOT}")

def audio_files_recursive(folder):
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ])

generator_rows = []
fake_rows = []

generator_dirs = sorted([p for p in FAKE_ROOT.iterdir() if p.is_dir()])

print("=" * 78)
print("SONICT F1B V7 - MLAAD-TINY ENGLISH INSPECTION")
print("=" * 78)
print(f"Fake root    : {FAKE_ROOT}")
print(f"Genuine root : {GENUINE_ROOT}")
print(f"Generators   : {len(generator_dirs)}")

for generator_dir in generator_dirs:
    files = audio_files_recursive(generator_dir)
    generator_name = generator_dir.name

    generator_rows.append({
        "generator": generator_name,
        "n_files": len(files),
    })

    for p in files:
        fake_rows.append({
            "path": str(p),
            "filename": p.name,
            "generator": generator_name,
            "label": 1,
            "label_name": "deepfake",
        })

generator_stats = pd.DataFrame(generator_rows).sort_values(
    ["n_files", "generator"], ascending=[False, True]
).reset_index(drop=True)

fake_manifest = pd.DataFrame(fake_rows)

if fake_manifest.empty:
    raise RuntimeError("No English fake audio files found.")

generator_stats.to_csv(GENERATOR_STATS_CSV, index=False)

print("\nGENERATOR COUNTS")
print("----------------")
print(generator_stats.to_string(index=False))
print("\nTotal fake files:", len(fake_manifest))
print("Unique generators:", fake_manifest["generator"].nunique())

eligible = generator_stats[generator_stats["n_files"] >= 20].copy()

if len(eligible) < 9:
    raise RuntimeError(
        f"Only {len(eligible)} generators have at least 20 files. "
        "Need at least 9 for train/validation/test generator split."
    )

generators = eligible["generator"].tolist()
rng.shuffle(generators)

n_total = len(generators)
n_test = max(3, int(round(n_total * 0.15)))
n_val = max(3, int(round(n_total * 0.15)))

if n_test + n_val >= n_total:
    raise RuntimeError("Not enough generators after split allocation.")

test_generators = set(generators[:n_test])
val_generators = set(generators[n_test:n_test + n_val])
train_generators = set(generators[n_test + n_val:])

TRAIN_PER_GENERATOR = 80
VAL_PER_GENERATOR = 60
TEST_PER_GENERATOR = 60

selected_fake_parts = []

for split_name, generator_set, cap in [
    ("train", train_generators, TRAIN_PER_GENERATOR),
    ("validation", val_generators, VAL_PER_GENERATOR),
    ("test", test_generators, TEST_PER_GENERATOR),
]:
    for generator in sorted(generator_set):
        subset = fake_manifest[fake_manifest["generator"] == generator].copy()

        if len(subset) > cap:
            subset = subset.sample(n=cap, random_state=SEED)

        subset["split"] = split_name
        selected_fake_parts.append(subset)

selected_fake = pd.concat(selected_fake_parts, ignore_index=True)
selected_fake.to_csv(FAKE_MANIFEST_CSV, index=False)

genuine_files = audio_files_recursive(GENUINE_ROOT)

if not genuine_files:
    raise RuntimeError("No English genuine audio files found.")

genuine_manifest = pd.DataFrame([
    {
        "path": str(p),
        "filename": p.name,
        "generator": "genuine_original",
        "label": 0,
        "label_name": "genuine",
    }
    for p in genuine_files
]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

fake_train_n = int((selected_fake["split"] == "train").sum())
fake_val_n = int((selected_fake["split"] == "validation").sum())
fake_test_n = int((selected_fake["split"] == "test").sum())

train_n = min(fake_train_n, len(genuine_manifest))
remaining_after_train = len(genuine_manifest) - train_n
val_n = min(fake_val_n, remaining_after_train)
remaining_after_val = remaining_after_train - val_n
test_n = min(fake_test_n, remaining_after_val)

train_genuine = genuine_manifest.iloc[:train_n].copy()
val_genuine = genuine_manifest.iloc[train_n:train_n + val_n].copy()
test_genuine = genuine_manifest.iloc[
    train_n + val_n:train_n + val_n + test_n
].copy()

train_genuine["split"] = "train"
val_genuine["split"] = "validation"
test_genuine["split"] = "test"

selected_genuine = pd.concat(
    [train_genuine, val_genuine, test_genuine],
    ignore_index=True
)

selected_genuine.to_csv(GENUINE_MANIFEST_CSV, index=False)

train_gen_check = set(
    selected_fake.loc[selected_fake["split"] == "train", "generator"]
)
val_gen_check = set(
    selected_fake.loc[selected_fake["split"] == "validation", "generator"]
)
test_gen_check = set(
    selected_fake.loc[selected_fake["split"] == "test", "generator"]
)

if train_gen_check & val_gen_check:
    raise RuntimeError("Generator leakage: train/validation overlap.")

if train_gen_check & test_gen_check:
    raise RuntimeError("Generator leakage: train/test overlap.")

if val_gen_check & test_gen_check:
    raise RuntimeError("Generator leakage: validation/test overlap.")

print("\n" + "=" * 78)
print("V7 GENERATOR-DISJOINT SPLIT")
print("=" * 78)

print(f"Eligible generators  : {len(generators)}")
print(f"Train generators     : {len(train_generators)}")
print(f"Validation generators: {len(val_generators)}")
print(f"Test generators      : {len(test_generators)}")

print("\nTRAIN GENERATORS")
print("----------------")
for name in sorted(train_generators):
    print(name)

print("\nVALIDATION GENERATORS")
print("---------------------")
for name in sorted(val_generators):
    print(name)

print("\nTEST GENERATORS - COMPLETELY UNSEEN")
print("-----------------------------------")
for name in sorted(test_generators):
    print(name)

summary_rows = []

for split_name in ["train", "validation", "test"]:
    fake_n = int((selected_fake["split"] == split_name).sum())
    genuine_n = int((selected_genuine["split"] == split_name).sum())
    fake_generators_n = int(
        selected_fake.loc[
            selected_fake["split"] == split_name, "generator"
        ].nunique()
    )

    summary_rows.append({
        "split": split_name,
        "fake_files": fake_n,
        "genuine_files": genuine_n,
        "fake_generators": fake_generators_n,
        "total_files": fake_n + genuine_n,
    })

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 78)
print("V7 FILE COUNTS")
print("=" * 78)
print(summary_df.to_string(index=False))

summary_df.to_csv(SPLIT_SUMMARY_CSV, index=False)

print("\n" + "=" * 78)
print("V7 DATA PREPARATION COMPLETE")
print("=" * 78)
print("\nSaved:")
print(GENERATOR_STATS_CSV)
print(FAKE_MANIFEST_CSV)
print(GENUINE_MANIFEST_CSV)
print(SPLIT_SUMMARY_CSV)
print("\nIMPORTANT:")
print("No audio files were copied or modified.")
print("Test generators are completely unseen during training.")
print("Do NOT train V7 yet until the split output is reviewed.")
