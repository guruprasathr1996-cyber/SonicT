"""
SonicT F1B V8 Phase 4 - Disagreement Gate Analysis
===================================================
NO TRAINING. NO MODEL MODIFICATION.

Uses the Phase 3 per-sample decisions to evaluate three-state rules:
    GENUINE
    DEEPFAKE
    UNCERTAIN-SUSPICIOUS

Special focus:
    V7 = DEEPFAKE while V7.2 = GENUINE

This is the exact pattern seen in the difficult compressed challenge.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

INPUT = Path(
    r"H:\SonicT_F1B_External_Test\v8_phase3_fusion\v8_phase3_all_decisions.csv"
)

OUT = Path(
    r"H:\SonicT_F1B_External_Test\v8_phase4_disagreement_gate"
)
OUT.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(INPUT)

required = {"truth", "v7", "v72", "v73"}
if not required.issubset(df.columns):
    raise RuntimeError(
        f"Missing required columns. Found: {df.columns.tolist()}"
    )

print("=" * 96)
print("SONICT F1B V8 PHASE 4 - DISAGREEMENT GATE ANALYSIS")
print("=" * 96)
print("Samples:", len(df))
print("Genuine:", int((df.truth == 0).sum()))
print("Deepfake:", int((df.truth == 1).sum()))
print("Training: NONE")

# Binary votes at the already-used 0.50 threshold.
df["v7_vote"] = (df.v7 >= .50).astype(int)
df["v72_vote"] = (df.v72 >= .50).astype(int)
df["v73_vote"] = (df.v73 >= .50).astype(int)

# Exact difficult-challenge disagreement pattern.
df["challenge_pattern"] = (
    (df.v7_vote == 1) &
    (df.v72_vote == 0)
)

# All three vote patterns.
df["vote_pattern"] = (
    df.v7_vote.astype(str) +
    df.v72_vote.astype(str) +
    df.v73_vote.astype(str)
)

# ------------------------------------------------------------
# Pattern statistics
# ------------------------------------------------------------
pattern_rows = []

for pattern, d in df.groupby("vote_pattern"):
    genuine = int((d.truth == 0).sum())
    deepfake = int((d.truth == 1).sum())
    total = len(d)

    pattern_rows.append({
        "vote_pattern_V7_V72_V73": pattern,
        "total": total,
        "genuine": genuine,
        "deepfake": deepfake,
        "deepfake_fraction": deepfake / total if total else 0
    })

patterns = pd.DataFrame(pattern_rows).sort_values(
    ["total", "vote_pattern_V7_V72_V73"],
    ascending=[False, True]
)

patterns.to_csv(
    OUT / "v8_phase4_vote_pattern_statistics.csv",
    index=False
)

# ------------------------------------------------------------
# Exact challenge pattern analysis
# ------------------------------------------------------------
cp = df[df.challenge_pattern]
cp_genuine = int((cp.truth == 0).sum())
cp_deepfake = int((cp.truth == 1).sum())

genuine_total = int((df.truth == 0).sum())
deepfake_total = int((df.truth == 1).sum())

cp_genuine_rate = cp_genuine / genuine_total if genuine_total else 0
cp_deepfake_rate = cp_deepfake / deepfake_total if deepfake_total else 0

# ------------------------------------------------------------
# Three-state rules
# ------------------------------------------------------------
# Rule G1:
# If V7 and V7.2 agree -> use that result.
# If they disagree -> UNCERTAIN.
def gate1(row):
    if row.v7_vote == row.v72_vote:
        return "DEEPFAKE" if row.v72_vote == 1 else "GENUINE"
    return "UNCERTAIN"

df["G1_V7_V72_disagreement"] = df.apply(gate1, axis=1)

# Rule G2:
# Majority deepfake -> DEEPFAKE.
# If majority says genuine BUT V7 alone strongly says deepfake -> UNCERTAIN.
# Otherwise genuine.
#
# "Strong V7" uses >= 0.90 only as an analysis threshold; it is NOT calibrated.
def gate2(row):
    votes = row.v7_vote + row.v72_vote + row.v73_vote

    if votes >= 2:
        return "DEEPFAKE"

    if row.v7 >= .90 and row.v72_vote == 0:
        return "UNCERTAIN"

    return "GENUINE"

df["G2_majority_plus_strong_V7"] = df.apply(gate2, axis=1)

# Rule G3:
# V7.2 is primary.
# - V7.2 deepfake -> DEEPFAKE
# - V7.2 genuine and V7 >= .90 -> UNCERTAIN
# - otherwise GENUINE
def gate3(row):
    if row.v72_vote == 1:
        return "DEEPFAKE"

    if row.v7 >= .90:
        return "UNCERTAIN"

    return "GENUINE"

df["G3_V72_primary_V7_gate"] = df.apply(gate3, axis=1)

def three_state_metrics(col):
    genuine = df[df.truth == 0]
    deepfake = df[df.truth == 1]

    genuine_correct = int((genuine[col] == "GENUINE").sum())
    genuine_uncertain = int((genuine[col] == "UNCERTAIN").sum())
    genuine_wrong = int((genuine[col] == "DEEPFAKE").sum())

    deepfake_correct = int((deepfake[col] == "DEEPFAKE").sum())
    deepfake_uncertain = int((deepfake[col] == "UNCERTAIN").sum())
    deepfake_wrong = int((deepfake[col] == "GENUINE").sum())

    # Safety capture counts DEEPFAKE or UNCERTAIN as successfully escalated
    # for known deepfake samples.
    deepfake_safety_capture = (
        deepfake_correct + deepfake_uncertain
    ) / len(deepfake)

    # Genuine clean acceptance means no warning/escalation.
    genuine_clean_accept = genuine_correct / len(genuine)

    return {
        "genuine_correct": genuine_correct,
        "genuine_uncertain": genuine_uncertain,
        "genuine_wrong_deepfake": genuine_wrong,
        "genuine_clean_accept_rate": genuine_clean_accept,

        "deepfake_correct": deepfake_correct,
        "deepfake_uncertain": deepfake_uncertain,
        "deepfake_missed_as_genuine": deepfake_wrong,
        "deepfake_direct_recall": deepfake_correct / len(deepfake),
        "deepfake_safety_capture_rate": deepfake_safety_capture,

        "overall_uncertain_rate": float(
            (df[col] == "UNCERTAIN").mean()
        )
    }

results = []

for name, col in [
    ("G1 - V7/V7.2 disagreement -> UNCERTAIN",
     "G1_V7_V72_disagreement"),
    ("G2 - Majority + strong V7 disagreement gate",
     "G2_majority_plus_strong_V7"),
    ("G3 - V7.2 primary + strong V7 gate",
     "G3_V72_primary_V7_gate")
]:
    m = three_state_metrics(col)
    m["rule"] = name
    results.append(m)

rdf = pd.DataFrame(results)

cols = [
    "rule",
    "genuine_correct",
    "genuine_uncertain",
    "genuine_wrong_deepfake",
    "genuine_clean_accept_rate",
    "deepfake_correct",
    "deepfake_uncertain",
    "deepfake_missed_as_genuine",
    "deepfake_direct_recall",
    "deepfake_safety_capture_rate",
    "overall_uncertain_rate"
]

rdf = rdf[cols]

rdf.to_csv(
    OUT / "v8_phase4_gate_comparison.csv",
    index=False
)

df.to_csv(
    OUT / "v8_phase4_all_gate_decisions.csv",
    index=False
)

with open(
    OUT / "v8_phase4_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "samples": len(df),
            "genuine": genuine_total,
            "deepfake": deepfake_total,
            "challenge_pattern": {
                "definition": "V7=DEEPFAKE and V7.2=GENUINE",
                "total": len(cp),
                "genuine": cp_genuine,
                "deepfake": cp_deepfake,
                "fraction_of_genuine": cp_genuine_rate,
                "fraction_of_deepfake": cp_deepfake_rate
            },
            "strong_v7_analysis_threshold": 0.90,
            "threshold_calibrated": False,
            "training_performed": False,
            "models_modified": False,
            "gate_results": rdf.to_dict(orient="records")
        },
        f,
        indent=2
    )

# ------------------------------------------------------------
# Console
# ------------------------------------------------------------
print("\n" + "=" * 96)
print("VOTE PATTERNS  (V7 / V7.2 / V7.3; 1=DEEPFAKE)")
print("=" * 96)

for _, r in patterns.iterrows():
    print(
        f"{r['vote_pattern_V7_V72_V73']} | "
        f"Total {int(r['total']):3d} | "
        f"Genuine {int(r['genuine']):3d} | "
        f"Deepfake {int(r['deepfake']):3d} | "
        f"DF fraction {r['deepfake_fraction']*100:6.2f}%"
    )

print("\n" + "=" * 96)
print("DIFFICULT-CHALLENGE PATTERN")
print("=" * 96)
print("Pattern: V7=DEEPFAKE, V7.2=GENUINE")
print("Total occurrences :", len(cp))
print("Genuine occurrences:", cp_genuine,
      f"({cp_genuine_rate*100:.2f}% of genuine)")
print("Deepfake occurrences:", cp_deepfake,
      f"({cp_deepfake_rate*100:.2f}% of deepfake)")

print("\n" + "=" * 96)
print("THREE-STATE GATE RESULTS")
print("=" * 96)

for _, r in rdf.iterrows():
    print(f"\n{r['rule']}")
    print(
        f"Genuine clean accept : "
        f"{r['genuine_clean_accept_rate']*100:6.2f}%"
    )
    print(
        f"Genuine uncertain    : "
        f"{int(r['genuine_uncertain'])}/50"
    )
    print(
        f"Genuine false alarm  : "
        f"{int(r['genuine_wrong_deepfake'])}/50"
    )
    print(
        f"Deepfake direct recall: "
        f"{r['deepfake_direct_recall']*100:6.2f}%"
    )
    print(
        f"Deepfake uncertain    : "
        f"{int(r['deepfake_uncertain'])}/250"
    )
    print(
        f"Deepfake missed       : "
        f"{int(r['deepfake_missed_as_genuine'])}/250"
    )
    print(
        f"DF safety capture     : "
        f"{r['deepfake_safety_capture_rate']*100:6.2f}%"
    )
    print(
        f"Overall UNCERTAIN     : "
        f"{r['overall_uncertain_rate']*100:6.2f}%"
    )

print("\nOutputs:", OUT)
print("No training performed. No models modified.")
print("NOTE: V7 >= 0.90 is an exploratory gate, not a calibrated threshold.")
