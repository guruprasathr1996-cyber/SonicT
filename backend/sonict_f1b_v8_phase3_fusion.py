"""
SonicT F1B - V8 Phase 3 Fusion Rule Evaluation
================================================
NO TRAINING. NO MODEL MODIFICATION.

Combines:
Phase 1: 50 external genuine-phone recordings
Phase 2: 250 conditioned deepfake evaluations

Evaluates:
A - V7.2 alone
B - V7 alone
C - V7.3 alone
D - 2-of-3 majority vote
E - V7.2 primary + V7 disagreement -> UNCERTAIN
F - weighted probability fusion

IMPORTANT:
Rule E is also evaluated with selective-classification metrics because
UNCERTAIN is an abstention, not automatically a wrong binary prediction.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

P1_DIR = Path(r"H:\SonicT_F1B_External_Test\v8_results")
P2_FILE = Path(r"H:\SonicT_F1B_External_Test\v8_phase2_results\v8_phase2_all_results.csv")
OUT = Path(r"H:\SonicT_F1B_External_Test\v8_phase3_fusion")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------- locate Phase 1 CSV ----------------------
candidates = list(P1_DIR.glob("*.csv"))
if not candidates:
    raise FileNotFoundError(f"No Phase-1 CSV found in {P1_DIR}")

def score_csv(p):
    try:
        d = pd.read_csv(p, nrows=3)
        cols = {c.lower() for c in d.columns}
        score = sum(any(k in c for c in cols) for k in ["v7", "v72", "v73"])
        return score
    except Exception:
        return -1

P1_FILE = max(candidates, key=score_csv)
print("Phase 1 CSV:", P1_FILE)
print("Phase 2 CSV:", P2_FILE)

p1 = pd.read_csv(P1_FILE)
p2 = pd.read_csv(P2_FILE)

print("\nPhase 1 columns:", p1.columns.tolist())
print("Phase 2 columns:", p2.columns.tolist())

# ---------------------- flexible column finder ----------------------
def find_col(df, alternatives, contains=None):
    lower = {c.lower(): c for c in df.columns}
    for a in alternatives:
        if a.lower() in lower:
            return lower[a.lower()]
    if contains:
        for c in df.columns:
            x = c.lower()
            if all(k.lower() in x for k in contains):
                return c
    return None

def resolve(df, model):
    # model keys: v7, v72, v73
    names = {
        "v7":  ["v7_median_df", "v7_median_deepfake", "v7_df_median"],
        "v72": ["v72_median_df", "v7.2_median_df", "v7_2_median_df",
                "v72_median_deepfake", "v7_2_df_median"],
        "v73": ["v73_median_df", "v7.3_median_df", "v7_3_median_df",
                "v73_median_deepfake", "v7_3_df_median"]
    }
    c = find_col(df, names[model])
    if c: return c

    tokens = {
        "v7":  [["v7","median"]],
        "v72": [["v72","median"],["v7.2","median"],["v7_2","median"]],
        "v73": [["v73","median"],["v7.3","median"],["v7_3","median"]]
    }
    for ks in tokens[model]:
        c = find_col(df, [], contains=ks)
        if c: return c
    raise RuntimeError(f"Cannot find {model} median deepfake-score column in: {df.columns.tolist()}")

p1cols = {m: resolve(p1,m) for m in ["v7","v72","v73"]}
p2cols = {m: resolve(p2,m) for m in ["v7","v72","v73"]}
print("\nResolved Phase 1 score columns:", p1cols)
print("Resolved Phase 2 score columns:", p2cols)

# ---------------------- normalize probability scales ----------------------
def prob(s):
    x = pd.to_numeric(s, errors="coerce").astype(float)
    # Some prior scripts may save 0..100 instead of 0..1.
    if x.dropna().max() > 1.5:
        x = x / 100.0
    return x.clip(0,1)

def make_frame(df, cols, truth):
    z = pd.DataFrame(index=df.index)

    z["truth"] = int(truth)

    z["v7"] = prob(df[cols["v7"]])
    z["v72"] = prob(df[cols["v72"]])
    z["v73"] = prob(df[cols["v73"]])

    if "condition" in df.columns:
        z["condition"] = df["condition"].astype(str)
    else:
        z["condition"] = "genuine_phone"

    if "generator" in df.columns:
        z["generator"] = df["generator"].astype(str)
    else:
        z["generator"] = ""

    return z.dropna(
        subset=["truth", "v7", "v72", "v73"]
    ).reset_index(drop=True)
genuine = make_frame(p1,p1cols,0)
deepfake = make_frame(p2,p2cols,1)
data = pd.concat([genuine,deepfake],ignore_index=True)

print(f"\nGenuine samples : {len(genuine)}")
print(f"Deepfake samples: {len(deepfake)}")
print(f"Total           : {len(data)}")

# ---------------------- decision rules ----------------------
data["A_V72"] = (data.v72 >= .50).astype(int)
data["B_V7"] = (data.v7 >= .50).astype(int)
data["C_V73"] = (data.v73 >= .50).astype(int)
data["D_Majority"] = ((data[["v7","v72","v73"]] >= .50).sum(axis=1) >= 2).astype(int)

# Rule E: V7.2 is primary. If V7 disagrees, abstain/UNCERTAIN.
e = np.where(
    (data.v72 >= .50) == (data.v7 >= .50),
    (data.v72 >= .50).astype(int),
    -1
)
data["E_V72_V7_Abstain"] = e

# Conservative weighted rule. V7.2 gets largest weight because Phase 1/2
# showed the best balance; V7 and V7.3 provide secondary evidence.
data["weighted_score"] = .60*data.v72 + .25*data.v7 + .15*data.v73
data["F_Weighted"] = (data.weighted_score >= .50).astype(int)

# ---------------------- metrics ----------------------
def binary_metrics(y,p):
    y=np.asarray(y); p=np.asarray(p)
    tp=int(((y==1)&(p==1)).sum())
    tn=int(((y==0)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum())
    acc=(tp+tn)/len(y) if len(y) else np.nan
    prec=tp/(tp+fp) if tp+fp else 0
    rec=tp/(tp+fn) if tp+fn else 0
    spec=tn/(tn+fp) if tn+fp else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    fpr=fp/(fp+tn) if fp+tn else 0
    fnr=fn/(fn+tp) if fn+tp else 0
    bal=(rec+spec)/2
    return dict(TP=tp,TN=tn,FP=fp,FN=fn,Accuracy=acc,Precision=prec,
                Recall=rec,Specificity=spec,F1=f1,FPR=fpr,FNR=fnr,
                Balanced_Accuracy=bal)

results=[]
for label,col in [
    ("A - V7.2 alone","A_V72"),
    ("B - V7 alone","B_V7"),
    ("C - V7.3 alone","C_V73"),
    ("D - 2-of-3 majority","D_Majority"),
    ("F - weighted 60/25/15","F_Weighted")
]:
    m=binary_metrics(data.truth,data[col])
    m["Rule"]=label
    m["Coverage"]=1.0
    m["Abstain_Rate"]=0.0
    results.append(m)

# Rule E: report only covered samples as classification metrics.
covered=data[data.E_V72_V7_Abstain!=-1]
m=binary_metrics(covered.truth,covered.E_V72_V7_Abstain)
m["Rule"]="E - V7.2 + V7 disagreement abstain"
m["Coverage"]=len(covered)/len(data)
m["Abstain_Rate"]=1-len(covered)/len(data)
results.append(m)

rdf=pd.DataFrame(results)
order=["Rule","Accuracy","Balanced_Accuracy","Precision","Recall","Specificity",
       "F1","FPR","FNR","Coverage","Abstain_Rate","TP","TN","FP","FN"]
rdf=rdf[order]
rdf.to_csv(OUT/"v8_phase3_rule_comparison.csv",index=False)

# Deepfake recall by condition for each non-abstaining binary rule
condition_rows=[]
for cond,d in data[data.truth==1].groupby("condition"):
    for label,col in [
        ("V7.2","A_V72"),("V7","B_V7"),("V7.3","C_V73"),
        ("Majority","D_Majority"),("Weighted","F_Weighted")
    ]:
        condition_rows.append({
            "condition":cond,"rule":label,"n":len(d),
            "deepfake_recall":float((d[col]==1).mean()),
            "false_negative_rate":float((d[col]==0).mean())
        })
pd.DataFrame(condition_rows).to_csv(OUT/"v8_phase3_condition_metrics.csv",index=False)

# Save per-sample decisions.
data.to_csv(OUT/"v8_phase3_all_decisions.csv",index=False)

# Human-readable JSON.
with open(OUT/"v8_phase3_summary.json","w",encoding="utf-8") as f:
    json.dump({
        "phase1_genuine_samples":len(genuine),
        "phase2_deepfake_samples":len(deepfake),
        "total_samples":len(data),
        "threshold":0.50,
        "weighted_rule":{"v7_2":0.60,"v7":0.25,"v7_3":0.15},
        "training_performed":False,
        "models_modified":False,
        "rules":rdf.to_dict(orient="records")
    },f,indent=2)

# ---------------------- console report ----------------------
print("\n"+"="*110)
print("V8 PHASE 3 - FUSION RULE RESULTS")
print("="*110)
for _,r in rdf.iterrows():
    print(f"\n{r['Rule']}")
    print(f"Accuracy       : {r['Accuracy']*100:6.2f}%")
    print(f"Balanced Acc   : {r['Balanced_Accuracy']*100:6.2f}%")
    print(f"Precision      : {r['Precision']*100:6.2f}%")
    print(f"Deepfake Recall: {r['Recall']*100:6.2f}%")
    print(f"Genuine Recall : {r['Specificity']*100:6.2f}%")
    print(f"F1             : {r['F1']*100:6.2f}%")
    print(f"False Positive : {r['FPR']*100:6.2f}%")
    print(f"False Negative : {r['FNR']*100:6.2f}%")
    if r["Abstain_Rate"]>0:
        print(f"Coverage       : {r['Coverage']*100:6.2f}%")
        print(f"UNCERTAIN      : {r['Abstain_Rate']*100:6.2f}%")
    print(f"TP/TN/FP/FN    : {int(r['TP'])}/{int(r['TN'])}/{int(r['FP'])}/{int(r['FN'])}")

print("\nOutputs:",OUT)
print("No training performed. No model files modified.")
