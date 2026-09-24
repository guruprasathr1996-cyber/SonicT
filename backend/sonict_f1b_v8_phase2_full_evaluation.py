"""
SonicT F1B V8 Phase 2 - Full Multi-Model Evaluation
====================================================
Evaluates the prepared 250 conditioned deepfake samples with V7, V7.2 and V7.3.

NO TRAINING.
NO MODEL MODIFICATION.
Ground truth for every row: DEEPFAKE.

Input manifest:
H:\SonicT_F1B_External_Test\v8_phase2_conditioned\v8_phase2_condition_manifest.csv
"""

from pathlib import Path
import json, subprocess, tempfile, warnings
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WavLMModel

warnings.filterwarnings("ignore")

SR=16000
WIN=6*SR
MIN=SR
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH=6 if DEVICE.type=="cuda" else 2

MANIFEST=Path(r"H:\SonicT_F1B_External_Test\v8_phase2_conditioned\v8_phase2_condition_manifest.csv")
OUT=Path(r"H:\SonicT_F1B_External_Test\v8_phase2_results")
OUT.mkdir(parents=True,exist_ok=True)

WAVLM=Path(r"C:\Users\Guruprasath\.cache\huggingface\hub\models--microsoft--wavlm-base-plus\snapshots\4c66d4806a428f2e922ccfa1a962776e232d487b")
V7=Path(r"H:\SonicT_Compression_Test\experiment_v7_output\wavlm_cross_generator_v7.pt")
V72=Path(r"H:\SonicT_Compression_Test\experiment_v7_2_symmetric_codec\wavlm_cross_generator_v7_2_symmetric_codec.pt")
V73=Path(r"H:\SonicT_Compression_Test\experiment_v7_3_telephony_branch\wavlm_acoustic_telephony_antispof_v7_3.pt")

for p,n in [(MANIFEST,"condition manifest"),(WAVLM,"local WavLM"),(V7,"V7"),(V72,"V7.2"),(V73,"V7.3")]:
    if not p.exists(): raise FileNotFoundError(f"{n} not found: {p}")

df=pd.read_csv(MANIFEST)
required={"conditioned_path","condition","generator"}
if not required.issubset(df.columns):
    raise RuntimeError(f"Manifest missing columns. Found: {df.columns.tolist()}")

print("="*92)
print("SONICT F1B V8 PHASE 2 - FULL DEEPFAKE EVALUATION")
print("="*92)
print("Rows:",len(df))
print("Device:",DEVICE)
print("Training: NONE")
print("Ground truth: DEEPFAKE")

def norm(y):
    y=np.asarray(y,np.float32)
    peak=np.max(np.abs(y)) if len(y) else 0
    return (y/peak if peak>0 else y).astype(np.float32)

def load_audio(p):
    try:
        y,_=librosa.load(str(p),sr=SR,mono=True)
    except Exception:
        f=tempfile.NamedTemporaryFile(suffix=".wav",delete=False)
        q=Path(f.name); f.close()
        try:
            subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(p),"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(q)],check=True)
            y,_=librosa.load(str(q),sr=SR,mono=True)
        finally:q.unlink(missing_ok=True)
    y=norm(y)
    if len(y)<MIN: raise ValueError("audio shorter than 1 second")
    return y

def windows(y):
    return [y[i:i+WIN] for i in range(0,len(y),WIN) if len(y[i:i+WIN])>=MIN]

print("\nLoading local WavLM...")
fe=AutoFeatureExtractor.from_pretrained(str(WAVLM),local_files_only=True)
wm=WavLMModel.from_pretrained(str(WAVLM),local_files_only=True).to(DEVICE)
wm.eval()
for p in wm.parameters():p.requires_grad=False

def embed(ws):
    outs=[]
    for i in range(0,len(ws),BATCH):
        b=ws[i:i+BATCH]
        z=fe(b,sampling_rate=SR,padding=True,return_attention_mask=True,return_tensors="pt")
        x=z["input_values"].to(DEVICE)
        mask=z.get("attention_mask")
        if mask is not None:mask=mask.to(DEVICE)
        with torch.no_grad():
            h=wm(input_values=x,attention_mask=mask).last_hidden_state
            try:
                fm=wm._get_feature_vector_attention_mask(h.shape[1],mask) if mask is not None else torch.ones(h.shape[:2],dtype=torch.bool,device=DEVICE)
            except Exception:
                fm=torch.ones(h.shape[:2],dtype=torch.bool,device=DEVICE)
            m=fm.unsqueeze(-1).float()
            c=m.sum(1).clamp(min=1)
            mean=(h*m).sum(1)/c
            var=(((h-mean[:,None])**2)*m).sum(1)/c
            outs.append(torch.cat([mean,torch.sqrt(var.clamp(min=1e-8))],1).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)

class MLP(nn.Module):
    def __init__(self,d=1536):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.30),
            nn.Linear(256,64),nn.ReLU(),nn.Dropout(.20),nn.Linear(64,2)
        )
    def forward(self,x):return self.net(x)

class TMLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d,384),nn.BatchNorm1d(384),nn.ReLU(),nn.Dropout(.35),
            nn.Linear(384,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.30),
            nn.Linear(128,32),nn.ReLU(),nn.Dropout(.20),nn.Linear(32,2)
        )
    def forward(self,x):return self.net(x)

def load_old(path):
    c=torch.load(path,map_location=DEVICE,weights_only=False)
    m=MLP().to(DEVICE);m.load_state_dict(c["model_state_dict"]);m.eval()
    mu=np.asarray(c["feature_mean"],np.float32)
    sd=np.asarray(c["feature_std"],np.float32);sd[sd<1e-6]=1
    return m,mu,sd

print("Loading V7...")
m7,mu7,sd7=load_old(V7)
print("Loading V7.2...")
m72,mu72,sd72=load_old(V72)
print("Loading V7.3...")
c73=torch.load(V73,map_location=DEVICE,weights_only=False)
m73=TMLP(int(c73["combined_feature_size"])).to(DEVICE)
m73.load_state_dict(c73["model_state_dict"]);m73.eval()
mw=np.asarray(c73["wavlm_mean"],np.float32)
sw=np.asarray(c73["wavlm_std"],np.float32);sw[sw<1e-6]=1
ma=np.asarray(c73["acoustic_mean"],np.float32)
sa=np.asarray(c73["acoustic_std"],np.float32);sa[sa<1e-6]=1
print("All models loaded.\n")

def band(y,a,b):
    S=np.abs(librosa.stft(y,n_fft=2048,hop_length=512))**2
    f=librosa.fft_frequencies(sr=SR,n_fft=2048)
    return float(S[(f>=a)&(f<b)].sum()/(S.sum()+1e-12))

def acoustic(y):
    cen=librosa.feature.spectral_centroid(y=y,sr=SR)[0]
    bw=librosa.feature.spectral_bandwidth(y=y,sr=SR)[0]
    ro=librosa.feature.spectral_rolloff(y=y,sr=SR,roll_percent=.85)[0]
    fl=librosa.feature.spectral_flatness(y=y)[0]
    z=librosa.feature.zero_crossing_rate(y)[0]
    r=librosa.feature.rms(y=y)[0]
    mf=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=13)
    v=[cen.mean(),cen.std(),bw.mean(),bw.std(),ro.mean(),ro.std(),fl.mean(),fl.std(),z.mean(),z.std(),r.mean(),r.std(),
       band(y,0,1000),band(y,1000,3000),band(y,3000,4000),band(y,4000,8000)]
    for x in mf:v += [x.mean(),x.std()]
    return np.asarray(v,np.float32)

def predict(m,x):
    with torch.no_grad():
        return torch.softmax(m(torch.tensor(x,dtype=torch.float32,device=DEVICE)),1)[:,1].cpu().numpy()

def summarize(p):
    p=np.asarray(p)
    return {
        "mean":float(np.mean(p)),
        "median":float(np.median(p)),
        "p90":float(np.percentile(p,90)),
        "max":float(np.max(p)),
        "suspicious_ratio":float(np.mean(p>=.50)),
        "prediction":"DEEPFAKE" if np.median(p)>=.50 else "GENUINE"
    }

rows=[];fails=[]
for i,r in df.iterrows():
    path=Path(r["conditioned_path"])
    try:
        y=load_audio(path);ws=windows(y)
        if not ws:raise RuntimeError("no valid windows")
        E=embed(ws)
        p7=predict(m7,(E-mu7)/sd7)
        p72=predict(m72,(E-mu72)/sd72)
        A=np.stack([acoustic(w) for w in ws])
        p73=predict(m73,np.concatenate([(E-mw)/sw,(A-ma)/sa],1))
        a,b,c=map(summarize,(p7,p72,p73))
        preds=[a["prediction"],b["prediction"],c["prediction"]]
        rows.append({
            "source":r.get("source",""),"conditioned_path":str(path),"generator":r["generator"],"condition":r["condition"],"windows":len(ws),
            "v7_prediction":preds[0],"v7_median_df":a["median"],"v7_p90_df":a["p90"],
            "v72_prediction":preds[1],"v72_median_df":b["median"],"v72_p90_df":b["p90"],
            "v73_prediction":preds[2],"v73_median_df":c["median"],"v73_p90_df":c["p90"],
            "deepfake_votes":sum(x=="DEEPFAKE" for x in preds),"disagreement":len(set(preds))>1
        })
        print(f"[{i+1:03d}/{len(df)}] {r['condition']:20s} | V7 {a['median']*100:6.2f}% | V7.2 {b['median']*100:6.2f}% | V7.3 {c['median']*100:6.2f}%")
    except Exception as e:
        fails.append({"path":str(path),"condition":r["condition"],"error":str(e)})
        print("FAILED:",path.name,e)

res=pd.DataFrame(rows)
res.to_csv(OUT/"v8_phase2_all_results.csv",index=False)
if fails:pd.DataFrame(fails).to_csv(OUT/"v8_phase2_failures.csv",index=False)

summary=[]
order=["original","narrowband_8k","telephone_band","mp3_16k","telephone_mp3_16k"]
for cond in order:
    d=res[res.condition==cond]
    for name,pre in [("V7","v7"),("V7.2","v72"),("V7.3","v73")]:
        recall=float((d[f"{pre}_prediction"]=="DEEPFAKE").mean())
        vals=d[f"{pre}_median_df"].to_numpy()
        summary.append({"condition":cond,"model":name,"n":len(d),"deepfake_recall":recall,
                        "false_negative_rate":1-recall,"median_df_score":float(np.median(vals)),
                        "p90_file_median_df":float(np.percentile(vals,90))})

sdf=pd.DataFrame(summary);sdf.to_csv(OUT/"v8_phase2_condition_summary.csv",index=False)

gen=[]
for g in sorted(res.generator.unique()):
    for cond in order:
        d=res[(res.generator==g)&(res.condition==cond)]
        for name,pre in [("V7","v7"),("V7.2","v72"),("V7.3","v73")]:
            gen.append({"generator":g,"condition":cond,"model":name,
                        "deepfake_recall":float((d[f"{pre}_prediction"]=="DEEPFAKE").mean()),
                        "median_df_score":float(np.median(d[f"{pre}_median_df"]))})
pd.DataFrame(gen).to_csv(OUT/"v8_phase2_generator_summary.csv",index=False)
res[res.disagreement].to_csv(OUT/"v8_phase2_disagreements.csv",index=False)

with open(OUT/"v8_phase2_summary.json","w",encoding="utf-8") as f:
    json.dump({"evaluations":len(res),"failures":len(fails),"results":summary,
               "training_performed":False,"models_modified":False},f,indent=2)

print("\n"+"="*92)
print("V8 PHASE 2 FINAL RESULTS")
print("="*92)
for cond in order:
    print("\n"+cond.upper())
    for r in [x for x in summary if x["condition"]==cond]:
        print(f"{r['model']:4s} | Deepfake Recall {r['deepfake_recall']*100:6.2f}% | False Negative {r['false_negative_rate']*100:6.2f}% | Median DF {r['median_df_score']*100:6.2f}% | P90 {r['p90_file_median_df']*100:6.2f}%")
print(f"\nModel disagreements: {int(res.disagreement.sum())}/{len(res)}")
print("Outputs:",OUT)
print("No training performed. No models modified.")
