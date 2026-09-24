"""
SonicT F1B V9 - External Cross-Corpus Evaluation
NO TRAINING. NO MODEL MODIFICATION. NO F1-F5 MODIFICATION.

Test:
- Existing external genuine-phone recordings
- 100 Wukong/EnvSDD TEST deepfakes:
    34 balanced, 33 bg_dominant, 33 voice_dominant
- Separate difficult compressed challenge (diagnostic)
"""

from pathlib import Path
import subprocess, tempfile, warnings, json, random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WavLMModel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
SEED = 42
SR = 16000
WIN = 6 * SR
MIN_WIN = SR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 6 if DEVICE.type == "cuda" else 2

PHONE_DIR = Path(r"H:\SonicT_F1B_External_Test\genuine_phone")
WUKONG_TEST = Path(r"H:\SonicT_F1B_V9\wukong_fake\wukong_envsdd_4s_fake_voice_real_bg\mixed_4s\test\fake_voice_real_bg")
CHALLENGE = Path(r"E:\Downloads\recording_6ee192c5-aab6-4a46-8afc-ee9985bb9aaf.mp3.mpeg")
OUT = Path(r"H:\SonicT_F1B_V9\results")
OUT.mkdir(parents=True, exist_ok=True)

WAVLM = Path(r"C:\Users\Guruprasath\.cache\huggingface\hub\models--microsoft--wavlm-base-plus\snapshots\4c66d4806a428f2e922ccfa1a962776e232d487b")
V7 = Path(r"H:\SonicT_Compression_Test\experiment_v7_output\wavlm_cross_generator_v7.pt")
V72 = Path(r"H:\SonicT_Compression_Test\experiment_v7_2_symmetric_codec\wavlm_cross_generator_v7_2_symmetric_codec.pt")
V73 = Path(r"H:\SonicT_Compression_Test\experiment_v7_3_telephony_branch\wavlm_acoustic_telephony_antispof_v7_3.pt")

for p, name in [(PHONE_DIR,"phone dir"),(WUKONG_TEST,"Wukong test"),(WAVLM,"WavLM"),(V7,"V7"),(V72,"V7.2"),(V73,"V7.3")]:
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")

AUDIO_EXTS = {".wav",".mp3",".mpeg",".m4a",".flac",".aac",".ogg",".opus",".wma",".webm"}
phone_files = sorted(p for p in PHONE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)

counts = {"balanced":34, "bg_dominant":33, "voice_dominant":33}
selected = []
for cond, n in counts.items():
    files = sorted((WUKONG_TEST / cond).glob("*.wav"))
    if len(files) < n:
        raise RuntimeError(f"{cond}: need {n}, found {len(files)}")
    rng = random.Random(SEED + sum(map(ord, cond)))
    for p in sorted(rng.sample(files, n)):
        selected.append((p, cond))

pd.DataFrame([{"path":str(p),"filename":p.name,"condition":c} for p,c in selected]).to_csv(
    OUT/"v9_selected_wukong_100.csv", index=False
)

print("="*96)
print("SONICT F1B V9 - EXTERNAL CROSS-CORPUS EVALUATION")
print("="*96)
print("Device:", DEVICE)
print("Genuine phone files:", len(phone_files))
print("Wukong deepfakes:", len(selected))
print("Selection:", counts)
print("Training: NONE")

def norm(y):
    y=np.asarray(y,np.float32)
    peak=np.max(np.abs(y)) if len(y) else 0
    return (y/peak if peak>0 else y).astype(np.float32)

def load_audio(path):
    try:
        y,_=librosa.load(str(path),sr=SR,mono=True)
        return norm(y)
    except Exception:
        f=tempfile.NamedTemporaryFile(suffix=".wav",delete=False); tmp=Path(f.name); f.close()
        try:
            subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(path),"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(tmp)],check=True)
            y,_=librosa.load(str(tmp),sr=SR,mono=True)
            return norm(y)
        finally:
            tmp.unlink(missing_ok=True)

def windows(y):
    # Critical for the 4-second Wukong files:
    # use one full-file window rather than skipping them.
    if len(y) < WIN:
        return [y] if len(y) >= MIN_WIN else []
    return [y[i:i+WIN] for i in range(0,len(y),WIN) if len(y[i:i+WIN])>=MIN_WIN]

print("\nLoading local WavLM...")
extractor=AutoFeatureExtractor.from_pretrained(str(WAVLM),local_files_only=True)
wavlm=WavLMModel.from_pretrained(str(WAVLM),local_files_only=True).to(DEVICE)
wavlm.eval()
for p in wavlm.parameters(): p.requires_grad=False

def embeddings(ws):
    out=[]
    for i in range(0,len(ws),BATCH):
        b=ws[i:i+BATCH]
        z=extractor(b,sampling_rate=SR,padding=True,return_attention_mask=True,return_tensors="pt")
        x=z["input_values"].to(DEVICE)
        mask=z.get("attention_mask")
        if mask is not None: mask=mask.to(DEVICE)
        with torch.no_grad():
            h=wavlm(input_values=x,attention_mask=mask).last_hidden_state
            try:
                fm=wavlm._get_feature_vector_attention_mask(h.shape[1],mask) if mask is not None else torch.ones(h.shape[:2],dtype=torch.bool,device=DEVICE)
            except Exception:
                fm=torch.ones(h.shape[:2],dtype=torch.bool,device=DEVICE)
            m=fm.unsqueeze(-1).float(); n=m.sum(1).clamp(min=1)
            mean=(h*m).sum(1)/n
            var=(((h-mean[:,None])**2)*m).sum(1)/n
            out.append(torch.cat([mean,torch.sqrt(var.clamp(min=1e-8))],1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

class MLP(nn.Module):
    def __init__(self,d=1536):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.30),nn.Linear(256,64),nn.ReLU(),nn.Dropout(.20),nn.Linear(64,2))
    def forward(self,x): return self.net(x)

class TelephonyMLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,384),nn.BatchNorm1d(384),nn.ReLU(),nn.Dropout(.35),nn.Linear(384,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.30),nn.Linear(128,32),nn.ReLU(),nn.Dropout(.20),nn.Linear(32,2))
    def forward(self,x): return self.net(x)

def load_std(path):
    c=torch.load(path,map_location=DEVICE,weights_only=False)
    m=MLP().to(DEVICE); m.load_state_dict(c["model_state_dict"]); m.eval()
    mu=np.asarray(c["feature_mean"],np.float32); sd=np.asarray(c["feature_std"],np.float32); sd[sd<1e-6]=1
    return m,mu,sd

print("Loading V7...")
m7,mu7,sd7=load_std(V7)
print("Loading V7.2...")
m72,mu72,sd72=load_std(V72)
print("Loading V7.3...")
c73=torch.load(V73,map_location=DEVICE,weights_only=False)
m73=TelephonyMLP(int(c73["combined_feature_size"])).to(DEVICE)
m73.load_state_dict(c73["model_state_dict"]); m73.eval()
mu73w=np.asarray(c73["wavlm_mean"],np.float32); sd73w=np.asarray(c73["wavlm_std"],np.float32)
mu73a=np.asarray(c73["acoustic_mean"],np.float32); sd73a=np.asarray(c73["acoustic_std"],np.float32)
sd73w[sd73w<1e-6]=1; sd73a[sd73a<1e-6]=1

def band(y,lo,hi):
    S=np.abs(librosa.stft(y,n_fft=2048,hop_length=512))**2
    f=librosa.fft_frequencies(sr=SR,n_fft=2048)
    return float(S[(f>=lo)&(f<hi)].sum()/(S.sum()+1e-12))

def acoustic(y):
    ce=librosa.feature.spectral_centroid(y=y,sr=SR)[0]
    bw=librosa.feature.spectral_bandwidth(y=y,sr=SR)[0]
    ro=librosa.feature.spectral_rolloff(y=y,sr=SR,roll_percent=.85)[0]
    fl=librosa.feature.spectral_flatness(y=y)[0]
    z=librosa.feature.zero_crossing_rate(y)[0]
    rms=librosa.feature.rms(y=y)[0]
    mf=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=13)
    f=[ce.mean(),ce.std(),bw.mean(),bw.std(),ro.mean(),ro.std(),fl.mean(),fl.std(),z.mean(),z.std(),rms.mean(),rms.std(),band(y,0,1000),band(y,1000,3000),band(y,3000,4000),band(y,4000,8000)]
    for x in mf: f += [x.mean(),x.std()]
    return np.asarray(f,np.float32)

def prob(model,x):
    with torch.no_grad():
        t=torch.tensor(x,dtype=torch.float32,device=DEVICE)
        return torch.softmax(model(t),1)[:,1].cpu().numpy()

def run_file(path):
    y=load_audio(path); ws=windows(y)
    if not ws: raise RuntimeError("audio shorter than 1 second")
    E=embeddings(ws)
    p7=prob(m7,(E-mu7)/sd7)
    p72=prob(m72,(E-mu72)/sd72)
    A=np.stack([acoustic(w) for w in ws])
    X73=np.concatenate([(E-mu73w)/sd73w,(A-mu73a)/sd73a],1)
    p73=prob(m73,X73)
    def s(p):
        return {"pred":int(np.median(p)>=.5),"median":float(np.median(p)),"p90":float(np.percentile(p,90)),"max":float(np.max(p))}
    return len(ws),s(p7),s(p72),s(p73)

jobs=[(p,0,"genuine_phone","") for p in phone_files]+[(p,1,"wukong_deepfake",c) for p,c in selected]
rows=[]; fails=[]

print("\nEvaluating benchmark...")
for i,(p,t,src,cond) in enumerate(jobs,1):
    try:
        nw,a,b,c=run_file(p)
        rows.append({"path":str(p),"filename":p.name,"truth":t,"source":src,"condition":cond,"windows":nw,
                     "v7_prediction":a["pred"],"v7_median_df":a["median"],"v7_p90_df":a["p90"],"v7_max_df":a["max"],
                     "v72_prediction":b["pred"],"v72_median_df":b["median"],"v72_p90_df":b["p90"],"v72_max_df":b["max"],
                     "v73_prediction":c["pred"],"v73_median_df":c["median"],"v73_p90_df":c["p90"],"v73_max_df":c["max"]})
    except Exception as e:
        fails.append({"path":str(p),"error":str(e)})
        print("FAILED:",p.name,"->",e)
    if i%10==0 or i==len(jobs): print(f"  {i}/{len(jobs)}")

df=pd.DataFrame(rows)
df["majority_prediction"]=((df.v7_prediction+df.v72_prediction+df.v73_prediction)>=2).astype(int)
df.to_csv(OUT/"v9_external_cross_corpus_results.csv",index=False)
if fails: pd.DataFrame(fails).to_csv(OUT/"v9_failures.csv",index=False)

def metrics(col):
    y=df.truth.to_numpy(); p=df[col].to_numpy()
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {"accuracy":accuracy_score(y,p),"balanced_accuracy":balanced_accuracy_score(y,p),
            "precision":precision_score(y,p,zero_division=0),"deepfake_recall":recall_score(y,p,zero_division=0),
            "genuine_recall":tn/(tn+fp),"f1":f1_score(y,p,zero_division=0),
            "fpr":fp/(fp+tn),"fnr":fn/(fn+tp),"tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn)}

models={"V7":"v7_prediction","V7.2":"v72_prediction","V7.3":"v73_prediction","2-of-3 majority":"majority_prediction"}
summary={n:metrics(c) for n,c in models.items()}

print("\n"+"="*96)
print("V9 EXTERNAL CROSS-CORPUS RESULTS")
print("="*96)
for n,m in summary.items():
    print(f"\n{n}")
    print(f"Accuracy       : {m['accuracy']*100:6.2f}%")
    print(f"Balanced Acc   : {m['balanced_accuracy']*100:6.2f}%")
    print(f"Precision      : {m['precision']*100:6.2f}%")
    print(f"Deepfake Recall: {m['deepfake_recall']*100:6.2f}%")
    print(f"Genuine Recall : {m['genuine_recall']*100:6.2f}%")
    print(f"F1             : {m['f1']*100:6.2f}%")
    print(f"False Positive : {m['fpr']*100:6.2f}%")
    print(f"False Negative : {m['fnr']*100:6.2f}%")
    print(f"TP/TN/FP/FN    : {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")

print("\n"+"="*96)
print("WUKONG DEEPFAKE RECALL BY CONDITION")
print("="*96)
fake=df[df.truth==1]
condition_summary={}
for cond in counts:
    d=fake[fake.condition==cond]
    condition_summary[cond]={}
    print(f"\n{cond} (n={len(d)})")
    for n,col in models.items():
        r=float(d[col].mean()) if len(d) else 0
        condition_summary[cond][n]=r
        print(f"  {n:18s}: {r*100:6.2f}%")

challenge=None
if CHALLENGE.exists():
    print("\n"+"="*96)
    print("SEPARATE DIFFICULT CHALLENGE")
    print("="*96)
    nw,a,b,c=run_file(CHALLENGE)
    rr={"V7":a,"V7.2":b,"V7.3":c}
    votes=sum(x["pred"] for x in rr.values())
    for n,x in rr.items():
        print(f"{n:5s}: {'DEEPFAKE' if x['pred'] else 'GENUINE':8s} | Median {x['median']*100:6.2f}% | P90 {x['p90']*100:6.2f}% | Max {x['max']*100:6.2f}%")
    print(f"2-of-3 majority: {'DEEPFAKE' if votes>=2 else 'GENUINE'} ({votes}/3 votes)")
    challenge={"windows":nw,"models":rr,"deepfake_votes":votes,"majority":"DEEPFAKE" if votes>=2 else "GENUINE"}

with open(OUT/"v9_summary.json","w",encoding="utf-8") as f:
    json.dump({"successful":len(df),"failures":len(fails),"genuine_phone":int((df.truth==0).sum()),
               "wukong_deepfake":int((df.truth==1).sum()),"metrics":summary,
               "condition_recall":condition_summary,"challenge":challenge,
               "training_performed":False,"models_modified":False},f,indent=2)

print("\nOutputs:",OUT)
print("Successful:",len(df)," Failures:",len(fails))
print("No training performed. No model files modified.")
