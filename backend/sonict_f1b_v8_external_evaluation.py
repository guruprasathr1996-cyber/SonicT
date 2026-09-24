# SonicT F1B V8 - external genuine-phone evaluation
from pathlib import Path
import json, tempfile, subprocess, warnings
import numpy as np, pandas as pd, librosa, torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WavLMModel
warnings.filterwarnings("ignore")

SR=16000; WIN=6*SR; MIN=SR
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT=Path(r"H:\SonicT_F1B_External_Test\genuine_phone")
OUT=Path(r"H:\SonicT_F1B_External_Test\v8_results"); OUT.mkdir(parents=True,exist_ok=True)
WAVLM=Path(r"C:\Users\Guruprasath\.cache\huggingface\hub\models--microsoft--wavlm-base-plus\snapshots\4c66d4806a428f2e922ccfa1a962776e232d487b")
V7=Path(r"H:\SonicT_Compression_Test\experiment_v7_output\wavlm_cross_generator_v7.pt")
V72=Path(r"H:\SonicT_Compression_Test\experiment_v7_2_symmetric_codec\wavlm_cross_generator_v7_2_symmetric_codec.pt")
V73=Path(r"H:\SonicT_Compression_Test\experiment_v7_3_telephony_branch\wavlm_acoustic_telephony_antispof_v7_3.pt")
EXT={".wav",".mp3",".mpeg",".mpg",".m4a",".flac",".aac",".ogg",".wma",".opus",".webm",".mov",".mkv",".avi",".3gp",".3g2",".aiff",".aif",".caf",".amr"}

for p,n in [(ROOT,"phone folder"),(WAVLM,"WavLM"),(V7,"V7"),(V72,"V7.2"),(V73,"V7.3")]:
    if not p.exists(): raise FileNotFoundError(f"{n} not found: {p}")
files=sorted(p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
if not files: raise RuntimeError("No supported audio files found.")

def norm(y):
    y=np.asarray(y,np.float32); peak=np.max(np.abs(y)) if len(y) else 0
    return (y/peak if peak>0 else y).astype(np.float32)

def load_audio(p):
    try: y,_=librosa.load(str(p),sr=SR,mono=True)
    except Exception:
        f=tempfile.NamedTemporaryFile(suffix=".wav",delete=False); q=Path(f.name); f.close()
        try:
            subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(p),"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(q)],check=True)
            y,_=librosa.load(str(q),sr=SR,mono=True)
        finally: q.unlink(missing_ok=True)
    y=norm(y)
    if len(y)<MIN: raise ValueError("audio shorter than 1 second")
    return y

def windows(y):
    return [y[i:i+WIN] for i in range(0,len(y),WIN) if len(y[i:i+WIN])>=MIN]

print("Loading local WavLM...")
fe=AutoFeatureExtractor.from_pretrained(str(WAVLM),local_files_only=True)
wm=WavLMModel.from_pretrained(str(WAVLM),local_files_only=True).to(DEVICE); wm.eval()
for p in wm.parameters(): p.requires_grad=False

def embed(ws):
    out=[]
    for i in range(0,len(ws),2 if DEVICE.type=="cpu" else 6):
        b=ws[i:i+(2 if DEVICE.type=="cpu" else 6)]
        z=fe(b,sampling_rate=SR,padding=True,return_attention_mask=True,return_tensors="pt")
        x=z["input_values"].to(DEVICE); mask=z.get("attention_mask")
        if mask is not None: mask=mask.to(DEVICE)
        with torch.no_grad():
            h=wm(input_values=x,attention_mask=mask).last_hidden_state
            try: fm=wm._get_feature_vector_attention_mask(h.shape[1],mask) if mask is not None else torch.ones(h.shape[:2],dtype=torch.bool,device=DEVICE)
            except: fm=torch.ones(h.shape[:2],dtype=torch.bool,device=DEVICE)
            m=fm.unsqueeze(-1).float(); c=m.sum(1).clamp(min=1)
            mean=(h*m).sum(1)/c; var=(((h-mean[:,None])**2)*m).sum(1)/c
            out.append(torch.cat([mean,torch.sqrt(var.clamp(min=1e-8))],1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)
class MLP(nn.Module):
    def __init__(self, d=1536):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d, 256),       # net.0
            nn.BatchNorm1d(256),     # net.1
            nn.ReLU(),               # net.2
            nn.Dropout(0.30),        # net.3

            nn.Linear(256, 64),      # net.4
            nn.ReLU(),               # net.5
            nn.Dropout(0.20),        # net.6

            nn.Linear(64, 2),        # net.7
        )

    def forward(self, x):
        return self.net(x)
class TMLP(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,384),nn.BatchNorm1d(384),nn.ReLU(),nn.Dropout(.35),nn.Linear(384,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,2))
    def forward(self,x): return self.net(x)

def ckpt(p): return torch.load(p,map_location=DEVICE,weights_only=False)
def state(c):
    for k in ("model_state_dict","state_dict","classifier_state_dict","mlp_state_dict"):
        if k in c: return c[k]
    return c
def arr(c,names,d):
    for n in names:
        if n in c:
            a=np.asarray(c[n],np.float32).reshape(-1)
            if len(a)==d:return a
    return None
def load_old(p,name):
    c=ckpt(p); s=state(c); m=MLP().to(DEVICE)
    try:m.load_state_dict(s)
    except:
        s={k[7:] if k.startswith("module.") else k:v for k,v in s.items()}; m.load_state_dict(s)
    mean=arr(c,("feature_mean","train_mean","mean","wavlm_mean","scaler_mean"),1536)
    std=arr(c,("feature_std","train_std","std","wavlm_std","scaler_std"),1536)
    if mean is None or std is None:
        print(name,"checkpoint keys:",list(c.keys()))
        raise KeyError(f"{name} normalization arrays not found; send the printed keys.")
    std[std<1e-6]=1; m.eval(); return m,mean,std

m7,mu7,sd7=load_old(V7,"V7"); m72,mu72,sd72=load_old(V72,"V7.2")
c73=ckpt(V73); m73=TMLP(int(c73["combined_feature_size"])).to(DEVICE); m73.load_state_dict(state(c73)); m73.eval()
mw=np.asarray(c73["wavlm_mean"],np.float32); sw=np.asarray(c73["wavlm_std"],np.float32); ma=np.asarray(c73["acoustic_mean"],np.float32); sa=np.asarray(c73["acoustic_std"],np.float32)
sw[sw<1e-6]=1; sa[sa<1e-6]=1

def band(y,a,b):
    S=np.abs(librosa.stft(y,n_fft=2048,hop_length=512))**2; f=librosa.fft_frequencies(sr=SR,n_fft=2048)
    return float(S[(f>=a)&(f<b)].sum()/(S.sum()+1e-12))
def acoustic(y):
    cen=librosa.feature.spectral_centroid(y=y,sr=SR)[0]; bw=librosa.feature.spectral_bandwidth(y=y,sr=SR)[0]
    ro=librosa.feature.spectral_rolloff(y=y,sr=SR,roll_percent=.85)[0]; fl=librosa.feature.spectral_flatness(y=y)[0]
    z=librosa.feature.zero_crossing_rate(y)[0]; r=librosa.feature.rms(y=y)[0]; mf=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=13)
    v=[cen.mean(),cen.std(),bw.mean(),bw.std(),ro.mean(),ro.std(),fl.mean(),fl.std(),z.mean(),z.std(),r.mean(),r.std(),band(y,0,1000),band(y,1000,3000),band(y,3000,4000),band(y,4000,8000)]
    for x in mf:v += [x.mean(),x.std()]
    return np.asarray(v,np.float32)
def pred(m,x):
    with torch.no_grad(): return torch.softmax(m(torch.tensor(x,dtype=torch.float32,device=DEVICE)),1)[:,1].cpu().numpy()
def summ(p):
    return dict(median=float(np.median(p)),mean=float(np.mean(p)),p90=float(np.percentile(p,90)),maximum=float(np.max(p)),prediction="DEEPFAKE" if np.median(p)>=.5 else "GENUINE")

rows=[]; fails=[]
print(f"\nEvaluating {len(files)} known-genuine external phone recordings...")
for i,p in enumerate(files,1):
    try:
        y=load_audio(p); ws=windows(y); e=embed(ws)
        p7=pred(m7,(e-mu7)/sd7); p72=pred(m72,(e-mu72)/sd72)
        a=np.stack([acoustic(w) for w in ws]); p73=pred(m73,np.concatenate([(e-mw)/sw,(a-ma)/sa],1))
        s7,s72,s73=map(summ,(p7,p72,p73)); pr=[s7["prediction"],s72["prediction"],s73["prediction"]]
        rows.append({"file":str(p.relative_to(ROOT)),"v7_prediction":pr[0],"v7_median_df":s7["median"],"v7_p90_df":s7["p90"],"v72_prediction":pr[1],"v72_median_df":s72["median"],"v72_p90_df":s72["p90"],"v73_prediction":pr[2],"v73_median_df":s73["median"],"v73_p90_df":s73["p90"],"deepfake_votes":sum(x=="DEEPFAKE" for x in pr),"disagreement":len(set(pr))>1})
        print(f"[{i:02d}/{len(files)}] {p.name} | V7 {s7['median']*100:.2f}% | V7.2 {s72['median']*100:.2f}% | V7.3 {s73['median']*100:.2f}%")
    except Exception as e:fails.append({"file":str(p),"error":str(e)}); print("FAILED",p.name,e)

df=pd.DataFrame(rows); df.to_csv(OUT/"v8_external_genuine_file_results.csv",index=False)
if fails:pd.DataFrame(fails).to_csv(OUT/"v8_failures.csv",index=False)
summary=[]
for name,pre in [("V7","v7"),("V7.2","v72"),("V7.3","v73")]:
    acc=float((df[f"{pre}_prediction"]=="GENUINE").mean()); med=df[f"{pre}_median_df"].to_numpy()
    summary.append({"model":name,"files":len(df),"genuine_accuracy":acc,"false_deepfake_rate":1-acc,"median_df_score":float(np.median(med)),"p90_file_median_df":float(np.percentile(med,90))})
sdf=pd.DataFrame(summary); sdf.to_csv(OUT/"v8_model_summary.csv",index=False)
df[df.disagreement].to_csv(OUT/"v8_model_disagreements.csv",index=False)
with open(OUT/"v8_external_genuine_summary.json","w") as f:json.dump({"files":len(df),"failures":len(fails),"models":summary,"disagreements":int(df.disagreement.sum()),"training_performed":False},f,indent=2)

print("\n"+"="*86+"\nV8 EXTERNAL GENUINE-PHONE RESULTS\n"+"="*86)
for r in summary:print(f"{r['model']:4s} | Genuine {r['genuine_accuracy']*100:.2f}% | False DF {r['false_deepfake_rate']*100:.2f}% | Median DF {r['median_df_score']*100:.2f}% | P90 {r['p90_file_median_df']*100:.2f}%")
print(f"Model disagreements: {int(df.disagreement.sum())}/{len(df)}")
print("Outputs:",OUT)
print("No model was trained or modified.")
