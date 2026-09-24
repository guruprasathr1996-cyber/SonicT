from pathlib import Path
import subprocess, tempfile, shutil, warnings, json, re, random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

warnings.filterwarnings("ignore")
SEED=42; SR=16000; WINDOW_SECONDS=3.0; WINDOW_SAMPLES=int(SR*WINDOW_SECONDS)
N_MELS=64; N_FFT=1024; HOP_LENGTH=256; MAX_WINDOWS_PER_FILE=5
EPOCHS=18; BATCH_SIZE=32; LR=1e-3; PATIENCE=4
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

ASV_ROOT=Path(r"H:\SonicT_Compression_Test\asvspoof_balanced")
BONAFIDE_DIR=ASV_ROOT/"bonafide"; SPOOF_DIR=ASV_ROOT/"spoof"; MANIFEST=ASV_ROOT/"manifest.csv"
PHONE_DIR=Path(r"H:\SonicT_Compression_Test\genuine")
CHALLENGE=Path(r"E:\Downloads\recording_6ee192c5-aab6-4a46-8afc-ee9985bb9aaf.mp3.mpeg")
OUT=Path(r"H:\SonicT_Compression_Test\experiment_v5_output"); OUT.mkdir(parents=True,exist_ok=True)
TMP=Path(tempfile.mkdtemp(prefix="sonict_v5_"))

CONDITIONS={
"clean16k":[],
"mp3_8k_16k":["-ar","8000","-ac","1","-b:a","16k","-c:a","libmp3lame"],
"mp3_16k_32k":["-ar","16000","-ac","1","-b:a","32k","-c:a","libmp3lame"],
"aac_16k_32k":["-ar","16000","-ac","1","-b:a","32k","-c:a","aac"],
"opus_16k_24k":["-ar","16000","-ac","1","-b:a","24k","-c:a","libopus"],
"telephone_bandpass":["-af","highpass=f=300,lowpass=f=3400","-ar","16000","-ac","1","-c:a","pcm_s16le"],
"telephone_mp3_8k_16k":["-af","highpass=f=300,lowpass=f=3400","-ar","8000","-ac","1","-b:a","16k","-c:a","libmp3lame"],
}
PHONE_CONDS=["clean16k","mp3_8k_16k","telephone_bandpass","telephone_mp3_8k_16k"]
EXTS={".wav",".mp3",".m4a",".flac",".aac",".ogg",".opus",".mpeg",".mpg",".wma",".amr",".3gp"}

def ff(cmd):
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return r.returncode==0,r.stderr
def rm(p):
    try:
        if Path(p).exists(): Path(p).unlink()
    except: pass
def suffix(c):
    if "mp3" in c:return ".mp3"
    if "aac" in c:return ".aac"
    if "opus" in c:return ".opus"
    return ".wav"
def transform(src,cond,outwav):
    mid=None
    try:
        if cond=="clean16k":
            return ff(["ffmpeg","-y","-loglevel","error","-i",str(src),"-ar",str(SR),"-ac","1","-c:a","pcm_s16le",str(outwav)])
        mid=outwav.with_suffix(suffix(cond))
        ok,e=ff(["ffmpeg","-y","-loglevel","error","-i",str(src)]+CONDITIONS[cond]+[str(mid)])
        if not ok:return False,e
        return ff(["ffmpeg","-y","-loglevel","error","-i",str(mid),"-ar",str(SR),"-ac","1","-c:a","pcm_s16le",str(outwav)])
    finally:
        if mid is not None and mid!=outwav: rm(mid)

def speaker_phone(p):
    n=re.sub(r"_\d{6}_\d{6}$","",Path(p).stem)
    if n.lower().startswith("call "):n=n[5:]
    n=re.sub(r"[^a-z0-9 ]+"," ",n.lower().strip())
    n=re.sub(r"\s+","_",n).strip("_")
    return "phone::"+(n or Path(p).stem.lower())

def specs(path,maxw=None):
    y,_=librosa.load(path,sr=SR,mono=True)
    y=np.asarray(y,np.float32)
    if len(y)<SR: raise ValueError("audio shorter than 1 second")
    peak=np.max(np.abs(y))
    if peak>0:y=y/peak
    ws=[y[s:s+WINDOW_SAMPLES] for s in range(0,len(y)-WINDOW_SAMPLES+1,WINDOW_SAMPLES)]
    if not ws:
        z=np.zeros(WINDOW_SAMPLES,np.float32);z[:len(y)]=y[:WINDOW_SAMPLES];ws=[z]
    if maxw and len(ws)>maxw:
        ids=np.linspace(0,len(ws)-1,maxw,dtype=int);ws=[ws[i] for i in ids]
    out=[]
    for w in ws:
        m=librosa.feature.melspectrogram(y=w,sr=SR,n_fft=N_FFT,hop_length=HOP_LENGTH,n_mels=N_MELS,fmin=20,fmax=SR//2,power=2)
        x=librosa.power_to_db(m+1e-10,ref=np.max)
        x=(x-x.mean())/(x.std()+1e-6);out.append(x.astype(np.float32))
    return out

man=pd.read_csv(MANIFEST); man["file_id"]=man["file_id"].astype(str); man["speaker"]=man["speaker"].astype(str)
meta={str(r.file_id):r for _,r in man.iterrows()}
rows=[]
for name,label,folder in [("genuine",0,BONAFIDE_DIR),("deepfake",1,SPOOF_DIR)]:
    for p in sorted(folder.glob("*.flac")):
        if p.stem in meta: rows.append({"path":p,"file_id":p.stem,"speaker":"asv::"+str(meta[p.stem].speaker),"label":label,"label_name":name})
asv=pd.DataFrame(rows)
g=asv.speaker.values;y=asv.label.values;dummy=np.zeros((len(asv),1))
for split_seed in range(42,142):
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=split_seed).split(dummy,y,g))
    if len(np.unique(y[tr]))==2 and len(np.unique(y[te]))==2:break
atr=asv.iloc[tr].reset_index(drop=True); ate=asv.iloc[te].reset_index(drop=True)

pf=sorted([p for p in PHONE_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
phone=pd.DataFrame([{"path":p,"file_id":p.stem,"speaker":speaker_phone(p),"label":0,"label_name":"genuine"} for p in pf])
sp=sorted(phone.speaker.unique()); rng=np.random.default_rng(SEED);rng.shuffle(sp)
nt=max(1,min(len(sp)-1,int(round(len(sp)*.35)))); tsp=set(sp[:nt])
ptr=phone[~phone.speaker.isin(tsp)].reset_index(drop=True); pte=phone[phone.speaker.isin(tsp)].reset_index(drop=True)

print("="*78);print("SONICT F1B V5 - WINDOW-LEVEL LOG-MEL CNN");print("="*78)
print("Device:",DEVICE);print("ASV train/test:",len(atr),len(ate));print("Phone train/test:",len(ptr),len(pte))

Xtr=[];Ytr=[];Mtr=[];Xte=[];Yte=[];Mte=[];fails=[]
def build(df,conds,X,Y,M,tag):
    for i,r in df.iterrows():
        print(f"[{tag} {i+1}/{len(df)}] {Path(r.path).name}")
        for c in conds:
            w=TMP/f"{tag}_{i}_{c}.wav"
            try:
                ok,e=transform(r.path,c,w)
                if not ok: fails.append({"split":tag,"file":Path(r.path).name,"condition":c,"error":e});continue
                for j,x in enumerate(specs(w,MAX_WINDOWS_PER_FILE)):
                    X.append(x);Y.append(int(r.label));M.append({"file_id":r.file_id,"speaker":r.speaker,"label":int(r.label),"label_name":r.label_name,"condition":c,"window_index":j})
            except Exception as e:fails.append({"split":tag,"file":Path(r.path).name,"condition":c,"error":str(e)})
            finally:rm(w)

print("\nEXTRACTING TRAIN WINDOWS")
build(atr,list(CONDITIONS),Xtr,Ytr,Mtr,"asv_train");build(ptr,PHONE_CONDS,Xtr,Ytr,Mtr,"phone_train")
print("\nEXTRACTING ASV TEST WINDOWS")
build(ate,list(CONDITIONS),Xte,Yte,Mte,"asv_test")
Xtr=np.asarray(Xtr,np.float32);Ytr=np.asarray(Ytr,np.int64);Xte=np.asarray(Xte,np.float32);Yte=np.asarray(Yte,np.int64)
pd.DataFrame(Mtr).to_csv(OUT/"train_window_manifest_v5.csv",index=False);pd.DataFrame(Mte).to_csv(OUT/"asv_test_window_manifest_v5.csv",index=False);pd.DataFrame(fails).to_csv(OUT/"failed_samples_v5.csv",index=False)
print("Train windows:",len(Xtr),"Test windows:",len(Xte),"Failures:",len(fails))

class DS(Dataset):
    def __init__(self,X,y):self.X=torch.from_numpy(X).unsqueeze(1);self.y=torch.from_numpy(y)
    def __len__(self):return len(self.y)
    def __getitem__(self,i):return self.X[i],self.y[i]
tl=DataLoader(DS(Xtr,Ytr),batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
vl=DataLoader(DS(Xte,Yte),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,96,3,padding=1),nn.BatchNorm2d(96),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.c=nn.Sequential(nn.Flatten(),nn.Linear(96*4*4,128),nn.ReLU(),nn.Dropout(.35),nn.Linear(128,2))
    def forward(self,x):return self.c(self.f(x))
model=CNN().to(DEVICE)
cnt=np.bincount(Ytr,minlength=2).astype(np.float32);weights=cnt.sum()/(2*np.maximum(cnt,1))
lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=DEVICE))
opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4)
best=-1;wait=0;hist=[];mp=OUT/"codec_robust_cnn_v5.pt"
def predict(loader):
    model.eval();T=[];P=[];Q=[]
    with torch.no_grad():
        for xb,yb in loader:
            q=torch.softmax(model(xb.to(DEVICE)),1);T+=yb.numpy().tolist();P+=torch.argmax(q,1).cpu().numpy().tolist();Q+=q[:,1].cpu().numpy().tolist()
    return np.array(T),np.array(P),np.array(Q)
print("\nTRAINING CNN")
for ep in range(1,EPOCHS+1):
    model.train();total=0
    for xb,yb in tl:
        xb=xb.to(DEVICE);yb=yb.to(DEVICE);opt.zero_grad();l=lossfn(model(xb),yb);l.backward();opt.step();total+=l.item()*len(yb)
    t,p,q=predict(vl);acc=accuracy_score(t,p);_,_,f1,_=precision_recall_fscore_support(t,p,average="macro",zero_division=0)
    hist.append({"epoch":ep,"loss":total/len(Ytr),"val_accuracy":acc,"val_macro_f1":f1});print(f"Epoch {ep:02d} | Loss {total/len(Ytr):.4f} | Acc {acc*100:.2f}% | F1 {f1*100:.2f}%")
    if f1>best+1e-4:best=f1;wait=0;torch.save({"model_state_dict":model.state_dict(),"best_f1":best},mp)
    else:wait+=1
    if wait>=PATIENCE:print("Early stopping.");break
pd.DataFrame(hist).to_csv(OUT/"training_history_v5.csv",index=False)
model.load_state_dict(torch.load(mp,map_location=DEVICE)["model_state_dict"])

t,p,q=predict(vl);acc=accuracy_score(t,p);pr,rc,f1,_=precision_recall_fscore_support(t,p,average="macro",zero_division=0)
print("\n"+"="*78);print("BEST CNN - ASV WINDOW TEST");print("="*78)
print(f"Accuracy: {acc*100:.2f}%  Macro F1: {f1*100:.2f}%");print(confusion_matrix(t,p,labels=[0,1]));print(classification_report(t,p,target_names=["genuine","deepfake"],digits=4,zero_division=0))

md=pd.DataFrame(Mte);md["deepfake_probability"]=q
fr=[]
for keys,z in md.groupby(["file_id","speaker","label","label_name","condition"]):
    pv=z.deepfake_probability.values;score=float(np.median(pv))
    fr.append({"file_id":keys[0],"speaker":keys[1],"label":keys[2],"label_name":keys[3],"condition":keys[4],"windows":len(pv),"mean":float(np.mean(pv)),"median":score,"max":float(np.max(pv)),"suspicious_ratio":float(np.mean(pv>=.5)),"high_ratio":float(np.mean(pv>=.7)),"prediction":int(score>=.5)})
fd=pd.DataFrame(fr);fa=accuracy_score(fd.label,fd.prediction);_,_,ff1,_=precision_recall_fscore_support(fd.label,fd.prediction,average="macro",zero_division=0)
print("\nASV FILE TEST - MEDIAN AGGREGATION");print(f"Accuracy: {fa*100:.2f}%  Macro F1: {ff1*100:.2f}%");print(confusion_matrix(fd.label,fd.prediction,labels=[0,1]))
fd.to_csv(OUT/"asv_file_results_v5.csv",index=False)
cr=[]
for c in sorted(fd.condition.unique()):
    z=fd[fd.condition==c];a=accuracy_score(z.label,z.prediction);_,_,f,_=precision_recall_fscore_support(z.label,z.prediction,average="macro",zero_division=0);cr.append({"condition":c,"n_files":len(z),"accuracy":a,"macro_f1":f})
cdf=pd.DataFrame(cr);print("\nPER-CODEC FILE PERFORMANCE");print(cdf.to_string(index=False));cdf.to_csv(OUT/"per_codec_file_results_v5.csv",index=False)

def external(path):
    w=TMP/f"ext_{abs(hash(str(path)))}.wav"
    try:
        ok,e=transform(path,"clean16k",w)
        if not ok:raise RuntimeError(e)
        xx=np.asarray(specs(w,None),np.float32);loader=DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(xx).unsqueeze(1)),batch_size=BATCH_SIZE)
        pv=[];model.eval()
        with torch.no_grad():
            for (xb,) in loader:pv+=torch.softmax(model(xb.to(DEVICE)),1)[:,1].cpu().numpy().tolist()
        pv=np.array(pv);med=float(np.median(pv))
        return {"file":Path(path).name,"windows":len(pv),"prediction":"DEEPFAKE" if med>=.5 else "GENUINE","mean":float(np.mean(pv)),"median":med,"max":float(np.max(pv)),"suspicious_ratio":float(np.mean(pv>=.5)),"high_ratio":float(np.mean(pv>=.7)),"window_probabilities":pv.tolist()}
    finally:rm(w)

print("\n"+"="*78);print("HELD-OUT PERSONAL GENUINE SPEAKER TEST");print("="*78)
pres=[];correct=0
for i,r in pte.iterrows():
    z=external(r.path);pres.append(z);correct+=z["prediction"]=="GENUINE";print(f"[{i+1}/{len(pte)}] {Path(r.path).name} | {z['prediction']} | median deepfake {z['median']*100:.2f}% | max {z['max']*100:.2f}%")
pacc=correct/len(pres) if pres else None;print(f"Held-out personal genuine accuracy: {pacc*100:.2f}%")
with open(OUT/"heldout_personal_results_v5.json","w",encoding="utf-8") as f:json.dump(pres,f,indent=2)

challenge=None
print("\n"+"="*78);print("DIFFICULT KNOWN DEEPFAKE - UNSEEN CHALLENGE");print("="*78)
if CHALLENGE.exists():
    challenge=external(CHALLENGE)
    print("File:",challenge["file"]);print("Windows:",challenge["windows"]);print("Prediction:",challenge["prediction"]);print(f"Mean Deepfake: {challenge['mean']*100:.2f}%");print(f"Median Deepfake: {challenge['median']*100:.2f}%");print(f"Maximum Evidence: {challenge['max']*100:.2f}%");print(f"Suspicious Ratio: {challenge['suspicious_ratio']*100:.2f}%");print(f"High-Risk Ratio: {challenge['high_ratio']*100:.2f}%")
    pd.DataFrame({"window_index":range(challenge["windows"]),"start_seconds":np.arange(challenge["windows"])*WINDOW_SECONDS,"end_seconds":(np.arange(challenge["windows"])+1)*WINDOW_SECONDS,"deepfake_probability":challenge["window_probabilities"]}).to_csv(OUT/"challenge_window_evidence_v5.csv",index=False)

summary={"version":"F1B_V5","device":str(DEVICE),"asv_split_seed":split_seed,"best_window_macro_f1":best,"asv_file_macro_f1":ff1,"heldout_phone_genuine_accuracy":pacc,"challenge":challenge,"failed_sample_count":len(fails)}
with open(OUT/"experiment_results_v5.json","w",encoding="utf-8") as f:json.dump(summary,f,indent=2)
shutil.rmtree(TMP,ignore_errors=True)
print("\n"+"="*78);print("F1B V5 EXPERIMENT COMPLETE");print("="*78)
print(f"Best ASV window Macro F1: {best*100:.2f}%");print(f"ASV file Macro F1: {ff1*100:.2f}%");print(f"Held-out phone genuine: {pacc*100:.2f}%")
if challenge:print(f"Challenge median deepfake: {challenge['median']*100:.2f}%");print(f"Challenge max evidence: {challenge['max']*100:.2f}%");print(f"Challenge suspicious: {challenge['suspicious_ratio']*100:.2f}% windows")
print("\nIMPORTANT: V5 is experimental. Do NOT integrate it into SonicT yet.")
