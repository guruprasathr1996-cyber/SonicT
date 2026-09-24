# SonicT V8 Phase 2 launcher/evaluator
# IMPORTANT: no training/model modification.
# Uses the already corrected V8 evaluator logic, with controlled FFmpeg conditions.
from pathlib import Path
import subprocess, pandas as pd, shutil

ROOT=Path(r"H:\SonicT_F1B_External_Test\compressed_deepfake")
MANIFEST=ROOT/"v8_deepfake_manifest.csv"
OUT=Path(r"H:\SonicT_F1B_External_Test\v8_phase2_conditioned")
OUT.mkdir(parents=True,exist_ok=True)

if not MANIFEST.exists():
    raise FileNotFoundError(MANIFEST)

df=pd.read_csv(MANIFEST)
print("="*80)
print("SONICT V8 PHASE 2 - CONDITION PREPARATION")
print("="*80)
print("Source deepfakes:",len(df))

conditions={
 "original":[],
 "narrowband_8k":[],
 "telephone_band":[],
 "mp3_16k":[],
 "telephone_mp3_16k":[]
}

rows=[]
for i,r in df.iterrows():
    src=Path(r["destination"])
    gen=str(r["generator"])
    if not src.exists():
        print("MISSING:",src); continue
    safe="".join(c if c.isalnum() or c in "-_" else "_" for c in (gen+"__"+src.stem))
    jobs=[
      ("narrowband_8k", OUT/(safe+"__narrowband8k.wav"),
       ["-ac","1","-ar","8000","-c:a","pcm_s16le"]),
      ("telephone_band", OUT/(safe+"__telephone.wav"),
       ["-ac","1","-ar","16000","-af","highpass=f=300,lowpass=f=3400","-c:a","pcm_s16le"]),
      ("mp3_16k", OUT/(safe+"__mp3_16k.mp3"),
       ["-ac","1","-ar","16000","-b:a","16k","-c:a","libmp3lame"]),
      ("telephone_mp3_16k", OUT/(safe+"__telephone_mp3_16k.mp3"),
       ["-ac","1","-ar","8000","-af","highpass=f=300,lowpass=f=3400","-b:a","16k","-c:a","libmp3lame"])
    ]
    rows.append({"source":str(src),"condition":"original","conditioned_path":str(src),"generator":gen})
    for name,dst,args in jobs:
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(src),"-vn"]+args+[str(dst)],check=True)
        rows.append({"source":str(src),"condition":name,"conditioned_path":str(dst),"generator":gen})
    print(f"[{i+1:02d}/{len(df)}] prepared {gen}")

m=pd.DataFrame(rows)
m.to_csv(OUT/"v8_phase2_condition_manifest.csv",index=False)
print("\nPrepared evaluations:",len(m))
print("Expected:",len(df)*5)
print("Manifest:",OUT/"v8_phase2_condition_manifest.csv")
print("\nNEXT: use the full V8 model evaluator on this manifest.")
