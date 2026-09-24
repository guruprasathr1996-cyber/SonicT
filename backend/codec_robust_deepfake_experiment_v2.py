from pathlib import Path
import subprocess, tempfile, shutil, warnings, json
import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings('ignore')

SR = 16000
MAX_DURATION = 15
RANDOM_STATE = 42

ROOT = Path(r'H:\SonicT_Compression_Test\asvspoof_balanced')
BONAFIDE_DIR = ROOT / 'bonafide'
SPOOF_DIR = ROOT / 'spoof'
MANIFEST = ROOT / 'manifest.csv'
OUTPUT_DIR = Path(r'H:\SonicT_Compression_Test\experiment_v2_output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PERSONAL_GENUINE_DIR = Path(r'H:\SonicT_Compression_Test\genuine')
DIFFICULT_DEEPFAKE = Path(r'E:\Downloads\recording_6ee192c5-aab6-4a46-8afc-ee9985bb9aaf.mp3.mpeg')

CODECS = {
    'wav16k': ['-ar','16000','-ac','1','-c:a','pcm_s16le'],
    'mp3_8k_16k': ['-ar','8000','-ac','1','-b:a','16k','-c:a','libmp3lame'],
    'mp3_8k_32k': ['-ar','8000','-ac','1','-b:a','32k','-c:a','libmp3lame'],
    'mp3_16k_32k': ['-ar','16000','-ac','1','-b:a','32k','-c:a','libmp3lame'],
    'aac_16k_32k': ['-ar','16000','-ac','1','-b:a','32k','-c:a','aac'],
    'opus_16k_24k': ['-ar','16000','-ac','1','-b:a','24k','-c:a','libopus'],
}

FEATURE_NAMES = [
    'spectral_centroid','spectral_bandwidth','rolloff_95','spectral_flatness','zcr',
    'f0_median_hz','f0_std_hz','voiced_ratio','jitter','shimmer',
    'tonality_ratio','noise_ratio','effective_bandwidth',
    'energy_0_1khz_ratio','energy_1_3khz_ratio','energy_3_4khz_ratio',
    'mfcc_std_mean','mfcc_delta_abs_mean','mfcc_delta2_abs_mean'
] + [f'mfcc_{i}_mean' for i in range(1,21)]

TEMP_ROOT = Path(tempfile.mkdtemp(prefix='sonict_codec_v2_'))

def run_ffmpeg(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode == 0, r.stderr

def encode_codec(inp, out, args):
    return run_ffmpeg(['ffmpeg','-y','-loglevel','error','-i',str(inp),'-t',str(MAX_DURATION),*args,str(out)])

def decode_to_wav(inp, out):
    return run_ffmpeg(['ffmpeg','-y','-loglevel','error','-i',str(inp),'-t',str(MAX_DURATION),'-ar',str(SR),'-ac','1','-c:a','pcm_s16le',str(out)])

def cleanup(path):
    try:
        if path.exists(): path.unlink()
    except Exception:
        pass

def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_DURATION)
    y = np.asarray(y, dtype=np.float32)
    if len(y) < SR:
        raise ValueError('Audio shorter than 1 second')
    peak = np.max(np.abs(y))
    if peak > 0: y = y / peak
    return y

def extract_features(path):
    y = load_audio(path)
    eps = 1e-10
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.95)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mean_flatness = float(np.mean(flatness))

    try:
        f0 = librosa.yin(y, fmin=60, fmax=500, sr=SR, frame_length=1024, hop_length=512)
        f0 = np.asarray(f0, dtype=np.float32)
        valid = np.isfinite(f0) & (f0 >= 60) & (f0 <= 500)
        vf0 = f0[valid]
        if len(vf0) >= 2:
            f0_median = float(np.median(vf0))
            f0_std = float(np.std(vf0))
            jitter = float(np.mean(np.abs(np.diff(vf0))) / (np.mean(vf0) + eps))
        else:
            f0_median = f0_std = jitter = 0.0
        vrms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
        threshold = max(float(np.median(vrms) * 0.5), 1e-5) if len(vrms) else 1e-5
        voiced_ratio = float(np.mean(vrms > threshold)) if len(vrms) else 0.0
    except Exception:
        f0_median = f0_std = jitter = voiced_ratio = 0.0

    rms = librosa.feature.rms(y=y)[0]
    shimmer = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + eps)) if len(rms) > 1 else 0.0

    # Fast replacement for slow HPSS/harmonic decomposition.
    tonality_ratio = float(np.clip(1.0 - mean_flatness, 0.0, 1.0))
    noise_ratio = float(np.clip(mean_flatness, 0.0, 1.0))

    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    total = np.sum(spectrum) + eps

    def band_ratio(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(np.sum(spectrum[m]) / total)

    e01, e13, e34 = band_ratio(0,1000), band_ratio(1000,3000), band_ratio(3000,4000)
    cumulative = np.cumsum(spectrum)
    idx = min(int(np.searchsorted(cumulative, 0.95 * cumulative[-1])), len(freqs)-1) if cumulative[-1] > 0 else 0
    effective_bandwidth = float(freqs[idx]) if len(freqs) else 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_means = np.mean(mfcc, axis=1)

    features = [
        float(np.mean(centroid)), float(np.mean(bandwidth)), float(np.mean(rolloff)),
        mean_flatness, float(np.mean(zcr)), f0_median, f0_std, voiced_ratio, jitter,
        shimmer, tonality_ratio, noise_ratio, effective_bandwidth, e01, e13, e34,
        float(np.mean(np.std(mfcc, axis=1))), float(np.mean(np.abs(delta))), float(np.mean(np.abs(delta2)))
    ] + mfcc_means.tolist()

    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

if not MANIFEST.exists():
    raise FileNotFoundError(f'Manifest not found: {MANIFEST}')

manifest = pd.read_csv(MANIFEST)
for c in ['speaker','file_id','codec','source','attack','label']:
    if c not in manifest.columns:
        raise RuntimeError(f'Manifest missing column: {c}')
manifest['file_id'] = manifest['file_id'].astype(str)
manifest['speaker'] = manifest['speaker'].astype(str)
metadata = {str(r['file_id']): r for _, r in manifest.iterrows()}
rows = []

def suffix_for(codec_name):
    if 'mp3' in codec_name: return '.mp3'
    if 'aac' in codec_name: return '.aac'
    if 'opus' in codec_name: return '.opus'
    return '.wav'

def process_class(folder, label_name, label_value):
    files = sorted(folder.glob('*.flac'))
    print(f'\nProcessing {label_name}: {len(files)} originals')
    for i, src in enumerate(files, 1):
        file_id = src.stem
        if file_id not in metadata:
            print(f'  Metadata missing: {file_id}')
            continue
        info = metadata[file_id]
        print(f'  [{i}/{len(files)}] {src.name}')
        for codec_name, codec_args in CODECS.items():
            compressed = TEMP_ROOT / f'{file_id}_{codec_name}{suffix_for(codec_name)}'
            decoded = TEMP_ROOT / f'{file_id}_{codec_name}_decoded.wav'
            try:
                ok, err = encode_codec(src, compressed, codec_args)
                if not ok:
                    print(f'    Encode failed {codec_name}: {err.strip()}')
                    continue
                ok, err = decode_to_wav(compressed, decoded)
                if not ok:
                    print(f'    Decode failed {codec_name}: {err.strip()}')
                    continue
                feat = extract_features(decoded)
                row = {
                    'file_id': file_id, 'speaker': str(info['speaker']), 'label': label_value,
                    'label_name': label_name, 'codec_condition': codec_name,
                    'original_codec': str(info['codec']), 'source': str(info['source']), 'attack': str(info['attack'])
                }
                for n, v in zip(FEATURE_NAMES, feat): row[n] = float(v)
                rows.append(row)
            except Exception as e:
                print(f'    Feature error {codec_name}: {e}')
            finally:
                cleanup(compressed); cleanup(decoded)

print('='*76)
print('SONICT CODEC-ROBUST DEEPFAKE EXPERIMENT V2')
print('='*76)
print(f'Analysis duration per file : {MAX_DURATION} sec')
print(f'Codec conditions           : {len(CODECS)}')
print('Expected maximum rows      : 1200')

process_class(BONAFIDE_DIR, 'genuine', 0)
process_class(SPOOF_DIR, 'deepfake', 1)

dataset = pd.DataFrame(rows)
if dataset.empty:
    shutil.rmtree(TEMP_ROOT, ignore_errors=True)
    raise RuntimeError('No feature rows were generated.')

DATASET_FILE = OUTPUT_DIR / 'codec_feature_dataset_v2.csv'
dataset.to_csv(DATASET_FILE, index=False)
print(f'\nFeature extraction complete. Rows={len(dataset)}, Originals={dataset.file_id.nunique()}')

X = dataset[FEATURE_NAMES].values
y = dataset['label'].values
groups = dataset['speaker'].values

split_found = False
for seed in range(RANDOM_STATE, RANDOM_STATE+100):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2:
        split_found = True
        SPLIT_SEED = seed
        break
if not split_found:
    raise RuntimeError('Could not create a valid speaker-grouped split.')

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
train_data, test_data = dataset.iloc[train_idx].copy(), dataset.iloc[test_idx].copy()

print('\n' + '='*76)
print('SPEAKER-GROUPED SPLIT')
print('='*76)
print('Split seed      :', SPLIT_SEED)
print('Train speakers  :', train_data.speaker.nunique())
print('Test speakers   :', test_data.speaker.nunique())
print('Train originals :', train_data.file_id.nunique())
print('Test originals  :', test_data.file_id.nunique())
print('\nTrain originals by class:')
print(train_data.groupby('label_name').file_id.nunique())
print('\nTest originals by class:')
print(test_data.groupby('label_name').file_id.nunique())

models = {
    'Random Forest': RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1, max_features='sqrt'),
    'Extra Trees': ExtraTreesClassifier(n_estimators=500, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1, max_features='sqrt'),
}

results = {}; best_model = None; best_name = None; best_f1 = -1
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[0,1])
    print('\n' + '='*60)
    print(name)
    print(f'Accuracy        : {acc*100:.2f}%')
    print(f'Macro Precision : {p*100:.2f}%')
    print(f'Macro Recall    : {r*100:.2f}%')
    print(f'Macro F1        : {f1*100:.2f}%')
    print('Confusion [genuine=0, deepfake=1]:')
    print(cm)
    print(classification_report(y_test, pred, labels=[0,1], target_names=['genuine','deepfake'], digits=4, zero_division=0))
    results[name] = {'accuracy':float(acc),'macro_precision':float(p),'macro_recall':float(r),'macro_f1':float(f1),'confusion_matrix':cm.tolist()}
    if f1 > best_f1:
        best_f1, best_model, best_name = f1, model, name

print('\n' + '='*76)
print('BEST MODEL:', best_name)
print(f'Macro F1  : {best_f1*100:.2f}%')

importance = pd.DataFrame({'feature':FEATURE_NAMES,'importance':best_model.feature_importances_}).sort_values('importance', ascending=False)
IMPORTANCE_FILE = OUTPUT_DIR / 'codec_feature_importance_v2.csv'
importance.to_csv(IMPORTANCE_FILE, index=False)
print('\nTop 15 features:')
print(importance.head(15).to_string(index=False))

codec_results = []
print('\n' + '='*76)
print('PER-CODEC TEST PERFORMANCE')
print('='*76)
test_codecs = test_data.codec_condition.values
for codec in sorted(test_data.codec_condition.unique()):
    mask = test_codecs == codec
    cp = best_model.predict(X_test[mask])
    acc = accuracy_score(y_test[mask], cp)
    _,_,f1,_ = precision_recall_fscore_support(y_test[mask], cp, average='macro', zero_division=0)
    codec_results.append({'codec_condition':codec,'n':int(mask.sum()),'accuracy':float(acc),'macro_f1':float(f1)})
codec_df = pd.DataFrame(codec_results)
print(codec_df.to_string(index=False))
CODEC_FILE = OUTPUT_DIR / 'codec_condition_results_v2.csv'
codec_df.to_csv(CODEC_FILE, index=False)

MODEL_FILE = OUTPUT_DIR / 'codec_robust_experimental_model_v2.pkl'
FEATURE_FILE = OUTPUT_DIR / 'codec_robust_features_v2.pkl'
joblib.dump(best_model, MODEL_FILE)
joblib.dump(FEATURE_NAMES, FEATURE_FILE)

def prepare_external_wav(path):
    out = TEMP_ROOT / ('external_' + path.stem.replace(' ','_') + '.wav')
    ok, err = decode_to_wav(path, out)
    if not ok: raise RuntimeError(f'FFmpeg decode failed: {err.strip()}')
    return out

def predict_external(path):
    wav = None
    try:
        wav = prepare_external_wav(path)
        feat = extract_features(wav)
        proba = best_model.predict_proba([feat])[0]
        return {
            'file':path.name,
            'prediction':'DEEPFAKE' if proba[1] >= 0.5 else 'GENUINE',
            'genuine_probability':float(proba[0]),
            'deepfake_probability':float(proba[1]),
        }
    finally:
        if wav is not None: cleanup(wav)

external_results = []
personal_genuine_accuracy = None
print('\n' + '='*76)
print('EXTERNAL VALIDATION')
print('='*76)

if PERSONAL_GENUINE_DIR.exists():
    allowed = {'.wav','.mp3','.m4a','.flac','.aac','.ogg','.opus','.mpeg','.mpg','.wma','.amr','.3gp'}
    personal_files = sorted([p for p in PERSONAL_GENUINE_DIR.iterdir() if p.is_file() and p.suffix.lower() in allowed])
    print(f'\nPersonal genuine files: {len(personal_files)}')
    correct = processed = 0
    for i, pth in enumerate(personal_files,1):
        print(f'[{i}/{len(personal_files)}] {pth.name}')
        try:
            res = predict_external(pth); external_results.append(res); processed += 1
            if res['prediction'] == 'GENUINE': correct += 1
            print(f"   Prediction : {res['prediction']}")
            print(f"   Genuine    : {res['genuine_probability']*100:.2f}%")
            print(f"   Deepfake   : {res['deepfake_probability']*100:.2f}%")
        except Exception as e:
            print('   ERROR:', e)
    if processed:
        personal_genuine_accuracy = correct / processed
        print(f'\nPersonal genuine accuracy: {personal_genuine_accuracy*100:.2f}%')
else:
    print('\nPersonal genuine directory not found:', PERSONAL_GENUINE_DIR)

print('\n' + '-'*76)
print('DIFFICULT KNOWN DEEPFAKE')
print('-'*76)
if DIFFICULT_DEEPFAKE.exists():
    try:
        res = predict_external(DIFFICULT_DEEPFAKE); external_results.append(res)
        print('File       :', DIFFICULT_DEEPFAKE.name)
        print('Prediction :', res['prediction'])
        print(f"Genuine    : {res['genuine_probability']*100:.2f}%")
        print(f"Deepfake   : {res['deepfake_probability']*100:.2f}%")
    except Exception as e:
        print('ERROR:', e)
else:
    print('Difficult deepfake file not found:', DIFFICULT_DEEPFAKE)

RESULT_JSON = OUTPUT_DIR / 'experiment_results_v2.json'
with open(RESULT_JSON, 'w', encoding='utf-8') as f:
    json.dump({
        'analysis_duration_seconds':MAX_DURATION,
        'speaker_group_split_seed':SPLIT_SEED,
        'best_model':best_name,
        'best_macro_f1':float(best_f1),
        'models':results,
        'personal_genuine_accuracy':None if personal_genuine_accuracy is None else float(personal_genuine_accuracy),
        'external_validation':external_results,
    }, f, indent=4)

shutil.rmtree(TEMP_ROOT, ignore_errors=True)

print('\n' + '='*76)
print('EXPERIMENT COMPLETE')
print('='*76)
print('Outputs:')
for p in [DATASET_FILE, IMPORTANCE_FILE, CODEC_FILE, MODEL_FILE, FEATURE_FILE, RESULT_JSON]: print(p)
print('\nIMPORTANT: This is still an experimental F1B model. Do NOT integrate it into SonicT yet.')
