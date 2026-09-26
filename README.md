# SonicT — Intelligent Voice Forensics & Identity Protection

SonicT is an explainable AI-powered audio forensic platform designed to detect voice-cloning, audio manipulation and replay-based impersonation attacks. The project combines a React web interface, a FastAPI inference backend and an Android call-monitoring application in one repository.

## Live Demo

- Web application: https://sonic-t.vercel.app
- Offline interactive demo: Select **View Interactive Demo** on the sign-in page.
- Source code: https://github.com/guruprasathr1996-cyber/SonicT

The offline demo uses predefined examples and remains available when the AI backend is offline. Real audio analysis requires the FastAPI backend.

## Five-Layer Forensic Analysis

| Feature | Purpose | Implementation |
|---|---|---|
| F1 | Voice-clone detection | WavLM with attention classifier |
| F2 | Spectrogram artifact detection | Log-Mel spectrogram CNN |
| F3 | Voice-feature analysis | MFCC and acoustic features with Random Forest |
| F4 | Tampering and splicing detection | Temporal CNN/CRNN with suspicious-region localization |
| F5 | Replay-attack detection | Replay CNN trained using replay-attack audio |

The individual feature probabilities are combined by a fusion model to produce the final classification, confidence, risk score and recommended action.

## Repository Structure

```text
SonicT/
├── frontend/   React and Vite web application
├── backend/    FastAPI API and SonicT inference pipeline
├── android/    SonicT Call Guard Android application
└── README.md
```

## Supported Results

SonicT classifies audio into:

- Genuine
- Deepfake
- Tampered
- Replay

The live analysis response includes all F1–F5 probabilities, final confidence, overall risk score, warning level, evidence disagreement alerts and verification recommendations.

## Run the Backend

Model weights are intentionally not stored in this Git repository because of their size. Configure the model paths before starting the API.

```bat
cd backend
pip install -r requirements.txt
set SONICT_API_KEY=sonict-demo-2026
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open the API documentation at:

```text
http://127.0.0.1:8000/docs
```

For mobile testing over the internet, expose port 8000 through a secure HTTPS tunnel and enter that address in the Android application.

## Run the Web Application

```bat
cd frontend
npm install
npm run dev
```

The production frontend is deployed through Vercel with `frontend` configured as the Root Directory.

## Run the Android Application

1. Open the `android` directory in Android Studio.
2. Allow Gradle synchronization to finish.
3. Connect an Android phone with USB debugging enabled.
4. Run the application.
5. Enter the HTTPS backend URL and API key.
6. Start five-second speaker-mode monitoring.
7. Expand the notification or tap **Refresh analysed result** to view F1–F5 scores.

Android security restrictions prevent ordinary applications from directly capturing the protected cellular call stream. SonicT therefore uses consent-based speaker-mode microphone monitoring.

## Evaluation

On the balanced 164-file evaluation set, the final fusion system achieved:

- Accuracy: 84.76%
- Macro Precision: 84.92%
- Macro Recall: 84.76%
- Macro F1-score: 84.66%

These results describe the evaluated prototype and should not be interpreted as a guarantee for every device, language, codec or recording environment.

## Technology Stack

- Frontend: React, Vite, JavaScript and CSS
- Backend: Python, FastAPI, PyTorch, Transformers, Librosa and scikit-learn
- Android: Kotlin and Android SDK
- Deployment: Vercel for the frontend and an HTTPS tunnel for the development backend

## Responsible Use

SonicT is a research and prototype forensic support system. Its output should be treated as decision-support evidence, not as the sole basis for legal, financial or identity decisions. Suspicious or conflicting results should be verified through an independent trusted channel.
