# SonicT Call Guard — Hybrid Android Prototype

This is a separate Android companion application. It does not modify the existing SonicT React/FastAPI project.

## Implemented

- Android call-screening role for normal incoming cellular calls.
- Offline caller lookup: number, saved-contact name, contact type and coarse country prefix.
- Offline Dynamic Risk Score using contact status, Android number verification, local block list and incident history.
- High-priority risk notification with **Report** and **Block future calls** actions.
- Optional automatic rejection for critical-risk callers.
- Consent-based microphone/speaker monitoring in five-second, 16 kHz mono chunks.
- Live connection to the existing SonicT FastAPI F1–F5 pipeline over local Wi-Fi.
- Automatic fallback to metadata screening when the analysis server is unavailable.
- Optional offline TFLite adapter for a future compatible mobile model.
- Audio chunks are processed in memory and are not stored by the Android app.

## Important limitation

A normal Play Store application cannot directly capture both sides of a cellular call. Speaker-mode analysis records the phone microphone and may vary by device. Reliable two-way audio analysis requires an in-app VoIP/SIP call or privileged/OEM integration.

## Where to place it

Keep this project in its own folder, for example:

```text
H:\SonicT_Public_Android
```

Do not place it inside `H:\SonicT_API` unless you intentionally want a monorepo.

## Run

1. Extract this update over the existing `H:\SonicT_Public_Android` folder.
2. Connect the Android 10+ phone with USB debugging enabled.
3. Double-click `APPLY_UPDATE_AND_INSTALL.bat`. It moves any old `java\in`
   sources to `legacy_source_backup`, then builds and installs the corrected app.
4. If needed, open the folder in Android Studio and allow Gradle to synchronize.
5. Grant Contacts, Microphone and Notification permissions.
6. Tap **Enable SonicT call screening** and select SonicT.
7. Call the phone from another number to test metadata risk.

## Connect the current SonicT models (recommended demo path)

1. Double-click `START_SONICT_SERVER.bat`, or start the existing API manually:

```powershell
cd H:\SonicT_API
uvicorn app:app --host 0.0.0.0 --port 8000
```

2. Keep the phone and laptop on the same Wi-Fi or phone hotspot.
3. Run `ipconfig` on the laptop and note its Wi-Fi IPv4 address.
4. In the Android app, enter `http://LAPTOP_IP:8000` and the API key, then tap
   **Save analysis connection**. The default development key is `sonict-demo-2026`.
5. Allow Python/port 8000 through Windows Firewall if prompted.
6. Put a test call on speaker and tap **Start 5-second SonicT monitoring**.

The app posts in-memory WAV chunks to `/analyze-live-chunk` and shows the final
classification and Voice Integrity Risk in its foreground notification.

## Optional true offline audio model

1. Export a mobile-sized, quantized SonicT model with this example contract:
   - Input: `[1, 80000]` Float32 PCM (five seconds at 16 kHz).
   - Output: `[1, 1]` deepfake probability from 0 to 1.
2. Create `app/src/main/assets/`.
3. Copy the model as `app/src/main/assets/sonict_mobile.tflite`.
4. If your model uses spectrograms or embeddings instead of waveform PCM, update `MobileModelRunner.kt` preprocessing and tensor shapes.

Without a compatible TFLite model, AI audio analysis requires the laptop backend.
When neither is available, the application clearly uses metadata-only screening;
it never invents a deepfake score.

## Production work still required

- Replace `SharedPreferences` with encrypted Room storage if retaining case history.
- Add signed model updates and model-version display.
- Calibrate risk thresholds on real phone/speaker data.
- Add a user-accessible delete/export data screen.
- Complete Play policy, privacy disclosure and local call-recording law review.
- Add an in-app VoIP route if reliable real-time two-way analysis is required.
