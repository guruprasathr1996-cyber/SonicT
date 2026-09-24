@echo off
setlocal
cd /d H:\SonicT_API
if errorlevel 1 (
  echo SonicT API folder was not found at H:\SonicT_API
  echo Edit this file if your API is stored elsewhere.
  pause
  exit /b 1
)

if "%SONICT_API_KEY%"=="" set "SONICT_API_KEY=sonict-demo-2026"
echo Starting SonicT at http://0.0.0.0:8000
echo Keep this window open while the Android app is monitoring.
python -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
