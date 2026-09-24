@echo off
setlocal
cd /d "%~dp0"

if exist "app\src\main\java\in" (
  if not exist "legacy_source_backup" mkdir "legacy_source_backup"
  if exist "legacy_source_backup\main_in" (
    echo A previous source backup already exists at legacy_source_backup\main_in
    echo Rename that backup folder, then run this script again.
    pause
    exit /b 1
  )
  move "app\src\main\java\in" "legacy_source_backup\main_in" >nul
)

if exist "app\src\test\java\in" (
  if not exist "legacy_source_backup" mkdir "legacy_source_backup"
  if exist "legacy_source_backup\test_in" (
    echo A previous test backup already exists at legacy_source_backup\test_in
    echo Rename that backup folder, then run this script again.
    pause
    exit /b 1
  )
  move "app\src\test\java\in" "legacy_source_backup\test_in" >nul
)

if not exist "gradlew.bat" (
  echo gradlew.bat was not found. Open this project in Android Studio and sync it first.
  pause
  exit /b 1
)

call gradlew.bat clean :app:installDebug
if errorlevel 1 (
  echo Build or installation failed. Review the error shown above.
  pause
  exit /b 1
)

echo SonicT Call Guard was built and installed successfully.
pause
