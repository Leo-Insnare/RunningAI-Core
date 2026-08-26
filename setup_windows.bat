@echo off
setlocal
cd /d %~dp0

echo [1/6] Checking Python Launcher...
py --version >nul 2>&1
if %errorlevel% neq 0 (
  echo Python Launcher was not found. Please install Python 3.11 x64 or 3.12 x64 from python.org.
  pause
  exit /b 1
)

set PYTHON_CMD=
py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 set PYTHON_CMD=py -3.11
if "%PYTHON_CMD%"=="" (
  py -3.12 --version >nul 2>&1
  if %errorlevel% equ 0 set PYTHON_CMD=py -3.12
)
if "%PYTHON_CMD%"=="" (
  echo Python 3.11 or 3.12 was not found.
  echo Skeleton Preview uses MediaPipe legacy Pose, so Python 3.11/3.12 x64 is recommended.
  pause
  exit /b 1
)

echo Using %PYTHON_CMD%
%PYTHON_CMD% --version

echo [2/6] Removing old virtual environment if it exists...
if exist .venv rmdir /s /q .venv

echo [3/6] Creating virtual environment .venv...
%PYTHON_CMD% -m venv .venv
if %errorlevel% neq 0 (
  echo Failed to create virtual environment.
  pause
  exit /b 1
)

echo [4/6] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
  echo Failed to upgrade pip.
  pause
  exit /b 1
)

echo [5/6] Installing compatible requirements...
.venv\Scripts\python.exe -m pip uninstall -y mediapipe numpy protobuf opencv-python opencv-python-headless opencv-contrib-python
.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
if %errorlevel% neq 0 (
  echo Failed to install requirements.
  echo Please check docs\Windows_설치_및_MediaPipe_문제해결.md
  pause
  exit /b 1
)

echo [6/6] Verifying MediaPipe legacy Pose backend...
.venv\Scripts\python.exe -c "import mediapipe as mp; print('mediapipe', mp.__version__); print('solutions', hasattr(mp, 'solutions')); import mediapipe as mp; mp.solutions.pose.Pose(static_image_mode=True).close(); print('pose backend ok')"
if %errorlevel% neq 0 (
  echo MediaPipe legacy Pose verification failed.
  echo The installed MediaPipe version must expose mp.solutions.pose.
  pause
  exit /b 1
)

echo Setup complete. Run run_windows.bat to start the app.
pause
