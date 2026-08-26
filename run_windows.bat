@echo off
setlocal
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo .venv was not found. Please run setup_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
