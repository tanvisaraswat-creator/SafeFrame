@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: run_windows.bat  ·  SafeFrame Quick-Start (Windows)
:: ─────────────────────────────────────────────────────────────────────────────

echo.
echo    SafeFrame - AI Content Moderation System
echo    -----------------------------------------

:: 1. Create venv if needed
if not exist venv (
  echo -- Creating virtual environment ...
  python -m venv venv
)

:: 2. Activate
call venv\Scripts\activate.bat

:: 3. Install deps
echo -- Installing dependencies ...
pip install --quiet -r requirements.txt

:: 4. Create runtime dirs
if not exist static\uploads mkdir static\uploads
if not exist logs mkdir logs

:: 5. Launch
echo.
echo    Server starting at http://localhost:5000
echo    Upload UI  -^>  http://localhost:5000
echo    Admin      -^>  http://localhost:5000/admin
echo.
python app.py
pause
