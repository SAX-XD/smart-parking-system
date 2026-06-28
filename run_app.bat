@echo off
echo ==========================================================
echo    Launching Smart Parking Analytics App...
echo ==========================================================
echo.

if not exist .venv (
    echo [ERROR] Virtual environment folder '.venv' not found!
    echo Please ensure dependencies are installed first.
    pause
    exit /b
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Starting Streamlit dashboard web server...
streamlit run app.py

pause