@echo off
title SatQuery AI Launcher
cls
echo ===================================================
echo             SatQuery AI System Launcher
echo ===================================================
echo.
echo Project Directory: "%~dp0"
echo.
echo [1/2] Starting Backend Server (FastAPI on Port 8000)...
start "SatQuery AI Backend" /d "%~dp0backend" cmd /k "python -m uvicorn app.main:app --reload --port 8000"

echo [2/2] Starting Frontend Server (Next.js on Port 3000)...
start "SatQuery AI Frontend" /d "%~dp0frontend" cmd /k "npm run dev"

echo.
echo Waiting 5 seconds for servers to initialize...
ping 127.0.0.1 -n 6 > nul

echo.
echo Opening SatQuery AI Dashboard in your default browser...
start http://localhost:3000

echo.
echo ===================================================
echo SatQuery AI is running!
echo - Web App Dashboard: http://localhost:3000
echo - API Docs: http://127.0.0.1:8000/docs
echo ===================================================
echo Keep the backend and frontend command windows open while using the app.
echo.
