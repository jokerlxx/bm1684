@echo off
title GrainAI Agent Proxy
cd /d "%~dp0"

echo ============================================================
echo   GrainAI Agent Proxy
echo ============================================================
echo   Config: .env
echo   Log:    logs\proxy.log
echo ============================================================
echo.
echo Starting Proxy...
echo.

python proxy_server.py
pause
