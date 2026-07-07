@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set SERVICE_NAME=GrainAI-AgentProxy
set PYTHON_EXE=python
set SCRIPT_PATH=%~dp0proxy_server.py
set WORK_DIR=%~dp0

echo ============================================================
echo   Install GrainAI Agent Proxy as Windows Service
echo ============================================================
echo.

REM --- Check admin rights ---
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Please run as Administrator (right-click - Run as Administrator)
    pause
    exit /b 1
)

REM --- Find NSSM ---
set NSSM=
where nssm >nul 2>&1 && set NSSM=nssm
if not defined NSSM (
    if exist "%~dp0nssm.exe" set NSSM=%~dp0nssm.exe
)
if not defined NSSM (
    if exist "%~dp0nssm-2.24\win64\nssm.exe" set NSSM=%~dp0nssm-2.24\win64\nssm.exe
)

if not defined NSSM (
    echo [!] NSSM not found. Downloading...
    powershell -Command "Invoke-WebRequest -Uri 'https://nssm.cc/release/nssm-2.24.zip' -OutFile 'nssm.zip'" 2>nul
    if exist nssm.zip (
        powershell -Command "Expand-Archive -Path 'nssm.zip' -DestinationPath '.' -Force" 2>nul
        del nssm.zip 2>nul
    )
    if exist "nssm-2.24\win64\nssm.exe" (
        set NSSM=nssm-2.24\win64\nssm.exe
    )
)

if not defined NSSM (
    echo.
    echo [MANUAL] Please download NSSM from https://nssm.cc/download
    echo          and place nssm.exe in this folder, then re-run.
    pause
    exit /b 1
)

echo NSSM: !NSSM!
echo.

REM --- Stop and remove existing service ---
!NSSM! stop %SERVICE_NAME% 2>nul
!NSSM! remove %SERVICE_NAME% confirm 2>nul

REM --- Install service ---
echo Installing service: %SERVICE_NAME%
!NSSM! install %SERVICE_NAME% "%PYTHON_EXE%" "%SCRIPT_PATH%"
!NSSM! set %SERVICE_NAME% AppDirectory "%WORK_DIR%"
!NSSM! set %SERVICE_NAME% DisplayName "GrainAI Agent Proxy"
!NSSM! set %SERVICE_NAME% Description "Grain Depot AI Monitoring - DeepSeek Agent Proxy Service"
!NSSM! set %SERVICE_NAME% Start SERVICE_AUTO_START

REM --- Auto restart on crash ---
!NSSM! set %SERVICE_NAME% AppExit Default Restart
!NSSM! set %SERVICE_NAME% AppRestartDelay 5000

REM --- Logging ---
!NSSM! set %SERVICE_NAME% AppStdout "%WORK_DIR%logs\stdout.log"
!NSSM! set %SERVICE_NAME% AppStderr "%WORK_DIR%logs\stderr.log"
!NSSM! set %SERVICE_NAME% AppRotateFiles 1
!NSSM! set %SERVICE_NAME% AppRotateOnline 1
!NSSM! set %SERVICE_NAME% AppRotateBytes 1048576

REM --- Start service ---
echo Starting service...
!NSSM! start %SERVICE_NAME%

echo.
echo ============================================================
echo   Service installed and started.
echo   Name:   %SERVICE_NAME%
echo   Status: !NSSM! status %SERVICE_NAME%
echo ============================================================
echo.
echo Manage the service:
echo   services.msc  (find "GrainAI Agent Proxy")
echo   or: nssm start/stop/restart %SERVICE_NAME%
echo.
pause
