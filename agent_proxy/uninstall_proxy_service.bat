@echo off
setlocal
cd /d "%~dp0"

set SERVICE_NAME=GrainAI-AgentProxy

echo ============================================================
echo   Uninstall GrainAI Agent Proxy Service
echo ============================================================
echo.

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Please run as Administrator
    pause
    exit /b 1
)

set NSSM=
where nssm >nul 2>&1 && set NSSM=nssm
if not defined NSSM if exist "%~dp0nssm.exe" set NSSM=%~dp0nssm.exe
if not defined NSSM if exist "%~dp0nssm-2.24\win64\nssm.exe" set NSSM=%~dp0nssm-2.24\win64\nssm.exe

if not defined NSSM (
    echo [ERROR] NSSM not found. Cannot uninstall service.
    pause
    exit /b 1
)

echo Stopping service...
!NSSM! stop %SERVICE_NAME% 2>nul
echo Removing service...
!NSSM! remove %SERVICE_NAME% confirm 2>nul

echo.
echo Service "%SERVICE_NAME%" uninstalled.
pause
