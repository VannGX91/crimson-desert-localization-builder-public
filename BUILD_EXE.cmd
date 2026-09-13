@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "BUILD_ENV=%~dp0.build-venv"
set "OUTPUT_ROOT=%~1"
if not defined OUTPUT_ROOT (
    for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "BUILD_STAMP=%%I"
    set "OUTPUT_ROOT=%~dp0..\release_onedir_v2.0.0_!BUILD_STAMP!"
)

if not exist "%BUILD_ENV%\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Creating isolated build environment with python...
        python -m venv "%BUILD_ENV%"
    ) else (
        echo Creating isolated build environment with py -3...
        py -3 -m venv "%BUILD_ENV%"
    )
    if errorlevel 1 goto :build_failed
)

set "BUILD_PYTHON=%BUILD_ENV%\Scripts\python.exe"

echo Installing locked build dependencies...
"%BUILD_PYTHON%" -m pip install --no-deps -r requirements-build.txt
if errorlevel 1 goto :build_failed

echo Building and smoke-testing the Windows one-folder release...
"%BUILD_PYTHON%" scripts\build_onedir.py --output-root "%OUTPUT_ROOT%"
if errorlevel 1 goto :build_failed

echo.
echo Build completed successfully.
echo Folder: %OUTPUT_ROOT%\CrimsonDesertLocalizationBuilder
echo ZIP:    %OUTPUT_ROOT%\CrimsonDesertLocalizationBuilder-windows-x64.zip
exit /b 0

:build_failed
echo.
echo Build failed. Review the error output above.
exit /b 1
