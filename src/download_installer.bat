@echo off
title Download Time Study Analyzer Installer
echo ============================================
echo Time Study Analyzer - Installer Downloader
echo ============================================
echo.
echo This will download the latest installer for Time Study Analyzer
echo.

REM Create temp directory for download
set TEMP_DIR=%TEMP%\TimeStudyInstaller
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

echo Downloading installer from GitHub...
echo.

REM Download the batch installer
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/Cayton3359/timestudyanalyzer/main/src/install_time_study.bat' -OutFile '%TEMP_DIR%\install_time_study.bat'"

if %errorlevel% neq 0 (
    echo ERROR: Failed to download installer!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo ✓ Installer downloaded successfully!
echo.
echo The installer has been saved to: %TEMP_DIR%\install_time_study.bat
echo.

REM Ask if user wants to run it now
set /p RUN_NOW="Would you like to run the installer now? (y/n): "
if /i "%RUN_NOW%"=="y" (
    echo.
    echo Starting installer...
    "%TEMP_DIR%\install_time_study.bat"
) else (
    echo.
    echo You can run the installer later from: %TEMP_DIR%\install_time_study.bat
    echo Or download it again by running this script.
)

echo.
pause
