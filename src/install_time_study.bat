@echo off
title Time Study Analyzer - One Click Installer
echo ========================================
echo Time Study Analyzer - One Click Installer
echo ========================================
echo.
echo This installer will:
echo   1. Check for Python installation
echo   2. Download the latest code from GitHub
echo   3. Install all required dependencies
echo   4. Create a desktop shortcut
echo   5. Launch the application
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python from https://python.org and try again.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo ✓ Python is installed
echo.

REM Set installation directory
set INSTALL_DIR=%USERPROFILE%\TimeStudyAnalyzer
echo Installation directory: %INSTALL_DIR%
echo.

REM Remove existing installation if it exists
if exist "%INSTALL_DIR%" (
    echo Removing existing installation...
    rmdir /s /q "%INSTALL_DIR%"
)

REM Download from GitHub
echo Downloading from GitHub...
git --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Using git to clone repository...
    git clone https://github.com/Cayton3359/timestudyanalyzer.git "%INSTALL_DIR%"
    if %errorlevel% neq 0 (
        echo Git clone failed, trying ZIP download...
        goto :download_zip
    )
    echo ✓ Repository cloned successfully
    goto :install_deps
) else (
    echo Git not found, downloading ZIP file...
    goto :download_zip
)

:download_zip
REM Download ZIP file using PowerShell
echo Downloading repository as ZIP...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/Cayton3359/timestudyanalyzer/archive/refs/heads/main.zip' -OutFile '%TEMP%\timestudyanalyzer.zip'"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download from GitHub!
    pause
    exit /b 1
)

echo Extracting files...
powershell -Command "Expand-Archive -Path '%TEMP%\timestudyanalyzer.zip' -DestinationPath '%USERPROFILE%' -Force"
if exist "%USERPROFILE%\timestudyanalyzer-main" (
    move "%USERPROFILE%\timestudyanalyzer-main" "%INSTALL_DIR%"
)

REM Clean up
del "%TEMP%\timestudyanalyzer.zip" >nul 2>&1

echo ✓ Files extracted successfully

:install_deps
REM Install dependencies
echo.
echo Installing Python dependencies...
cd /d "%INSTALL_DIR%"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo ✓ All dependencies installed successfully

REM Create desktop shortcut
echo.
echo Creating desktop shortcut...

REM Create batch file to run the application
echo @echo off > "%INSTALL_DIR%\run_time_study.bat"
echo cd /d "%INSTALL_DIR%" >> "%INSTALL_DIR%\run_time_study.bat"
echo python main.py >> "%INSTALL_DIR%\run_time_study.bat"
echo pause >> "%INSTALL_DIR%\run_time_study.bat"

REM Create desktop shortcut using PowerShell
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\Time Study Analyzer.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\run_time_study.bat'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'Time Study Analyzer Application'; $Shortcut.Save()"

if %errorlevel% equ 0 (
    echo ✓ Desktop shortcut created
) else (
    echo Creating desktop batch file as fallback...
    copy "%INSTALL_DIR%\run_time_study.bat" "%USERPROFILE%\Desktop\Time Study Analyzer.bat"
    echo ✓ Desktop batch file created
)

REM Test installation
echo.
echo Testing installation...
python -c "import sys; sys.path.insert(0, '.'); import main; print('SUCCESS: Installation test passed')" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Installation test passed
) else (
    echo ! Installation test failed, but continuing...
)

REM Installation complete
echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Time Study Analyzer has been installed to:
echo %INSTALL_DIR%
echo.
echo You can now:
echo   • Use the desktop shortcut to launch the application
echo   • Or run main.py from the installation directory
echo.

REM Ask if user wants to launch now
set /p LAUNCH="Would you like to launch Time Study Analyzer now? (y/n): "
if /i "%LAUNCH%"=="y" (
    echo.
    echo Launching Time Study Analyzer...
    start /d "%INSTALL_DIR%" python main.py
)

echo.
echo Installation completed successfully!
pause
