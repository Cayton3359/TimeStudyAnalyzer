# Time Study Analyzer - One Click Installers

This folder contains multiple installer options for easy deployment of the Time Study Analyzer application.

## 🚀 Quick Start Options

### Option 1: Batch File Installer (Recommended)
**File:** `install_time_study.bat`

Simply download and double-click this batch file. It will:
- ✅ Check for Python installation
- ✅ Download the latest code from GitHub
- ✅ Install all dependencies automatically
- ✅ Create a desktop shortcut
- ✅ Launch the application

**Requirements:** Python must be installed on the system

### Option 2: Python GUI Installer
**File:** `one_click_installer.py`

A graphical Python installer with progress tracking:
- Modern GUI with progress bar
- Real-time installation logging
- Error handling and recovery
- Automatic shortcut creation

**To use:** 
```bash
python one_click_installer.py
```

### Option 3: Build Standalone Executable
**File:** `build_installer.py`

Creates a standalone .exe file that can be distributed:
```bash
python build_installer.py
```

This creates `TimeStudyAnalyzer_Installer.exe` in the `dist` folder.

## 📝 What the Installers Do

1. **Environment Check**: Verifies Python is installed
2. **Download**: Gets the latest code from GitHub
3. **Dependencies**: Installs all required Python packages:
   - opencv-python (computer vision)
   - PyQt5 (GUI framework)
   - numpy (numerical processing)
   - pandas (data handling)
   - openpyxl (Excel output)
   - pytesseract (OCR)
   - pillow (image processing)
   - requests (web requests)
4. **Shortcuts**: Creates desktop shortcut for easy access
5. **Testing**: Verifies installation works correctly
6. **Launch**: Optionally starts the application

## 🎯 Installation Directory

By default, the application is installed to:
```
%USERPROFILE%\TimeStudyAnalyzer
```
(Usually `C:\Users\[YourName]\TimeStudyAnalyzer`)

## 🔧 Manual Installation

If the automatic installers don't work, you can install manually:

1. Clone the repository:
   ```bash
   git clone https://github.com/Cayton3359/timestudyanalyzer.git
   ```

2. Install dependencies:
   ```bash
   cd timestudyanalyzer
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## 📋 System Requirements

- **Python 3.7+** (Download from [python.org](https://python.org))
- **Windows 10/11** (tested)
- **Internet connection** (for initial download)
- **~100MB disk space**

## 🆘 Troubleshooting

### Python Not Found
If you get "Python is not installed":
1. Download Python from [python.org](https://python.org)
2. During installation, check "Add Python to PATH"
3. Restart your computer
4. Try the installer again

### Dependencies Fail to Install
If package installation fails:
1. Open Command Prompt as Administrator
2. Run: `python -m pip install --upgrade pip`
3. Try the installer again

### Application Won't Start
1. Check that all files were downloaded correctly
2. Verify the desktop shortcut points to the right location
3. Try running `python main.py` directly from the installation folder

## 📞 Support

If you encounter issues:
1. Check the installation log in the GUI installer
2. Try the batch file installer as an alternative
3. Report issues on the GitHub repository

---

**Repository:** https://github.com/Cayton3359/timestudyanalyzer
