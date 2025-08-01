@echo off
echo === GitHub Push All Installer Files ===
cd /d "M:\Engineering\Lindsey\12. Code\Time Study Camera\TimeStudyAnalyzer"

echo.
echo Current directory: %CD%
echo.

echo === Current Git Status ===
git status

echo.
echo === Adding all files ===
git add .

echo.
echo === Committing changes ===
git commit -m "Add complete Time Study Analyzer installer system

Features added:
- GUI installer with progress bar and error handling
- Batch file installer for simple one-click installation
- Download script to get installer from GitHub
- Manual dependency installer for troubleshooting
- Complete documentation and setup guides
- Professional UI with proper dimensions
- Robust error handling and recovery options"

echo.
echo === Pushing to GitHub ===
git push origin main

echo.
echo === Verification ===
echo Recent commits:
git log --oneline -n 3

echo.
echo Files in src directory:
dir src

echo.
echo === DONE ===
echo Check your GitHub repository at:
echo https://github.com/Cayton3359/timestudyanalyzer
echo.

pause
