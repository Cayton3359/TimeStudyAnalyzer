@echo off
echo === Git Status Check and Commit Script ===
cd /d "M:\Engineering\Lindsey\12. Code\Time Study Camera\TimeStudyAnalyzer"

echo Current directory:
cd

echo.
echo === Git Status ===
git status

echo.
echo === Adding all files ===
git add .

echo.
echo === Committing installer files ===
git commit -m "Add complete one-click installer system with GUI and batch options"

echo.
echo === Pushing to GitHub ===
git push origin main

echo.
echo === Done! ===
pause
