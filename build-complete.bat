@echo off
echo ========================================
echo Time Study Analyzer - Complete Build
echo ========================================
echo.

echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller
echo.

echo Building complete package...
pyinstaller TimeStudyAnalyzer.spec
echo.

echo Creating release folder...
mkdir "TimeStudyAnalyzer-Release" 2>nul
copy "dist\TimeStudyAnalyzer-Complete\TimeStudyAnalyzer-GUI.exe" "TimeStudyAnalyzer-Release\"
copy "dist\TimeStudyAnalyzer-Complete\TimeStudyAnalyzer-CLI.exe" "TimeStudyAnalyzer-Release\"
copy "Que Cards.pptx" "TimeStudyAnalyzer-Release\"
copy "User-Guide.md" "TimeStudyAnalyzer-Release\"
copy "README.md" "TimeStudyAnalyzer-Release\"

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo Package location: TimeStudyAnalyzer-Release\
echo.
echo Contents:
echo - TimeStudyAnalyzer-GUI.exe (Graphical interface)
echo - TimeStudyAnalyzer-CLI.exe (Command line)
echo - Que Cards.pptx (Printable cards)
echo - User-Guide.md (Setup instructions)
echo - README.md (Technical documentation)
echo.
echo Ready for distribution!
echo ========================================

pause
