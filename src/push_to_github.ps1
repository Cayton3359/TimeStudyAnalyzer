#!/usr/bin/env powershell

Write-Host "=== Time Study Analyzer - Push to GitHub ===" -ForegroundColor Green
Write-Host ""

# Navigate to project directory
$projectPath = "M:\Engineering\Lindsey\12. Code\Time Study Camera\TimeStudyAnalyzer"
Set-Location $projectPath

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Show current status
Write-Host "=== Current Git Status ===" -ForegroundColor Cyan
git status

Write-Host ""
Write-Host "=== Adding all files ===" -ForegroundColor Cyan
git add .

Write-Host ""
Write-Host "=== Committing installer files ===" -ForegroundColor Cyan
$commitMessage = @"
Add complete Time Study Analyzer installer system

New installer features:
- Professional GUI installer with progress tracking
- Simple batch file installer for easy deployment
- Download script to get installer from GitHub
- Manual dependency installer for troubleshooting
- Complete installation documentation
- Improved error handling and user feedback
- Professional UI layout and dimensions
- Multiple installation methods for different user needs
"@

git commit -m $commitMessage

Write-Host ""
Write-Host "=== Pushing to GitHub ===" -ForegroundColor Cyan
git push origin main

Write-Host ""
Write-Host "=== Verification ===" -ForegroundColor Cyan
Write-Host "Recent commits:"
git log --oneline -n 3

Write-Host ""
Write-Host "=== SUCCESS! ===" -ForegroundColor Green
Write-Host "Your installer files should now be available at:" -ForegroundColor Yellow
Write-Host "https://github.com/Cayton3359/timestudyanalyzer" -ForegroundColor Blue

Read-Host "`nPress Enter to close"
