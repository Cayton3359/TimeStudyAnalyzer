# PowerShell script to commit and push installer files to GitHub
Write-Host "=== Time Study Analyzer - Git Commit Script ===" -ForegroundColor Green
Write-Host ""

# Navigate to project directory
$projectPath = "M:\Engineering\Lindsey\12. Code\Time Study Camera\TimeStudyAnalyzer"
Set-Location $projectPath

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Check git status
Write-Host "=== Git Status ===" -ForegroundColor Cyan
git status

Write-Host ""
Write-Host "=== Adding all files ===" -ForegroundColor Cyan
git add .

Write-Host ""
Write-Host "=== Committing installer files ===" -ForegroundColor Cyan
git commit -m "Add complete one-click installer system with GUI and batch options"

Write-Host ""
Write-Host "=== Pushing to GitHub ===" -ForegroundColor Cyan
git push origin main

Write-Host ""
Write-Host "=== Done! Check your GitHub repository ===" -ForegroundColor Green
Write-Host "Repository: https://github.com/Cayton3359/timestudyanalyzer" -ForegroundColor Blue

Read-Host "Press Enter to close"
