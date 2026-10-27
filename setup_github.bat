@echo off
echo =====================================================================
echo  Guardex AI - Git LFS ^& GitHub Setup Script
echo =====================================================================
echo.

:: 1. Check if Git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not found in your PATH. Please install Git: https://git-scm.com/
    pause
    exit /b 1
)

:: 2. Check Git LFS
where git-lfs >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Git LFS is not detected in your PATH!
    echo ResNet weights (222MB) and RF-DETR weights (121MB) exceed GitHub's 100MB limit.
    echo Please install Git LFS from: https://git-lfs.github.com/
    echo.
    echo Once installed, run: git lfs install
    echo.
) else (
    echo [*] Initializing Git LFS...
    git lfs install
)

:: 3. Initialize git repository if not already initialized
if not exist ".git" (
    echo [*] Initializing Git repository...
    git init
    git branch -M main
) else (
    echo [*] Existing Git repository detected.
)

:: 4. Ensure large files are tracked by LFS
echo [*] Tracking model weights via Git LFS...
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.onnx"

:: 5. Add and commit files
echo [*] Staging files...
git add .gitattributes
git add .

echo.
echo =====================================================================
echo  Repository is prepared for commit and push!
echo =====================================================================
echo.
echo Next steps to push to your GitHub:
echo  1. git commit -m "Initial commit: Production Tin Counter with RF-DETR Small and ResNet152"
echo  2. git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
echo  3. git push -u origin main
echo.
pause
