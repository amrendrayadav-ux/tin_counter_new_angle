#!/usr/bin/env bash
set -e

echo "====================================================================="
echo " Guardex AI - Git LFS & GitHub Setup Script"
echo "====================================================================="
echo ""

# 1. Check if git is installed
if ! command -v git &> /dev/null; then
    echo "[ERROR] git could not be found. Please install git."
    exit 1
fi

# 2. Check Git LFS
if command -v git-lfs &> /dev/null; then
    echo "[*] Initializing Git LFS..."
    git lfs install
else
    echo "[WARNING] git-lfs not found! Install Git LFS: https://git-lfs.github.com/"
    echo "ResNet weights (222MB) and RF-DETR weights (121MB) exceed GitHub's 100MB limit."
fi

# 3. Initialize git repository
if [ ! -d ".git" ]; then
    echo "[*] Initializing Git repository..."
    git init
    git branch -M main
fi

# 4. Track model files with Git LFS
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.onnx"

# 5. Stage files
echo "[*] Staging files..."
git add .gitattributes
git add .

echo ""
echo "====================================================================="
echo " Repository is prepared for commit and push!"
echo "====================================================================="
echo ""
echo "Next steps:"
echo " 1. git commit -m 'Initial commit: Production Tin Counter with RF-DETR Small and ResNet152'"
echo " 2. git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git"
echo " 3. git push -u origin main"
echo ""
