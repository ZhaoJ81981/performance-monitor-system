#!/bin/bash
# Script to push Performance Monitor System to GitHub
# Run this in terminal, it will use your browser login for authentication

cd "$(dirname "$0")"

echo "=== Pushing Performance Monitor System to GitHub ==="
echo "Repository: https://github.com/ZhaoJ81981/performance-monitor-system"
echo ""

# Check if git remote is set
if ! git remote -v | grep -q "origin"; then
    echo "Setting remote origin..."
    git remote add origin https://github.com/ZhaoJ81981/performance-monitor-system.git
fi

# Rename branch to main if needed
if git branch | grep -q "master"; then
    echo "Renaming master branch to main..."
    git branch -M main
fi

echo "Pushing code to GitHub..."
echo "Note: A browser window may open for authentication."
echo ""

# Try push with credential helper
git config credential.helper store
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Success! Code pushed to GitHub."
    echo "Visit: https://github.com/ZhaoJ81981/performance-monitor-system"
else
    echo ""
    echo "❌ Push failed. Possible solutions:"
    echo "1. Open https://github.com/login/device and enter code: B4D1-B185"
    echo "2. Generate a Personal Access Token at https://github.com/settings/tokens"
    echo "   Then run: git remote set-url origin https://<token>@github.com/ZhaoJ81981/performance-monitor-system.git"
    echo "   Then run: git push -u origin main"
    echo "3. Manually upload files via GitHub web interface"
fi