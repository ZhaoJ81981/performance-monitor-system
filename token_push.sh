#!/bin/bash
# Script to push using GitHub Personal Access Token
# Usage: ./token_push.sh YOUR_TOKEN_HERE

if [ -z "$1" ]; then
    echo "Usage: $0 <github_token>"
    echo "Example: ./token_push.sh ghp_xxxxxxxxxxxx"
    exit 1
fi

TOKEN="$1"
REPO_URL="https://${TOKEN}@github.com/ZhaoJ81981/performance-monitor-system.git"

cd "$(dirname "$0")"

echo "=== Pushing with Token ==="
echo "Repository: https://github.com/ZhaoJ81981/performance-monitor-system"
echo ""

# Set remote with token
git remote set-url origin "$REPO_URL"

# Push
echo "Pushing code to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Success! Code pushed to GitHub."
    echo "Visit: https://github.com/ZhaoJ81981/performance-monitor-system"
    
    # Reset remote to regular URL (without token in config)
    git remote set-url origin "https://github.com/ZhaoJ81981/performance-monitor-system.git"
    echo "Remote URL reset to regular (token removed from config)."
else
    echo ""
    echo "❌ Push failed with token."
    echo "Check token permissions (must have 'repo' scope)."
    exit 1
fi