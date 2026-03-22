#!/usr/bin/env python3
"""
Script to create GitHub repository and push code for Performance Monitor System.
"""

import os
import sys
import json
import subprocess
import requests
from pathlib import Path
from getpass import getpass

def get_github_token():
    """Get GitHub token from environment or user input."""
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        print("Found GitHub token in environment")
        return token
    
    # Try to read from gh CLI config
    gh_config = Path.home() / '.config' / 'gh' / 'hosts.yml'
    if gh_config.exists():
        import yaml
        with open(gh_config, 'r') as f:
            config = yaml.safe_load(f)
            if config and 'github.com' in config:
                token = config['github.com'].get('oauth_token')
                if token:
                    print("Found GitHub token in gh config")
                    return token
    
    # Prompt user
    print("GitHub token not found.")
    print("Please create a personal access token with 'repo' scope at:")
    print("https://github.com/settings/tokens/new")
    token = getpass("Enter GitHub token: ").strip()
    
    if not token:
        print("Error: Token is required")
        sys.exit(1)
    
    return token

def check_repo_exists(owner, repo_name, token):
    """Check if repository already exists."""
    url = f"https://api.github.com/repos/{owner}/{repo_name}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return True, response.json()
    elif response.status_code == 404:
        return False, None
    else:
        print(f"Error checking repo: {response.status_code} - {response.text}")
        return False, None

def create_github_repo(owner, repo_name, token, description=""):
    """Create a new GitHub repository."""
    url = "https://api.github.com/user/repos"
    if owner != "user":
        url = f"https://api.github.com/orgs/{owner}/repos"
    
    data = {
        "name": repo_name,
        "description": description,
        "private": False,
        "has_issues": True,
        "has_projects": True,
        "has_wiki": True,
        "auto_init": False,
        "gitignore_template": "Python",
        "license_template": "mit"
    }
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    print(f"Creating repository {owner}/{repo_name}...")
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 201:
        repo_info = response.json()
        print(f"Repository created: {repo_info['html_url']}")
        return repo_info
    else:
        print(f"Error creating repository: {response.status_code} - {response.text}")
        return None

def setup_git_remote(repo_url, repo_name="origin"):
    """Set up git remote and push code."""
    # Check if remote already exists
    result = subprocess.run(
        ["git", "remote", "get-url", repo_name],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"Remote '{repo_name}' already exists: {result.stdout.strip()}")
        choice = input("Replace remote? (y/N): ").lower()
        if choice != 'y':
            return False
        subprocess.run(["git", "remote", "remove", repo_name])
    
    # Add remote
    print(f"Adding remote '{repo_name}' -> {repo_url}")
    subprocess.run(["git", "remote", "add", repo_name, repo_url], check=True)
    
    # Rename default branch to main if needed
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True
    )
    current_branch = result.stdout.strip()
    
    if current_branch == "master":
        print("Renaming branch from 'master' to 'main'...")
        subprocess.run(["git", "branch", "-M", "main"], check=True)
        current_branch = "main"
    
    # Push code
    print(f"Pushing code to {repo_name}/{current_branch}...")
    result = subprocess.run(
        ["git", "push", "-u", repo_name, current_branch],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("Push successful!")
        return True
    else:
        print(f"Push failed: {result.stderr}")
        return False

def create_github_workflows(repo_path):
    """Create GitHub Actions workflows for CI/CD."""
    workflows_dir = repo_path / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # CI workflow
    ci_workflow = workflows_dir / "ci.yml"
    ci_workflow.write_text("""name: CI

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Test with pytest
      run: |
        pip install pytest pytest-cov
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  docker:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build Docker images
      run: |
        docker build -f Dockerfile.api -t pms-api:latest .
        docker build -f Dockerfile.ml -t pms-ml:latest .
    
    - name: Run security scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'pms-api:latest'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload security scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
""")
    
    # CD workflow
    cd_workflow = workflows_dir / "cd.yml"
    cd_workflow.write_text("""name: CD

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Build and push Docker images
      run: |
        docker build -f Dockerfile.api -t ${{ secrets.DOCKERHUB_USERNAME }}/pms-api:${{ github.sha }} .
        docker build -f Dockerfile.ml -t ${{ secrets.DOCKERHUB_USERNAME }}/pms-ml:${{ github.sha }} .
        
        docker push ${{ secrets.DOCKERHUB_USERNAME }}/pms-api:${{ github.sha }}
        docker push ${{ secrets.DOCKERHUB_USERNAME }}/pms-ml:${{ github.sha }}
        
        # Tag with latest
        docker tag ${{ secrets.DOCKERHUB_USERNAME }}/pms-api:${{ github.sha }} ${{ secrets.DOCKERHUB_USERNAME }}/pms-api:latest
        docker tag ${{ secrets.DOCKERHUB_USERNAME }}/pms-ml:${{ github.sha }} ${{ secrets.DOCKERHUB_USERNAME }}/pms-ml:latest
        
        docker push ${{ secrets.DOCKERHUB_USERNAME }}/pms-api:latest
        docker push ${{ secrets.DOCKERHUB_USERNAME }}/pms-ml:latest
    
    - name: Update deployment
      run: |
        echo "Deployment completed for ${{ github.sha }}"
""")
    
    print(f"Created GitHub Actions workflows in {workflows_dir}")

def create_readme_badge(repo_owner, repo_name):
    """Update README with badges."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        return
    
    readme_content = readme_path.read_text()
    
    badges = f"""
## Badges

[![CI](https://github.com/{repo_owner}/{repo_name}/workflows/CI/badge.svg)](https://github.com/{repo_owner}/{repo_name}/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/{repo_owner}/{repo_name}/branch/main/graph/badge.svg)](https://codecov.io/gh/{repo_owner}/{repo_name})
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://hub.docker.com/)
"""
    
    # Insert badges after the main title
    if "## Badges" not in readme_content:
        lines = readme_content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('# ') and i < len(lines) - 1:
                lines.insert(i + 1, badges.strip())
                break
        readme_path.write_text('\n'.join(lines))
        print("Added badges to README")

def main():
    """Main function."""
    # Configuration
    repo_owner = "ZhaoJ81981"
    repo_name = "performance-monitor-system"
    repo_description = "Real‑time system monitoring and predictive analytics platform with Telegraf, InfluxDB, Grafana, and Python ML"
    
    # Get current directory
    repo_path = Path.cwd()
    if not (repo_path / ".git").exists():
        print("Error: Not a git repository")
        sys.exit(1)
    
    print("=" * 60)
    print("GitHub Repository Setup for Performance Monitor System")
    print("=" * 60)
    
    # Get GitHub token
    token = get_github_token()
    
    # Check if repository already exists
    exists, repo_info = check_repo_exists(repo_owner, repo_name, token)
    
    if exists:
        print(f"Repository {repo_owner}/{repo_name} already exists")
        print(f"URL: {repo_info['html_url']}")
        
        choice = input("Continue with existing repository? (y/N): ").lower()
        if choice != 'y':
            print("Aborting")
            sys.exit(0)
        
        repo_url = repo_info['clone_url']
    else:
        # Create new repository
        repo_info = create_github_repo(repo_owner, repo_name, token, repo_description)
        if not repo_info:
            print("Failed to create repository")
            sys.exit(1)
        
        repo_url = repo_info['clone_url']
    
    # Setup git remote and push
    if not setup_git_remote(repo_url):
        print("Failed to setup git remote")
        sys.exit(1)
    
    # Create GitHub Actions workflows
    create_github_workflows(repo_path)
    
    # Add badges to README
    create_readme_badge(repo_owner, repo_name)
    
    # Commit and push workflows
    subprocess.run(["git", "add", ".github/"], check=True)
    subprocess.run(["git", "add", "README.md"], check=True)
    subprocess.run(["git", "commit", "-m", "Add GitHub Actions workflows and badges"], check=True)
    subprocess.run(["git", "push"], check=True)
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print(f"Repository: https://github.com/{repo_owner}/{repo_name}")
    print("GitHub Actions: https://github.com/{repo_owner}/{repo_name}/actions")
    print("=" * 60)

if __name__ == "__main__":
    main()