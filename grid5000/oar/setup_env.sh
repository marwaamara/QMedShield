#!/usr/bin/env bash
# One-time environment build on the Grid5000 frontend.
set -e
python3 -m venv ~/qmed_env
source ~/qmed_env/bin/activate
pip install --upgrade pip
pip install -r "$(dirname "$0")/../requirements.txt"
echo
echo "Environment ready. Before each job:  source ~/qmed_env/bin/activate"
echo "Then verify:                         python verify_deploy.py"
