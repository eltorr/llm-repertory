#!/bin/bash
# Repertory on Linux: the manager (dashboard, OpenAI API, model lifecycle, setup) with its one dependency (psutil).
#   ./repertory.sh                 start the service and print the dashboard URL
#   ./repertory.sh detect | setup [STEP] | status | plan | open      see: ./repertory.sh --help
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [ ! -x .venv-manager/bin/python ]; then
  echo "creating the manager's Python environment (.venv-manager)..."
  python3 -m venv .venv-manager
  .venv-manager/bin/pip install -q --require-hashes -r repertory/requirements.txt
fi
exec .venv-manager/bin/python -m repertory "$@"
