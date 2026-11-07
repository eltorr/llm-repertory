#!/usr/bin/env bash
# The checks the Checks workflow runs on GitHub, run locally (Linux, or Git Bash on Windows):
#   lint (ruff), shell scripts (shellcheck), secrets in tracked files, known CVEs in the pinned packages (pip-audit),
#   and the unit tests that need no GPU.
#
#   bash tools/scan.sh            everything
#   bash tools/scan.sh --offline  skip the CVE lookup, which needs the network
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
offline=0; [ "${1:-}" = "--offline" ] && offline=1
windows=0; case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) windows=1 ;; esac

# --- the scanners live in .venv-scan, reinstalled when tools/scan-requirements.txt changes
VENV=.venv-scan
BIN=$VENV/bin; [ "$windows" = 1 ] && BIN=$VENV/Scripts
PY=$BIN/python
pick_python() {   # a Python 3.10+ to create the environment with (never the Windows Store stub)
  for p in "${PYTHON:-}" .venv/Scripts/python.exe python3 python; do
    if [ -n "$p" ] && command -v "$p" >/dev/null 2>&1 \
        && "$p" -c "import sys; sys.exit(sys.version_info < (3, 10))" </dev/null >/dev/null 2>&1; then
      echo "$p"; return
    fi
  done
  echo "tools/scan.sh: no Python 3.10+ found (set PYTHON=...)" >&2; exit 1
}
if ! cmp -s tools/scan-requirements.txt "$VENV/.installed" 2>/dev/null; then
  echo "== installing the scanners into $VENV"
  [ -x "$PY" ] || [ -x "$PY.exe" ] || "$(pick_python)" -m venv "$VENV" || exit 1
  "$PY" -m pip install -q --disable-pip-version-check --require-hashes -r tools/scan-requirements.txt || exit 1
  cp tools/scan-requirements.txt "$VENV/.installed"
fi

failed=()
step() {
  local name=$1; shift
  echo "== $name"
  if "$@"; then echo "   ok"; else failed+=("$name"); fi
}

# --- Repertory's Python (ruff.toml lists the files)
step "lint (ruff)" "$BIN/ruff" check --no-cache .

# --- Repertory's shell scripts
step "shell scripts (shellcheck)" "$BIN/shellcheck" -S warning setup/*.sh engines/exl3_offload/run.sh repertory.sh \
  bin/repertory-backend tools/*.sh tools/pre-push

# --- tokens and keys in tracked files
no_secrets() {
  ! git grep -n -I -E '(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{60,}|hf_[A-Za-z0-9]{34}'\
'|sk-[A-Za-z0-9_-]{32,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----'
}
step "secrets in tracked files" no_secrets

# --- known vulnerabilities in the pinned packages
PINNED=(setup/requirements.txt repertory/requirements.txt tools/scan-requirements.txt)
# Accepted advisories, as advisory ids, each listed here with its reason (none at the moment). Remove one once its fix
# can be installed.
IGNORE=()
audit() {
  local f rc=0 tmp
  tmp=$(mktemp)
  for f in "${PINNED[@]}"; do
    echo "   $f"
    sed -E 's/\+cu[0-9]+//' "$f" > "$tmp"     # torch's CUDA builds (2.15.0+cu132) are audited as the PyPI release
    "$BIN/pip-audit" --progress-spinner off --disable-pip --no-deps -r "$tmp" "${IGNORE[@]/#/--ignore-vuln=}" || rc=1
  done
  rm -f "$tmp"
  return $rc
}
if [ "$offline" = 1 ]; then echo "== known vulnerabilities (pip-audit): skipped (--offline)"
else step "known vulnerabilities (pip-audit)" audit; fi

# --- unit tests that need no GPU (the ones that need the engine's environment or Linux skip themselves)
step "unit tests" "$PY" -m unittest tests/test_*.py

if [ ${#failed[@]} -gt 0 ]; then
  printf '\nFailed: %s\n' "${failed[*]}"
  exit 1
fi
printf '\nAll checks passed.\n'
