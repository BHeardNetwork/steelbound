#!/bin/sh
# Steelbound — installer for the systemd deployment layout.
#
# Default (system) layout:
#   PREFIX   /opt/steelbound        venv at $PREFIX/venv
#   DATA_DIR /var/lib/steelbound    ledger DB + workspace
#   Unit     /etc/systemd/system/steelbound.service (if systemd present)
#
# Writing to /opt, /var/lib and /etc requires root. Either run this
# script with sudo, or do a user-level install by overriding PREFIX
# and DATA_DIR, e.g.:
#   PREFIX="$HOME/.local/steelbound" DATA_DIR="$HOME/.local/share/steelbound" ./deploy/install.sh
# A user-level install skips the system-wide unit install (the unit
# assumes a system "steelbound" user) and prints how to run manually.
#
# Idempotent-ish: re-running upgrades the package in the existing
# venv, keeps the existing data dir and its contents, and reinstalls
# the unit file.
set -eu

PREFIX="${PREFIX:-/opt/steelbound}"
DATA_DIR="${DATA_DIR:-/var/lib/steelbound}"
REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

# --- privilege check (explain, never assume) -------------------------
SYSTEM_INSTALL=1
case "$PREFIX" in
  /opt/*|/usr/*|/var/*|/etc/*) ;;
  *) SYSTEM_INSTALL=0 ;;
esac
if [ "$SYSTEM_INSTALL" -eq 1 ] && [ "$(id -u)" -ne 0 ]; then
  echo "install.sh: the default layout writes to $PREFIX, $DATA_DIR and" >&2
  echo "  /etc/systemd/system, which needs root. Re-run with sudo, or" >&2
  echo "  set PREFIX and DATA_DIR under your home directory for a" >&2
  echo "  user-level install (see the header of this script)." >&2
  exit 1
fi

command -v python3 >/dev/null 2>&1 || { echo "install.sh: python3 not found (Steelbound needs Python >= 3.12)" >&2; exit 1; }
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
  || { echo "install.sh: Python >= 3.12 is required" >&2; exit 1; }

# --- dedicated user (system installs only) ---------------------------
if [ "$SYSTEM_INSTALL" -eq 1 ]; then
  if ! id steelbound >/dev/null 2>&1; then
    if command -v useradd >/dev/null 2>&1; then
      useradd --system --no-create-home --shell /usr/sbin/nologin steelbound 2>/dev/null \
        || useradd --system --no-create-home steelbound
    else
      echo "install.sh: no useradd available; create a 'steelbound' user manually" >&2
      exit 1
    fi
  fi
fi

# --- venv + package ----------------------------------------------------
mkdir -p "$PREFIX"
if [ ! -x "$PREFIX/venv/bin/python" ]; then
  python3 -m venv "$PREFIX/venv"
fi
"$PREFIX/venv/bin/pip" install --quiet --upgrade pip
# Non-editable install from this checkout, exactly as packaged.
"$PREFIX/venv/bin/pip" install --quiet "$REPO_ROOT"

# --- data dir ------------------------------------------------------------
mkdir -p "$DATA_DIR"
if [ "$SYSTEM_INSTALL" -eq 1 ]; then
  chown steelbound:steelbound "$DATA_DIR" 2>/dev/null || chown steelbound "$DATA_DIR"
  chmod 0750 "$DATA_DIR"
fi

# --- systemd unit (guarded; skip gracefully without systemd) -------------
UNIT_INSTALLED=0
if [ "$SYSTEM_INSTALL" -eq 1 ] && command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
  # The shipped unit hard-codes the default layout; only install it
  # verbatim when PREFIX/DATA_DIR are the defaults it describes.
  if [ "$PREFIX" = "/opt/steelbound" ] && [ "$DATA_DIR" = "/var/lib/steelbound" ]; then
    cp "$REPO_ROOT/deploy/steelbound.service" /etc/systemd/system/steelbound.service
    systemctl daemon-reload
    UNIT_INSTALLED=1
  else
    echo "install.sh: PREFIX/DATA_DIR differ from the unit's defaults;"
    echo "  copy deploy/steelbound.service, edit ExecStart/ReadWritePaths"
    echo "  to match, then: systemctl daemon-reload"
  fi
fi

# --- next steps ------------------------------------------------------------
echo ""
echo "Steelbound installed:"
echo "  binary:  $PREFIX/venv/bin/steelbound"
echo "  data:    $DATA_DIR"
echo ""
echo "Verify (offline, safe):"
echo "  $PREFIX/venv/bin/steelbound --data-dir \"$DATA_DIR\" evals"
if [ "$UNIT_INSTALLED" -eq 1 ]; then
  echo ""
  echo "Start the API (loopback only, http://127.0.0.1:8765):"
  echo "  systemctl enable --now steelbound"
  echo "  systemctl status steelbound"
else
  echo ""
  echo "No systemd unit installed (no systemd, or a user-level/custom"
  echo "layout). Run the API directly (loopback only):"
  echo "  $PREFIX/venv/bin/steelbound --data-dir \"$DATA_DIR\" serve --host 127.0.0.1 --port 8765"
fi
