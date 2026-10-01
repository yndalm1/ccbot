#!/usr/bin/env bash
set -euo pipefail

# NOTE: routine restarts are non-destructive — KillMode=process leaves ccbot's
# dedicated-socket tmux server (and every Claude session) running, and ccbot
# reattaches on start. The ONE exception is the very first restart that switches
# ccbot onto the dedicated socket: tmux cannot move sessions between sockets, so
# sessions on the old (default) socket are left behind. Do that first switch at
# a clean checkpoint.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Reinstall the package so the systemd service picks up code changes
echo "Installing ccbot from ${PROJECT_DIR}..."
uv tool install --force --reinstall "$PROJECT_DIR" 2>&1

# Restart the systemd user service
echo "Restarting ccbot service..."
systemctl --user restart ccbot

# Wait for startup and verify
sleep 2
if systemctl --user is-active --quiet ccbot; then
    echo "ccbot restarted successfully. Recent logs:"
    echo "----------------------------------------"
    tail -n 20 "${CCBOT_DIR:-$HOME/.ccbot}/ccbot.log" 2>/dev/null || true
    echo "----------------------------------------"
else
    echo "Warning: ccbot service failed to start. Logs:"
    echo "----------------------------------------"
    journalctl --user -u ccbot --since "10 sec ago" --no-pager | tail -30
    echo "----------------------------------------"
    exit 1
fi
