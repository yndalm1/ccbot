#!/usr/bin/env bash
set -euo pipefail

# NOTE: routine restarts are non-destructive — AbandonProcessGroup leaves
# ccbot's dedicated-socket tmux server (and every Claude session) running, and
# ccbot reattaches on start. The ONE exception is the very first restart that
# switches ccbot onto the dedicated socket: tmux cannot move sessions between
# sockets, so sessions on the old (default) socket are left behind. Do that
# first switch at a clean checkpoint.

LABEL="${CCBOT_LAUNCHD_LABEL:-com.ccbot}"
PLIST="${CCBOT_LAUNCHD_PLIST:-$HOME/Library/LaunchAgents/${LABEL}.plist}"

if [[ ! -f "$PLIST" ]]; then
    echo "Error: launchd plist not found: $PLIST"
    exit 1
fi

echo "Reloading launchd agent: $LABEL"
launchctl bootout "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

sleep 2

if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
    echo "ccbot restarted successfully."
    echo "----------------------------------------"
    tail -n 20 "${CCBOT_DIR:-$HOME/.ccbot}/ccbot.log" 2>/dev/null || true
    echo "----------------------------------------"
else
    # A crash before logging starts lands in launchd's stderr file, not ccbot.log.
    echo "Error: ccbot failed to start. Startup output:"
    tail -n 30 "$HOME/.ccbot/ccbot.stderr.log" 2>/dev/null || true
    exit 1
fi
