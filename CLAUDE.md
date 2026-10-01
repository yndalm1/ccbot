# CLAUDE.md

ccmux — Telegram bot that bridges Telegram Forum topics to Claude Code sessions via tmux windows. Each topic is bound to one tmux window running one Claude Code instance.

Tech stack: Python, python-telegram-bot, tmux, uv.

## Common Commands

```bash
uv run ruff check src/ tests/         # Lint — MUST pass before committing
uv run ruff format src/ tests/        # Format — auto-fix, then verify with --check
uv run pyright src/ccbot/             # Type check — MUST be 0 errors before committing
uv run python -m pytest               # Run full test suite in repo-local .venv
./scripts/restart.sh                  # Linux: reinstall + restart systemd service
./scripts/restart-macos.sh            # macOS: reload launchd agent
ccbot hook --install                  # Auto-install Claude Code SessionStart hook
```

## Core Design Constraints

- **1 Topic = 1 Window = 1 Session** — all internal routing keyed by tmux window ID (`@0`, `@12`), not window name. Window names kept as display names. Same directory can have multiple windows.
- **Topic-only** — no backward-compat for non-topic mode. No `active_sessions`, no `/list`, no General topic routing.
- **No message truncation** at parse layer — splitting only at send layer (`split_message`, 4096 char limit).
- **MarkdownV2 only** — use `safe_reply`/`safe_edit`/`safe_send` helpers (auto fallback to plain text). Internal queue/UI code calls bot API directly with its own fallback.
- **Hook-based session tracking** — `SessionStart` hook writes `session_map.json`; monitor polls it to detect session changes.
- **Message queue per user** — FIFO ordering, message merging (3800 char limit), tool_use/tool_result pairing.
- **Rate limiting** — `AIORateLimiter(max_retries=5)` on the Application (30/s global). On restart, the global bucket is pre-filled to avoid burst against Telegram's server-side counter.

## Code Conventions

- Every `.py` file starts with a module-level docstring: purpose clear within 10 lines, one-sentence summary first line, then core responsibilities and key components.
- Telegram interaction: prefer inline keyboards over reply keyboards; use `edit_message_text` for in-place updates; keep callback data under 64 bytes; use `answer_callback_query` for instant feedback.

## Configuration

- Config directory: `~/.ccbot/` by default, override with `CCBOT_DIR` env var.
- `.env` loading priority: local `.env` > config dir `.env`.
- State files: `state.json` (thread bindings), `session_map.json` (hook-generated), `monitor_state.json` (byte offsets).
- Log file: `ccbot.log` in the config dir — rotates at 50 MB with one backup (`ccbot.log.1`), so logs never exceed 100 MB. The service's own stdout/stderr (launchd `ccbot.std*.log`, systemd journal) only hold output from before logging starts, such as a startup crash.
- Service management is platform-specific: Linux uses `systemd`; macOS uses `launchd`.
- For Linux, use `deploy/linux/ccbot.service` as the unit template; for macOS, keep a local agent plist in `~/Library/LaunchAgents/` and use `deploy/macos/com.ccbot.plist` as the template.
- **Dedicated tmux socket:** ccbot runs its tmux server on a dedicated socket (`TMUX_SOCKET_NAME`, default `ccbot`) so it is isolated from the user's interactive tmux — foreign sessions can't leak into `session_map`, and the service can restart without killing sessions (unit uses `KillMode=process`; launchd uses `AbandonProcessGroup`). Attach over SSH with `tmux -L ccbot attach -t ccbot` (handy alias: `alias ctmux='tmux -L ccbot'`). On restart, ccbot reattaches to the existing server and resumes monitoring. The systemd unit is a user unit (`WantedBy=default.target`): run `loginctl enable-linger <user>`, or the service (and this survival story) stops at logout.
  - ⚠️ **The first deploy/restart onto the dedicated socket is one-time destructive.** tmux cannot move sessions between sockets, so sessions running on the old (default) socket do **not** carry over — they are left behind on the old socket. Make this first switch at a clean checkpoint. **Every restart after that is non-destructive** (`KillMode=process` leaves the tmux server running; ccbot reattaches and resumes from saved byte offsets).

## Hook Configuration

Auto-install: `ccbot hook --install`

Or manually in `~/.claude/settings.json`:
```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [{ "type": "command", "command": "ccbot hook", "timeout": 5 }]
      }
    ]
  }
}
```

## Architecture Details

See @.claude/rules/architecture.md for full system diagram and module inventory.
See @.claude/rules/topic-architecture.md for topic→window→session mapping details.
See @.claude/rules/message-handling.md for message queue, merging, and rate limiting.
