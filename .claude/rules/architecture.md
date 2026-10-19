# System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Telegram Bot (bot.py)                       │
│  - Topic-based routing: 1 topic = 1 window = 1 session             │
│  - /history: Paginated message history (default: latest page)      │
│  - /screenshot: Capture tmux pane as PNG                           │
│  - /esc: Send Escape to interrupt Claude                           │
│  - /restart: Restart session in place (respawn-pane + --resume)    │
│  - /kill: Kill session and delete topic                            │
│  - /unbind: Detach topic from session, tmux window kept running    │
│  - /usage: Show Claude Code usage/cost stats from the TUI          │
│  - Send text → Claude Code via tmux keystrokes                     │
│  - Forward /commands to Claude Code                                │
│  - Create sessions via directory browser in unbound topics         │
│  - Tool use → tool result: edit message in-place                   │
│  - Interactive UI: AskUserQuestion / ExitPlanMode / Permission     │
│  - Per-user message queue + worker (merge, rate limit)             │
│  - MarkdownV2 output with auto fallback to plain text              │
├──────────────────────┬──────────────────────────────────────────────┤
│  markdown_v2.py      │  telegram_sender.py                         │
│  MD → MarkdownV2     │  split_message (4096 limit)                 │
│  + expandable quotes │                                             │
├──────────────────────┴──────────────────────────────────────────────┤
│  terminal_parser.py                                                 │
│  - Detect interactive UIs (AskUserQuestion, ExitPlanMode, etc.)    │
│  - Parse status line (spinner + working text) + turn-end summary   │
│  - Parse statusline footer from bottom chrome (verbatim)           │
└──────────┬──────────────────────────────────────────────────────────┘
           │                              │
           │ Notify (NewMessage callback) │ Send (tmux keys)
           │                              │
┌──────────┴──────────────┐    ┌──────────┴──────────────────────┐
│  SessionMonitor         │    │  TmuxManager (tmux_manager.py)  │
│  (session_monitor.py)   │    │  - list/find/create/kill windows│
│  - Poll JSONL every 2s  │    │  - send_keys to pane            │
│  - Detect mtime changes │    │  - capture_pane for screenshot  │
│  - Parse new lines      │    └──────────────┬─────────────────┘
│  - Track pending tools  │                   │
│    across poll cycles   │                   │
└──────────┬──────────────┘                   │
           │                                  │
           ▼                                  ▼
┌────────────────────────┐         ┌─────────────────────────┐
│  TranscriptParser      │         │  Tmux Windows           │
│  (transcript_parser.py)│         │  - Claude Code process  │
│  - Parse JSONL entries │         │  - One window per       │
│  - Pair tool_use ↔     │         │    topic/session        │
│    tool_result         │         └────────────┬────────────┘
│  - Format expandable   │                      │
│    quotes for thinking │              SessionStart hook
│  - Extract history     │                      │
└────────────────────────┘                      ▼
                                    ┌────────────────────────┐
┌────────────────────────┐         │  Hook (hook.py)        │
│  SessionManager        │◄────────│  - Receive hook stdin  │
│  (session.py)          │  reads  │  - Write session_map   │
│  - Window ↔ Session    │  map    │    .json               │
│    resolution          │         └────────────────────────┘
│  - Thread bindings     │
│    (topic → window)    │         ┌────────────────────────┐
│  - Message history     │────────►│  Claude Sessions       │
│    retrieval           │  reads  │  ~/.claude/projects/   │
└────────────────────────┘  JSONL  │  - sessions-index      │
                                   │  - *.jsonl files       │
┌────────────────────────┐         └────────────────────────┘
│  MonitorState          │
│  (monitor_state.py)    │
│  - Track byte offset   │
│  - Prevent duplicates  │
│    after restart       │
└────────────────────────┘

Additional modules:
  screenshot.py       ─ Terminal text → PNG rendering (ANSI color, font fallback)
  transcribe.py       ─ Voice-to-text transcription via OpenAI API (gpt-4o-transcribe)
  update_watcher.py   ─ Notifies on Claude Code version drift / a stuck session; drives in-place /restart (respawn-pane + --resume)
  main.py             ─ CLI entry point
  utils.py            ─ Shared utilities (ccbot_dir, atomic_write_json)

Handler modules (handlers/):
  message_sender.py   ─ safe_reply/safe_edit/safe_send + rate_limit_send
  message_queue.py    ─ Per-user queue + worker (merge, status dedup)
  status_polling.py   ─ Background status line polling (1s interval); turn-end
                        detection → statusline footer on the final message
  response_builder.py ─ Response pagination and formatting
  interactive_ui.py   ─ AskUserQuestion / ExitPlanMode / Permission UI
  directory_browser.py─ Directory selection + session picker UI for new topics
  history.py          ─ Message history pagination (/history rendering)
  maintenance.py      ─ Periodic session_map sweep, hook-failure notices, and session-divergence detection
  cleanup.py          ─ Topic state cleanup on close/delete
  callback_data.py    ─ Callback data constants

State files (~/.ccbot/ or $CCBOT_DIR/):
  state.json         ─ thread bindings + window states + display names + read offsets
  session_map.json   ─ hook-generated window_id→session mapping
  monitor_state.json ─ poll progress (byte offset) per JSONL file
```

## Key Design Decisions

- **Topic-centric** — Each Telegram topic binds to one tmux window. No centralized session list; topics *are* the session list.
- **Window ID-centric** — All internal state keyed by tmux window ID (e.g. `@0`, `@12`), not window names. Window IDs are guaranteed unique within a tmux server session. Window names are kept as display names via `window_display_names` map. Same directory can have multiple windows.
- **Hook-based session tracking** — Claude Code `SessionStart` hook writes `session_map.json` (plus `transcript_size_at_start`, used to seed a newly-noticed session's read offset). The monitor does not read that file directly: `session_manager.load_session_map()` reconciles it into `window_states`, applying any manual pin (`WindowState.pinned_over`, e.g. from a `--resume`) on top, and the monitor watches that reconciled map.
- **Tool use ↔ tool result pairing** — `tool_use_id` tracked across poll cycles; tool result edits the original tool_use Telegram message in-place.
- **MarkdownV2 with fallback** — All messages go through `safe_reply`/`safe_edit`/`safe_send` which convert via `telegramify-markdown` and fall back to plain text on parse failure.
- **No truncation at parse layer** — Full content preserved; splitting at send layer respects Telegram's 4096 char limit with expandable quote atomicity.
- Only sessions present in `session_manager.window_states` (session_map.json reconciled with pins) are monitored.
- Notifications delivered to users via thread bindings (topic → window_id → session).
- **Startup re-resolution** — Window IDs reset on tmux server restart. On startup, `resolve_stale_ids()` matches persisted display names against live windows to re-map IDs. Old state.json files keyed by window name are auto-migrated.
- **Restore after server death** — `state.json` records the identity of the tmux server that issued the persisted window IDs (`tmux_server`, `<pid>:<start_time>`). If the live server differs at startup (`TmuxManager.server_identity()`; e.g. after a reboot), `resolve_stale_ids(server=...)` first recreates each persisted Claude window at its cwd running `claude --resume <session_id>`, and moves window_states, thread_bindings, read offsets and display names to the new window IDs by exact old→new mapping. Anything keyed by a dead window ID with no restored window is dropped, since a fresh server reuses low IDs. The new identity is saved together with the remap, so a crash before that restores again on the next start, adopting the windows the interrupted run already created (found by their `@ccbot_resume` tmux window option) instead of duplicating them. The restore set stays equal to the windows alive: killing a window (/kill, topic closed or deleted) drops its state at once (`forget_window`), the maintenance loop's window_state sweep drops states of windows that died otherwise, and both it and status polling's stale-binding cleanup skip while `live_window_ids()` is unknown (server unreachable or replaced).
