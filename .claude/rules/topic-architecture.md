# Topic-Only Architecture

The bot operates exclusively in Telegram Forum (topics) mode. There is **no** `active_sessions` mapping, **no** `/list` command, **no** General topic routing, and **no** backward-compatibility logic for older non-topic modes. Every code path assumes named topics.

## 1 Topic = 1 Window = 1 Session

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Topic ID   │ ───▶ │ Window ID   │ ───▶ │ Session ID  │
│  (Telegram) │      │ (tmux @id)  │      │  (Claude)   │
└─────────────┘      └─────────────┘      └─────────────┘
     thread_bindings      session_map.json
     (state.json)         (written by hook)
```

Window IDs (e.g. `@0`, `@12`) are guaranteed unique within a tmux server session. Window names are stored separately as display names (`window_display_names` map).

## Mapping 1: Topic → Window ID (thread_bindings)

```python
# session.py: SessionManager
thread_bindings: dict[int, dict[int, str]]  # user_id → {thread_id → window_id}
window_display_names: dict[str, str]        # window_id → window_name (for display)
```

- Storage: memory + `state.json`
- Written when: user creates a new session via the directory browser in a topic
- Purpose: route user messages to the correct tmux window

## Mapping 2: Window ID → Session (session_map.json)

```python
# session_map.json (key format: "tmux_session:window_id")
{
  "ccbot:@0": {"session_id": "uuid-xxx", "cwd": "/path/to/project", "window_name": "project", "transcript_size_at_start": 4821},
  "ccbot:@5": {"session_id": "uuid-yyy", "cwd": "/path/to/project", "window_name": "project-2", "transcript_size_at_start": 0}
}
```

- Storage: `session_map.json`
- Written when: Claude Code's `SessionStart` hook fires. Also records `transcript_size_at_start` — the transcript's byte size at that exact moment — which seeds a newly-noticed session's read offset so a reply already on disk before the monitor's first poll of it isn't silently skipped.
- Property: one window maps to one session; session_id changes after `/clear`
- Purpose: the monitor does not read this file directly. `session_manager.load_session_map()` reconciles it into `window_states` (a hook entry is skipped while it still reports a manually-pinned `session_id`, see `WindowState.pinned_over`), and SessionMonitor watches `window_states` — the reconciled authority — to decide which sessions to watch.

## Message Flows

**Outbound** (user → Claude):
```
User sends "hello" in topic (thread_id=42)
  → thread_bindings[user_id][42] → "@0"
  → send_to_window("@0", "hello")   # resolves via find_window_by_id
```

**Inbound** (Claude → user):
```
SessionMonitor reads new message (session_id = "uuid-xxx")
  → Iterate thread_bindings, find (user, thread) whose window_id maps to this session
  → Deliver message to user in the correct topic (thread_id)
```

**New topic flow**: First message in an unbound topic → directory browser → select directory → session picker (if existing sessions found) or create window → bind topic → forward pending message.

**Resume session flow**: When selecting a directory with existing Claude sessions, a session picker UI is shown. Choosing a session runs `claude --resume <session_id>`. The window must stay on the resumed session, whose JSONL file receives the messages. If the hook reports that same session_id, nothing more is needed. If it reports a different one, the bot keeps `session_id` on the resumed session and records the hook's id in `WindowState.pinned_over`. `load_session_map()` keeps honoring that pin — ignoring hook entries that still report the pinned-over id — until the hook reports yet another session_id (e.g. a later `/clear`), at which point the pin is cleared automatically. Windows created with `--resume` carry the tmux window option `@ccbot_resume=<session_id>`.

**Topic lifecycle**: Closing/deleting a topic auto-kills the associated tmux window and unbinds the thread. Stale bindings (window deleted externally) are cleaned up by the status polling loop.

## Session Lifecycle

**Startup cleanup**: On bot startup, all tracked sessions not present in the reconciled `window_states` map are cleaned up, preventing monitoring of closed sessions.

**Runtime change detection**: Each polling cycle reloads `session_map.json` into `window_states` (via `load_session_map()`) and checks the reconciled map for changes:
- Window's session_id changed (e.g., after `/clear`) → clean up old session
- Window deleted → clean up corresponding session
