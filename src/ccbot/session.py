"""Claude Code session management — the core state hub.

Manages the key mappings:
  Window→Session (window_states): which Claude session_id a window holds (keyed by window_id).
  User→Thread→Window (thread_bindings): topic-to-window bindings (1 topic = 1 window_id).

Responsibilities:
  - Persist/load state to ~/.ccbot/state.json.
  - Sync window↔session bindings from session_map.json (written by hook).
  - Resolve window IDs to ClaudeSession objects (JSONL file reading).
  - Track per-user read offsets for unread-message detection.
  - Manage thread↔window bindings for Telegram topic routing.
  - Send keystrokes to tmux windows and retrieve message history.
  - Maintain window_id→display name mapping for UI display.
  - Re-resolve stale window IDs on startup (tmux server restart recovery).

Key class: SessionManager (singleton instantiated as `session_manager`).
Key methods for thread binding access:
  - resolve_window_for_thread: Get window_id for a user's thread
  - iter_thread_bindings: Generator for iterating all (user_id, thread_id, window_id)
  - find_users_for_session: Find all users bound to a session_id
"""

import asyncio
import fcntl
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any

import aiofiles

from .config import config
from .tmux_manager import tmux_manager
from .transcript_parser import TranscriptParser
from .utils import atomic_write_json

logger = logging.getLogger(__name__)


@dataclass
class WindowState:
    """Persistent state for a tmux window.

    Attributes:
        session_id: Associated Claude session ID (empty if not yet detected)
        cwd: Working directory for direct file path construction
        window_name: Display name of the window
        claude_launch_version: Installed `claude --version` at the moment this
            window's claude process was launched. Compared against the live
            installed version on each turn-end to decide whether to notify the
            user that an update is available. Per-window (not global) so
            independent sessions still on the old version each get their own
            notice.
        update_notified_version: The installed version we last sent an
            "update available" notice about for this window. Prevents re-nagging
            the same drift every turn; a *newer* version re-triggers exactly one
            fresh notice. Persisted so a ccbot restart does not re-notify.
        failure_notified: Whether we have already surfaced a "session looks
            broken" notice for the current failure. Reset when the pane is clean
            again (or on /restart) so a recurrence re-notifies once.
        pinned_over: When not None, session_id/cwd were set manually (resume
            override or hook-timeout recovery) and outrank hook entries in
            session_map.json that still report session_id == pinned_over. A
            hook entry with a DIFFERENT non-empty session_id (e.g. after
            /clear) unpins and is accepted normally. An empty string pins
            over nothing, so any future hook entry with a non-empty
            session_id unpins it.
        session_start_size: The session's transcript file size at the
            SessionStart moment the hook recorded it, or -1 if unknown (old
            session_map entry predating this field, or a non-int value on
            disk). Lets the monitor seed a newly-noticed session's read
            offset at start-of-session instead of current EOF, so a reply
            that landed before the monitor ever polled it is not silently
            skipped (review f17/RC38).
    """

    session_id: str = ""
    cwd: str = ""
    window_name: str = ""
    claude_launch_version: str = ""
    update_notified_version: str = ""
    failure_notified: bool = False
    pinned_over: str | None = None
    session_start_size: int = -1

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "cwd": self.cwd,
        }
        if self.window_name:
            d["window_name"] = self.window_name
        if self.claude_launch_version:
            d["claude_launch_version"] = self.claude_launch_version
        if self.update_notified_version:
            d["update_notified_version"] = self.update_notified_version
        if self.failure_notified:
            d["failure_notified"] = self.failure_notified
        if self.pinned_over is not None:
            d["pinned_over"] = self.pinned_over
        if self.session_start_size >= 0:
            d["session_start_size"] = self.session_start_size
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowState":
        return cls(
            session_id=data.get("session_id", ""),
            cwd=data.get("cwd", ""),
            window_name=data.get("window_name", ""),
            claude_launch_version=data.get("claude_launch_version", ""),
            update_notified_version=data.get("update_notified_version", ""),
            failure_notified=data.get("failure_notified", False),
            pinned_over=data.get("pinned_over"),
            session_start_size=data.get("session_start_size", -1),
        )


@dataclass
class ClaudeSession:
    """Information about a Claude Code session."""

    session_id: str
    summary: str
    message_count: int
    file_path: str
    # Claude Code session title (v2.1.75+). Prefer this over `summary` for
    # display when non-empty. Sourced from JSONL entries of type
    # "custom-title" (user-set via /rename or -n) or "agent-name"
    # (auto-generated); custom-title wins.
    name: str = ""


@dataclass
class SessionManager:
    """Manages session state for Claude Code.

    All internal keys use window_id (e.g. '@0', '@12') for uniqueness.
    Display names (window_name) are stored separately for UI presentation.

    window_states: window_id -> WindowState (session_id, cwd, window_name)
    user_window_offsets: user_id -> {window_id -> byte_offset}
    thread_bindings: user_id -> {thread_id -> window_id}
    window_display_names: window_id -> window_name (for display)
    group_chat_ids: "user_id:thread_id" -> group chat_id (for supergroup routing)
    """

    window_states: dict[str, WindowState] = field(default_factory=dict)
    user_window_offsets: dict[int, dict[str, int]] = field(default_factory=dict)
    thread_bindings: dict[int, dict[int, str]] = field(default_factory=dict)
    # window_id -> display name (window_name)
    window_display_names: dict[str, str] = field(default_factory=dict)
    # "user_id:thread_id" -> group chat_id (for supergroup forum topic routing)
    # IMPORTANT: This mapping is essential for supergroup/forum topic support.
    # Telegram Bot API requires group chat_id (negative number like -100xxx)
    # as the chat_id parameter when sending messages to forum topics.
    # Using user_id as chat_id will fail with "Message thread not found".
    # See: https://core.telegram.org/bots/api#sendmessage
    # History: originally added in 5afc111, erroneously removed in 26cb81f,
    # restored in PR #23.
    group_chat_ids: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._load_state()

    def _save_state(self) -> None:
        state: dict[str, Any] = {
            "window_states": {k: v.to_dict() for k, v in self.window_states.items()},
            "user_window_offsets": {
                str(uid): offsets for uid, offsets in self.user_window_offsets.items()
            },
            "thread_bindings": {
                str(uid): {str(tid): wid for tid, wid in bindings.items()}
                for uid, bindings in self.thread_bindings.items()
            },
            "window_display_names": self.window_display_names,
            "group_chat_ids": self.group_chat_ids,
        }
        atomic_write_json(config.state_file, state)
        logger.debug("State saved to %s", config.state_file)

    def _is_window_id(self, key: str) -> bool:
        """Check if a key looks like a tmux window ID (e.g. '@0', '@12')."""
        return key.startswith("@") and len(key) > 1 and key[1:].isdigit()

    def _load_state(self) -> None:
        """Load state synchronously during initialization.

        Detects old-format state (window_name keys without '@' prefix) and
        marks for migration on next startup re-resolution.
        """
        if config.state_file.exists():
            try:
                state = json.loads(config.state_file.read_text())
                self.window_states = {
                    k: WindowState.from_dict(v)
                    for k, v in state.get("window_states", {}).items()
                }
                self.user_window_offsets = {
                    int(uid): offsets
                    for uid, offsets in state.get("user_window_offsets", {}).items()
                }
                self.thread_bindings = {}
                for uid, bindings in state.get("thread_bindings", {}).items():
                    parsed: dict[int, str] = {}
                    for tid, wid in bindings.items():
                        if isinstance(wid, dict):
                            # Old format: {"window_id": "@4", "chat_id": ...}
                            parsed[int(tid)] = wid["window_id"]
                            # Migrate chat_id to group_chat_ids
                            chat_id = wid.get("chat_id")
                            if chat_id and int(chat_id) < 0:
                                key = f"{uid}:{tid}"
                                self.group_chat_ids[key] = int(chat_id)
                        else:
                            parsed[int(tid)] = wid
                    self.thread_bindings[int(uid)] = parsed
                self.window_display_names = state.get("window_display_names", {})
                self.group_chat_ids = {
                    k: int(v) for k, v in state.get("group_chat_ids", {}).items()
                }

                # Detect old format: keys that don't look like window IDs
                needs_migration = False
                for k in self.window_states:
                    if not self._is_window_id(k):
                        needs_migration = True
                        break
                if not needs_migration:
                    for bindings in self.thread_bindings.values():
                        for wid in bindings.values():
                            if not self._is_window_id(wid):
                                needs_migration = True
                                break
                        if needs_migration:
                            break

                if needs_migration:
                    logger.info(
                        "Detected old-format state (window_name keys), "
                        "will re-resolve on startup"
                    )
                    pass

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to load state: %s", e)
                self.window_states = {}
                self.user_window_offsets = {}
                self.thread_bindings = {}
                self.window_display_names = {}
                self.group_chat_ids = {}
                pass

    async def resolve_stale_ids(self) -> None:
        """Re-resolve persisted window IDs against live tmux windows.

        Called on startup. Handles two cases:
        1. Old-format migration: window_name keys → window_id keys
        2. Stale IDs: window_id no longer exists but display name matches a live window

        Builds {window_name: [window_id, ...]} from live windows, then remaps
        or drops entries. A display name shared by more than one live window
        is ambiguous — silently picking one (e.g. "last one wins") risks
        cross-wiring a stale entry onto the WRONG live window/session (RC3/RC4:
        window renames are never checked for uniqueness), so ambiguous names
        are dropped exactly like a no-match, never guessed.
        """
        windows = await tmux_manager.list_windows()
        live_by_name: dict[str, list[str]] = {}  # window_name -> [window_id, ...]
        live_ids: set[str] = set()
        for w in windows:
            live_by_name.setdefault(w.window_name, []).append(w.window_id)
            live_ids.add(w.window_id)

        def resolve_name(name: str) -> str | None:
            """Resolve a display name to a live window_id, refusing ambiguity."""
            candidates = live_by_name.get(name)
            if not candidates:
                return None
            if len(candidates) > 1:
                logger.warning(
                    "ambiguous display name %s matches %d windows; "
                    "dropping stale entry",
                    name,
                    len(candidates),
                )
                return None
            return candidates[0]

        changed = False

        # --- Migrate window_states ---
        new_window_states: dict[str, WindowState] = {}
        for key, ws in self.window_states.items():
            if self._is_window_id(key):
                if key in live_ids:
                    new_window_states[key] = ws
                else:
                    # Stale ID — try re-resolve by display name
                    display = self.window_display_names.get(key, ws.window_name or key)
                    new_id = resolve_name(display)
                    if new_id:
                        logger.info(
                            "Re-resolved stale window_id %s -> %s (name=%s)",
                            key,
                            new_id,
                            display,
                        )
                        new_window_states[new_id] = ws
                        ws.window_name = display
                        self.window_display_names[new_id] = display
                        self.window_display_names.pop(key, None)
                        changed = True
                    else:
                        logger.info(
                            "Dropping stale window_state: %s (name=%s)", key, display
                        )
                        changed = True
            else:
                # Old format: key is window_name
                new_id = resolve_name(key)
                if new_id:
                    logger.info("Migrating window_state key %s -> %s", key, new_id)
                    ws.window_name = key
                    new_window_states[new_id] = ws
                    self.window_display_names[new_id] = key
                    changed = True
                else:
                    logger.info(
                        "Dropping old-format window_state: %s (no live window)", key
                    )
                    changed = True
        self.window_states = new_window_states

        # --- Migrate thread_bindings ---
        for uid, bindings in self.thread_bindings.items():
            new_bindings: dict[int, str] = {}
            for tid, val in bindings.items():
                if self._is_window_id(val):
                    if val in live_ids:
                        new_bindings[tid] = val
                    else:
                        display = self.window_display_names.get(val, val)
                        new_id = resolve_name(display)
                        if new_id:
                            logger.info(
                                "Re-resolved thread binding %s -> %s (name=%s)",
                                val,
                                new_id,
                                display,
                            )
                            new_bindings[tid] = new_id
                            self.window_display_names[new_id] = display
                            changed = True
                        else:
                            logger.info(
                                "Dropping stale thread binding: user=%d, thread=%d, wid=%s",
                                uid,
                                tid,
                                val,
                            )
                            changed = True
                else:
                    # Old format: val is window_name
                    new_id = resolve_name(val)
                    if new_id:
                        logger.info("Migrating thread binding %s -> %s", val, new_id)
                        new_bindings[tid] = new_id
                        self.window_display_names[new_id] = val
                        changed = True
                    else:
                        logger.info(
                            "Dropping old-format thread binding: user=%d, thread=%d, name=%s",
                            uid,
                            tid,
                            val,
                        )
                        changed = True
            self.thread_bindings[uid] = new_bindings

        # Remove empty user entries
        empty_users = [uid for uid, b in self.thread_bindings.items() if not b]
        for uid in empty_users:
            del self.thread_bindings[uid]

        # --- Migrate user_window_offsets ---
        for uid, offsets in self.user_window_offsets.items():
            new_offsets: dict[str, int] = {}
            for key, offset in offsets.items():
                if self._is_window_id(key):
                    if key in live_ids:
                        new_offsets[key] = offset
                    else:
                        display = self.window_display_names.get(key, key)
                        new_id = resolve_name(display)
                        if new_id:
                            new_offsets[new_id] = offset
                            changed = True
                        else:
                            changed = True
                else:
                    new_id = resolve_name(key)
                    if new_id:
                        new_offsets[new_id] = offset
                        changed = True
                    else:
                        changed = True
            self.user_window_offsets[uid] = new_offsets

        if changed:
            self._save_state()
            logger.info("Startup re-resolution complete")

        # Clean up session_map.json: stale window IDs and old-format keys
        await self.sweep_stale_session_map_entries()
        await self._cleanup_old_format_session_map_keys()

    async def _accepted_session_map_names(self) -> set[str]:
        """Return the configured tmux session plus any grouped peers."""

        try:
            names = await tmux_manager.list_group_session_names()
        except Exception as e:
            logger.debug("Failed to list grouped tmux sessions: %s", e)
            names = set()
        return names or {config.tmux_session_name}

    @staticmethod
    def _split_session_map_key(key: str) -> tuple[str, str] | None:
        """Split a session_map key into (session_name, window_key)."""

        session_name, sep, window_key = key.partition(":")
        if not sep or not session_name or not window_key:
            return None
        return session_name, window_key

    def _select_canonical_session_map_entries(
        self,
        session_map: dict[str, Any],
        accepted_names: set[str],
    ) -> dict[str, tuple[str, dict[str, Any]]]:
        """Pick one (session_name, info) per window_id from accepted entries.

        Grouped tmux sessions share windows, so the SessionStart hook can write
        the same window_id under multiple session-name prefixes (e.g.
        ``ccbot:@48`` and ``ccbot-2:@48``). Iterating all of them would
        ping-pong ``window_state.session_id`` every poll cycle and flood the
        log. The configured ``tmux_session_name`` wins; otherwise pick the
        alphabetically-first peer so the choice is deterministic across runs.
        """

        candidates: dict[str, dict[str, dict[str, Any]]] = {}
        for key, info in session_map.items():
            parts = self._split_session_map_key(key)
            if parts is None:
                continue
            session_name, window_id = parts
            if session_name not in accepted_names:
                continue
            if not self._is_window_id(window_id):
                continue
            candidates.setdefault(window_id, {})[session_name] = info

        primary = config.tmux_session_name
        canonical: dict[str, tuple[str, dict[str, Any]]] = {}
        for window_id, by_name in candidates.items():
            if primary in by_name:
                canonical[window_id] = (primary, by_name[primary])
            else:
                chosen = sorted(by_name)[0]
                canonical[window_id] = (chosen, by_name[chosen])
        return canonical

    @staticmethod
    def _locked_session_map_update(
        mutate: Callable[[dict[str, Any]], bool],
    ) -> None:
        """Read-modify-write session_map.json under the hook's file lock.

        The SessionStart hook takes `session_map.lock` for its writes; bot-side
        writers must take the same lock or a hook write landing mid-update gets
        clobbered. Blocking I/O — call via asyncio.to_thread.
        ``mutate`` edits the dict in place and returns True if it changed.
        """
        map_file = config.session_map_file
        if not map_file.exists():
            return
        lock_path = map_file.with_suffix(".lock")
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                try:
                    session_map = json.loads(map_file.read_text())
                except (json.JSONDecodeError, OSError):
                    return
                if mutate(session_map):
                    atomic_write_json(map_file, session_map)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    async def _cleanup_old_format_session_map_keys(self) -> None:
        """Remove old-format keys (window_name instead of @window_id) from session_map.json."""
        accepted_names = await self._accepted_session_map_names()

        def mutate(session_map: dict[str, Any]) -> bool:
            old_keys = [
                key
                for key in session_map
                for parts in [self._split_session_map_key(key)]
                if parts is not None
                and parts[0] in accepted_names
                and not self._is_window_id(parts[1])
            ]
            for key in old_keys:
                del session_map[key]
            if old_keys:
                logger.info(
                    "Cleaned up %d old-format session_map keys: %s",
                    len(old_keys),
                    old_keys,
                )
            return bool(old_keys)

        await asyncio.to_thread(self._locked_session_map_update, mutate)

    async def sweep_stale_session_map_entries(self) -> None:
        """Remove entries for tmux windows that no longer exist on the socket.

        Window IDs are unique per tmux server and ccbot is session_map's only
        consumer, so any window-id-keyed entry whose window is gone is garbage
        regardless of its session-name prefix — including entries written by
        grouped peers, other tmux sessions on the socket, or (historically)
        other servers. Runs at startup and periodically from the maintenance
        loop, so entries disappear shortly after their window dies instead of
        accumulating until the next restart.
        """
        live_ids = await tmux_manager.list_all_window_ids()
        if live_ids is None:
            # Server unreachable — "unknown", not "no windows". Never sweep.
            return

        def mutate(session_map: dict[str, Any]) -> bool:
            stale_keys = [
                key
                for key in session_map
                for parts in [self._split_session_map_key(key)]
                if parts is not None
                and self._is_window_id(parts[1])
                and parts[1] not in live_ids
            ]
            for key in stale_keys:
                del session_map[key]
                logger.info("Removed stale session_map entry: %s", key)
            return bool(stale_keys)

        await asyncio.to_thread(self._locked_session_map_update, mutate)

    async def repoint_window_session(self, window_id: str, session_id: str) -> bool:
        """Re-point a window's session_map entry at a different session id.

        User-approved recovery for stale tracking (the divergence notice's
        button). Updates every key for this window_id; the monitor observes
        the change on its next poll, drops the old session, and starts
        tracking the new one at end-of-file. The old session's
        transcript_size_at_start is dropped with it: left in place, it would
        seed the new session's read offset and replay its whole transcript.
        Returns False when no entry exists for the window.
        """
        repointed = False

        def mutate(session_map: dict[str, Any]) -> bool:
            nonlocal repointed
            for key, info in session_map.items():
                parts = self._split_session_map_key(key)
                if parts is not None and parts[1] == window_id:
                    info["session_id"] = session_id
                    info.pop("transcript_size_at_start", None)
                    repointed = True
            if repointed:
                logger.info("Re-pointed window %s -> session %s", window_id, session_id)
            return repointed

        await asyncio.to_thread(self._locked_session_map_update, mutate)
        return repointed

    # --- Display name management ---

    def get_display_name(self, window_id: str) -> str:
        """Get display name for a window_id, fallback to window_id itself."""
        return self.window_display_names.get(window_id, window_id)

    def update_display_name(self, window_id: str, new_name: str) -> None:
        """Update the display name for a window and persist state."""
        self.window_display_names[window_id] = new_name
        # Also update WindowState.window_name if it exists
        if window_id in self.window_states:
            self.window_states[window_id].window_name = new_name
        self._save_state()
        logger.info("Updated display name: window_id %s -> '%s'", window_id, new_name)

    # --- Group chat ID management (supergroup forum topic routing) ---

    def set_group_chat_id(
        self, user_id: int, thread_id: int | None, chat_id: int
    ) -> None:
        """Store the group chat_id for a user+thread combination.

        In supergroups with forum topics, messages must be sent to the group's
        chat_id (negative number like -100xxx) rather than the user's personal ID.
        Telegram's Bot API rejects message_thread_id when chat_id is a private
        user ID — the thread only exists within the group context.

        DO NOT REMOVE this method or the group_chat_ids mapping.
        Without it, all outbound messages in forum topics fail with
        "Message thread not found". See commit history: 5afc111 → 26cb81f → PR #23.
        """
        tid = thread_id or 0
        key = f"{user_id}:{tid}"
        if self.group_chat_ids.get(key) != chat_id:
            self.group_chat_ids[key] = chat_id
            self._save_state()
            logger.debug(
                "Stored group chat_id: user=%d, thread=%s, chat_id=%d",
                user_id,
                thread_id,
                chat_id,
            )

    def resolve_chat_id(self, user_id: int, thread_id: int | None = None) -> int:
        """Resolve the correct chat_id for sending messages.

        Returns the stored group chat_id when a thread_id is present and a
        mapping exists, otherwise falls back to user_id (for private chats).

        Every outbound Telegram API call (send_message, edit_message_text,
        delete_message, send_chat_action, edit_forum_topic, etc.) MUST use
        this method instead of raw user_id. Using user_id directly breaks
        supergroup forum topic routing.
        """
        if thread_id is not None:
            key = f"{user_id}:{thread_id}"
            group_id = self.group_chat_ids.get(key)
            if group_id is not None:
                return group_id
        return user_id

    async def wait_for_session_map_entry(
        self, window_id: str, timeout: float = 5.0, interval: float = 0.5
    ) -> bool:
        """Poll session_map.json until an entry for window_id appears.

        Accepts the configured tmux session name plus any grouped peers.
        Returns True if the entry was found within timeout, False otherwise.
        """
        logger.debug(
            "Waiting for session_map entry: window_id=%s, timeout=%.1f",
            window_id,
            timeout,
        )
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                if config.session_map_file.exists():
                    async with aiofiles.open(config.session_map_file, "r") as f:
                        content = await f.read()
                    session_map = json.loads(content)
                    accepted_names = await self._accepted_session_map_names()
                    for session_name in accepted_names:
                        info = session_map.get(f"{session_name}:{window_id}", {})
                        if info.get("session_id"):
                            # Found — load into window_states immediately
                            logger.debug(
                                "session_map entry found for window_id %s", window_id
                            )
                            await self.load_session_map()
                            return True
            except (json.JSONDecodeError, OSError):
                pass
            await asyncio.sleep(interval)
        logger.warning(
            "Timed out waiting for session_map entry: window_id=%s", window_id
        )
        return False

    async def load_session_map(self) -> None:
        """Read session_map.json and update window_states with new session associations.

        Keys in session_map are formatted as "tmux_session:window_id" (e.g. "ccbot:@12").
        Accepts entries under our tmux_session_name or any grouped peer session.
        Also cleans up window_states entries not in current session_map.
        Updates window_display_names from the "window_name" field in values.
        When a new session_id is applied to an unpinned window, also records
        the hook's "transcript_size_at_start" as WindowState.session_start_size
        for the session monitor to seed its read offset from.
        """
        if not config.session_map_file.exists():
            return
        try:
            async with aiofiles.open(config.session_map_file, "r") as f:
                content = await f.read()
            session_map = json.loads(content)
        except (json.JSONDecodeError, OSError):
            return

        accepted_names = await self._accepted_session_map_names()
        canonical = self._select_canonical_session_map_entries(
            session_map, accepted_names
        )
        valid_wids: set[str] = set()
        changed = False

        for window_id, (_session_name, info) in canonical.items():
            valid_wids.add(window_id)
            new_sid = info.get("session_id", "")
            new_cwd = info.get("cwd", "")
            new_wname = info.get("window_name", "")
            if not new_sid:
                continue
            state = self.get_window_state(window_id)
            if state.pinned_over is not None:
                if new_sid == state.pinned_over:
                    # Hook still reports the outranked sid — keep the manual
                    # override in place, only sync display name below.
                    pass
                else:
                    logger.info(
                        "unpinning window %s: hook reports new session %s",
                        window_id,
                        new_sid,
                    )
                    state.pinned_over = None
                    if state.session_id != new_sid or state.cwd != new_cwd:
                        logger.info(
                            "Session map: window_id %s updated sid=%s, cwd=%s",
                            window_id,
                            new_sid,
                            new_cwd,
                        )
                        state.session_id = new_sid
                        state.cwd = new_cwd
                    changed = True
            elif state.session_id != new_sid or state.cwd != new_cwd:
                logger.info(
                    "Session map: window_id %s updated sid=%s, cwd=%s",
                    window_id,
                    new_sid,
                    new_cwd,
                )
                state.session_id = new_sid
                state.cwd = new_cwd
                raw_start_size = info.get("transcript_size_at_start", -1)
                state.session_start_size = (
                    raw_start_size
                    if isinstance(raw_start_size, int)
                    and not isinstance(raw_start_size, bool)
                    else -1
                )
                changed = True
            # Update display name
            if new_wname:
                state.window_name = new_wname
                if self.window_display_names.get(window_id) != new_wname:
                    self.window_display_names[window_id] = new_wname
                    changed = True

        # Clean up window_states entries not in current session_map. Pinned
        # entries (manual override / hook-timeout recovery) may legitimately
        # have no session_map entry yet — keep them; resolve_stale_ids' live
        # tmux-window check still drops truly dead windows at startup.
        stale_wids = [
            w
            for w, ws in self.window_states.items()
            if w and w not in valid_wids and ws.pinned_over is None
        ]
        for wid in stale_wids:
            logger.info("Removing stale window_state: %s", wid)
            del self.window_states[wid]
            changed = True

        if changed:
            self._save_state()

    # --- Window state management ---

    def get_window_state(self, window_id: str) -> WindowState:
        """Get or create window state."""
        if window_id not in self.window_states:
            self.window_states[window_id] = WindowState()
        return self.window_states[window_id]

    def clear_window_session(self, window_id: str) -> None:
        """Clear session association for a window (e.g., after /clear command)."""
        state = self.get_window_state(window_id)
        state.session_id = ""
        state.pinned_over = None
        self._save_state()
        logger.info("Cleared session for window_id %s", window_id)

    def set_claude_launch_version(self, window_id: str, version: str) -> None:
        """Record the installed claude version active when this window launched.

        Used by `update_watcher.maybe_notify_update_or_failure` to decide
        whether to send this window a one-time "update available" notice.
        Persisted per-window so an upgrade in one session does not silence the
        upgrade signal for others.
        """
        state = self.get_window_state(window_id)
        if state.claude_launch_version == version:
            return
        state.claude_launch_version = version
        self._save_state()
        logger.info(
            "Set claude_launch_version for window_id %s -> %s", window_id, version
        )

    @staticmethod
    def _encode_cwd(cwd: str) -> str:
        """Encode a cwd path to match Claude Code's project directory naming.

        Replaces all non-alphanumeric characters (except dash) with dashes.
        E.g. /home/user_name/Code/project -> -home-user-name-Code-project
        """
        return re.sub(r"[^a-zA-Z0-9-]", "-", cwd)

    def _build_session_file_path(self, session_id: str, cwd: str) -> Path | None:
        """Build the direct file path for a session from session_id and cwd."""
        if not session_id or not cwd:
            return None
        encoded_cwd = self._encode_cwd(cwd)
        return config.claude_projects_path / encoded_cwd / f"{session_id}.jsonl"

    async def _get_session_direct(
        self, session_id: str, cwd: str
    ) -> ClaudeSession | None:
        """Get a ClaudeSession directly from session_id and cwd (no scanning)."""
        file_path = self._build_session_file_path(session_id, cwd)

        # Fallback: glob search if direct path doesn't exist
        if not file_path or not file_path.exists():
            pattern = f"*/{session_id}.jsonl"
            matches = list(config.claude_projects_path.glob(pattern))
            if matches:
                file_path = matches[0]
                logger.debug("Found session via glob: %s", file_path)
            else:
                return None

        # Single pass: read file once, extract summary + name + count messages.
        # Keep the LAST occurrence of custom-title / agent-name (users can
        # rename multiple times; the latest one wins).
        summary = ""
        custom_title = ""
        agent_name = ""
        last_user_msg = ""
        message_count = 0
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    message_count += 1
                    try:
                        data = json.loads(line)
                        entry_type = data.get("type")
                        if entry_type == "summary":
                            s = data.get("summary", "")
                            if s:
                                summary = s
                        elif entry_type == "custom-title":
                            t = data.get("customTitle", "")
                            if t:
                                custom_title = t
                        elif entry_type == "agent-name":
                            n = data.get("agentName", "")
                            if n:
                                agent_name = n
                        elif TranscriptParser.is_user_message(data):
                            parsed = TranscriptParser.parse_message(data)
                            if parsed and parsed.text.strip():
                                last_user_msg = parsed.text.strip()
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return None

        if not summary:
            summary = last_user_msg[:50] if last_user_msg else "Untitled"

        return ClaudeSession(
            session_id=session_id,
            summary=summary,
            message_count=message_count,
            file_path=str(file_path),
            name=custom_title or agent_name,
        )

    # --- Directory session listing ---

    async def list_sessions_for_directory(self, cwd: str) -> list[ClaudeSession]:
        """List existing Claude sessions for a directory.

        Encodes the cwd path to find the project directory under
        ~/.claude/projects/{encoded_cwd}/, globs *.jsonl files, and
        extracts summary info from each.

        Returns a list sorted by mtime (most recent first), capped at 10.
        """
        encoded_cwd = self._encode_cwd(cwd)
        project_dir = config.claude_projects_path / encoded_cwd
        if not project_dir.is_dir():
            return []

        # Collect JSONL files sorted by mtime (newest first)
        jsonl_files = sorted(
            project_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Skip sessions-index and cap at 10
        sessions: list[ClaudeSession] = []
        for f in jsonl_files:
            if f.stem == "sessions-index":
                continue
            if len(sessions) >= 10:
                break
            session_id = f.stem
            session = await self._get_session_direct(session_id, cwd)
            if session and session.message_count > 0:
                sessions.append(session)
        return sessions

    # --- Window → Session resolution ---

    async def resolve_session_for_window(self, window_id: str) -> ClaudeSession | None:
        """Resolve a tmux window to the best matching Claude session.

        Uses persisted session_id + cwd to construct file path directly.
        Returns None if no session is associated with this window.
        """
        state = self.get_window_state(window_id)

        if not state.session_id or not state.cwd:
            return None

        session = await self._get_session_direct(state.session_id, state.cwd)
        if session:
            return session

        # File no longer exists, clear state
        logger.warning(
            "Session file no longer exists for window_id %s (sid=%s, cwd=%s)",
            window_id,
            state.session_id,
            state.cwd,
        )
        state.session_id = ""
        state.cwd = ""
        self._save_state()
        return None

    # --- User window offset management ---

    def update_user_window_offset(
        self, user_id: int, window_id: str, offset: int
    ) -> None:
        """Update the user's last read offset for a window."""
        if user_id not in self.user_window_offsets:
            self.user_window_offsets[user_id] = {}
        self.user_window_offsets[user_id][window_id] = offset
        self._save_state()

    # --- Thread binding management ---

    def bind_thread(
        self, user_id: int, thread_id: int, window_id: str, window_name: str = ""
    ) -> bool:
        """Bind a Telegram topic thread to a tmux window.

        Enforces the '1 topic = 1 window' invariant at the source: a
        window_id already bound to a DIFFERENT (user_id, thread_id) pair is
        refused rather than silently double-bound (RC3/RC4 — without this,
        two topics can end up routed to the same tmux window/session, so
        Claude's replies bleed into both and either topic's input lands in
        the shared session). Re-binding the SAME (user_id, thread_id) to the
        same window_id (e.g. re-confirming an existing binding) stays
        allowed and idempotent.

        Args:
            user_id: Telegram user ID
            thread_id: Telegram topic thread ID
            window_id: Tmux window ID (e.g. '@0')
            window_name: Display name for the window (optional)

        Returns:
            True if the binding was made. False if window_id is already
            bound to a different (user_id, thread_id) pair — the caller must
            treat the topic as NOT bound in that case.
        """
        for bound_user, bound_thread, bound_window in self.iter_thread_bindings():
            if bound_window == window_id and (bound_user, bound_thread) != (
                user_id,
                thread_id,
            ):
                logger.warning(
                    "Refusing to bind thread %d (user %d) to window_id %s: "
                    "already bound to thread %d (user %d)",
                    thread_id,
                    user_id,
                    window_id,
                    bound_thread,
                    bound_user,
                )
                return False

        if user_id not in self.thread_bindings:
            self.thread_bindings[user_id] = {}
        self.thread_bindings[user_id][thread_id] = window_id
        if window_name:
            self.window_display_names[window_id] = window_name
        self._save_state()
        display = window_name or self.get_display_name(window_id)
        logger.info(
            "Bound thread %d -> window_id %s (%s) for user %d",
            thread_id,
            window_id,
            display,
            user_id,
        )
        return True

    def unbind_thread(self, user_id: int, thread_id: int) -> str | None:
        """Remove a thread binding. Returns the previously bound window_id, or None."""
        bindings = self.thread_bindings.get(user_id)
        if not bindings or thread_id not in bindings:
            return None
        window_id = bindings.pop(thread_id)
        if not bindings:
            del self.thread_bindings[user_id]
        self._save_state()
        logger.info(
            "Unbound thread %d (was %s) for user %d",
            thread_id,
            window_id,
            user_id,
        )
        return window_id

    def get_window_for_thread(self, user_id: int, thread_id: int) -> str | None:
        """Look up the window_id bound to a thread."""
        bindings = self.thread_bindings.get(user_id)
        if not bindings:
            return None
        return bindings.get(thread_id)

    def resolve_window_for_thread(
        self,
        user_id: int,
        thread_id: int | None,
    ) -> str | None:
        """Resolve the tmux window_id for a user's thread.

        Returns None if thread_id is None or the thread is not bound.
        """
        if thread_id is None:
            return None
        return self.get_window_for_thread(user_id, thread_id)

    def iter_thread_bindings(self) -> Iterator[tuple[int, int, str]]:
        """Iterate all thread bindings as (user_id, thread_id, window_id).

        Provides encapsulated access to thread_bindings without exposing
        the internal data structure directly.
        """
        for user_id, bindings in self.thread_bindings.items():
            for thread_id, window_id in bindings.items():
                yield user_id, thread_id, window_id

    async def find_users_for_session(
        self,
        session_id: str,
    ) -> list[tuple[int, str, int]]:
        """Find all users whose thread-bound window maps to the given session_id.

        Returns list of (user_id, window_id, thread_id) tuples.
        """
        result: list[tuple[int, str, int]] = []
        # Materialize before awaiting inside the loop: a concurrent unbind
        # mutating thread_bindings mid-iteration would otherwise raise
        # RuntimeError (dict changed size during iteration), which the
        # dispatch path swallows, silently dropping the message (RC7/f45).
        for user_id, thread_id, window_id in list(self.iter_thread_bindings()):
            resolved = await self.resolve_session_for_window(window_id)
            if resolved and resolved.session_id == session_id:
                result.append((user_id, window_id, thread_id))
        return result

    # --- Tmux helpers ---

    async def send_to_window(self, window_id: str, text: str) -> tuple[bool, str]:
        """Send text to a tmux window by ID."""
        display = self.get_display_name(window_id)
        logger.debug(
            "send_to_window: window_id=%s (%s), text_len=%d",
            window_id,
            display,
            len(text),
        )
        window = await tmux_manager.find_window_by_id(window_id)
        if not window:
            return False, "Window not found (may have been closed)"
        success = await tmux_manager.send_keys(window.window_id, text)
        if success:
            return True, f"Sent to {display}"
        return False, "Failed to send keys"

    # --- Message history ---

    async def get_recent_messages(
        self,
        window_id: str,
        *,
        start_byte: int = 0,
        end_byte: int | None = None,
    ) -> tuple[list[dict], int]:
        """Get user/assistant messages for a window's session.

        Resolves window → session, then reads the JSONL.
        Supports byte range filtering via start_byte/end_byte.
        Returns (messages, total_count).
        """
        session = await self.resolve_session_for_window(window_id)
        if not session or not session.file_path:
            return [], 0

        file_path = Path(session.file_path)
        if not file_path.exists():
            return [], 0

        # Read JSONL entries (optionally filtered by byte range)
        entries: list[dict] = []
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                if start_byte > 0:
                    await f.seek(start_byte)

                while True:
                    # Check byte limit before reading
                    if end_byte is not None:
                        current_pos = await f.tell()
                        if current_pos >= end_byte:
                            break

                    line = await f.readline()
                    if not line:
                        break

                    data = TranscriptParser.parse_line(line)
                    if data:
                        entries.append(data)
        except OSError as e:
            logger.error("Error reading session file %s: %s", file_path, e)
            return [], 0

        parsed_entries, _ = TranscriptParser.parse_entries(entries)
        all_messages = [
            {
                "role": e.role,
                "text": e.text,
                "content_type": e.content_type,
                "timestamp": e.timestamp,
            }
            for e in parsed_entries
        ]

        return all_messages, len(all_messages)


session_manager = SessionManager()
