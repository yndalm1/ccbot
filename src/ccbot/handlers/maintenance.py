"""Periodic local-state maintenance, decoupled from Telegram-facing polling.

A dedicated background loop for hygiene and staleness detection on ccbot's
own state. Kept separate from status_polling on purpose: that loop's 60s
block makes network calls against Telegram (topic probes) and demonstrably
stalls on timeouts; maintenance is local I/O and must not queue behind it.

Steps (each independently guarded, one failure never blocks the others):
  - session_map sweep: drop entries whose tmux window no longer exists.
  - hook-failure notices: tail <CCBOT_DIR>/hook_failures.jsonl (written by
    the SessionStart hook when it cannot map a session to a window) and
    surface each failure in the topic(s) bound to that cwd.
  - divergence notices: a bound window whose tracked transcript has frozen
    while a different, untracked transcript grows in the same project
    directory has probably had its session re-created behind ccbot's back
    (the 2026-07-04 daemon incident signature). A tracked transcript that
    never appeared on disk counts as frozen too. Post a notice with a
    human-approved "Re-point" button — never rebind automatically.

Key components: MAINTENANCE_INTERVAL, run_maintenance_once(), maintenance_loop().
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import RetryAfter

from ..config import config
from ..monitor_state import MonitorState
from ..session import session_manager
from .callback_data import CB_REPOINT
from .message_sender import safe_send

logger = logging.getLogger(__name__)

# Local-I/O hygiene cadence. Unrelated to status_polling's TOPIC_CHECK_INTERVAL
# (a Telegram API budget) even though the values coincide.
MAINTENANCE_INTERVAL = 60.0  # seconds

# Divergence detection: how long the tracked transcript must be quiet before
# a growing sibling counts as evidence, and how fresh that sibling must be.
# Long thinking turns legitimately pause transcript writes for hours, so the
# signal is never "tracked file quiet" alone — only "quiet while another,
# untracked file in the same project dir is actively growing".
DIVERGENCE_FROZEN_AFTER = 300.0  # seconds
DIVERGENCE_CANDIDATE_FRESH = 300.0  # seconds
# Candidate must grow across this many consecutive maintenance ticks.
DIVERGENCE_GROWTH_TICKS = 2

# --- module state (reset only on process restart) ---

# Byte offset into hook_failures.jsonl. None = first run: seek to EOF so a
# restart never replays historical failures into topics.
_hook_failures_offset: int | None = None

# window_id -> divergence episode: {"sid", "size", "ticks", "notified"}
_divergence: dict[str, dict[str, Any]] = {}

# window_id -> (tracked session_id, when its transcript was first seen missing)
_missing_since: dict[str, tuple[str, float]] = {}


def _hook_failures_file() -> Path:
    return config.session_map_file.parent / "hook_failures.jsonl"


# --- step 1: session_map sweep ------------------------------------------


async def _sweep_session_map() -> None:
    await session_manager.sweep_stale_session_map_entries()


# --- step 2: hook-failure notices ----------------------------------------


async def _check_hook_failures(bot: Bot) -> None:
    """Tail hook_failures.jsonl and surface new failures in bound topics."""
    global _hook_failures_offset
    path = _hook_failures_file()
    try:
        size = path.stat().st_size
    except OSError:
        return

    if _hook_failures_offset is None or _hook_failures_offset > size:
        # First run (skip history) or truncated file.
        _hook_failures_offset = size
        return
    if size == _hook_failures_offset:
        return

    offset = _hook_failures_offset

    def _read() -> bytes:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read()

    data = await asyncio.to_thread(_read)
    _hook_failures_offset = offset + len(data)

    for line in data.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            failure = json.loads(line)
        except json.JSONDecodeError:
            continue
        cwd = failure.get("cwd", "")
        reason = failure.get("reason", "unknown")
        session_id = failure.get("session_id", "")
        if not cwd:
            continue
        text = (
            "⚠️ A Claude session in this directory couldn't be mapped to "
            f"this window: {reason}.\n"
            f"Session {session_id[:8]}… may be untracked — messages from it "
            "won't reach Telegram."
        )
        await _notify_topics_for_cwd(bot, cwd, text)


async def _notify_topics_for_cwd(bot: Bot, cwd: str, text: str) -> None:
    """Send a notice (MarkdownV2, falling back to plain text) to every topic
    bound to a window in cwd."""
    for user_id, thread_id, window_id in list(session_manager.iter_thread_bindings()):
        ws = session_manager.window_states.get(window_id)
        if ws is None or ws.cwd != cwd:
            continue
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        try:
            await safe_send(bot, chat_id, text, message_thread_id=thread_id)
        except RetryAfter as e:
            logger.warning(
                "Hook-failure notice rate-limited for chat %s, skipping: %s",
                chat_id,
                e,
            )
        except Exception as e:
            logger.error("Failed to send hook-failure notice: %s", e)


# --- step 3: divergence notices -------------------------------------------


def _newest_untracked_sibling(
    tracked_path: Path,
    tracked_sid: str,
    other_tracked_sids: set[str],
    tracked_mtime: float,
    now: float,
) -> tuple[str, int] | None:
    """Newest actively-written .jsonl in the tracked file's project dir that
    no window tracks. Returns (session_id, size) or None. Blocking I/O —
    call via asyncio.to_thread."""
    best: tuple[float, str, int] | None = None
    try:
        siblings = list(tracked_path.parent.glob("*.jsonl"))
    except OSError:
        return None
    for p in siblings:
        sid = p.stem
        if sid == tracked_sid or sid in other_tracked_sids:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime <= tracked_mtime or now - st.st_mtime > (
            DIVERGENCE_CANDIDATE_FRESH
        ):
            continue
        if best is None or st.st_mtime > best[0]:
            best = (st.st_mtime, sid, st.st_size)
    if best is None:
        return None
    return best[1], best[2]


async def _check_divergence(bot: Bot) -> None:
    """Detect bound windows whose session moved to an untracked transcript."""
    now = time.time()
    monitor_state = MonitorState(state_file=config.monitor_state_file)
    await asyncio.to_thread(monitor_state.load)

    bindings = []
    cwd_counts: dict[str, int] = {}
    for user_id, thread_id, window_id in list(session_manager.iter_thread_bindings()):
        ws = session_manager.window_states.get(window_id)
        if ws is None or not ws.session_id or not ws.cwd:
            continue
        cwd_counts[ws.cwd] = cwd_counts.get(ws.cwd, 0) + 1
        bindings.append((user_id, thread_id, window_id, ws))

    all_tracked_sids = {
        ws.session_id for ws in session_manager.window_states.values() if ws.session_id
    }

    for user_id, thread_id, window_id, ws in bindings:
        if cwd_counts[ws.cwd] > 1:
            # Two bound windows on one directory: a growing sibling can't be
            # attributed to either. Same refusal principle as the hook.
            _divergence.pop(window_id, None)
            continue

        tracked = monitor_state.get_session(ws.session_id)
        if tracked is not None:
            tracked_path: Path | None = Path(tracked.file_path)
        else:
            tracked_path = session_manager._build_session_file_path(
                ws.session_id, ws.cwd
            )
        if tracked_path is None:
            _divergence.pop(window_id, None)
            continue
        try:
            tracked_mtime = tracked_path.stat().st_mtime
            tracked_missing = False
            _missing_since.pop(window_id, None)
        except OSError:
            # A transcript that never appeared on disk (a mapping stolen by a
            # `claude -p --no-session-persistence` child) is frozen from the
            # moment it was first seen missing. A fresh session writes its
            # transcript only with its first message, so the same quiet
            # period applies before it counts.
            seen = _missing_since.get(window_id)
            if seen is None or seen[0] != ws.session_id:
                seen = (ws.session_id, now)
                _missing_since[window_id] = seen
            tracked_mtime = seen[1]
            tracked_missing = True
        if now - tracked_mtime < DIVERGENCE_FROZEN_AFTER:
            _divergence.pop(window_id, None)
            continue

        candidate = await asyncio.to_thread(
            _newest_untracked_sibling,
            tracked_path,
            ws.session_id,
            all_tracked_sids - {ws.session_id},
            tracked_mtime,
            now,
        )
        if candidate is None:
            _divergence.pop(window_id, None)
            continue
        cand_sid, cand_size = candidate

        episode = _divergence.get(window_id)
        if episode is None or episode["sid"] != cand_sid:
            _divergence[window_id] = {
                "sid": cand_sid,
                "size": cand_size,
                "ticks": 1,
                "notified": False,
            }
            continue
        if cand_size > episode["size"]:
            episode["ticks"] += 1
            episode["size"] = cand_size
        if episode["ticks"] >= DIVERGENCE_GROWTH_TICKS and not episode["notified"]:
            episode["notified"] = True
            await _send_divergence_notice(
                bot,
                user_id,
                thread_id,
                window_id,
                tracked_sid=ws.session_id,
                candidate_sid=cand_sid,
                frozen_secs=now - tracked_mtime,
                tracked_missing=tracked_missing,
            )


async def _send_divergence_notice(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
    *,
    tracked_sid: str,
    candidate_sid: str,
    frozen_secs: float,
    tracked_missing: bool = False,
) -> None:
    """One notice per episode: stale tracking suspected, re-point on approval."""
    minutes = int(frozen_secs // 60)
    if tracked_missing:
        tracked_state = f"has had no transcript on disk for {minutes} min"
    else:
        tracked_state = f"last wrote {minutes} min ago"
    text = (
        "⚠️ Session tracking for this window may be stale.\n"
        f"Tracked session {tracked_sid[:8]}… {tracked_state}, "
        f"while {candidate_sid[:8]}… is actively writing in the same "
        "directory.\n"
        "If this window's conversation moved (e.g. /clear or resume), tap "
        "to re-point."
    )
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"Re-point to {candidate_sid[:8]}…",
                    callback_data=f"{CB_REPOINT}{window_id}:{candidate_sid}"[:64],
                )
            ]
        ]
    )
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    try:
        await safe_send(
            bot,
            chat_id,
            text,
            message_thread_id=thread_id,
            reply_markup=keyboard,
        )
        logger.info(
            "Divergence notice sent: window=%s tracked=%s candidate=%s",
            window_id,
            tracked_sid,
            candidate_sid,
        )
    except RetryAfter as e:
        logger.warning(
            "Divergence notice rate-limited for chat %s, skipping: %s",
            chat_id,
            e,
        )
    except Exception as e:
        logger.error("Failed to send divergence notice: %s", e)


# --- loop ------------------------------------------------------------------


async def run_maintenance_once(bot: Bot) -> None:
    """Run every maintenance step, isolating failures per step."""
    try:
        await _sweep_session_map()
    except Exception as e:
        logger.error("session_map sweep failed: %s", e)
    try:
        await _check_hook_failures(bot)
    except Exception as e:
        logger.error("hook-failure check failed: %s", e)
    try:
        await _check_divergence(bot)
    except Exception as e:
        logger.error("divergence check failed: %s", e)


async def maintenance_loop(bot: Bot) -> None:
    """Background task: run maintenance steps every MAINTENANCE_INTERVAL."""
    logger.info("Maintenance loop started (interval: %ss)", MAINTENANCE_INTERVAL)
    while True:
        await asyncio.sleep(MAINTENANCE_INTERVAL)
        await run_maintenance_once(bot)
