"""Tmux session/window management via libtmux.

Wraps libtmux to provide async-friendly operations on a single tmux session:
  - list_windows / find_window_by_name: discover Claude Code windows.
  - capture_pane: read terminal content (plain or with ANSI colors).
  - send_keys: forward user input or control keys to a window.
  - create_window / kill_window: lifecycle management.

All blocking libtmux calls are wrapped in asyncio.to_thread() and bounded by
`TmuxManager._bounded()` (timeout `_TMUX_SUBPROCESS_TIMEOUT`), so a wedged
tmux server can't hang a caller coroutine forever.

`send_keys` treats delivery as an *observed* effect, not an assumed one:
literal text goes through a raw `tmux send-keys -l --` subprocess (so its
exit code — swallowed by vendored libtmux — is checked, and a leading '-'
in the text can't be mis-parsed as an option), the text is polled for in
`capture-pane` before Enter is sent (so a slow-redrawing TUI can't turn
Enter into a stray newline), and the whole per-window sequence runs under
a lock (`_get_send_lock`) so concurrent sends to one window can't interleave.

Key class: TmuxManager (singleton instantiated as `tmux_manager`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import pwd
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import libtmux

from .config import SENSITIVE_ENV_VARS, config
from .utils import parse_group_session_names

logger = logging.getLogger(__name__)

# Common claude install locations prepended to PATH inside the window's shell
# command. Mirrors update_watcher._FALLBACK_BIN_DIRS — they exist for the same
# reason (service-manager PATH is often stripped of ~/.local/bin etc.).
_FALLBACK_PATH = "$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin"

# Upper bound for blocking raw `tmux` subprocess calls, so a wedged tmux server
# can't pin a thread-pool thread (and hang /restart) indefinitely.
_TMUX_SUBPROCESS_TIMEOUT = 10.0

# Verify-before-Enter: number of capture-pane polls used to confirm literal
# text actually landed in the pane before submitting Enter, and the spacing
# between them. A wedged/slow-redrawing TUI would otherwise turn a
# fixed-delay Enter into a stray newline inside the input box. The ~4s
# total window absorbs a TUI that is slow to echo a multi-KB burst (a
# 1.2k-char send was observed to miss the previous ~0.6s window); on
# success the loop exits at first sight of the text, so the widening
# costs nothing on the happy path.
_SEND_VERIFY_ATTEMPTS = 8
_SEND_VERIFY_POLL_INTERVAL = 0.5

_T = TypeVar("_T")


def _user_shell() -> str:
    """The user's login shell, with /bin/zsh as a last-resort fallback."""
    try:
        shell = pwd.getpwuid(os.getuid()).pw_shell
        if shell:
            return shell
    except (KeyError, OSError):
        pass
    return "/bin/zsh"


def build_claude_command(
    base_command: str,
    *,
    permission_mode: str = "",
    resume_session_id: str | None = None,
) -> str:
    """Compose the shell command used to launch Claude Code in a new window.

    `--permission-mode` is placed before `--resume` so mode applies to the
    resumed session. Both args are only appended when set.
    """
    cmd = base_command
    if permission_mode:
        cmd = f"{cmd} --permission-mode {permission_mode}"
    if resume_session_id:
        cmd = f"{cmd} --resume {resume_session_id}"
    return cmd


def build_window_shell_cmd(claude_cmd: str, user_shell: str) -> str:
    """Compose the value passed as `window_shell` to tmux's new-window.

    tmux executes this via `/bin/sh -c "<value>"`, which is non-interactive
    and skips zsh init entirely — eliminating the shell-init race where
    prompts during `.zshrc` (e.g. oh-my-zsh's update prompt) consume the
    first keystrokes of a `send_keys` payload.

    PATH is prepended with common claude install locations so the binary
    is findable even when the bot runs under a service manager with a
    stripped-down PATH. The trailing `; exec <user_shell>` keeps a debug
    shell in the pane after claude exits, instead of the window closing.

    CLAUDE_CODE_ENTRYPOINT is unset so the pane's claude computes its own
    ("cli" for the TUI): Claude Code keeps an inherited value, and the hook
    refuses to register a session whose entrypoint reads "sdk*".
    """
    return (
        f"unset CLAUDE_CODE_ENTRYPOINT; "
        f'PATH="{_FALLBACK_PATH}:$PATH" {claude_cmd}; exec {user_shell}'
    )


def _text_visible_in_pane(pane_text: str, sent_text: str) -> bool:
    """Whitespace-insensitive tail check: did `sent_text` reach the pane?

    TUIs re-wrap/re-flow text as it's typed (word-wrap, prompt padding), so
    an exact substring match on raw captured text would false-negative on
    legitimately-landed input, and a needle that lands split across two
    wrapped pane lines would false-negative too. Comparing only the trailing
    slice of what was sent, with all whitespace stripped from both the
    needle and the last ~15 lines of the pane, tolerates wrapping while
    still confirming the text actually reached the pane rather than being
    swallowed by a slow-redrawing TUI.
    """
    needle = "".join(sent_text[-40:].split())
    if not needle:
        return True
    haystack = "".join("".join(pane_text.splitlines()[-15:]).split())
    return needle in haystack


@dataclass
class TmuxWindow:
    """Information about a tmux window."""

    window_id: str
    window_name: str
    cwd: str  # Current working directory
    pane_current_command: str = ""  # Process running in active pane


class TmuxManager:
    """Manages tmux windows for Claude Code sessions."""

    def __init__(self, session_name: str | None = None):
        """Initialize tmux manager.

        Args:
            session_name: Name of the tmux session to use (default from config)
        """
        self.session_name = session_name or config.tmux_session_name
        self.socket_name = config.tmux_socket_name
        self._server: libtmux.Server | None = None
        # One lock per window, serializing send_keys sequences so concurrent
        # sends to the same window (e.g. a queued message racing an
        # interactive-UI button press) can't interleave their keystrokes and
        # Enter presses (RC11 / f37).
        self._send_locks: dict[str, asyncio.Lock] = {}

    @property
    def server(self) -> libtmux.Server:
        """Get or create tmux server connection on ccbot's dedicated socket."""
        if self._server is None:
            self._server = libtmux.Server(socket_name=self.socket_name)
        return self._server

    def _tmux_argv(self, *args: str) -> list[str]:
        """Build a `tmux -L <socket> ...` argv for raw subprocess calls.

        Daemon-side `tmux` invocations run outside any pane, so (unlike the hook,
        which inherits ``$TMUX``) they have no way to find ccbot's server and
        would default to the shared socket. Every raw call must go through here.
        """
        return ["tmux", "-L", self.socket_name, *args]

    def _get_send_lock(self, window_id: str) -> asyncio.Lock:
        """Return (creating if needed) the lock serializing send_keys for one
        window. See `_send_locks` for why this exists."""
        lock = self._send_locks.get(window_id)
        if lock is None:
            lock = asyncio.Lock()
            self._send_locks[window_id] = lock
        return lock

    async def _bounded(self, label: str, fn: Callable[[], _T], default: _T) -> _T:
        """Run a blocking libtmux/tmux call in a thread, bounded by
        `_TMUX_SUBPROCESS_TIMEOUT`.

        A wedged tmux server can leave `fn` blocked in libtmux's untimed
        `Popen.communicate()` forever. `asyncio.wait_for` can't cancel that
        thread — the pool thread is abandoned running `fn` — but that's
        deliberate: leaking one thread-pool thread is far cheaper than
        permanently freezing the caller coroutine (this backs the
        session-monitor and status-polling loops).
        """
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(fn), timeout=_TMUX_SUBPROCESS_TIMEOUT
            )
        except TimeoutError:
            logger.error(
                "tmux call %s timed out after %.0fs", label, _TMUX_SUBPROCESS_TIMEOUT
            )
            return default

    def get_session(self) -> libtmux.Session | None:
        """Get the tmux session if it exists."""
        try:
            return self.server.sessions.get(session_name=self.session_name)
        except Exception:
            return None

    def get_or_create_session(self) -> libtmux.Session:
        """Get existing session or create a new one."""
        session = self.get_session()
        if session:
            self._scrub_session_env(session)
            return session

        # Create new session. We deliberately do NOT pass -x/-y or set
        # window-size — pane sizing tracks the attached client per tmux's
        # global window-size setting, which is what users expect when they
        # `tmux attach`. Robust UI detection (bottom-anchored + degraded
        # backstop) handles small panes, so pinning a size is not needed.
        # A prior global window-size mutation on a grouped session crashed
        # tmux on 2026-05-27 — leaving the option alone avoids that class of
        # bug entirely.
        session = self.server.new_session(
            session_name=self.session_name,
            start_directory=str(Path.home()),
        )
        # Rename the default window to the main window name
        if session.windows:
            session.windows[0].rename_window(config.tmux_main_window_name)
        self._scrub_session_env(session)
        return session

    @staticmethod
    def _scrub_session_env(session: libtmux.Session) -> None:
        """Remove sensitive env vars from the tmux session environment.

        Prevents new windows (and their child processes like Claude Code)
        from inheriting secrets such as TELEGRAM_BOT_TOKEN.
        """
        for var in SENSITIVE_ENV_VARS:
            try:
                session.unset_environment(var)
            except Exception:
                pass  # var not set in session env — nothing to remove

    async def list_windows(self) -> list[TmuxWindow]:
        """List all windows in the session with their working directories.

        Returns:
            List of TmuxWindow with window info and cwd
        """

        def _sync_list_windows() -> list[TmuxWindow]:
            windows = []
            session = self.get_session()

            if not session:
                return windows

            for window in session.windows:
                name = window.window_name or ""
                # Skip the main window (placeholder window)
                if name == config.tmux_main_window_name:
                    continue

                try:
                    # Get the active pane's current path and command
                    pane = window.active_pane
                    if pane:
                        cwd = pane.pane_current_path or ""
                        # tmux appends " (deleted)" when cwd no longer exists
                        if cwd.endswith(" (deleted)"):
                            cwd = cwd.removesuffix(" (deleted)")
                        pane_cmd = pane.pane_current_command or ""
                    else:
                        cwd = ""
                        pane_cmd = ""

                    windows.append(
                        TmuxWindow(
                            window_id=window.window_id or "",
                            window_name=name,
                            cwd=cwd,
                            pane_current_command=pane_cmd,
                        )
                    )
                except Exception as e:
                    logger.debug(f"Error getting window info: {e}")

            return windows

        return await self._bounded("list_windows", _sync_list_windows, [])

    async def list_all_window_ids(self) -> set[str] | None:
        """Window IDs of every window on ccbot's tmux server, across all
        tmux sessions on the socket (not just the configured one).

        Used by the session_map sweep: window IDs are unique per server, so
        any map entry whose ID is absent here belongs to a dead window.
        Returns None when the server can't be queried — callers must treat
        that as "unknown", never as "no windows".
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._tmux_argv("list-windows", "-a", "-F", "#{window_id}"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_TMUX_SUBPROCESS_TIMEOUT
            )
        except Exception as e:
            logger.debug("list_all_window_ids failed: %s", e)
            return None
        if proc.returncode != 0:
            logger.debug(
                "list_all_window_ids failed (rc=%s): %s",
                proc.returncode,
                stderr.decode("utf-8", "replace").strip(),
            )
            return None
        return {ln.strip() for ln in stdout.decode("utf-8").splitlines() if ln.strip()}

    async def find_window_by_name(self, window_name: str) -> TmuxWindow | None:
        """Find a window by its name.

        Args:
            window_name: The window name to match

        Returns:
            TmuxWindow if found, None otherwise
        """
        windows = await self.list_windows()
        for window in windows:
            if window.window_name == window_name:
                return window
        logger.debug("Window not found by name: %s", window_name)
        return None

    async def find_window_by_id(self, window_id: str) -> TmuxWindow | None:
        """Find a window by its tmux window ID (e.g. '@0', '@12').

        Args:
            window_id: The tmux window ID to match

        Returns:
            TmuxWindow if found, None otherwise
        """
        windows = await self.list_windows()
        for window in windows:
            if window.window_id == window_id:
                return window
        logger.debug("Window not found by id: %s", window_id)
        return None

    async def get_pane_current_command(self, window_id: str) -> str | None:
        """Return the foreground process name in the window's active pane.

        Returns None if the window or pane is gone. Used by the auto-restart
        health check to verify claude actually started.
        """

        def _sync_get() -> str | None:
            session = self.get_session()
            if not session:
                return None
            try:
                window = session.windows.get(window_id=window_id)
                if not window:
                    return None
                pane = window.active_pane
                if not pane:
                    return None
                return pane.pane_current_command or ""
            except Exception as e:
                logger.debug(f"get_pane_current_command({window_id}) failed: {e}")
                return None

        return await self._bounded(
            f"get_pane_current_command({window_id})", _sync_get, None
        )

    async def get_pane_pid(self, window_id: str) -> int | None:
        """Return the PID of the window's active pane.

        When claude is launched via `window_shell` (tmux's `new-window <cmd>`
        form, executed by `/bin/sh -c`), the shell is the pane_pid and claude
        runs as a child. `get_pane_current_command` reports the shell's name,
        so the restart health check uses this PID to walk the process tree
        for claude. Returns None if the pane is gone or the PID is missing.
        """

        def _sync_get() -> int | None:
            session = self.get_session()
            if not session:
                return None
            try:
                window = session.windows.get(window_id=window_id)
                if not window:
                    return None
                pane = window.active_pane
                if not pane:
                    return None
                raw = pane.pane_pid
                if raw is None:
                    return None
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    return None
            except Exception as e:
                logger.debug(f"get_pane_pid({window_id}) failed: {e}")
                return None

        return await self._bounded(f"get_pane_pid({window_id})", _sync_get, None)

    async def list_group_session_names(self) -> set[str]:
        """Return the configured tmux session plus any grouped peers."""

        def _sync_list() -> set[str]:
            try:
                result = subprocess.run(
                    self._tmux_argv(
                        "list-sessions",
                        "-F",
                        "#{session_name}|#{session_group}",
                    ),
                    capture_output=True,
                    text=True,
                    timeout=_TMUX_SUBPROCESS_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                logger.error(
                    "list_group_session_names: tmux timed out after %.0fs",
                    _TMUX_SUBPROCESS_TIMEOUT,
                )
                return {self.session_name}
            except OSError as e:
                logger.debug("list_group_session_names failed to exec tmux: %s", e)
                return {self.session_name}

            if result.returncode != 0:
                logger.debug(
                    "list_group_session_names failed: %s",
                    result.stderr.strip() or f"exit {result.returncode}",
                )
                return {self.session_name}

            return parse_group_session_names(result.stdout, self.session_name)

        return await self._bounded(
            "list_group_session_names", _sync_list, {self.session_name}
        )

    async def capture_pane(self, window_id: str, with_ansi: bool = False) -> str | None:
        """Capture the visible text content of a window's active pane.

        Args:
            window_id: The window ID to capture
            with_ansi: If True, capture with ANSI color codes

        Returns:
            The captured text, or None on failure.
        """
        if with_ansi:
            # Use async subprocess to call tmux capture-pane -e for ANSI colors
            try:
                proc = await asyncio.create_subprocess_exec(
                    *self._tmux_argv("capture-pane", "-e", "-p", "-t", window_id),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=_TMUX_SUBPROCESS_TIMEOUT
                    )
                except TimeoutError:
                    logger.error(
                        "capture_pane(%s): tmux timed out after %.0fs",
                        window_id,
                        _TMUX_SUBPROCESS_TIMEOUT,
                    )
                    proc.kill()
                    with contextlib.suppress(Exception):
                        await proc.wait()
                    return None
                if proc.returncode == 0:
                    return stdout.decode("utf-8")
                logger.error(
                    f"Failed to capture pane {window_id}: {stderr.decode('utf-8')}"
                )
                return None
            except Exception as e:
                logger.error(f"Unexpected error capturing pane {window_id}: {e}")
                return None

        # Original implementation for plain text - wrap in thread
        def _sync_capture() -> str | None:
            session = self.get_session()
            if not session:
                return None
            try:
                window = session.windows.get(window_id=window_id)
                if not window:
                    return None
                pane = window.active_pane
                if not pane:
                    return None
                lines = pane.capture_pane()
                return "\n".join(lines) if isinstance(lines, list) else str(lines)
            except Exception as e:
                logger.error(f"Failed to capture pane {window_id}: {e}")
                return None

        return await self._bounded(f"capture_pane({window_id})", _sync_capture, None)

    async def send_keys(
        self, window_id: str, text: str, enter: bool = True, literal: bool = True
    ) -> bool:
        """Send keys to a specific window.

        Runs the whole sequence (literal text + verify + Enter, or a
        special-key send) under the window's send lock, so two concurrent
        send_keys calls to the same window can never interleave their
        keystrokes/Enter presses.

        Args:
            window_id: The window ID to send to
            text: Text to send
            enter: Whether to press enter after the text
            literal: If True, send text literally. If False, interpret special keys
                     like "Up", "Down", "Left", "Right", "Escape", "Enter".

        Returns:
            True if successful, False otherwise
        """
        async with self._get_send_lock(window_id):
            if literal and enter:
                return await self._send_literal_with_enter(window_id, text)
            return await self._send_special_or_no_enter(window_id, text, enter, literal)

    async def _send_literal_with_enter(self, window_id: str, text: str) -> bool:
        """Send literal text, verify it actually landed, then send Enter.

        Claude Code's TUI sometimes interprets a rapid-fire Enter (arriving
        before the TUI has redrawn the text) as a newline inside the input
        box rather than submit — a silent stall for the user. Polling
        capture-pane for the sent text before sending Enter turns "assume it
        landed" into "observe that it landed".
        """
        # Claude Code's ! command mode: send "!" first so the TUI switches
        # to bash mode, wait 1s, then send the rest — same two-step
        # semantics as before, just via the raw/verified send.
        if text.startswith("!"):
            if not await self._send_literal_raw(window_id, "!"):
                return False
            rest = text[1:]
            sent_text = "!"
            if rest:
                await asyncio.sleep(1.0)
                if not await self._send_literal_raw(window_id, rest):
                    return False
                sent_text = rest
        else:
            if not await self._send_literal_raw(window_id, text):
                return False
            sent_text = text

        return await self._verify_and_send_enter(window_id, sent_text)

    async def _verify_and_send_enter(self, window_id: str, sent_text: str) -> bool:
        """Poll capture-pane until `sent_text` is visible, then send Enter.

        Returns False without sending Enter if the text never becomes
        visible — a silent half-typed submit is worse than the honest
        '❌ Failed to send keys' the caller (session.send_to_window)
        already surfaces to the user.
        """
        visible = False
        for attempt in range(_SEND_VERIFY_ATTEMPTS):
            if attempt:
                await asyncio.sleep(_SEND_VERIFY_POLL_INTERVAL)
            pane_text = await self.capture_pane(window_id)
            if pane_text is not None and _text_visible_in_pane(pane_text, sent_text):
                visible = True
                break

        if not visible:
            logger.warning(
                "send_keys(%s): text not visible in pane after %d attempts, "
                "not sending Enter (prefix=%r)",
                window_id,
                _SEND_VERIFY_ATTEMPTS,
                sent_text[:40],
            )
            return False

        return await self._send_enter_raw(window_id)

    async def _send_literal_raw(self, window_id: str, chars: str) -> bool:
        """Send literal text via a raw `tmux send-keys -l --` subprocess.

        The vendored libtmux `Pane.send_keys` discards the tmux subprocess's
        exit code, so a send tmux rejects (e.g. text starting with '-'
        mis-parsed as an option when no `--` separator is used) silently
        reported success. `--` ends option parsing, making leading-'-' text
        safe, and checking `returncode` surfaces what libtmux swallowed.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._tmux_argv("send-keys", "-t", window_id, "-l", "--", chars),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_TMUX_SUBPROCESS_TIMEOUT
            )
        except Exception as e:
            logger.error("send_keys:literal(%s) failed to exec tmux: %s", window_id, e)
            return False
        if proc.returncode != 0:
            logger.error(
                "send_keys:literal(%s) rejected (rc=%s): %s",
                window_id,
                proc.returncode,
                stderr.decode("utf-8", "replace").strip(),
            )
            return False
        return True

    async def _send_enter_raw(self, window_id: str) -> bool:
        """Send Enter via a raw `tmux send-keys` subprocess (same
        returncode handling as `_send_literal_raw`)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._tmux_argv("send-keys", "-t", window_id, "Enter"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_TMUX_SUBPROCESS_TIMEOUT
            )
        except Exception as e:
            logger.error("send_keys:enter(%s) failed to exec tmux: %s", window_id, e)
            return False
        if proc.returncode != 0:
            logger.error(
                "send_keys:enter(%s) rejected (rc=%s): %s",
                window_id,
                proc.returncode,
                stderr.decode("utf-8", "replace").strip(),
            )
            return False
        return True

    async def _send_special_or_no_enter(
        self, window_id: str, text: str, enter: bool, literal: bool
    ) -> bool:
        """Special keys (arrows/Escape/Enter from UI buttons) or a literal
        send with no trailing Enter.

        Stays on libtmux via `_bounded` — these are single fire-and-forget
        key sends, not the text-then-Enter sequence the verify-before-Enter
        path guards.
        """

        def _sync_send_keys() -> bool:
            session = self.get_session()
            if not session:
                logger.error("No tmux session found")
                return False

            try:
                window = session.windows.get(window_id=window_id)
                if not window:
                    logger.error(f"Window {window_id} not found")
                    return False

                pane = window.active_pane
                if not pane:
                    logger.error(f"No active pane in window {window_id}")
                    return False

                pane.send_keys(text, enter=enter, literal=literal)
                return True

            except Exception as e:
                logger.error(f"Failed to send keys to window {window_id}: {e}")
                return False

        return await self._bounded(f"send_keys({window_id})", _sync_send_keys, False)

    async def rename_window(self, window_id: str, new_name: str) -> bool:
        """Rename a tmux window by its ID."""

        def _sync_rename() -> bool:
            session = self.get_session()
            if not session:
                return False
            try:
                window = session.windows.get(window_id=window_id)
                if not window:
                    return False
                window.rename_window(new_name)
                logger.info("Renamed window %s to '%s'", window_id, new_name)
                return True
            except Exception as e:
                logger.error(f"Failed to rename window {window_id}: {e}")
                return False

        return await self._bounded(f"rename_window({window_id})", _sync_rename, False)

    async def kill_window(self, window_id: str) -> bool:
        """Kill a tmux window by its ID."""

        def _sync_kill() -> bool:
            session = self.get_session()
            if not session:
                return False
            try:
                window = session.windows.get(window_id=window_id)
                if not window:
                    return False
                window.kill()
                logger.info("Killed window %s", window_id)
                return True
            except Exception as e:
                logger.error(f"Failed to kill window {window_id}: {e}")
                return False

        return await self._bounded(f"kill_window({window_id})", _sync_kill, False)

    async def respawn_pane(
        self,
        window_id: str,
        work_dir: str,
        resume_session_id: str | None = None,
    ) -> bool:
        """Restart the claude process in place, reusing the same tmux window.

        Unlike kill_window + create_window, `respawn-pane -k` reuses the
        existing window (and therefore the same @window_id), so the topic
        binding and session_map entry stay valid — only the Claude session_id
        rolls. This is the no-churn path used by `/restart` and avoids the
        leaked-window-id problem that kill+create caused on every upgrade.

        The launch command matches create_window (no explicit --model), so the
        respawned session uses the user's current settings.json default model.
        """
        path = Path(work_dir).expanduser().resolve()
        if not path.is_dir():
            logger.error("respawn_pane: not a directory: %s", work_dir)
            return False

        inner = build_claude_command(
            config.claude_command,
            permission_mode=config.claude_permission_mode,
            resume_session_id=resume_session_id,
        )
        window_shell_arg = build_window_shell_cmd(inner, _user_shell())

        def _sync_respawn() -> bool:
            try:
                result = subprocess.run(
                    self._tmux_argv(
                        "respawn-pane",
                        "-k",
                        "-c",
                        str(path),
                        "-t",
                        window_id,
                        window_shell_arg,
                    ),
                    capture_output=True,
                    text=True,
                    timeout=_TMUX_SUBPROCESS_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                logger.error(
                    "respawn_pane(%s): tmux timed out after %.0fs",
                    window_id,
                    _TMUX_SUBPROCESS_TIMEOUT,
                )
                return False
            except OSError as e:
                logger.error("respawn_pane: failed to exec tmux: %s", e)
                return False
            if result.returncode != 0:
                logger.error(
                    "respawn_pane(%s) failed: %s",
                    window_id,
                    result.stderr.strip() or f"exit {result.returncode}",
                )
                return False
            logger.info(
                "Respawned pane in window %s (resume=%s)", window_id, resume_session_id
            )
            return True

        return await self._bounded(f"respawn_pane({window_id})", _sync_respawn, False)

    async def create_window(
        self,
        work_dir: str,
        window_name: str | None = None,
        start_claude: bool = True,
        resume_session_id: str | None = None,
    ) -> tuple[bool, str, str, str]:
        """Create a new tmux window and optionally start Claude Code.

        Args:
            work_dir: Working directory for the new window
            window_name: Optional window name (defaults to directory name)
            start_claude: Whether to start claude command
            resume_session_id: If set, append --resume <id> to claude command

        Returns:
            Tuple of (success, message, window_name, window_id)
        """
        # Validate directory first
        path = Path(work_dir).expanduser().resolve()
        if not path.exists():
            return False, f"Directory does not exist: {work_dir}", "", ""
        if not path.is_dir():
            return False, f"Not a directory: {work_dir}", "", ""

        # Create window name, adding suffix if name already exists
        final_window_name = window_name if window_name else path.name

        # Check for existing window name
        base_name = final_window_name
        counter = 2
        while await self.find_window_by_name(final_window_name):
            final_window_name = f"{base_name}-{counter}"
            counter += 1

        # Build the window's primary command up-front so it runs as the pane's
        # process (via tmux's `new-window <shell-command>`, executed by /bin/sh
        # -c). This bypasses the user's interactive shell init entirely —
        # no .zshrc, no oh-my-zsh prompts — eliminating the race where
        # init-time `read` calls eat the first chars of a `send_keys` payload.
        window_shell_arg: str | None = None
        if start_claude:
            inner = build_claude_command(
                config.claude_command,
                permission_mode=config.claude_permission_mode,
                resume_session_id=resume_session_id,
            )
            window_shell_arg = build_window_shell_cmd(inner, _user_shell())

        # Create window in thread
        def _create() -> tuple[bool, str, str, str]:
            session = self.get_or_create_session()
            try:
                window = session.new_window(
                    window_name=final_window_name,
                    start_directory=str(path),
                    window_shell=window_shell_arg,
                )

                wid = window.window_id or ""

                # Prevent Claude Code from overriding window name. Defense in
                # depth — the global `allow-rename` is typically already off.
                window.set_window_option("allow-rename", "off")

                logger.info(
                    "Created window '%s' (id=%s) at %s",
                    final_window_name,
                    wid,
                    path,
                )
                return (
                    True,
                    f"Created window '{final_window_name}' at {path}",
                    final_window_name,
                    wid,
                )

            except Exception as e:
                logger.error(f"Failed to create window: {e}")
                return False, f"Failed to create window: {e}", "", ""

        return await self._bounded(
            f"create_window({final_window_name})",
            _create,
            (False, "tmux timed out", "", ""),
        )


# Global instance with default session name
tmux_manager = TmuxManager()
