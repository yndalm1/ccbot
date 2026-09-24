"""Hook subcommand for Claude Code session tracking.

Called by Claude Code's SessionStart hook to maintain a window↔session
mapping in <CCBOT_DIR>/session_map.json. Skips panes belonging to a foreign
tmux session sharing ccbot's dedicated socket (e.g. the user's own
interactive tmux sessions), so their entries never pollute the map, and
skips claude sessions that cannot own a window: non-interactive ones
(`claude -p`, SDK hosts) and ones nested inside another claude. Also
provides `--install` to auto-configure the hook in ~/.claude/settings.json.

This module must NOT import config.py (which requires TELEGRAM_BOT_TOKEN),
since hooks run inside tmux panes where bot env vars are not set.
Config directory resolution uses utils.ccbot_dir() (shared with config.py).

Key functions: hook_main() (CLI entry), _install_hook().
"""

import argparse
import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Validate session_id looks like a UUID
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

_CLAUDE_SETTINGS_FILE = Path.home() / ".claude" / "settings.json"

# The hook command suffix for detection
_HOOK_COMMAND_SUFFIX = "ccbot hook"


def _find_ccbot_path() -> str:
    """Find the full path to the ccbot executable.

    Priority:
    1. shutil.which("ccbot") - if ccbot is in PATH
    2. Same directory as the Python interpreter (for venv installs)
    """
    # Try PATH first
    ccbot_path = shutil.which("ccbot")
    if ccbot_path:
        return ccbot_path

    # Fall back to the directory containing the Python interpreter
    # This handles the case where ccbot is installed in a venv
    python_dir = Path(sys.executable).parent
    ccbot_in_venv = python_dir / "ccbot"
    if ccbot_in_venv.exists():
        return str(ccbot_in_venv)

    # Last resort: assume it will be in PATH
    return "ccbot"


def _is_hook_installed(settings: dict) -> bool:
    """Check if ccbot hook is already installed in the settings.

    Detects both 'ccbot hook' and full paths like '/path/to/ccbot hook'.
    """
    hooks = settings.get("hooks", {})
    session_start = hooks.get("SessionStart", [])

    for entry in session_start:
        if not isinstance(entry, dict):
            continue
        inner_hooks = entry.get("hooks", [])
        for h in inner_hooks:
            if not isinstance(h, dict):
                continue
            cmd = h.get("command", "")
            # Match 'ccbot hook' or paths ending with 'ccbot hook'
            if cmd == _HOOK_COMMAND_SUFFIX or cmd.endswith("/" + _HOOK_COMMAND_SUFFIX):
                return True
    return False


def _install_hook() -> int:
    """Install the ccbot hook into Claude's settings.json.

    Returns 0 on success, 1 on error.
    """
    settings_file = _CLAUDE_SETTINGS_FILE
    settings_file.parent.mkdir(parents=True, exist_ok=True)

    # Read existing settings
    settings: dict = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Error reading %s: %s", settings_file, e)
            print(f"Error reading {settings_file}: {e}", file=sys.stderr)
            return 1

    # Check if already installed
    if _is_hook_installed(settings):
        logger.info("Hook already installed in %s", settings_file)
        print(f"Hook already installed in {settings_file}")
        return 0

    # Find the full path to ccbot
    ccbot_path = _find_ccbot_path()
    hook_command = f"{ccbot_path} hook"
    hook_config = {"type": "command", "command": hook_command, "timeout": 5}
    logger.info("Installing hook command: %s", hook_command)

    # Install the hook
    if "hooks" not in settings:
        settings["hooks"] = {}
    if "SessionStart" not in settings["hooks"]:
        settings["hooks"]["SessionStart"] = []

    settings["hooks"]["SessionStart"].append({"hooks": [hook_config]})

    # Write back
    try:
        settings_file.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
        )
    except OSError as e:
        logger.error("Error writing %s: %s", settings_file, e)
        print(f"Error writing {settings_file}: {e}", file=sys.stderr)
        return 1

    logger.info("Hook installed successfully in %s", settings_file)
    print(f"Hook installed successfully in {settings_file}")
    return 0


def _env_or_dotenv(key: str, default: str) -> str:
    """Resolve a config value the way this module must: env var, then the
    config dir's .env, then a default.

    config.py can't be imported here (it requires TELEGRAM_BOT_TOKEN), so
    this hand-rolls the same env/.env precedence for the couple of
    tmux-identifying vars the hook needs directly (socket name, session
    name).
    """
    value = os.environ.get(key, "")
    if value:
        return value

    from .utils import ccbot_dir

    env_file = ccbot_dir() / ".env"
    try:
        for line in env_file.read_text().splitlines():
            file_key, sep, file_value = line.strip().partition("=")
            if sep and file_key == key:
                file_value = file_value.strip().strip("'\"")
                if file_value:
                    return file_value
    except OSError:
        pass
    return default


def _tmux_socket_name() -> str:
    """Resolve ccbot's dedicated tmux socket name (mirrors config.tmux_socket_name)."""
    return _env_or_dotenv("TMUX_SOCKET_NAME", "ccbot")


def _tmux_session_name() -> str:
    """Resolve ccbot's configured tmux session name (mirrors config.tmux_session_name)."""
    return _env_or_dotenv("TMUX_SESSION_NAME", "ccbot")


def _accepted_session_names() -> set[str] | None:
    """Tmux session names ccbot's session_map accepts: the configured
    session plus any grouped peers.

    Returns None if the tmux query itself failed (transient hiccup, socket
    not up yet, etc.) — callers must treat that as "unknown, accept
    anything" rather than silently dropping a legitimate registration.
    """
    try:
        result = subprocess.run(
            [
                "tmux",
                "-L",
                _tmux_socket_name(),
                "list-sessions",
                "-F",
                "#{session_name}|#{session_group}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("_accepted_session_names: tmux query failed: %s", e)
        return None
    if result.returncode != 0:
        logger.debug(
            "_accepted_session_names: list-sessions failed (rc=%s): %s",
            result.returncode,
            result.stderr.strip(),
        )
        return None

    from .utils import parse_group_session_names

    return parse_group_session_names(result.stdout, _tmux_session_name())


def _record_hook_failure(cwd: str, session_id: str, reason: str) -> None:
    """Append a window-mapping failure to <CCBOT_DIR>/hook_failures.jsonl.

    The bot's maintenance loop tails this file and surfaces the failure in
    the topic(s) bound to this cwd — otherwise the failure dies in hook
    stderr, only visible by transcript archaeology, and the topic goes
    silent with no explanation (the 2026-07-04 incident mode).
    """
    from .utils import ccbot_dir

    try:
        line = json.dumps(
            {"ts": time.time(), "cwd": cwd, "session_id": session_id, "reason": reason}
        )
        with open(ccbot_dir() / "hook_failures.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        logger.debug("Failed to record hook failure: %s", e)


def _resolve_window_by_pane(pane_id: str) -> tuple[str, str, str] | None:
    """Resolve (session_name, window_id, window_name) from a tmux pane id."""
    result = subprocess.run(
        [
            "tmux",
            "display-message",
            "-t",
            pane_id,
            "-p",
            "#{session_name}:#{window_id}:#{window_name}",
        ],
        capture_output=True,
        text=True,
    )
    raw_output = result.stdout.strip()
    # Expected format: "session_name:@id:window_name"
    parts = raw_output.split(":", 2)
    if len(parts) < 3:
        logger.warning(
            "Failed to parse session:window_id:window_name from tmux (pane=%s, output=%s)",
            pane_id,
            raw_output,
        )
        return None
    tmux_session_name, window_id, window_name = parts
    return tmux_session_name, window_id, window_name


def _panes_running_claude(pane_pids: list[str]) -> set[str]:
    """Return the subset of pane PIDs with a live ``claude`` process at or
    below them.

    Even under the daemon architecture the pane keeps a thin ``claude``
    client running, so this distinguishes a window hosting a Claude Code UI
    from an idle shell parked in the same directory.
    """
    result = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,comm="],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    children: dict[str, list[str]] = {}
    comm: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, name = parts[0], parts[1], parts[2].strip()
        children.setdefault(ppid, []).append(pid)
        # basename: Linux comm is the bare name, macOS comm is the full path
        comm[pid] = os.path.basename(name)

    matched: set[str] = set()
    for pane_pid in pane_pids:
        stack = [pane_pid]
        while stack:
            p = stack.pop()
            if comm.get(p) == "claude":
                matched.add(pane_pid)
                break
            stack.extend(children.get(p, []))
    return matched


def _count_claude_ancestors() -> int | None:
    """Count ``claude`` processes among this hook process's ancestors.

    The pane's own session runs its hooks under exactly one claude — the
    one hosting the pane. A child session (e.g. a tool inside the pane
    shelling out to ``claude -p``) runs them under two or more, because the
    child inherits the pane's environment and fires the same hook. The
    count is the discriminator; callers skip registration at >= 2.

    Returns None when the process table can't be read — unknown must fail
    open (register as before) rather than silently drop a legitimate
    registration, mirroring _accepted_session_names.
    """
    result = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,comm="],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    parent: dict[str, str] = {}
    comm: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        parent[parts[0]] = parts[1]
        # basename: Linux comm is the bare name, macOS comm is the full path
        comm[parts[0]] = os.path.basename(parts[2].strip())

    count = 0
    seen: set[str] = set()
    p = str(os.getpid())
    while p in parent and p not in seen:
        seen.add(p)
        if comm.get(p) == "claude":
            count += 1
        p = parent[p]
    return count


def _resolve_window_by_cwd(cwd: str) -> tuple[str, str, str] | None:
    """Resolve (session_name, window_id, window_name) by matching the cwd.

    Daemon-hosted Claude Code sessions (``claude --bg-pty-host …``) run hooks
    with TMUX/TMUX_PANE stripped from the environment, so the pane cannot be
    read from env. Ask ccbot's tmux server directly (explicit ``-L`` socket,
    since $TMUX is also absent) which live window's pane sits in the session's
    cwd. Shells merely parked in the directory are filtered out by requiring
    a running claude client; only a unique match resolves — with two claude
    windows on the same directory the window cannot be named, and a wrong
    guess is worse than a stale map.
    """
    if not cwd:
        return None
    result = subprocess.run(
        [
            "tmux",
            "-L",
            _tmux_socket_name(),
            "list-panes",
            "-a",
            "-F",
            "#{session_name}\t#{window_id}\t#{window_name}\t"
            "#{pane_current_path}\t#{pane_pid}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(
            "tmux list-panes failed (rc=%s): %s",
            result.returncode,
            result.stderr.strip(),
        )
        return None

    target = os.path.realpath(cwd)
    matches: dict[str, tuple[str, str, str]] = {}
    pane_pids: dict[str, list[str]] = {}  # window_id -> its panes' PIDs
    for line in result.stdout.splitlines():
        parts = line.split("\t", 4)
        if len(parts) != 5:
            continue
        session_name, window_id, window_name, pane_path, pane_pid = parts
        if os.path.realpath(pane_path) == target:
            matches[window_id] = (session_name, window_id, window_name)
            pane_pids.setdefault(window_id, []).append(pane_pid)

    if len(matches) > 1:
        running = _panes_running_claude(
            [pid for pids in pane_pids.values() for pid in pids]
        )
        matches = {
            wid: info
            for wid, info in matches.items()
            if any(pid in running for pid in pane_pids[wid])
        }

    if len(matches) != 1:
        logger.warning(
            "TMUX_PANE not set and cwd %s matches %d live claude windows; "
            "cannot determine window",
            cwd,
            len(matches),
        )
        return None
    resolved = next(iter(matches.values()))
    logger.info(
        "Resolved window by cwd fallback: %s:%s (%s)",
        resolved[0],
        resolved[1],
        resolved[2],
    )
    return resolved


def hook_main() -> None:
    """Process a Claude Code hook event from stdin, or install the hook."""
    # Configure logging for the hook subprocess (main.py logging doesn't apply here)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG,
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        prog="ccbot hook",
        description="Claude Code session tracking hook",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the hook into ~/.claude/settings.json",
    )
    # Parse only known args to avoid conflicts with stdin JSON
    args, _ = parser.parse_known_args(sys.argv[2:])

    if args.install:
        logger.info("Hook install requested")
        sys.exit(_install_hook())

    # Normal hook processing: read JSON from stdin
    logger.debug("Processing hook event from stdin")
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse stdin JSON: %s", e)
        return

    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd", "")
    event = payload.get("hook_event_name", "")
    transcript_path = payload.get("transcript_path", "")

    if not session_id or not event:
        logger.debug("Empty session_id or event, ignoring")
        return

    # Validate session_id format
    if not _UUID_RE.match(session_id):
        logger.warning("Invalid session_id format: %s", session_id)
        return

    # Validate cwd is an absolute path (if provided)
    if cwd and not os.path.isabs(cwd):
        logger.warning("cwd is not absolute: %s", cwd)
        return

    if event != "SessionStart":
        logger.debug("Ignoring non-SessionStart event: %s", event)
        return

    # Only an interactive session can own a window. A non-interactive one
    # (`claude -p`, an SDK host) started from inside a pane inherits
    # TMUX_PANE and would steal the window's mapping; with
    # --no-session-persistence its transcript never exists, so the topic
    # goes silent. Claude Code sets CLAUDE_CODE_ENTRYPOINT for its own hooks
    # ("cli" for the TUI, "sdk-cli" for `-p`, "sdk-*" for SDK hosts),
    # independent of how the process was launched. Unset fails open.
    entrypoint = os.environ.get("CLAUDE_CODE_ENTRYPOINT", "")
    if entrypoint.startswith("sdk"):
        logger.info(
            "Hook fired by a non-interactive claude (entrypoint %s); not registering",
            entrypoint,
        )
        return

    # Get tmux session:window key for the pane running this hook.
    # TMUX_PANE is set by tmux for every process inside a pane. Daemon-hosted
    # sessions (claude --bg-pty-host …) strip it, so fall back to matching the
    # session cwd against live panes — but only for continuation sources
    # (clear/compact/resume): a fresh `claude` started outside tmux also lacks
    # TMUX_PANE, and must not steal a window's mapping.
    source = payload.get("source", "")
    pane_id = os.environ.get("TMUX_PANE", "")
    by_cwd_fallback = False
    if pane_id:
        # A child claude spawned from inside the pane inherits TMUX_PANE and
        # reaches this path looking exactly like the pane's own session —
        # its SessionStart would steal the window's mapping and the topic
        # would go silent until the divergence notice. The entrypoint check
        # above catches non-interactive children; this catches the rest.
        # The pane's own session runs this hook under exactly one claude
        # ancestor; a child session under two or more. Skip those; unknown
        # (None) fails open. The count only sees processes named "claude":
        # one launched by its versioned path shows as e.g. "2.1.280".
        claude_ancestors = _count_claude_ancestors()
        if claude_ancestors is not None and claude_ancestors >= 2:
            logger.info(
                "Hook fired by a claude nested inside another claude "
                "(%d claude ancestors); not registering",
                claude_ancestors,
            )
            return
        # session_map is ccbot's private state: only panes on ccbot's own
        # tmux server belong in it. $TMUX is "<socket_path>,<pid>,<session>";
        # a different socket basename means a foreign server — writing its
        # windows would pollute the map with IDs that can collide with ours.
        tmux_env = os.environ.get("TMUX", "")
        if tmux_env:
            socket_path = tmux_env.split(",", 1)[0]
            if os.path.basename(socket_path) != _tmux_socket_name():
                logger.debug("Pane is on foreign tmux socket %s, skipping", socket_path)
                return
        resolved = _resolve_window_by_pane(pane_id)
    elif source and source != "startup":
        resolved = _resolve_window_by_cwd(cwd)
        by_cwd_fallback = True
    else:
        logger.warning("TMUX_PANE not set, cannot determine window")
        return
    if resolved is None:
        if by_cwd_fallback:
            _record_hook_failure(
                cwd,
                session_id,
                "no TMUX_PANE and no unique live claude window for this cwd",
            )
        return
    tmux_session_name, window_id, window_name = resolved

    # A pane can resolve to a tmux session that isn't ccbot's own — e.g. the
    # user's personal interactive tmux sessions sharing this dedicated
    # socket. session_map is ccbot's private state; readers only ever look
    # at the configured session (plus grouped peers), so an entry from any
    # other session is inert pollution. Skip it here instead of writing it
    # and relying on readers to ignore it.
    accepted_session_names = _accepted_session_names()
    if (
        accepted_session_names is not None
        and tmux_session_name not in accepted_session_names
    ):
        logger.info(
            "Pane belongs to foreign tmux session %r on ccbot's socket; "
            "not registering",
            tmux_session_name,
        )
        return

    # Key uses window_id for uniqueness
    session_window_key = f"{tmux_session_name}:{window_id}"

    # The transcript's size at this exact SessionStart moment — only the hook
    # can observe this. The monitor seeds a newly-noticed session's read
    # offset here instead of at current EOF, so a reply that lands before
    # the monitor's first poll (or before it ever notices the session) is
    # not silently skipped (review f17/RC38).
    transcript_size_at_start = 0
    if (
        transcript_path
        and os.path.isabs(transcript_path)
        and os.path.exists(transcript_path)
    ):
        try:
            transcript_size_at_start = os.path.getsize(transcript_path)
        except OSError:
            transcript_size_at_start = 0

    logger.debug(
        "tmux key=%s, window_name=%s, session_id=%s, cwd=%s",
        session_window_key,
        window_name,
        session_id,
        cwd,
    )

    # Read-modify-write with file locking to prevent concurrent hook races
    from .utils import ccbot_dir

    map_file = ccbot_dir() / "session_map.json"
    map_file.parent.mkdir(parents=True, exist_ok=True)

    lock_path = map_file.with_suffix(".lock")
    try:
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            logger.debug("Acquired lock on %s", lock_path)
            try:
                session_map: dict[str, dict[str, Any]] = {}
                if map_file.exists():
                    try:
                        session_map = json.loads(map_file.read_text())
                    except (json.JSONDecodeError, OSError):
                        logger.warning(
                            "Failed to read existing session_map, starting fresh"
                        )

                if by_cwd_fallback:
                    # A cwd match may only re-point a window this hook has
                    # already bound in-pane (first registration always runs
                    # inside the pane, where TMUX_PANE is set). Creating or
                    # re-purposing entries from a cwd guess would let any
                    # outside-tmux claude in a coincidental directory hijack
                    # a window's mapping.
                    prior_cwd = session_map.get(session_window_key, {}).get("cwd", "")
                    if not prior_cwd or os.path.realpath(prior_cwd) != os.path.realpath(
                        cwd
                    ):
                        logger.warning(
                            "cwd fallback matched %s but it has no prior entry "
                            "for cwd %s; refusing to bind without TMUX_PANE",
                            session_window_key,
                            cwd,
                        )
                        _record_hook_failure(
                            cwd,
                            session_id,
                            f"cwd fallback matched {session_window_key} but it "
                            "has no prior entry for this cwd",
                        )
                        return

                session_map[session_window_key] = {
                    "session_id": session_id,
                    "cwd": cwd,
                    "window_name": window_name,
                    "transcript_size_at_start": transcript_size_at_start,
                }

                # Clean up old-format key ("session:window_name") if it exists.
                # Previous versions keyed by window_name instead of window_id.
                old_key = f"{tmux_session_name}:{window_name}"
                if old_key != session_window_key and old_key in session_map:
                    del session_map[old_key]
                    logger.info("Removed old-format session_map key: %s", old_key)

                # Drop any other entries for the same physical window.
                # Grouped tmux sessions share windows, so a previous hook fire
                # under a peer session name (e.g. `ccbot:@48`) leaves a stale
                # entry when the next fire reports a different `#{session_name}`
                # (e.g. `ccbot-2:@48`). Without this dedup, session_map.json
                # accumulates multiple entries for one physical window with
                # diverging session_ids, and readers have to guess which is
                # current. Window IDs are unique within one tmux server, so any
                # other key ending in `:<window_id>` is necessarily this same
                # window under a different session name — safe to remove.
                stale_peer_keys = [
                    k
                    for k in session_map
                    if k != session_window_key
                    and ":" in k
                    and k.split(":", 1)[1] == window_id
                ]
                for k in stale_peer_keys:
                    del session_map[k]
                    logger.info("Removed stale grouped-peer session_map key: %s", k)

                from .utils import atomic_write_json

                atomic_write_json(map_file, session_map)
                logger.info(
                    "Updated session_map: %s -> session_id=%s, cwd=%s",
                    session_window_key,
                    session_id,
                    cwd,
                )
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
    except OSError as e:
        logger.error("Failed to write session_map: %s", e)
