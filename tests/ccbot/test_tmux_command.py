"""Unit tests for build_claude_command — shell command string assembly."""

import asyncio
import os
import subprocess
import time
from unittest.mock import patch

import pytest

from ccbot import tmux_manager as tm
from ccbot.config import config
from ccbot.tmux_manager import (
    TmuxManager,
    _text_visible_in_pane,
    build_claude_command,
    build_window_shell_cmd,
)


class TestDedicatedSocket:
    """ccbot's tmux server lives on a dedicated socket; every daemon-side raw
    `tmux` call must carry `-L <socket>` (the hook is exempt — it inherits $TMUX)."""

    def test_socket_name_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "tmux_socket_name", "ccbot-test")
        mgr = TmuxManager()
        assert mgr.socket_name == "ccbot-test"

    def test_tmux_argv_prepends_socket_flag(self, monkeypatch):
        monkeypatch.setattr(config, "tmux_socket_name", "ccbot")
        mgr = TmuxManager()
        assert mgr._tmux_argv("capture-pane", "-p", "-t", "@9") == [
            "tmux",
            "-L",
            "ccbot",
            "capture-pane",
            "-p",
            "-t",
            "@9",
        ]

    def test_tmux_argv_follows_custom_socket(self, monkeypatch):
        monkeypatch.setattr(config, "tmux_socket_name", "myproj")
        mgr = TmuxManager()
        assert mgr._tmux_argv("list-sessions")[:3] == ["tmux", "-L", "myproj"]

    def test_server_uses_named_socket(self, monkeypatch):
        monkeypatch.setattr(config, "tmux_socket_name", "ccbot")
        created = {}

        class FakeServer:
            def __init__(self, socket_name=None):
                created["socket_name"] = socket_name

        monkeypatch.setattr(tm.libtmux, "Server", FakeServer)
        mgr = TmuxManager()
        _ = mgr.server  # triggers lazy creation
        assert created["socket_name"] == "ccbot"


class TestBuildClaudeCommand:
    def test_plain_command(self):
        assert build_claude_command("claude") == "claude"

    def test_preserves_custom_base_command(self):
        # Matches the README pattern `IS_SANDBOX=1 claude`.
        assert build_claude_command("IS_SANDBOX=1 claude") == "IS_SANDBOX=1 claude"

    def test_permission_mode_appended(self):
        assert (
            build_claude_command("claude", permission_mode="auto")
            == "claude --permission-mode auto"
        )

    def test_empty_permission_mode_is_no_op(self):
        assert build_claude_command("claude", permission_mode="") == "claude"

    def test_resume_only(self):
        assert (
            build_claude_command("claude", resume_session_id="abc-123")
            == "claude --resume abc-123"
        )

    def test_permission_mode_precedes_resume(self):
        # Placing --permission-mode before --resume ensures the flag applies
        # to the resumed session (CLI order matters for some claude versions).
        assert (
            build_claude_command(
                "claude", permission_mode="auto", resume_session_id="abc-123"
            )
            == "claude --permission-mode auto --resume abc-123"
        )

    @pytest.mark.parametrize(
        "mode", ["default", "acceptEdits", "plan", "auto", "bypassPermissions"]
    )
    def test_each_mode_roundtrips(self, mode):
        result = build_claude_command("claude", permission_mode=mode)
        assert result == f"claude --permission-mode {mode}"


class TestBuildWindowShellCmd:
    """The window_shell value tmux runs as the pane's primary process.

    Structure: `PATH="<fallback>:$PATH" <inner>; exec <user_shell>`. tmux
    invokes this via `/bin/sh -c`, so shell features (`PATH=`, `;`, `exec`)
    are interpreted by the shell — verified via a tmux probe before the
    refactor landed.
    """

    def test_contains_inner_command_verbatim(self):
        result = build_window_shell_cmd("claude --resume abc", "/bin/zsh")
        assert "claude --resume abc" in result

    def test_prepends_fallback_path(self):
        result = build_window_shell_cmd("claude", "/bin/zsh")
        # The fallback dirs guard against launchd/systemd PATH being
        # stripped; same dirs as update_watcher's binary-resolution helper.
        assert ".local/bin" in result
        assert "/opt/homebrew/bin" in result
        assert "/usr/local/bin" in result
        assert "$PATH" in result

    def test_exec_user_shell_after_claude(self):
        # The trailing `; exec <shell>` keeps a debug shell in the pane after
        # claude exits — otherwise the window would silently disappear.
        result = build_window_shell_cmd("claude", "/bin/zsh")
        assert result.rstrip().endswith("; exec /bin/zsh")

    def test_claude_starts_without_inherited_entrypoint(self):
        """An inherited "sdk-cli" would survive into the pane's interactive
        claude (Claude Code keeps an inherited entrypoint) and the hook would
        refuse to register it. Run the real shell string under /bin/sh, as
        tmux does, with a stand-in claude that reports what it inherited."""
        probe = "sh -c 'echo entry=${CLAUDE_CODE_ENTRYPOINT-unset}'"
        result = subprocess.run(
            ["/bin/sh", "-c", build_window_shell_cmd(probe, "true")],
            env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "sdk-cli"},
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "entry=unset"

    def test_preserves_env_var_prefix(self):
        # The README documents `CLAUDE_COMMAND=IS_SANDBOX=1 claude`; this
        # should pass through unchanged so the shell interprets the env var.
        result = build_window_shell_cmd("IS_SANDBOX=1 claude", "/bin/zsh")
        assert "IS_SANDBOX=1 claude" in result


class TestUserShell:
    def test_falls_back_to_zsh_when_pwd_lookup_raises(self):
        with patch("ccbot.tmux_manager.pwd.getpwuid", side_effect=KeyError("nope")):
            assert tm._user_shell() == "/bin/zsh"

    def test_falls_back_to_zsh_when_shell_field_empty(self):
        class _Stub:
            pw_shell = ""

        with patch("ccbot.tmux_manager.pwd.getpwuid", return_value=_Stub()):
            assert tm._user_shell() == "/bin/zsh"

    def test_returns_pwd_shell_when_set(self):
        class _Stub:
            pw_shell = "/usr/bin/fish"

        with patch("ccbot.tmux_manager.pwd.getpwuid", return_value=_Stub()):
            assert tm._user_shell() == "/usr/bin/fish"


class TestBounded:
    """`_bounded` must return the caller's default (not hang) when the
    wrapped blocking call outlives `_TMUX_SUBPROCESS_TIMEOUT` — this is the
    guard against a wedged tmux server freezing session-monitor/status-poll
    callers forever (review findings f19/f60)."""

    async def test_returns_default_and_logs_on_timeout(self, monkeypatch, caplog):
        monkeypatch.setattr(tm, "_TMUX_SUBPROCESS_TIMEOUT", 0.05)
        mgr = TmuxManager()

        def _blocking() -> str:
            time.sleep(0.3)  # far longer than the patched timeout
            return "unreachable"

        start = time.monotonic()
        with caplog.at_level("ERROR", logger="ccbot.tmux_manager"):
            result = await mgr._bounded("test-call", _blocking, "fallback")
        elapsed = time.monotonic() - start

        assert result == "fallback"
        # Returned near the (tiny) timeout, not after the blocking call finished.
        assert elapsed < 0.2
        assert any(
            "test-call" in record.getMessage() and "timed out" in record.getMessage()
            for record in caplog.records
        )

    async def test_returns_real_value_when_call_completes_in_time(self):
        mgr = TmuxManager()
        result = await mgr._bounded("fast-call", lambda: "value", "fallback")
        assert result == "value"


class TestListWindowsTimeout:
    """One representative call-site test: a wedged `get_session()` must not
    hang `list_windows` — it should come back with the safe default `[]`."""

    async def test_list_windows_returns_empty_list_on_timeout(self, monkeypatch):
        monkeypatch.setattr(tm, "_TMUX_SUBPROCESS_TIMEOUT", 0.05)
        mgr = TmuxManager()

        def _blocking_get_session():
            time.sleep(0.3)
            return None

        monkeypatch.setattr(mgr, "get_session", _blocking_get_session)

        result = await mgr.list_windows()

        assert result == []


class TestTextVisibleInPane:
    """`_text_visible_in_pane` is the pure predicate `send_keys` polls before
    sending Enter — a whitespace-insensitive tail check tolerant of TUI
    re-wrapping (RC10/RC11, f38/f62/f37)."""

    def test_empty_needle_is_trivially_true(self):
        # An empty (or all-whitespace) sent_text has nothing to verify —
        # skip verification rather than false-negative on it.
        assert _text_visible_in_pane("anything here", "") is True
        assert _text_visible_in_pane("anything here", "   \n\t  ") is True

    def test_absent_needle_is_false(self):
        assert _text_visible_in_pane("hello world", "goodbye") is False

    def test_present_needle_is_true(self):
        pane = "some text\nhello world\nmore"
        assert _text_visible_in_pane(pane, "hello world") is True

    def test_needle_split_across_wrapped_lines_still_matches(self):
        # A TUI word-wraps "hello world" across two pane lines with no
        # intervening space — whitespace-stripped tail matching must still
        # find it.
        pane = "prompt> hel\nlo world"
        assert _text_visible_in_pane(pane, "hello world") is True

    def test_only_last_40_chars_of_sent_text_are_checked(self):
        tail = "z" * 40
        # The prefix is deliberately absent from the pane; only the exact
        # last-40-chars tail should be required to match.
        long_text = "unrelated-garbage-not-in-pane-at-all" + tail
        pane = f"screen shows: {tail}"
        assert _text_visible_in_pane(pane, long_text) is True

    def test_only_last_15_pane_lines_are_checked(self):
        # A needle that scrolled off the visible 15-line window must not
        # match, even though it's present earlier in the full capture.
        lines = ["needle-line"] + [f"line{i}" for i in range(20)]
        pane = "\n".join(lines)
        assert _text_visible_in_pane(pane, "needle-line") is False


class _FakeCompletedProc:
    """Stand-in for asyncio.subprocess.Process — just communicate()+returncode."""

    def __init__(self, returncode: int = 0, stderr: bytes = b""):
        self.returncode = returncode
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", self._stderr


class TestSendKeysVerifyBeforeEnter:
    """send_keys must observe the text landed (via capture-pane) before
    submitting Enter, instead of assuming it after a fixed sleep — a slow
    TUI redraw would otherwise turn Enter into a stray newline and the user
    gets a silent stall (RC10/RC11, f38/f62/f37)."""

    @pytest.fixture(autouse=True)
    def _fast_poll(self, monkeypatch):
        # Keep the (patched-small) poll spacing fast so tests don't sleep
        # for real; attempt *count* is left at its default (3) since several
        # tests assert on it directly.
        monkeypatch.setattr(tm, "_SEND_VERIFY_POLL_INTERVAL", 0.001)

    @staticmethod
    def _patch_subprocess(monkeypatch, returncode: int = 0, stderr: bytes = b""):
        calls: list[list[str]] = []

        async def _fake_create_subprocess_exec(*args, **kwargs):
            calls.append(list(args))
            return _FakeCompletedProc(returncode=returncode, stderr=stderr)

        monkeypatch.setattr(
            tm.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
        )
        return calls

    async def test_returns_false_and_does_not_send_enter_when_never_visible(
        self, monkeypatch
    ):
        mgr = TmuxManager()
        calls = self._patch_subprocess(monkeypatch)

        async def _never_visible(window_id, with_ansi=False):
            return "prompt> (nothing typed yet)"

        monkeypatch.setattr(mgr, "capture_pane", _never_visible)

        result = await mgr.send_keys("@3", "hello there")

        assert result is False
        assert not any("Enter" in call for call in calls)
        # The literal text itself was still attempted.
        assert any("-l" in call for call in calls)

    async def test_returns_true_and_sends_enter_when_visible(self, monkeypatch):
        mgr = TmuxManager()
        calls = self._patch_subprocess(monkeypatch)

        async def _visible(window_id, with_ansi=False):
            return "prompt> hello there"

        monkeypatch.setattr(mgr, "capture_pane", _visible)

        result = await mgr.send_keys("@3", "hello there")

        assert result is True
        assert any("Enter" in call for call in calls)

    async def test_literal_argv_uses_dash_l_and_separator(self, monkeypatch):
        mgr = TmuxManager()
        calls = self._patch_subprocess(monkeypatch)

        async def _visible(window_id, with_ansi=False):
            return "prompt> -not-an-option"

        monkeypatch.setattr(mgr, "capture_pane", _visible)

        result = await mgr.send_keys("@3", "-not-an-option")

        assert result is True
        literal_call = next(
            call for call in calls if "send-keys" in call and "Enter" not in call
        )
        assert "-l" in literal_call
        assert "--" in literal_call
        # The text must be its own argv element right after '--' (not glued
        # to a flag) — this is what makes a leading '-' safe from being
        # parsed as an option.
        assert literal_call[literal_call.index("--") + 1] == "-not-an-option"
        assert literal_call[-1] == "-not-an-option"

    async def test_retries_polling_until_text_becomes_visible(self, monkeypatch):
        mgr = TmuxManager()
        calls = self._patch_subprocess(monkeypatch)
        attempts = {"n": 0}

        async def _visible_on_third_poll(window_id, with_ansi=False):
            attempts["n"] += 1
            return "hello there" if attempts["n"] >= 3 else "prompt> "

        monkeypatch.setattr(mgr, "capture_pane", _visible_on_third_poll)

        result = await mgr.send_keys("@3", "hello there")

        assert result is True
        assert attempts["n"] == 3
        assert any("Enter" in call for call in calls)

    async def test_gives_up_after_max_attempts(self, monkeypatch):
        mgr = TmuxManager()
        self._patch_subprocess(monkeypatch)
        attempts = {"n": 0}

        async def _never(window_id, with_ansi=False):
            attempts["n"] += 1
            return "prompt> "

        monkeypatch.setattr(mgr, "capture_pane", _never)

        result = await mgr.send_keys("@3", "hello there")

        assert result is False
        assert attempts["n"] == tm._SEND_VERIFY_ATTEMPTS

    async def test_nonzero_returncode_fails_without_retrying_capture(self, monkeypatch):
        # A rejected tmux send-keys (e.g. bad target) must surface as False
        # immediately, not proceed to poll/verify at all.
        mgr = TmuxManager()
        self._patch_subprocess(monkeypatch, returncode=1, stderr=b"can't find pane")
        capture_calls = {"n": 0}

        async def _capture(window_id, with_ansi=False):
            capture_calls["n"] += 1
            return "irrelevant"

        monkeypatch.setattr(mgr, "capture_pane", _capture)

        result = await mgr.send_keys("@999", "hello")

        assert result is False
        assert capture_calls["n"] == 0

    async def test_special_key_path_still_used_when_enter_is_false(self, monkeypatch):
        # literal=False (or enter=False) sends must keep going through the
        # libtmux/_bounded path, not the raw-subprocess verify path.
        mgr = TmuxManager()
        monkeypatch.setattr(mgr, "get_session", lambda: None)

        result = await mgr.send_keys("@3", "Escape", enter=False, literal=False)

        assert result is False

    async def test_concurrent_sends_to_same_window_do_not_interleave(self, monkeypatch):
        mgr = TmuxManager()
        order: list[str] = []

        async def _fake_literal_with_enter(window_id, text):
            order.append(f"start:{text}")
            await asyncio.sleep(0.01)
            order.append(f"end:{text}")
            return True

        monkeypatch.setattr(mgr, "_send_literal_with_enter", _fake_literal_with_enter)

        await asyncio.gather(
            mgr.send_keys("@3", "first"),
            mgr.send_keys("@3", "second"),
        )

        # Whichever request wins the lock, its start/end pair must not be
        # split by the other request's start.
        assert order in (
            ["start:first", "end:first", "start:second", "end:second"],
            ["start:second", "end:second", "start:first", "end:first"],
        )
