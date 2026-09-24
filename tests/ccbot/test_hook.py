"""Tests for Claude Code session tracking hook."""

import io
import json
import os
import subprocess
import sys

import pytest

from ccbot.hook import _UUID_RE, _is_hook_installed, hook_main


@pytest.fixture(autouse=True)
def _interactive_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every hook as the pane's interactive session unless a test says
    otherwise: pytest inherits CLAUDE_CODE_ENTRYPOINT from whatever claude
    launched it, and "sdk-cli" would short-circuit every registration."""
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")


class TestUuidRegex:
    @pytest.mark.parametrize(
        "value",
        [
            "550e8400-e29b-41d4-a716-446655440000",
            "00000000-0000-0000-0000-000000000000",
            "abcdef01-2345-6789-abcd-ef0123456789",
        ],
        ids=["standard", "all-zeros", "all-hex"],
    )
    def test_valid_uuid_matches(self, value: str) -> None:
        assert _UUID_RE.match(value) is not None

    @pytest.mark.parametrize(
        "value",
        [
            "not-a-uuid",
            "550e8400-e29b-41d4-a716",
            "550e8400-e29b-41d4-a716-44665544000g",
            "",
        ],
        ids=["gibberish", "truncated", "invalid-hex-char", "empty"],
    )
    def test_invalid_uuid_no_match(self, value: str) -> None:
        assert _UUID_RE.match(value) is None


class TestIsHookInstalled:
    def test_hook_present(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {"type": "command", "command": "ccbot hook", "timeout": 5}
                        ]
                    }
                ]
            }
        }
        assert _is_hook_installed(settings) is True

    def test_no_hooks_key(self) -> None:
        assert _is_hook_installed({}) is False

    def test_different_hook_command(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "other-tool hook"}]}
                ]
            }
        }
        assert _is_hook_installed(settings) is False

    def test_full_path_matches(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "/usr/bin/ccbot hook",
                                "timeout": 5,
                            }
                        ]
                    }
                ]
            }
        }
        assert _is_hook_installed(settings) is True


class TestHookMainValidation:
    def _run_hook_main(
        self, monkeypatch: pytest.MonkeyPatch, payload: dict, *, tmux_pane: str = ""
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["ccbot", "hook"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        if tmux_pane:
            monkeypatch.setenv("TMUX_PANE", tmux_pane)
        else:
            monkeypatch.delenv("TMUX_PANE", raising=False)
        hook_main()

    def test_missing_session_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {"cwd": "/tmp", "hook_event_name": "SessionStart"},
        )
        assert not (tmp_path / "session_map.json").exists()

    def test_invalid_uuid_format(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {
                "session_id": "not-a-uuid",
                "cwd": "/tmp",
                "hook_event_name": "SessionStart",
            },
        )
        assert not (tmp_path / "session_map.json").exists()

    def test_relative_cwd(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "cwd": "relative/path",
                "hook_event_name": "SessionStart",
            },
        )
        assert not (tmp_path / "session_map.json").exists()

    def test_non_session_start_event(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "cwd": "/tmp",
                "hook_event_name": "Stop",
            },
        )
        assert not (tmp_path / "session_map.json").exists()


class TestHookMainCwdFallback:
    """SessionStart under a daemon-hosted claude (--bg-pty-host) runs with
    TMUX/TMUX_PANE stripped. The hook falls back to matching the session cwd
    against live panes on ccbot's socket, and only re-points a window entry
    that a prior in-pane fire already created — never creates or re-purposes
    one from a cwd guess."""

    SESSION_MAP = {
        "ccbot:@41": {
            "session_id": "11111111-1111-1111-1111-111111111111",
            "cwd": "/proj",
            "window_name": "job",
        },
        "ccbot:@49": {
            "session_id": "22222222-2222-2222-2222-222222222222",
            "cwd": "/other",
            "window_name": "other",
        },
    }

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        *,
        panes: str,
        source: str = "clear",
        seed_map: dict | None = None,
        ps_output: str = "",
    ) -> dict | None:
        """Run hook_main without TMUX_PANE; tmux list-panes returns `panes`,
        ps (used to filter idle shells on ambiguity) returns `ps_output`."""

        def fake_run(cmd, *args, **kwargs):
            if cmd[0] == "ps":
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=ps_output, stderr=""
                )
            assert cmd[:2] == ["tmux", "-L"], cmd
            if "list-sessions" in cmd:
                # Every resolved window in this test class reports back the
                # configured session name ("ccbot"), so a trivial ungrouped
                # entry is enough to have it accepted.
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="ccbot|\n", stderr=""
                )
            assert "list-panes" in cmd
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=panes, stderr=""
            )

        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        if seed_map is not None:
            (tmp_path / "session_map.json").write_text(json.dumps(seed_map))
        monkeypatch.setattr("ccbot.hook.subprocess.run", fake_run)
        monkeypatch.setattr(sys, "argv", ["ccbot", "hook"])
        payload = {
            "session_id": "33333333-3333-3333-3333-333333333333",
            "cwd": "/proj",
            "hook_event_name": "SessionStart",
            "source": source,
        }
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.delenv("TMUX", raising=False)
        hook_main()
        map_file = tmp_path / "session_map.json"
        return json.loads(map_file.read_text()) if map_file.exists() else None

    def test_unique_cwd_match_repoints_existing_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        result = self._run(
            monkeypatch,
            tmp_path,
            panes=("ccbot\t@41\tjob\t/proj\t100\nccbot\t@49\tother\t/other\t200\n"),
            seed_map=self.SESSION_MAP,
        )
        assert result is not None
        assert result["ccbot:@41"]["session_id"] == (
            "33333333-3333-3333-3333-333333333333"
        )
        # Unrelated entry untouched
        assert result["ccbot:@49"] == self.SESSION_MAP["ccbot:@49"]

    def test_no_prior_entry_refuses_to_bind(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """An outside-tmux claude in a directory matching some live pane must
        not create a mapping for that window."""
        seed = {"ccbot:@49": self.SESSION_MAP["ccbot:@49"]}
        result = self._run(
            monkeypatch,
            tmp_path,
            panes="ccbot\t@41\tjob\t/proj\t100\n",
            seed_map=seed,
        )
        assert result == seed

    def test_idle_shell_in_same_dir_filtered_out(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A bare shell parked in the project directory (no claude below it)
        must not block resolution — only the window running a claude client
        counts."""
        result = self._run(
            monkeypatch,
            tmp_path,
            panes=("7\t@29\tzsh\t/proj\t100\nccbot\t@41\tjob\t/proj\t200\n"),
            # pane 100 is an idle zsh; pane 200 has a claude child (201)
            ps_output=("100 1 zsh\n200 1 zsh\n201 200 claude\n"),
            seed_map=self.SESSION_MAP,
        )
        assert result is not None
        assert result["ccbot:@41"]["session_id"] == (
            "33333333-3333-3333-3333-333333333333"
        )

    def test_ambiguous_claude_windows_refuse(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Two live windows on the same directory, both running claude — the
        window cannot be named, so the map must stay untouched."""
        result = self._run(
            monkeypatch,
            tmp_path,
            panes=("ccbot\t@41\tjob\t/proj\t100\nccbot\t@42\tjob-2\t/proj\t200\n"),
            ps_output=("100 1 zsh\n101 100 claude\n200 1 zsh\n201 200 claude\n"),
            seed_map=self.SESSION_MAP,
        )
        assert result == self.SESSION_MAP
        # The refusal is recorded for the maintenance loop to surface.
        failures = (tmp_path / "hook_failures.jsonl").read_text().splitlines()
        assert len(failures) == 1
        failure = json.loads(failures[0])
        assert failure["cwd"] == "/proj"
        assert failure["session_id"] == "33333333-3333-3333-3333-333333333333"

    def test_startup_source_never_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A fresh `claude` launched outside tmux fires source=startup; the
        fallback is reserved for continuations (clear/compact/resume)."""
        result = self._run(
            monkeypatch,
            tmp_path,
            panes="ccbot\t@41\tjob\t/proj\t100\n",
            source="startup",
            seed_map=self.SESSION_MAP,
        )
        assert result == self.SESSION_MAP


class TestHookSocketGate:
    """session_map is ccbot's private state: panes on a foreign tmux server
    (different socket basename in $TMUX) must not be written — their window
    IDs can collide with ccbot's own."""

    PAYLOAD = {
        "session_id": "33333333-3333-3333-3333-333333333333",
        "cwd": "/proj",
        "hook_event_name": "SessionStart",
    }

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        *,
        tmux_env: str,
    ) -> dict | None:
        def fake_run(cmd, *args, **kwargs):
            if cmd[0] == "ps":
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="", stderr=""
                )
            if "list-sessions" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="ccbot|\n", stderr=""
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="ccbot:@41:job\n", stderr=""
            )

        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        monkeypatch.setattr("ccbot.hook.subprocess.run", fake_run)
        monkeypatch.setattr(sys, "argv", ["ccbot", "hook"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(self.PAYLOAD)))
        monkeypatch.setenv("TMUX_PANE", "%9")
        monkeypatch.setenv("TMUX", tmux_env)
        monkeypatch.delenv("TMUX_SOCKET_NAME", raising=False)
        monkeypatch.delenv("TMUX_SESSION_NAME", raising=False)
        hook_main()
        map_file = tmp_path / "session_map.json"
        return json.loads(map_file.read_text()) if map_file.exists() else None

    def test_foreign_socket_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        result = self._run(
            monkeypatch, tmp_path, tmux_env="/tmp/tmux-1002/default,999,0"
        )
        assert result is None

    def test_ccbot_socket_is_written(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        result = self._run(monkeypatch, tmp_path, tmux_env="/tmp/tmux-1002/ccbot,999,0")
        assert result is not None
        assert result["ccbot:@41"]["session_id"] == (
            "33333333-3333-3333-3333-333333333333"
        )


class TestHookForeignSessionGate:
    """The socket gate (TestHookSocketGate) only catches a foreign tmux
    *server*. A user's personal interactive tmux session (e.g. a
    default-named session "7") can perfectly well live on ccbot's own
    dedicated socket alongside ccbot's session — same $TMUX socket basename,
    different #{session_name}. Claude Code started in one of those panes
    must not get a session_map entry: readers only ever look at the
    configured session (plus grouped peers), so anything else is inert
    pollution the hook should just not write."""

    PAYLOAD = {
        "session_id": "33333333-3333-3333-3333-333333333333",
        "cwd": "/proj",
        "hook_event_name": "SessionStart",
    }

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        *,
        pane_output: str,
        list_sessions_output: str,
        list_sessions_rc: int = 0,
    ) -> dict | None:
        def fake_run(cmd, *args, **kwargs):
            if cmd[0] == "ps":
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="", stderr=""
                )
            if "list-sessions" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=list_sessions_rc,
                    stdout=list_sessions_output,
                    stderr="" if list_sessions_rc == 0 else "tmux exploded",
                )
            assert "display-message" in cmd, cmd
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=pane_output, stderr=""
            )

        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        monkeypatch.setattr("ccbot.hook.subprocess.run", fake_run)
        monkeypatch.setattr(sys, "argv", ["ccbot", "hook"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(self.PAYLOAD)))
        monkeypatch.setenv("TMUX_PANE", "%9")
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.delenv("TMUX_SOCKET_NAME", raising=False)
        monkeypatch.delenv("TMUX_SESSION_NAME", raising=False)
        hook_main()
        map_file = tmp_path / "session_map.json"
        return json.loads(map_file.read_text()) if map_file.exists() else None

    def test_foreign_ungrouped_session_is_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level("INFO", logger="ccbot.hook"):
            result = self._run(
                monkeypatch,
                tmp_path,
                pane_output="7:@29:zsh\n",
                list_sessions_output="ccbot|\n7|\n",
            )
        assert result is None
        assert any(
            "foreign tmux session" in record.getMessage()
            and "'7'" in record.getMessage()
            for record in caplog.records
        )

    def test_grouped_peer_session_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A session sharing the configured session's group (e.g. reattached
        via `tmux new-session -t ccbot`) is accepted even though its name
        differs from TMUX_SESSION_NAME — it's the same logical ccbot session,
        not a foreign one."""
        result = self._run(
            monkeypatch,
            tmp_path,
            pane_output="ccbot-2:@7:job\n",
            list_sessions_output="ccbot|grp1\nccbot-2|grp1\n",
        )
        assert result is not None
        assert result["ccbot-2:@7"]["session_id"] == (
            "33333333-3333-3333-3333-333333333333"
        )

    def test_list_sessions_failure_fails_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A transient tmux hiccup querying list-sessions must not block a
        legitimate registration — unknown is treated as accept-anything."""
        result = self._run(
            monkeypatch,
            tmp_path,
            pane_output="ccbot:@41:job\n",
            list_sessions_output="",
            list_sessions_rc=1,
        )
        assert result is not None
        assert result["ccbot:@41"]["session_id"] == (
            "33333333-3333-3333-3333-333333333333"
        )


class TestHookNestedClaudeGate:
    """A child claude spawned from inside a pane (a tool shelling out to
    ``claude -p``) inherits TMUX_PANE, so its SessionStart hook reaches the
    pane path looking exactly like the pane's own session — and would steal
    the window's mapping (the 2026-07-15 lateen incident: every Game Master
    turn re-pointed the window at a throwaway session, silencing the topic).
    The pane's own session runs the hook under exactly one claude ancestor;
    a child session under two or more. A failed ps read must fail open."""

    PAYLOAD = {
        "session_id": "33333333-3333-3333-3333-333333333333",
        "cwd": "/proj",
        "hook_event_name": "SessionStart",
    }

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        *,
        ps_output: str,
        ps_rc: int = 0,
    ) -> dict | None:
        def fake_run(cmd, *args, **kwargs):
            if cmd[0] == "ps":
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=ps_rc,
                    stdout=ps_output,
                    stderr="" if ps_rc == 0 else "ps exploded",
                )
            if "list-sessions" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="ccbot|\n", stderr=""
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="ccbot:@41:job\n", stderr=""
            )

        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        monkeypatch.setattr("ccbot.hook.subprocess.run", fake_run)
        monkeypatch.setattr(sys, "argv", ["ccbot", "hook"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(self.PAYLOAD)))
        monkeypatch.setenv("TMUX_PANE", "%9")
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.delenv("TMUX_SOCKET_NAME", raising=False)
        monkeypatch.delenv("TMUX_SESSION_NAME", raising=False)
        hook_main()
        map_file = tmp_path / "session_map.json"
        return json.loads(map_file.read_text()) if map_file.exists() else None

    def _ps_table(self, *, nested: bool) -> str:
        """A process table rooting this test process in a pane's claude —
        with or without a second claude between them (the child-session
        case: hook <- sh <- claude -p <- python tool <- claude <- pane)."""
        me = str(os.getpid())
        lines = [
            f"{me} 77770 /opt/homebrew/bin/python3",
            "77770 77771 sh",
            "77771 77772 claude",
        ]
        if nested:
            lines += [
                "77772 77773 python3",
                "77773 77774 claude",
                "77774 77775 -zsh",
            ]
        else:
            lines += ["77772 77775 -zsh"]
        lines += ["77775 1 tmux", "1 0 /sbin/launchd"]
        return "\n".join(lines) + "\n"

    def test_nested_claude_is_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level("INFO", logger="ccbot.hook"):
            result = self._run(
                monkeypatch, tmp_path, ps_output=self._ps_table(nested=True)
            )
        assert result is None
        assert any(
            "nested inside another claude" in record.getMessage()
            for record in caplog.records
        )

    def test_single_claude_ancestor_registers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        result = self._run(
            monkeypatch, tmp_path, ps_output=self._ps_table(nested=False)
        )
        assert result is not None
        assert result["ccbot:@41"]["session_id"] == (
            "33333333-3333-3333-3333-333333333333"
        )

    @pytest.mark.parametrize("entrypoint", ["sdk-cli", "sdk-py", "sdk-ts"])
    def test_non_interactive_claude_is_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        caplog: pytest.LogCaptureFixture,
        entrypoint: str,
    ) -> None:
        """The 2026-09-24 overworld incident: `claude -p` children of the
        pane's session stole its mapping although ps showed a single claude
        ancestor. The entrypoint marks them regardless of process ancestry."""
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", entrypoint)
        with caplog.at_level("INFO", logger="ccbot.hook"):
            result = self._run(
                monkeypatch, tmp_path, ps_output=self._ps_table(nested=False)
            )
        assert result is None
        assert any(
            "non-interactive claude" in record.getMessage() for record in caplog.records
        )

    def test_unset_entrypoint_fails_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)
        result = self._run(
            monkeypatch, tmp_path, ps_output=self._ps_table(nested=False)
        )
        assert result is not None
        assert result["ccbot:@41"]["session_id"] == (
            "33333333-3333-3333-3333-333333333333"
        )

    def test_ps_failure_fails_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A transient ps failure must not block a legitimate registration —
        unknown ancestry is treated as not nested."""
        result = self._run(monkeypatch, tmp_path, ps_output="", ps_rc=1)
        assert result is not None
        assert result["ccbot:@41"]["session_id"] == (
            "33333333-3333-3333-3333-333333333333"
        )


class TestHookMainWritePath:
    """Tests that exercise the session_map write path with tmux mocked.

    These cover behavior the validation tests can't reach because they all
    short-circuit before the tmux query.
    """

    def _run_hook_main_with_tmux(
        self,
        monkeypatch: pytest.MonkeyPatch,
        payload: dict,
        *,
        tmux_pane: str,
        tmux_output: str,
        list_sessions_output: str = "ccbot|grp1\nccbot-2|grp1\n",
    ) -> None:
        """Run hook_main with `subprocess.run` mocked to return `tmux_output`
        for the pane resolution call, and `list_sessions_output` for the
        session-acceptance query. The default groups "ccbot" and "ccbot-2"
        together so tests resolving to either name are accepted.
        """

        def fake_run(cmd, *args, **kwargs):
            if cmd[0] == "ps":
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="", stderr=""
                )
            if "list-sessions" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=list_sessions_output, stderr=""
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=tmux_output, stderr=""
            )

        monkeypatch.setattr("ccbot.hook.subprocess.run", fake_run)
        monkeypatch.setattr(sys, "argv", ["ccbot", "hook"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        monkeypatch.setenv("TMUX_PANE", tmux_pane)
        hook_main()

    def test_dedups_grouped_peer_entries_for_same_window_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Grouped tmux sessions share windows, so the hook can fire under
        peer A in one attach and peer B in another — both targeting the
        same window @48. Without dedup the old peer's key (with a now-stale
        session_id) lingers forever and downstream readers must guess which
        is current. Hook must atomically drop other-prefix entries for the
        same window_id when it writes."""
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@48": {
                        "session_id": "11111111-1111-1111-1111-111111111111",
                        "cwd": "/proj",
                        "window_name": "job",
                    },
                    "ccbot:@49": {  # different window — MUST be preserved
                        "session_id": "22222222-2222-2222-2222-222222222222",
                        "cwd": "/other",
                        "window_name": "other",
                    },
                }
            )
        )

        self._run_hook_main_with_tmux(
            monkeypatch,
            {
                "session_id": "33333333-3333-3333-3333-333333333333",
                "cwd": "/proj",
                "hook_event_name": "SessionStart",
            },
            tmux_pane="%99",
            tmux_output="ccbot-2:@48:job\n",
        )

        result = json.loads(session_map_file.read_text())
        assert result == {
            "ccbot-2:@48": {
                "session_id": "33333333-3333-3333-3333-333333333333",
                "cwd": "/proj",
                "window_name": "job",
                "transcript_size_at_start": 0,
            },
            "ccbot:@49": {
                "session_id": "22222222-2222-2222-2222-222222222222",
                "cwd": "/other",
                "window_name": "other",
            },
        }

    def test_overwrite_same_key_does_not_remove_unrelated_entries(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """When the hook writes the same key it already had (claude restart
        in the same session), nothing else should be touched. Guards the
        dedup loop against a `k != session_window_key` slip that would
        delete the key it just wrote."""
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@48": {
                        "session_id": "11111111-1111-1111-1111-111111111111",
                        "cwd": "/proj",
                        "window_name": "job",
                    },
                    "ccbot:@49": {
                        "session_id": "22222222-2222-2222-2222-222222222222",
                        "cwd": "/other",
                        "window_name": "other",
                    },
                }
            )
        )

        self._run_hook_main_with_tmux(
            monkeypatch,
            {
                "session_id": "33333333-3333-3333-3333-333333333333",
                "cwd": "/proj",
                "hook_event_name": "SessionStart",
            },
            tmux_pane="%99",
            tmux_output="ccbot:@48:job\n",
        )

        result = json.loads(session_map_file.read_text())
        assert result == {
            "ccbot:@48": {
                "session_id": "33333333-3333-3333-3333-333333333333",
                "cwd": "/proj",
                "window_name": "job",
                "transcript_size_at_start": 0,
            },
            "ccbot:@49": {
                "session_id": "22222222-2222-2222-2222-222222222222",
                "cwd": "/other",
                "window_name": "other",
            },
        }


class TestHookMainTranscriptSizeAtStart:
    """The SessionStart stdin payload carries transcript_path; the hook
    records the transcript's size at that exact moment as
    transcript_size_at_start, since only the hook can observe it. The
    session monitor later seeds a newly-noticed session's read offset there
    instead of at current EOF (review f17/RC38) — otherwise a reply that
    landed before the monitor's first poll is silently skipped."""

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        *,
        transcript_path: str,
    ) -> dict:
        def fake_run(cmd, *args, **kwargs):
            if cmd[0] == "ps":
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="", stderr=""
                )
            if "list-sessions" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="ccbot|\n", stderr=""
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="ccbot:@41:job\n", stderr=""
            )

        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        monkeypatch.setattr("ccbot.hook.subprocess.run", fake_run)
        monkeypatch.setattr(sys, "argv", ["ccbot", "hook"])
        payload = {
            "session_id": "33333333-3333-3333-3333-333333333333",
            "cwd": "/proj",
            "hook_event_name": "SessionStart",
            "transcript_path": transcript_path,
        }
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        monkeypatch.setenv("TMUX_PANE", "%9")
        hook_main()
        map_file = tmp_path / "session_map.json"
        return json.loads(map_file.read_text())

    def test_existing_transcript_records_its_size(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_bytes(b"x" * 123)

        result = self._run(monkeypatch, tmp_path, transcript_path=str(transcript))

        assert result["ccbot:@41"]["transcript_size_at_start"] == 123

    def test_missing_transcript_records_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        result = self._run(
            monkeypatch,
            tmp_path,
            transcript_path=str(tmp_path / "does-not-exist.jsonl"),
        )

        assert result["ccbot:@41"]["transcript_size_at_start"] == 0

    def test_relative_transcript_path_records_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        result = self._run(monkeypatch, tmp_path, transcript_path="relative.jsonl")

        assert result["ccbot:@41"]["transcript_size_at_start"] == 0

    def test_no_transcript_path_records_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        result = self._run(monkeypatch, tmp_path, transcript_path="")

        assert result["ccbot:@41"]["transcript_size_at_start"] == 0
