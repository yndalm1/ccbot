"""Tests for maintenance — hook-failure notices and divergence detection."""

import json
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from telegram.error import RetryAfter

import ccbot.handlers.maintenance as maintenance
from ccbot.config import config


@pytest.fixture(autouse=True)
def _reset_module_state():
    maintenance._hook_failures_offset = None
    maintenance._divergence.clear()
    maintenance._missing_since.clear()
    yield
    maintenance._hook_failures_offset = None
    maintenance._divergence.clear()
    maintenance._missing_since.clear()


def _fake_session_manager(bindings, window_states):
    sm = SimpleNamespace()
    sm.iter_thread_bindings = lambda: iter(bindings)
    sm.window_states = window_states
    sm.resolve_chat_id = lambda user_id, thread_id=None: 100
    return sm


class TestHookFailureNotices:
    @pytest.mark.asyncio
    async def test_first_run_skips_history_then_tails_new_lines(
        self, tmp_path, monkeypatch
    ):
        """A restart must not replay historical failures; only lines appended
        after the first check are surfaced, and only to topics bound to the
        failure's cwd."""
        monkeypatch.setattr(config, "session_map_file", tmp_path / "session_map.json")
        failures = tmp_path / "hook_failures.jsonl"
        failures.write_text(
            json.dumps({"ts": 1.0, "cwd": "/proj", "session_id": "old", "reason": "x"})
            + "\n"
        )
        bot = AsyncMock()
        sm = _fake_session_manager(
            bindings=[(1, 42, "@41"), (1, 43, "@50")],
            window_states={
                "@41": SimpleNamespace(session_id="sid-a", cwd="/proj"),
                "@50": SimpleNamespace(session_id="sid-b", cwd="/other"),
            },
        )

        with (
            patch("ccbot.handlers.maintenance.session_manager", sm),
            patch(
                "ccbot.handlers.maintenance.safe_send", new_callable=AsyncMock
            ) as mock_safe_send,
        ):
            await maintenance._check_hook_failures(bot)  # first run: seek EOF
            mock_safe_send.assert_not_called()

            with open(failures, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": 2.0,
                            "cwd": "/proj",
                            "session_id": "44444444-4444-4444-4444-444444444444",
                            "reason": "no unique claude window",
                        }
                    )
                    + "\n"
                )
            await maintenance._check_hook_failures(bot)

        mock_safe_send.assert_called_once()
        args, kwargs = mock_safe_send.call_args
        assert args[1] == 100  # chat_id from resolve_chat_id
        text = args[2]
        assert kwargs["message_thread_id"] == 42  # /proj topic only
        assert "no unique claude window" in text
        assert "44444444" in text

    @pytest.mark.asyncio
    async def test_processed_lines_are_not_resent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "session_map_file", tmp_path / "session_map.json")
        failures = tmp_path / "hook_failures.jsonl"
        failures.write_text("")
        bot = AsyncMock()
        sm = _fake_session_manager(
            bindings=[(1, 42, "@41")],
            window_states={"@41": SimpleNamespace(session_id="s", cwd="/proj")},
        )

        with (
            patch("ccbot.handlers.maintenance.session_manager", sm),
            patch(
                "ccbot.handlers.maintenance.safe_send", new_callable=AsyncMock
            ) as mock_safe_send,
        ):
            await maintenance._check_hook_failures(bot)
            with open(failures, "a") as f:
                f.write(json.dumps({"cwd": "/proj", "reason": "r"}) + "\n")
            await maintenance._check_hook_failures(bot)
            await maintenance._check_hook_failures(bot)

        assert mock_safe_send.call_count == 1

    @pytest.mark.asyncio
    async def test_notice_falls_back_to_plain_text_on_markdown_failure(
        self, tmp_path, monkeypatch
    ):
        """Exercises the real safe_send wiring (not mocked): MarkdownV2 first,
        plain text fallback on failure — proving the notice is no longer a
        bare bot.send_message call with no fallback."""
        monkeypatch.setattr(config, "session_map_file", tmp_path / "session_map.json")
        failures = tmp_path / "hook_failures.jsonl"
        failures.write_text("")
        bot = AsyncMock()
        calls: list[dict] = []

        async def fake_send(*_args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ValueError("bad markdown entities")
            return SimpleNamespace(message_id=1)

        bot.send_message = AsyncMock(side_effect=fake_send)
        sm = _fake_session_manager(
            bindings=[(1, 42, "@41")],
            window_states={"@41": SimpleNamespace(session_id="s", cwd="/proj")},
        )

        with patch("ccbot.handlers.maintenance.session_manager", sm):
            await maintenance._check_hook_failures(bot)
            with open(failures, "a") as f:
                f.write(json.dumps({"cwd": "/proj", "reason": "r"}) + "\n")
            await maintenance._check_hook_failures(bot)

        assert len(calls) == 2
        assert calls[0]["parse_mode"] == "MarkdownV2"
        assert "parse_mode" not in calls[1]
        assert calls[1]["message_thread_id"] == 42

    @pytest.mark.asyncio
    async def test_retryafter_on_one_topic_does_not_block_another(
        self, tmp_path, monkeypatch
    ):
        """A RetryAfter delivering to one topic must be logged and skipped,
        not crash the sweep or block delivery to a second bound topic."""
        monkeypatch.setattr(config, "session_map_file", tmp_path / "session_map.json")
        failures = tmp_path / "hook_failures.jsonl"
        failures.write_text("")
        bot = AsyncMock()
        sm = _fake_session_manager(
            bindings=[(1, 42, "@41"), (1, 43, "@42")],
            window_states={
                "@41": SimpleNamespace(session_id="s1", cwd="/proj"),
                "@42": SimpleNamespace(session_id="s2", cwd="/proj"),
            },
        )
        delivered: list[int | None] = []

        async def fake_safe_send(_bot, _chat_id, _text, **kwargs):
            thread_id = kwargs.get("message_thread_id")
            delivered.append(thread_id)
            if thread_id == 42:
                raise RetryAfter(retry_after=1)

        with (
            patch("ccbot.handlers.maintenance.session_manager", sm),
            patch("ccbot.handlers.maintenance.safe_send", side_effect=fake_safe_send),
        ):
            await maintenance._check_hook_failures(bot)  # first run: seek EOF
            with open(failures, "a") as f:
                f.write(json.dumps({"cwd": "/proj", "reason": "r"}) + "\n")
            await maintenance._check_hook_failures(bot)

        assert delivered == [42, 43]


class TestDivergenceDetection:
    def _setup(self, tmp_path, monkeypatch, *, bindings=None, window_states=None):
        """Project dir with a frozen tracked transcript and a fresh sibling."""
        proj = tmp_path / "proj"
        proj.mkdir()
        tracked = proj / "sid-a.jsonl"
        tracked.write_text("{}\n")
        stale = time.time() - 600
        os.utime(tracked, (stale, stale))
        candidate = proj / "sid-b.jsonl"
        candidate.write_text("{}\n")

        monitor_state_file = tmp_path / "monitor_state.json"
        monitor_state_file.write_text(
            json.dumps(
                {
                    "tracked_sessions": {
                        "sid-a": {
                            "session_id": "sid-a",
                            "file_path": str(tracked),
                            "last_byte_offset": 0,
                        }
                    }
                }
            )
        )
        monkeypatch.setattr(config, "monitor_state_file", monitor_state_file)

        sm = _fake_session_manager(
            bindings=bindings or [(1, 42, "@41")],
            window_states=window_states
            or {"@41": SimpleNamespace(session_id="sid-a", cwd="/proj")},
        )
        return candidate, sm

    @staticmethod
    def _grow(path):
        with open(path, "a") as f:
            f.write("{}\n")

    @pytest.mark.asyncio
    async def test_notice_after_two_growth_ticks_then_once(self, tmp_path, monkeypatch):
        candidate, sm = self._setup(tmp_path, monkeypatch)
        bot = AsyncMock()

        with (
            patch("ccbot.handlers.maintenance.session_manager", sm),
            patch(
                "ccbot.handlers.maintenance.safe_send", new_callable=AsyncMock
            ) as mock_safe_send,
        ):
            await maintenance._check_divergence(bot)  # tick 1: observe
            mock_safe_send.assert_not_called()

            self._grow(candidate)
            await maintenance._check_divergence(bot)  # tick 2: growth -> notice
            mock_safe_send.assert_called_once()

            self._grow(candidate)
            await maintenance._check_divergence(bot)  # tick 3: no duplicate
            mock_safe_send.assert_called_once()

        args, kwargs = mock_safe_send.call_args
        assert "sid-b" in args[2]
        assert kwargs["message_thread_id"] == 42
        button = kwargs["reply_markup"].inline_keyboard[0][0]
        assert button.callback_data == "rp:@41:sid-b"

    @pytest.mark.asyncio
    async def test_no_notice_without_growth(self, tmp_path, monkeypatch):
        """A static sibling file (e.g. an old finished session) never fires."""
        _candidate, sm = self._setup(tmp_path, monkeypatch)
        bot = AsyncMock()

        with (
            patch("ccbot.handlers.maintenance.session_manager", sm),
            patch(
                "ccbot.handlers.maintenance.safe_send", new_callable=AsyncMock
            ) as mock_safe_send,
        ):
            await maintenance._check_divergence(bot)
            await maintenance._check_divergence(bot)
            await maintenance._check_divergence(bot)

        mock_safe_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_ambiguous_directory_suppresses_notice(self, tmp_path, monkeypatch):
        """Two bound windows on one directory: the growing sibling can't be
        attributed to either, so no notice (same refusal as the hook)."""
        candidate, _ = self._setup(tmp_path, monkeypatch)
        sm = _fake_session_manager(
            bindings=[(1, 42, "@41"), (1, 43, "@42")],
            window_states={
                "@41": SimpleNamespace(session_id="sid-a", cwd="/proj"),
                "@42": SimpleNamespace(session_id="sid-c", cwd="/proj"),
            },
        )
        bot = AsyncMock()

        with (
            patch("ccbot.handlers.maintenance.session_manager", sm),
            patch(
                "ccbot.handlers.maintenance.safe_send", new_callable=AsyncMock
            ) as mock_safe_send,
        ):
            await maintenance._check_divergence(bot)
            self._grow(candidate)
            await maintenance._check_divergence(bot)
            self._grow(candidate)
            await maintenance._check_divergence(bot)

        mock_safe_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_recently_active_tracked_file_clears_episode(
        self, tmp_path, monkeypatch
    ):
        """A tracked transcript that is still writing is not stale, no matter
        what its siblings do (long thinking turns must never false-alarm)."""
        candidate, sm = self._setup(tmp_path, monkeypatch)
        # Tracked file wrote just now.
        tracked = tmp_path / "proj" / "sid-a.jsonl"
        now = time.time()
        os.utime(tracked, (now, now))
        bot = AsyncMock()

        with (
            patch("ccbot.handlers.maintenance.session_manager", sm),
            patch(
                "ccbot.handlers.maintenance.safe_send", new_callable=AsyncMock
            ) as mock_safe_send,
        ):
            await maintenance._check_divergence(bot)
            self._grow(candidate)
            await maintenance._check_divergence(bot)

        mock_safe_send.assert_not_called()
        assert maintenance._divergence == {}

    def _setup_missing(self, tmp_path, monkeypatch):
        """Window tracks a session whose transcript never reached disk (a
        mapping stolen by a `claude -p --no-session-persistence` child), while
        the window's real transcript keeps growing next door."""
        proj = tmp_path / "proj"
        proj.mkdir()
        candidate = proj / "sid-real.jsonl"
        candidate.write_text("{}\n")
        monitor_state_file = tmp_path / "monitor_state.json"
        monitor_state_file.write_text(json.dumps({"tracked_sessions": {}}))
        monkeypatch.setattr(config, "monitor_state_file", monitor_state_file)
        sm = _fake_session_manager(
            bindings=[(1, 42, "@41")],
            window_states={"@41": SimpleNamespace(session_id="sid-ghost", cwd="/proj")},
        )
        sm._build_session_file_path = lambda sid, cwd: proj / f"{sid}.jsonl"
        return candidate, sm

    @pytest.mark.asyncio
    async def test_missing_transcript_counts_as_frozen_after_quiet_period(
        self, tmp_path, monkeypatch
    ):
        candidate, sm = self._setup_missing(tmp_path, monkeypatch)
        maintenance._missing_since["@41"] = ("sid-ghost", time.time() - 600)
        bot = AsyncMock()

        with (
            patch("ccbot.handlers.maintenance.session_manager", sm),
            patch(
                "ccbot.handlers.maintenance.safe_send", new_callable=AsyncMock
            ) as mock_safe_send,
        ):
            await maintenance._check_divergence(bot)
            self._grow(candidate)
            await maintenance._check_divergence(bot)

        mock_safe_send.assert_called_once()
        args, kwargs = mock_safe_send.call_args
        assert "no transcript on disk" in args[2]
        button = kwargs["reply_markup"].inline_keyboard[0][0]
        assert button.callback_data == "rp:@41:sid-real"

    @pytest.mark.asyncio
    async def test_newly_missing_transcript_waits_out_quiet_period(
        self, tmp_path, monkeypatch
    ):
        """A fresh session has no transcript until its first message, so a
        just-noticed missing file must not fire even beside a growing sibling."""
        candidate, sm = self._setup_missing(tmp_path, monkeypatch)
        bot = AsyncMock()

        with (
            patch("ccbot.handlers.maintenance.session_manager", sm),
            patch(
                "ccbot.handlers.maintenance.safe_send", new_callable=AsyncMock
            ) as mock_safe_send,
        ):
            await maintenance._check_divergence(bot)
            self._grow(candidate)
            await maintenance._check_divergence(bot)

        mock_safe_send.assert_not_called()
        assert maintenance._missing_since["@41"][0] == "sid-ghost"

    @pytest.mark.asyncio
    async def test_retryafter_is_logged_and_swallowed(self, tmp_path, monkeypatch):
        """A RetryAfter delivering a divergence notice must not propagate out
        of _check_divergence (run_maintenance_once relies on each step being
        isolated)."""
        candidate, sm = self._setup(tmp_path, monkeypatch)
        bot = AsyncMock()

        with (
            patch("ccbot.handlers.maintenance.session_manager", sm),
            patch(
                "ccbot.handlers.maintenance.safe_send",
                side_effect=RetryAfter(retry_after=1),
            ),
        ):
            await maintenance._check_divergence(bot)
            self._grow(candidate)
            await maintenance._check_divergence(bot)  # must not raise
