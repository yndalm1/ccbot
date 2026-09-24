"""Tests for SessionManager pure dict operations."""

import json
from unittest.mock import AsyncMock

import pytest

from ccbot.config import config
from ccbot.session import SessionManager


@pytest.fixture
def mgr(monkeypatch) -> SessionManager:
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    return SessionManager()


class TestThreadBindings:
    def test_bind_and_get(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        assert mgr.get_window_for_thread(100, 1) == "@1"

    def test_bind_unbind_get_returns_none(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.unbind_thread(100, 1)
        assert mgr.get_window_for_thread(100, 1) is None

    def test_unbind_nonexistent_returns_none(self, mgr: SessionManager) -> None:
        assert mgr.unbind_thread(100, 999) is None

    def test_iter_thread_bindings(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.bind_thread(100, 2, "@2")
        mgr.bind_thread(200, 3, "@3")
        result = set(mgr.iter_thread_bindings())
        assert result == {(100, 1, "@1"), (100, 2, "@2"), (200, 3, "@3")}

    def test_bind_thread_returns_true_on_fresh_bind(self, mgr: SessionManager) -> None:
        assert mgr.bind_thread(100, 1, "@1") is True

    def test_bind_thread_returns_false_on_cross_topic_conflict(
        self, mgr: SessionManager
    ) -> None:
        """Owner-enforced '1 topic = 1 window' (RC3/RC4/f46): a window
        already bound to a different (user, thread) must be refused, not
        silently double-bound."""
        assert mgr.bind_thread(100, 1, "@1") is True

        result = mgr.bind_thread(200, 2, "@1")

        assert result is False
        # The original binding is untouched; the conflicting one never lands.
        assert mgr.get_window_for_thread(100, 1) == "@1"
        assert mgr.get_window_for_thread(200, 2) is None

    def test_bind_thread_returns_true_on_idempotent_rebind(
        self, mgr: SessionManager
    ) -> None:
        """Re-binding the SAME (user, thread) to the same window_id stays
        allowed — e.g. re-confirming an existing binding."""
        assert mgr.bind_thread(100, 1, "@1") is True

        result = mgr.bind_thread(100, 1, "@1", window_name="renamed")

        assert result is True
        assert mgr.get_window_for_thread(100, 1) == "@1"
        assert mgr.get_display_name("@1") == "renamed"


class TestFindUsersForSessionConcurrentUnbind:
    """Regression (f45/RC7): `find_users_for_session` used to iterate
    `self.iter_thread_bindings()` directly while awaiting inside the loop.
    A concurrent unbind mutating `thread_bindings` mid-iteration raised
    RuntimeError ("dictionary changed size during iteration"), which the
    dispatch path swallowed, silently dropping the message. Materializing
    the bindings into a list first (as status_polling.py already does)
    must survive the same race.
    """

    @pytest.mark.asyncio
    async def test_survives_unbind_during_iteration(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.bind_thread(200, 2, "@2")

        async def fake_resolve_session_for_window(window_id: str):
            if window_id == "@1":
                # Simulate a concurrent unbind racing this await — removing
                # user 200's only binding also deletes their key from the
                # outer thread_bindings dict (unbind_thread), which is the
                # exact mutation that crashes a live (non-materialized)
                # iterator over it.
                mgr.unbind_thread(200, 2)
            return None

        mgr.resolve_session_for_window = fake_resolve_session_for_window  # type: ignore[method-assign]

        # Must complete without raising RuntimeError.
        result = await mgr.find_users_for_session("some-session")

        assert result == []
        # The race actually happened (sanity check the test reproduces it).
        assert mgr.get_window_for_thread(200, 2) is None


class TestGroupChatId:
    """Tests for group chat_id routing (supergroup forum topic support).

    IMPORTANT: These tests protect against regression. The group_chat_ids
    mapping is required for Telegram supergroup forum topics — without it,
    all outbound messages fail with "Message thread not found". This was
    erroneously removed once (26cb81f) and restored in PR #23. Do NOT
    delete these tests or the underlying functionality.
    """

    def test_resolve_with_stored_group_id(self, mgr: SessionManager) -> None:
        """resolve_chat_id returns stored group chat_id for known thread."""
        mgr.set_group_chat_id(100, 1, -1001234567890)
        assert mgr.resolve_chat_id(100, 1) == -1001234567890

    def test_resolve_without_group_id_falls_back_to_user_id(
        self, mgr: SessionManager
    ) -> None:
        """resolve_chat_id falls back to user_id when no group_id stored."""
        assert mgr.resolve_chat_id(100, 1) == 100

    def test_resolve_none_thread_id_falls_back_to_user_id(
        self, mgr: SessionManager
    ) -> None:
        """resolve_chat_id returns user_id when thread_id is None (private chat)."""
        mgr.set_group_chat_id(100, 1, -1001234567890)
        assert mgr.resolve_chat_id(100) == 100

    def test_set_group_chat_id_overwrites(self, mgr: SessionManager) -> None:
        """set_group_chat_id updates the stored value on change."""
        mgr.set_group_chat_id(100, 1, -999)
        mgr.set_group_chat_id(100, 1, -888)
        assert mgr.resolve_chat_id(100, 1) == -888

    def test_multiple_threads_independent(self, mgr: SessionManager) -> None:
        """Different threads for the same user store independent group chat_ids."""
        mgr.set_group_chat_id(100, 1, -111)
        mgr.set_group_chat_id(100, 2, -222)
        assert mgr.resolve_chat_id(100, 1) == -111
        assert mgr.resolve_chat_id(100, 2) == -222

    def test_multiple_users_independent(self, mgr: SessionManager) -> None:
        """Different users store independent group chat_ids."""
        mgr.set_group_chat_id(100, 1, -111)
        mgr.set_group_chat_id(200, 1, -222)
        assert mgr.resolve_chat_id(100, 1) == -111
        assert mgr.resolve_chat_id(200, 1) == -222

    def test_set_group_chat_id_with_none_thread(self, mgr: SessionManager) -> None:
        """set_group_chat_id handles None thread_id (mapped to 0)."""
        mgr.set_group_chat_id(100, None, -999)
        # thread_id=None in resolve falls back to user_id (by design)
        assert mgr.resolve_chat_id(100, None) == 100
        # The stored key is "100:0", only accessible with explicit thread_id=0
        assert mgr.group_chat_ids.get("100:0") == -999


class TestWindowState:
    def test_get_creates_new(self, mgr: SessionManager) -> None:
        state = mgr.get_window_state("@0")
        assert state.session_id == ""
        assert state.cwd == ""

    def test_get_returns_existing(self, mgr: SessionManager) -> None:
        state = mgr.get_window_state("@1")
        state.session_id = "abc"
        assert mgr.get_window_state("@1").session_id == "abc"

    def test_clear_window_session(self, mgr: SessionManager) -> None:
        state = mgr.get_window_state("@1")
        state.session_id = "abc"
        mgr.clear_window_session("@1")
        assert mgr.get_window_state("@1").session_id == ""

    def test_clear_window_session_unpins(self, mgr: SessionManager) -> None:
        # /clear forwarded to a pinned window must drop the pin too, or the
        # next load_session_map has no way to accept the fresh post-clear sid.
        state = mgr.get_window_state("@1")
        state.session_id = "abc"
        state.pinned_over = "hook-sid"
        mgr.clear_window_session("@1")
        assert mgr.get_window_state("@1").pinned_over is None


class TestResolveWindowForThread:
    def test_none_thread_id_returns_none(self, mgr: SessionManager) -> None:
        assert mgr.resolve_window_for_thread(100, None) is None

    def test_unbound_thread_returns_none(self, mgr: SessionManager) -> None:
        assert mgr.resolve_window_for_thread(100, 42) is None

    def test_bound_thread_returns_window(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 42, "@3")
        assert mgr.resolve_window_for_thread(100, 42) == "@3"


class TestDisplayNames:
    def test_get_display_name_fallback(self, mgr: SessionManager) -> None:
        """get_display_name returns window_id when no display name is set."""
        assert mgr.get_display_name("@99") == "@99"

    def test_set_and_get_display_name(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1", window_name="myproject")
        assert mgr.get_display_name("@1") == "myproject"

    def test_set_display_name_update(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1", window_name="old-name")
        mgr.window_display_names["@1"] = "new-name"
        assert mgr.get_display_name("@1") == "new-name"

    def test_bind_thread_sets_display_name(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1", window_name="proj")
        assert mgr.get_display_name("@1") == "proj"

    def test_bind_thread_without_name_no_display(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        # No display name set, fallback to window_id
        assert mgr.get_display_name("@1") == "@1"


class TestIsWindowId:
    def test_valid_ids(self, mgr: SessionManager) -> None:
        assert mgr._is_window_id("@0") is True
        assert mgr._is_window_id("@12") is True
        assert mgr._is_window_id("@999") is True

    def test_invalid_ids(self, mgr: SessionManager) -> None:
        assert mgr._is_window_id("myproject") is False
        assert mgr._is_window_id("@") is False
        assert mgr._is_window_id("") is False
        assert mgr._is_window_id("@abc") is False


class TestGroupedSessionMapHandling:
    @pytest.mark.asyncio
    async def test_load_session_map_accepts_grouped_session_prefix(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@5": {"session_id": "sid-1", "cwd": "/one"},
                    "ccbot-2:@7": {"session_id": "sid-2", "cwd": "/two"},
                    "other:@9": {"session_id": "sid-3", "cwd": "/three"},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot", "ccbot-2"}),
        )

        await mgr.load_session_map()

        assert mgr.get_window_state("@5").session_id == "sid-1"
        assert mgr.get_window_state("@7").session_id == "sid-2"
        assert "@9" not in mgr.window_states

    @pytest.mark.asyncio
    async def test_load_session_map_survives_tmux_query_failure(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@5": {"session_id": "sid-1", "cwd": "/one"},
                    "ccbot-2:@7": {"session_id": "sid-2", "cwd": "/two"},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(side_effect=RuntimeError("tmux down")),
        )

        await mgr.load_session_map()

        assert mgr.get_window_state("@5").session_id == "sid-1"
        assert "@7" not in mgr.window_states

    @pytest.mark.asyncio
    async def test_load_session_map_does_not_wipe_state_from_grouped_prefix(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot-2:@28": {
                        "session_id": "sid-28",
                        "cwd": "/proj",
                        "window_name": "proposal",
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot", "ccbot-2"}),
        )
        mgr.get_window_state("@28").session_id = "old-sid"

        await mgr.load_session_map()

        assert mgr.window_states["@28"].session_id == "sid-28"

    @pytest.mark.asyncio
    async def test_wait_for_session_map_entry_accepts_grouped_session_prefix(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps({"ccbot-2:@28": {"session_id": "sid-28"}}), encoding="utf-8"
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot", "ccbot-2"}),
        )
        load_session_map = AsyncMock()
        monkeypatch.setattr(mgr, "load_session_map", load_session_map)

        ok = await mgr.wait_for_session_map_entry("@28", timeout=0.1, interval=0.01)

        assert ok is True
        load_session_map.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sweep_removes_dead_windows_regardless_of_prefix(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        """Window IDs are unique per tmux server and ccbot is the map's only
        consumer, so a dead window's entry is garbage no matter which
        session-name prefix wrote it (grouped peer, foreign tmux session).
        Live windows survive even under foreign prefixes."""
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@5": {"session_id": "sid-5"},
                    "ccbot-2:@7": {"session_id": "sid-7"},
                    "other:@9": {"session_id": "sid-9"},
                    "other:@11": {"session_id": "sid-11"},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_all_window_ids",
            AsyncMock(return_value={"@5", "@11"}),
        )

        await mgr.sweep_stale_session_map_entries()

        remaining = json.loads(session_map_file.read_text(encoding="utf-8"))
        assert remaining == {
            "ccbot:@5": {"session_id": "sid-5"},
            "other:@11": {"session_id": "sid-11"},
        }

    @pytest.mark.asyncio
    async def test_repoint_updates_every_prefix_for_the_window(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@41": {"session_id": "sid-old", "cwd": "/proj"},
                    "ccbot-2:@41": {"session_id": "sid-old", "cwd": "/proj"},
                    "ccbot:@49": {"session_id": "sid-49", "cwd": "/other"},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)

        ok = await mgr.repoint_window_session("@41", "sid-new")

        assert ok is True
        result = json.loads(session_map_file.read_text(encoding="utf-8"))
        assert result["ccbot:@41"]["session_id"] == "sid-new"
        assert result["ccbot-2:@41"]["session_id"] == "sid-new"
        assert result["ccbot:@49"]["session_id"] == "sid-49"

    @pytest.mark.asyncio
    async def test_repoint_drops_old_session_start_size(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        """The old session's start size would seed the new session's read
        offset (0 = replay the whole transcript); without it the monitor
        starts at end-of-file."""
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@41": {
                        "session_id": "sid-old",
                        "cwd": "/proj",
                        "transcript_size_at_start": 0,
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)

        await mgr.repoint_window_session("@41", "sid-new")

        result = json.loads(session_map_file.read_text(encoding="utf-8"))
        assert result["ccbot:@41"] == {"session_id": "sid-new", "cwd": "/proj"}

    @pytest.mark.asyncio
    async def test_repoint_returns_false_when_window_absent(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        original = {"ccbot:@49": {"session_id": "sid-49"}}
        session_map_file.write_text(json.dumps(original), encoding="utf-8")
        monkeypatch.setattr(config, "session_map_file", session_map_file)

        ok = await mgr.repoint_window_session("@77", "sid-new")

        assert ok is False
        assert json.loads(session_map_file.read_text(encoding="utf-8")) == original

    @pytest.mark.asyncio
    async def test_sweep_is_a_noop_when_tmux_unreachable(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        """An unreachable tmux server means "unknown", not "no windows" —
        sweeping on it would delete every entry."""
        session_map_file = tmp_path / "session_map.json"
        original = {"ccbot:@5": {"session_id": "sid-5"}}
        session_map_file.write_text(json.dumps(original), encoding="utf-8")
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_all_window_ids",
            AsyncMock(return_value=None),
        )

        await mgr.sweep_stale_session_map_entries()

        assert json.loads(session_map_file.read_text(encoding="utf-8")) == original

    @pytest.mark.asyncio
    async def test_load_session_map_dedups_same_window_id_across_grouped_peers(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        """Grouped tmux sessions share windows, so the SessionStart hook can
        write the same window_id under multiple session-name prefixes. If
        load_session_map applied both, window_state.session_id would ping-pong
        every poll cycle and flood the log. Configured tmux_session_name wins;
        a re-load on unchanged input must be a no-op (no _save_state call).
        """
        session_map_file = tmp_path / "session_map.json"
        # Write in primary-then-peer order so a buggy last-write-wins picks
        # `sid-peer` — the fix must still pick `sid-primary`.
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@48": {
                        "session_id": "sid-primary",
                        "cwd": "/proj",
                        "window_name": "name",
                    },
                    "ccbot-2:@48": {
                        "session_id": "sid-peer",
                        "cwd": "/proj",
                        "window_name": "name",
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot", "ccbot-2"}),
        )

        await mgr.load_session_map()
        assert mgr.get_window_state("@48").session_id == "sid-primary"

        saves: list[None] = []
        monkeypatch.setattr(mgr, "_save_state", lambda: saves.append(None))

        await mgr.load_session_map()
        assert mgr.get_window_state("@48").session_id == "sid-primary"
        assert saves == [], (
            "second load_session_map on unchanged input must not mutate state"
        )

    @pytest.mark.asyncio
    async def test_load_session_map_falls_back_to_peer_when_primary_missing(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        """When only a grouped peer has an entry for a window_id (no entry
        under the configured tmux_session_name), the peer's entry must still
        be applied — grouped peers share windows."""
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot-2:@7": {"session_id": "sid-peer", "cwd": "/x"},
                    "ccbot-3:@7": {"session_id": "sid-peer-3", "cwd": "/x"},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot", "ccbot-2", "ccbot-3"}),
        )

        await mgr.load_session_map()
        # Deterministic peer pick (sorted name → ccbot-2 wins).
        assert mgr.get_window_state("@7").session_id == "sid-peer"


class TestLoadSessionMapStartSize:
    """load_session_map propagates the hook's transcript_size_at_start into
    WindowState.session_start_size for the monitor to seed offsets from
    (f17/RC38) — but only when applying a new session_id/cwd to an unpinned
    window."""

    @pytest.mark.asyncio
    async def test_new_session_propagates_start_size(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@5": {
                        "session_id": "sid-1",
                        "cwd": "/one",
                        "transcript_size_at_start": 4096,
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot"}),
        )

        await mgr.load_session_map()

        assert mgr.get_window_state("@5").session_start_size == 4096

    @pytest.mark.asyncio
    async def test_missing_start_size_defaults_to_unknown(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        """An older hook (or entry predating this field) has no
        transcript_size_at_start — session_start_size must fall back to -1
        (unknown), not some other value that would wrongly seed an offset."""
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps({"ccbot:@5": {"session_id": "sid-1", "cwd": "/one"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot"}),
        )

        await mgr.load_session_map()

        assert mgr.get_window_state("@5").session_start_size == -1

    @pytest.mark.asyncio
    async def test_non_int_start_size_defaults_to_unknown(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@5": {
                        "session_id": "sid-1",
                        "cwd": "/one",
                        "transcript_size_at_start": "not-a-number",
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot"}),
        )

        await mgr.load_session_map()

        assert mgr.get_window_state("@5").session_start_size == -1

    @pytest.mark.asyncio
    async def test_pinned_window_unaffected_by_start_size(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        """A pinned window (resume override) skips the hook-sync sid/cwd
        update entirely, so its session_start_size must stay untouched too —
        it does not belong to the session the hook is reporting."""
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@5": {
                        "session_id": "hook-sid",
                        "cwd": "/hook-cwd",
                        "transcript_size_at_start": 999,
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot"}),
        )
        state = mgr.get_window_state("@5")
        state.session_id = "resume-sid"
        state.cwd = "/resume-cwd"
        state.pinned_over = "hook-sid"
        state.session_start_size = 42

        await mgr.load_session_map()

        assert mgr.get_window_state("@5").session_start_size == 42


class TestResolveStaleIdsAmbiguousNames:
    """resolve_stale_ids must refuse to remap a stale entry when its display
    name matches more than one live window (RC3/RC4/f14/f36/f46) — silently
    picking one (e.g. "last one wins" on a dict) can cross-wire a topic's
    thread binding onto the WRONG tmux window/session after a restart."""

    @staticmethod
    def _mock_tmux(monkeypatch, windows) -> None:
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_windows",
            AsyncMock(return_value=windows),
        )
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_all_window_ids",
            AsyncMock(return_value={w.window_id for w in windows}),
        )
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={config.tmux_session_name}),
        )

    @pytest.mark.asyncio
    async def test_ambiguous_name_drops_stale_thread_binding(
        self, mgr: SessionManager, monkeypatch, tmp_path
    ) -> None:
        from ccbot.tmux_manager import TmuxWindow

        # Two LIVE windows happen to share the display name "proj" (e.g. a
        # racy create or an unchecked topic rename let this happen).
        live_windows = [
            TmuxWindow(window_id="@10", window_name="proj", cwd="/x"),
            TmuxWindow(window_id="@11", window_name="proj", cwd="/x"),
        ]
        self._mock_tmux(monkeypatch, live_windows)
        monkeypatch.setattr(config, "session_map_file", tmp_path / "session_map.json")

        # A stale thread binding whose window_id is gone, but whose recorded
        # display name ("proj") now matches BOTH live windows above.
        mgr.thread_bindings[100] = {5: "@99"}
        mgr.window_display_names["@99"] = "proj"

        await mgr.resolve_stale_ids()

        # Dropped outright — never remapped to either @10 or @11.
        assert mgr.get_window_for_thread(100, 5) is None
        assert 100 not in mgr.thread_bindings

    @pytest.mark.asyncio
    async def test_ambiguous_name_drops_stale_window_state(
        self, mgr: SessionManager, monkeypatch, tmp_path
    ) -> None:
        from ccbot.tmux_manager import TmuxWindow

        live_windows = [
            TmuxWindow(window_id="@10", window_name="proj", cwd="/x"),
            TmuxWindow(window_id="@11", window_name="proj", cwd="/x"),
        ]
        self._mock_tmux(monkeypatch, live_windows)
        monkeypatch.setattr(config, "session_map_file", tmp_path / "session_map.json")

        mgr.get_window_state("@99").window_name = "proj"
        mgr.window_display_names["@99"] = "proj"

        await mgr.resolve_stale_ids()

        assert "@99" not in mgr.window_states
        assert "@10" not in mgr.window_states
        assert "@11" not in mgr.window_states

    @pytest.mark.asyncio
    async def test_unambiguous_name_still_remaps(
        self, mgr: SessionManager, monkeypatch, tmp_path
    ) -> None:
        """Control: a single live window matching the display name still
        gets re-resolved normally — the fix must not break the common case."""
        from ccbot.tmux_manager import TmuxWindow

        live_windows = [TmuxWindow(window_id="@10", window_name="proj", cwd="/x")]
        self._mock_tmux(monkeypatch, live_windows)
        monkeypatch.setattr(config, "session_map_file", tmp_path / "session_map.json")

        mgr.thread_bindings[100] = {5: "@99"}
        mgr.window_display_names["@99"] = "proj"

        await mgr.resolve_stale_ids()

        assert mgr.get_window_for_thread(100, 5) == "@10"


class TestPinnedOverSessionRouting:
    """WindowState.pinned_over gives manually-established routing (resume
    override / hook-timeout recovery) provenance so load_session_map's
    hook-sync loop doesn't revert it. See RC2 / f30 / f35."""

    @pytest.mark.asyncio
    async def test_pinned_state_survives_repeated_hook_report_of_pinned_sid(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps({"ccbot:@5": {"session_id": "hook-sid", "cwd": "/hook-cwd"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot"}),
        )
        state = mgr.get_window_state("@5")
        state.session_id = "resume-sid"
        state.cwd = "/resume-cwd"
        state.pinned_over = "hook-sid"

        # Poll repeatedly — the hook keeps reporting the outranked sid.
        await mgr.load_session_map()
        await mgr.load_session_map()
        await mgr.load_session_map()

        assert mgr.get_window_state("@5").session_id == "resume-sid"
        assert mgr.get_window_state("@5").cwd == "/resume-cwd"
        assert mgr.get_window_state("@5").pinned_over == "hook-sid"

    @pytest.mark.asyncio
    async def test_hook_entry_with_different_sid_unpins_and_applies(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {"ccbot:@5": {"session_id": "post-clear-sid", "cwd": "/new-cwd"}}
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot"}),
        )
        state = mgr.get_window_state("@5")
        state.session_id = "resume-sid"
        state.cwd = "/resume-cwd"
        state.pinned_over = "hook-sid"

        await mgr.load_session_map()

        state = mgr.get_window_state("@5")
        assert state.session_id == "post-clear-sid"
        assert state.cwd == "/new-cwd"
        assert state.pinned_over is None

    @pytest.mark.asyncio
    async def test_pinned_window_state_with_no_session_map_entry_survives_sweep(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        # session_map.json has no entry at all for @5 (e.g. daemon hook
        # blindspot: SessionStart never fired for this window).
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot"}),
        )
        state = mgr.get_window_state("@5")
        state.session_id = "manual-sid"
        state.cwd = "/manual-cwd"
        state.pinned_over = ""  # hook-timeout recovery: pins over nothing

        await mgr.load_session_map()

        assert "@5" in mgr.window_states
        assert mgr.get_window_state("@5").session_id == "manual-sid"

    @pytest.mark.asyncio
    async def test_unpinned_window_state_absent_from_session_map_is_swept(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(json.dumps({}), encoding="utf-8")
        monkeypatch.setattr(config, "session_map_file", session_map_file)
        monkeypatch.setattr(config, "tmux_session_name", "ccbot")
        monkeypatch.setattr(
            "ccbot.session.tmux_manager.list_group_session_names",
            AsyncMock(return_value={"ccbot"}),
        )
        state = mgr.get_window_state("@5")
        state.session_id = "old-sid"
        state.cwd = "/old-cwd"

        await mgr.load_session_map()

        assert "@5" not in mgr.window_states


def _write_jsonl(path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


class TestSessionNameParsing:
    """Extracting custom-title / agent-name from session JSONL."""

    @pytest.fixture
    def session_layout(self, mgr: SessionManager, tmp_path, monkeypatch):
        """Build ~/.claude/projects/<encoded-cwd>/<sid>.jsonl under tmp_path."""
        cwd = "/proj/demo"
        sid = "11111111-2222-3333-4444-555555555555"
        project_dir = tmp_path / mgr._encode_cwd(cwd)
        project_dir.mkdir(parents=True)
        monkeypatch.setattr(config, "claude_projects_path", tmp_path)
        return cwd, sid, project_dir / f"{sid}.jsonl"

    @pytest.mark.asyncio
    async def test_custom_title_extracted(
        self, mgr: SessionManager, session_layout
    ) -> None:
        cwd, sid, path = session_layout
        _write_jsonl(
            path,
            [
                {"type": "summary", "summary": "an old summary"},
                {"type": "custom-title", "customTitle": "my-feature"},
            ],
        )
        result = await mgr._get_session_direct(sid, cwd)
        assert result is not None
        assert result.name == "my-feature"
        assert result.summary == "an old summary"

    @pytest.mark.asyncio
    async def test_agent_name_extracted(
        self, mgr: SessionManager, session_layout
    ) -> None:
        cwd, sid, path = session_layout
        _write_jsonl(
            path,
            [
                {"type": "agent-name", "agentName": "auto-generated-name"},
                {"type": "summary", "summary": "something"},
            ],
        )
        result = await mgr._get_session_direct(sid, cwd)
        assert result is not None
        assert result.name == "auto-generated-name"

    @pytest.mark.asyncio
    async def test_custom_title_wins_over_agent_name(
        self, mgr: SessionManager, session_layout
    ) -> None:
        cwd, sid, path = session_layout
        _write_jsonl(
            path,
            [
                {"type": "agent-name", "agentName": "auto-name"},
                {"type": "custom-title", "customTitle": "user-chose-this"},
            ],
        )
        result = await mgr._get_session_direct(sid, cwd)
        assert result is not None
        assert result.name == "user-chose-this"

    @pytest.mark.asyncio
    async def test_latest_custom_title_wins(
        self, mgr: SessionManager, session_layout
    ) -> None:
        cwd, sid, path = session_layout
        _write_jsonl(
            path,
            [
                {"type": "custom-title", "customTitle": "first"},
                {"type": "custom-title", "customTitle": "renamed-later"},
            ],
        )
        result = await mgr._get_session_direct(sid, cwd)
        assert result is not None
        assert result.name == "renamed-later"

    @pytest.mark.asyncio
    async def test_no_name_entries_leaves_name_empty(
        self, mgr: SessionManager, session_layout
    ) -> None:
        cwd, sid, path = session_layout
        _write_jsonl(
            path,
            [
                {"type": "summary", "summary": "just a summary"},
                {"type": "user", "message": {"content": "hi"}},
            ],
        )
        result = await mgr._get_session_direct(sid, cwd)
        assert result is not None
        assert result.name == ""


class TestWindowStateSerialization:
    """claude_launch_version round-trips through to_dict/from_dict."""

    def test_to_dict_omits_empty_launch_version(self) -> None:
        from ccbot.session import WindowState

        ws = WindowState(session_id="sid", cwd="/x")
        assert "claude_launch_version" not in ws.to_dict()

    def test_to_dict_includes_set_launch_version(self) -> None:
        from ccbot.session import WindowState

        ws = WindowState(session_id="sid", cwd="/x", claude_launch_version="2.1.118")
        assert ws.to_dict()["claude_launch_version"] == "2.1.118"

    def test_from_dict_reads_launch_version(self) -> None:
        from ccbot.session import WindowState

        ws = WindowState.from_dict(
            {"session_id": "sid", "cwd": "/x", "claude_launch_version": "2.1.117"}
        )
        assert ws.claude_launch_version == "2.1.117"

    def test_from_dict_missing_launch_version_defaults_empty(self) -> None:
        # Existing on-disk state.json files have no field — must load cleanly.
        from ccbot.session import WindowState

        ws = WindowState.from_dict({"session_id": "sid", "cwd": "/x"})
        assert ws.claude_launch_version == ""

    def test_to_dict_omits_pinned_over_when_none(self) -> None:
        from ccbot.session import WindowState

        ws = WindowState(session_id="sid", cwd="/x")
        assert "pinned_over" not in ws.to_dict()

    def test_pinned_over_round_trips_including_empty_string(self) -> None:
        # "" is a meaningful value (pins over nothing) distinct from None
        # (not pinned) — must not be conflated by to_dict/from_dict.
        from ccbot.session import WindowState

        for value in ("hook-sid", ""):
            ws = WindowState(session_id="sid", cwd="/x", pinned_over=value)
            d = ws.to_dict()
            assert d["pinned_over"] == value
            assert WindowState.from_dict(d).pinned_over == value

    def test_from_dict_missing_pinned_over_defaults_none(self) -> None:
        from ccbot.session import WindowState

        ws = WindowState.from_dict({"session_id": "sid", "cwd": "/x"})
        assert ws.pinned_over is None

    def test_to_dict_omits_session_start_size_when_unknown(self) -> None:
        from ccbot.session import WindowState

        ws = WindowState(session_id="sid", cwd="/x")
        assert "session_start_size" not in ws.to_dict()

    def test_session_start_size_round_trips(self) -> None:
        from ccbot.session import WindowState

        for value in (0, 4096):
            ws = WindowState(session_id="sid", cwd="/x", session_start_size=value)
            d = ws.to_dict()
            assert d["session_start_size"] == value
            assert WindowState.from_dict(d).session_start_size == value

    def test_from_dict_missing_session_start_size_defaults_unknown(self) -> None:
        # Existing on-disk state.json files predate this field.
        from ccbot.session import WindowState

        ws = WindowState.from_dict({"session_id": "sid", "cwd": "/x"})
        assert ws.session_start_size == -1


class TestSetClaudeLaunchVersion:
    def test_sets_field_on_existing_state(self, mgr: SessionManager) -> None:
        mgr.get_window_state("@1")  # ensure exists
        mgr.set_claude_launch_version("@1", "2.1.118")
        assert mgr.get_window_state("@1").claude_launch_version == "2.1.118"

    def test_creates_state_if_missing(self, mgr: SessionManager) -> None:
        # Backfill path: window_id may not have an entry yet.
        mgr.set_claude_launch_version("@99", "2.1.118")
        assert mgr.get_window_state("@99").claude_launch_version == "2.1.118"

    def test_no_op_when_unchanged_skips_save(self, monkeypatch) -> None:
        # Verify we don't churn state.json on every turn-end when there's
        # nothing to write.
        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        saved: list[bool] = []

        def fake_save(self) -> None:
            saved.append(True)

        monkeypatch.setattr(SessionManager, "_save_state", fake_save)
        m = SessionManager()
        m.set_claude_launch_version("@1", "2.1.118")
        m.set_claude_launch_version("@1", "2.1.118")  # no-op
        assert len(saved) == 1
