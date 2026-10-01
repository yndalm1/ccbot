"""Tests for forward_command_handler — command forwarding to Claude Code."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot.tmux_manager import SendResult


def _make_update(text: str, user_id: int = 1, thread_id: int = 42) -> MagicMock:
    """Build a minimal mock Update with message text in a forum topic."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.message = MagicMock()
    update.message.text = text
    update.message.message_thread_id = thread_id
    update.message.chat = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.type = "supergroup"
    update.effective_chat.id = 100
    return update


def _make_context() -> MagicMock:
    """Build a minimal mock context."""
    context = MagicMock()
    context.bot = AsyncMock()
    context.user_data = {}
    return context


class TestForwardCommand:
    @pytest.mark.asyncio
    async def test_model_sends_command_to_tmux(self):
        """/model → send_to_window called with "/model"."""
        update = _make_update("/model")
        context = _make_context()

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock),
        ):
            mock_sm.resolve_window_for_thread.return_value = "@5"
            mock_sm.get_display_name.return_value = "project"
            mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_sm.send_to_window = AsyncMock(return_value=SendResult.SENT)

            from ccbot.bot import forward_command_handler

            await forward_command_handler(update, context)

            mock_sm.send_to_window.assert_called_once_with("@5", "/model")

    @pytest.mark.asyncio
    async def test_cost_sends_command_to_tmux(self):
        """/cost → send_to_window called with "/cost"."""
        update = _make_update("/cost")
        context = _make_context()

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock),
        ):
            mock_sm.resolve_window_for_thread.return_value = "@5"
            mock_sm.get_display_name.return_value = "project"
            mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_sm.send_to_window = AsyncMock(return_value=SendResult.SENT)

            from ccbot.bot import forward_command_handler

            await forward_command_handler(update, context)

            mock_sm.send_to_window.assert_called_once_with("@5", "/cost")

    @pytest.mark.asyncio
    async def test_clear_clears_session(self):
        """/clear → send_to_window + clear_window_session."""
        update = _make_update("/clear")
        context = _make_context()

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock),
        ):
            mock_sm.resolve_window_for_thread.return_value = "@5"
            mock_sm.get_display_name.return_value = "project"
            mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_sm.send_to_window = AsyncMock(return_value=SendResult.SENT)

            from ccbot.bot import forward_command_handler

            await forward_command_handler(update, context)

            mock_sm.send_to_window.assert_called_once_with("@5", "/clear")
            mock_sm.clear_window_session.assert_called_once_with("@5")

    @pytest.mark.asyncio
    async def test_at_sign_in_arguments_is_not_truncated(self):
        """'/compact keep @decorators intact' must forward the arguments in
        full — only a mention on the command TOKEN gets stripped."""
        update = _make_update("/compact keep @decorators intact")
        context = _make_context()

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock),
        ):
            mock_sm.resolve_window_for_thread.return_value = "@5"
            mock_sm.get_display_name.return_value = "project"
            mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_sm.send_to_window = AsyncMock(return_value=SendResult.SENT)

            from ccbot.bot import forward_command_handler

            await forward_command_handler(update, context)

            mock_sm.send_to_window.assert_called_once_with(
                "@5", "/compact keep @decorators intact"
            )

    @pytest.mark.asyncio
    async def test_bot_mention_on_command_token_is_still_stripped(self):
        """'/clear@my_bot' still has the mention stripped before forwarding
        and before the clear-session check fires."""
        update = _make_update("/clear@my_bot")
        context = _make_context()

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock),
        ):
            mock_sm.resolve_window_for_thread.return_value = "@5"
            mock_sm.get_display_name.return_value = "project"
            mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_sm.send_to_window = AsyncMock(return_value=SendResult.SENT)

            from ccbot.bot import forward_command_handler

            await forward_command_handler(update, context)

            mock_sm.send_to_window.assert_called_once_with("@5", "/clear")
            mock_sm.clear_window_session.assert_called_once_with("@5")


class TestStripBotMention:
    """Direct coverage of the pure mention-stripping helper."""

    def test_plain_command_unchanged(self):
        from ccbot.bot import _strip_bot_mention

        assert _strip_bot_mention("/model") == "/model"

    def test_command_with_args_unchanged(self):
        from ccbot.bot import _strip_bot_mention

        assert _strip_bot_mention("/compact foo bar") == "/compact foo bar"

    def test_mention_on_token_stripped(self):
        from ccbot.bot import _strip_bot_mention

        assert _strip_bot_mention("/clear@my_bot") == "/clear"

    def test_mention_on_token_with_args_leaves_args_untouched(self):
        from ccbot.bot import _strip_bot_mention

        assert (
            _strip_bot_mention("/compact@my_bot keep @decorators intact")
            == "/compact keep @decorators intact"
        )

    def test_at_sign_in_arguments_survives(self):
        from ccbot.bot import _strip_bot_mention

        assert (
            _strip_bot_mention("/compact keep @decorators intact")
            == "/compact keep @decorators intact"
        )

    def test_email_argument_survives(self):
        from ccbot.bot import _strip_bot_mention

        assert (
            _strip_bot_mention("/note contact me@example.com please")
            == "/note contact me@example.com please"
        )

    def test_multiple_at_signs_in_arguments_all_survive(self):
        from ccbot.bot import _strip_bot_mention

        assert (
            _strip_bot_mention("/note cc a@x.com and b@y.com")
            == "/note cc a@x.com and b@y.com"
        )

    def test_empty_string(self):
        from ccbot.bot import _strip_bot_mention

        assert _strip_bot_mention("") == ""

    def test_whitespace_only(self):
        from ccbot.bot import _strip_bot_mention

        assert _strip_bot_mention("   ") == "   "
