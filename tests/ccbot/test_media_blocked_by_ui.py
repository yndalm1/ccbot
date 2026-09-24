"""Tests for _media_blocked_by_ui — f63/RC30: photo captions and voice
transcripts must never be typed blindly into a pending interactive dialog
(permission prompt, AskUserQuestion, ...). Unlike text_handler, media can
never BE the menu answer, so blocking (rather than forwarding) is correct.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot.bot import _media_blocked_by_ui, image_handler, voice_handler


class TestMediaBlockedByUiHelper:
    @pytest.mark.asyncio
    async def test_ui_present_returns_true_and_shows_dialog(self):
        bot = AsyncMock()
        message = MagicMock()

        with (
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.is_interactive_ui", return_value=True) as mock_is_ui,
            patch(
                "ccbot.bot.handle_interactive_ui", new_callable=AsyncMock
            ) as mock_handle_ui,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            mock_tmux.capture_pane = AsyncMock(return_value="Do you want to proceed?")

            result = await _media_blocked_by_ui(bot, message, 7, "@5", 42)

        assert result is True
        mock_is_ui.assert_called_once_with("Do you want to proceed?")
        mock_handle_ui.assert_awaited_once_with(bot, 7, "@5", 42)
        mock_reply.assert_awaited_once()
        warning = mock_reply.await_args.args[1]
        assert "interactive prompt" in warning

    @pytest.mark.asyncio
    async def test_no_ui_returns_false(self):
        bot = AsyncMock()
        message = MagicMock()

        with (
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.is_interactive_ui", return_value=False),
            patch(
                "ccbot.bot.handle_interactive_ui", new_callable=AsyncMock
            ) as mock_handle_ui,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            mock_tmux.capture_pane = AsyncMock(return_value="$ some prompt")

            result = await _media_blocked_by_ui(bot, message, 7, "@5", 42)

        assert result is False
        mock_handle_ui.assert_not_called()
        mock_reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_pane_returns_false_without_calling_is_interactive_ui(self):
        bot = AsyncMock()
        message = MagicMock()

        with (
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.is_interactive_ui") as mock_is_ui,
            patch(
                "ccbot.bot.handle_interactive_ui", new_callable=AsyncMock
            ) as mock_handle_ui,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            mock_tmux.capture_pane = AsyncMock(return_value=None)

            result = await _media_blocked_by_ui(bot, message, 7, "@5", 42)

        assert result is False
        mock_is_ui.assert_not_called()
        mock_handle_ui.assert_not_called()
        mock_reply.assert_not_called()


def _make_update_with_photo(user_id: int = 1, thread_id: int = 42) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.message = MagicMock()
    photo = MagicMock()
    photo.file_size = 1000
    photo.get_file = AsyncMock()
    update.message.photo = [photo]
    update.message.caption = None
    update.message.chat = MagicMock()
    update.message.chat.type = "supergroup"
    update.message.chat.id = 100
    return update


def _make_update_with_voice(user_id: int = 1, thread_id: int = 42) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.message = MagicMock()
    update.message.voice = MagicMock()
    update.message.chat = MagicMock()
    update.message.chat.type = "supergroup"
    update.message.chat.id = 100
    return update


def _make_context() -> MagicMock:
    context = MagicMock()
    context.bot = AsyncMock()
    return context


class TestPhotoHandlerBlockedByUi:
    @pytest.mark.asyncio
    async def test_pending_ui_blocks_send_and_skips_download(self):
        update = _make_update_with_photo()
        context = _make_context()

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.is_interactive_ui", return_value=True),
            patch(
                "ccbot.bot.handle_interactive_ui", new_callable=AsyncMock
            ) as mock_handle_ui,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            mock_sm.get_window_for_thread.return_value = "@5"
            mock_sm.send_to_window = AsyncMock(return_value=(True, "ok"))
            mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_tmux.capture_pane = AsyncMock(return_value="1. Yes  2. No")

            await image_handler(update, context)

            mock_handle_ui.assert_awaited_once()
            mock_sm.send_to_window.assert_not_called()
            update.message.photo[-1].get_file.assert_not_called()
            assert mock_reply.await_count == 1
            assert "interactive prompt" in mock_reply.await_args.args[1]


class TestVoiceHandlerBlockedByUi:
    @pytest.mark.asyncio
    async def test_pending_ui_blocks_send_and_skips_transcription(self):
        update = _make_update_with_voice()
        context = _make_context()

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.config") as mock_config,
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.is_interactive_ui", return_value=True),
            patch(
                "ccbot.bot.handle_interactive_ui", new_callable=AsyncMock
            ) as mock_handle_ui,
            patch(
                "ccbot.bot.transcribe_voice", new_callable=AsyncMock
            ) as mock_transcribe,
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            mock_config.openai_api_key = "sk-test"
            mock_sm.get_window_for_thread.return_value = "@5"
            mock_sm.send_to_window = AsyncMock(return_value=(True, "ok"))
            mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_tmux.capture_pane = AsyncMock(return_value="1. Yes  2. No")

            await voice_handler(update, context)

            mock_handle_ui.assert_awaited_once()
            mock_transcribe.assert_not_called()
            mock_sm.send_to_window.assert_not_called()
            assert mock_reply.await_count == 1
            assert "interactive prompt" in mock_reply.await_args.args[1]
