"""Tests for the per-cause notice shown when a send to a session fails.

Each SendResult cause gets its own message naming what ccbot observed and
the action that fits it. Text that was typed but never showed up in the
input box (NOT_VISIBLE) also gets a ♻️ Restart button and a screenshot of the
pane carrying the screenshot control keys, so a dialog ccbot doesn't
recognize can still be seen and answered from the phone.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot.bot import _report_send_failure, callback_handler, text_handler
from ccbot.handlers.callback_data import CB_KEYS_PREFIX, CB_SEND_FAIL_RESTART
from ccbot.tmux_manager import SendResult


def _callback_data(markup) -> list[str]:
    return [btn.callback_data for row in markup.inline_keyboard for btn in row]


@pytest.fixture
def sender():
    with (
        patch("ccbot.bot.safe_send", new_callable=AsyncMock) as mock_send,
        patch("ccbot.bot.tmux_manager") as mock_tmux,
        patch("ccbot.bot.text_to_image", new_callable=AsyncMock, return_value=b"png"),
    ):
        mock_tmux.capture_pane = AsyncMock(return_value="❯ Keep xhigh")
        yield mock_send, mock_tmux


class TestReportSendFailure:
    @pytest.mark.parametrize(
        ("result", "phrase"),
        [
            (SendResult.WINDOW_GONE, "window no longer exists"),
            (SendResult.REJECTED, "tmux rejected the keystrokes"),
        ],
    )
    async def test_text_only_causes_send_one_message(self, sender, result, phrase):
        mock_send, mock_tmux = sender
        bot = MagicMock(send_document=AsyncMock())

        await _report_send_failure(bot, 100, 42, "@5", result)

        mock_send.assert_awaited_once()
        text = mock_send.await_args.args[2]
        assert text.startswith("❌ Not delivered") and phrase in text
        assert mock_send.await_args.kwargs["message_thread_id"] == 42
        mock_tmux.capture_pane.assert_not_called()
        bot.send_document.assert_not_called()

    async def test_not_visible_sends_restart_button_then_screenshot(self, sender):
        mock_send, mock_tmux = sender
        bot = MagicMock(send_document=AsyncMock())

        await _report_send_failure(bot, 100, 42, "@5", SendResult.NOT_VISIBLE)

        text = mock_send.await_args.args[2]
        assert text.startswith("⚠️ Not submitted")
        restart = mock_send.await_args.kwargs["reply_markup"]
        assert _callback_data(restart) == [f"{CB_SEND_FAIL_RESTART}@5"]

        mock_tmux.capture_pane.assert_awaited_once_with("@5", with_ansi=True)
        doc_kwargs = bot.send_document.await_args.kwargs
        assert doc_kwargs["chat_id"] == 100
        assert doc_kwargs["message_thread_id"] == 42
        keys = _callback_data(doc_kwargs["reply_markup"])
        assert f"{CB_KEYS_PREFIX}ent:@5" in keys
        assert f"{CB_KEYS_PREFIX}dn:@5" in keys

    async def test_not_visible_without_capture_points_to_screenshot(self, sender):
        mock_send, mock_tmux = sender
        mock_tmux.capture_pane = AsyncMock(return_value=None)
        bot = MagicMock(send_document=AsyncMock())

        await _report_send_failure(bot, 100, 42, "@5", SendResult.NOT_VISIBLE)

        assert mock_send.await_count == 2
        assert "/screenshot" in mock_send.await_args.args[2]
        bot.send_document.assert_not_called()


class TestTextHandlerReportsCause:
    async def test_not_visible_send_is_reported_with_its_cause(self):
        update = MagicMock()
        update.effective_user.id = 1
        update.message.text = "Hi"
        update.message.chat_id = 100
        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch("ccbot.bot.is_interactive_ui", return_value=False),
            patch("ccbot.bot.enqueue_status_update", new_callable=AsyncMock),
            patch("ccbot.bot.clear_status_msg_info"),
            patch("ccbot.bot.get_interactive_window", return_value=None),
            patch(
                "ccbot.bot._report_send_failure", new_callable=AsyncMock
            ) as mock_report,
        ):
            mock_sm.get_window_for_thread.return_value = "@5"
            mock_sm.resolve_window_for_thread.return_value = "@5"
            mock_sm.send_to_window = AsyncMock(return_value=SendResult.NOT_VISIBLE)
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@5")
            )
            mock_tmux.capture_pane = AsyncMock(return_value="❯ Keep xhigh")

            await text_handler(update, MagicMock())

        mock_report.assert_awaited_once()
        assert mock_report.await_args.args[1:] == (
            100,
            42,
            "@5",
            SendResult.NOT_VISIBLE,
        )


def _restart_update(window_id: str = "@5") -> MagicMock:
    update = MagicMock()
    update.effective_user.id = 1
    update.effective_chat.type = "supergroup"
    update.callback_query.data = f"{CB_SEND_FAIL_RESTART}{window_id}"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_reply_markup = AsyncMock()
    return update


class TestRestartButton:
    async def test_bound_window_restarts_in_place_and_drops_button(self):
        update = _restart_update()
        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch("ccbot.bot.tmux_manager") as mock_tmux,
            patch(
                "ccbot.update_watcher.restart_topic_in_place", new_callable=AsyncMock
            ) as mock_restart,
        ):
            mock_sm.resolve_window_for_thread.return_value = "@5"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@5")
            )
            context = MagicMock()

            await callback_handler(update, context)

        mock_restart.assert_awaited_once_with(context.bot, 1, 42, "@5")
        update.callback_query.edit_message_reply_markup.assert_awaited_once_with(
            reply_markup=None
        )

    async def test_topic_rebound_elsewhere_refuses_restart(self):
        update = _restart_update("@5")
        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager") as mock_sm,
            patch(
                "ccbot.update_watcher.restart_topic_in_place", new_callable=AsyncMock
            ) as mock_restart,
        ):
            mock_sm.resolve_window_for_thread.return_value = "@9"

            await callback_handler(update, MagicMock())

        mock_restart.assert_not_called()
        assert update.callback_query.answer.await_args.kwargs["show_alert"] is True
