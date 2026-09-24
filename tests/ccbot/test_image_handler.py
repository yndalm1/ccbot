"""Tests for image_handler — photos and image files forwarded to Claude Code.

The session is asked, in the user's own words, to open the one saved file:
a bare "(image attached: <path>)" reads as a description and the session may
stop to ask before reading outside its working directory. Image files keep
their format; formats Claude can't read are refused before any download.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot.bot import image_handler


def _make_update(
    *, photo: bool = False, mime_type: str | None = None, file_size: int = 1000
) -> MagicMock:
    update = MagicMock()
    update.effective_user.id = 1
    update.message.caption = None
    update.message.chat.type = "supergroup"
    update.message.chat.id = 100
    image = MagicMock()
    image.file_unique_id = "uniq"
    image.file_size = file_size
    image.get_file = AsyncMock(return_value=MagicMock(download_to_drive=AsyncMock()))
    if photo:
        update.message.photo = [MagicMock(), image]
        update.message.document = None
    else:
        update.message.photo = []
        image.mime_type = mime_type
        update.message.document = image
    return update


@contextmanager
def _bound_topic(tmp_path):
    """A topic bound to a live window with no pending interactive UI."""
    with (
        patch("ccbot.bot.is_user_allowed", return_value=True),
        patch("ccbot.bot._get_thread_id", return_value=42),
        patch("ccbot.bot._IMAGES_DIR", tmp_path),
        patch("ccbot.bot.session_manager") as mock_sm,
        patch("ccbot.bot.tmux_manager") as mock_tmux,
        patch("ccbot.bot.is_interactive_ui", return_value=False),
        patch("ccbot.bot.clear_status_msg_info"),
        patch("ccbot.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
    ):
        mock_sm.get_window_for_thread.return_value = "@5"
        mock_sm.send_to_window = AsyncMock(return_value=(True, "ok"))
        mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
        mock_tmux.capture_pane = AsyncMock(return_value="$ ")
        yield mock_sm, mock_reply


def _image(update: MagicMock) -> MagicMock:
    return update.message.photo[-1] if update.message.photo else update.message.document


class TestImageHandler:
    @pytest.mark.asyncio
    async def test_photo_without_caption_asks_claude_to_open_it(self, tmp_path):
        update = _make_update(photo=True)
        with _bound_topic(tmp_path) as (mock_sm, _reply):
            await image_handler(update, MagicMock())

        saved = _image(update).get_file.return_value.download_to_drive
        path = saved.await_args.args[0]
        assert path.parent == tmp_path and path.suffix == ".jpg"
        mock_sm.send_to_window.assert_awaited_once_with(
            "@5", f"Open and look at the image I sent: {path}"
        )

    @pytest.mark.asyncio
    async def test_caption_leads_and_open_request_follows(self, tmp_path):
        update = _make_update(photo=True)
        update.message.caption = "why is this red?"
        with _bound_topic(tmp_path) as (mock_sm, _reply):
            await image_handler(update, MagicMock())

        path = _image(update).get_file.return_value.download_to_drive.await_args.args[0]
        mock_sm.send_to_window.assert_awaited_once_with(
            "@5", f"why is this red?\n\n(Open the image I sent: {path})"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mime_type", "suffix"),
        [
            ("image/png", ".png"),
            ("image/jpeg", ".jpg"),
            ("image/gif", ".gif"),
            ("image/webp", ".webp"),
        ],
    )
    async def test_image_file_keeps_its_format(self, tmp_path, mime_type, suffix):
        update = _make_update(mime_type=mime_type)
        with _bound_topic(tmp_path) as (mock_sm, _reply):
            await image_handler(update, MagicMock())

        path = _image(update).get_file.return_value.download_to_drive.await_args.args[0]
        assert path.suffix == suffix
        mock_sm.send_to_window.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unreadable_format_is_refused_before_download(self, tmp_path):
        update = _make_update(mime_type="image/heic")
        with _bound_topic(tmp_path) as (mock_sm, mock_reply):
            await image_handler(update, MagicMock())

        _image(update).get_file.assert_not_called()
        mock_sm.send_to_window.assert_not_called()
        warning = mock_reply.await_args.args[1]
        assert "image/heic" in warning and "Send it as a photo" in warning

    @pytest.mark.asyncio
    async def test_file_over_bot_download_limit_is_refused(self, tmp_path):
        update = _make_update(mime_type="image/png", file_size=21 * 1024 * 1024)
        with _bound_topic(tmp_path) as (mock_sm, mock_reply):
            await image_handler(update, MagicMock())

        _image(update).get_file.assert_not_called()
        mock_sm.send_to_window.assert_not_called()
        assert "20 MB" in mock_reply.await_args.args[1]
