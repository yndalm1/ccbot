"""Telegram bot handlers — the main UI layer of CCBot.

Registers all command/callback/message handlers and manages the bot lifecycle.
Each Telegram topic maps 1:1 to a tmux window (Claude session).

Core responsibilities:
  - Command handlers: /start, /history, /screenshot, /esc, /kill, /unbind,
    plus forwarding unknown /commands to Claude Code via tmux.
  - Callback query handler: directory browser, history pagination,
    interactive UI navigation, screenshot refresh.
  - Topic-based routing: each named topic binds to one tmux window.
    Unbound topics trigger the directory browser to create a new session.
  - Image handling: photos and image files (PNG/JPEG/GIF/WebP) sent by
    the user are downloaded, and Claude Code is asked to open the saved
    file (image_handler).
  - Voice handling: voice messages are transcribed via OpenAI API and
    forwarded as text (voice_handler).
  - Automatic cleanup: closing a topic kills the associated window
    (topic_closed_handler). Unsupported content (stickers, etc.)
    is rejected with a warning (unsupported_content_handler).
  - Bot lifecycle management: post_init, post_shutdown, create_bot.

Handler modules (in handlers/):
  - callback_data: Callback data constants
  - message_queue: Per-user message queue management
  - message_sender: Safe message sending helpers
  - history: Message history pagination
  - directory_browser: Directory browser UI
  - interactive_ui: Interactive UI handling
  - status_polling: Terminal status polling
  - response_builder: Response message building

Key functions: create_bot(), handle_new_message().
"""

import asyncio
import io
import logging
import re
import time
from pathlib import Path

from aiolimiter import AsyncLimiter
from telegram import (
    Bot,
    BotCommand,
    Document,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaDocument,
    Message,
    PhotoSize,
    Update,
)
from telegram.error import RetryAfter
from telegram.ext import (
    AIORateLimiter,
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import config
from .handlers.callback_data import (
    CB_ASK_DOWN,
    CB_ASK_ENTER,
    CB_ASK_ESC,
    CB_ASK_LEFT,
    CB_ASK_REFRESH,
    CB_ASK_RIGHT,
    CB_ASK_SPACE,
    CB_ASK_TAB,
    CB_ASK_UP,
    CB_DIR_CANCEL,
    CB_DIR_CONFIRM,
    CB_DIR_PAGE,
    CB_DIR_SELECT,
    CB_DIR_UP,
    CB_HISTORY_NEXT,
    CB_HISTORY_PREV,
    CB_REPOINT,
    CB_SESSION_CANCEL,
    CB_SESSION_NEW,
    CB_SESSION_SELECT,
    CB_KEYS_PREFIX,
    CB_SCREENSHOT_REFRESH,
    CB_WIN_BIND,
    CB_WIN_CANCEL,
    CB_WIN_NEW,
)
from .handlers.directory_browser import (
    BROWSE_DIRS_KEY,
    BROWSE_PAGE_KEY,
    BROWSE_PATH_KEY,
    PENDING_TEXT_KEY,
    SELECTED_PATH_KEY,
    SESSIONS_KEY,
    STATE_BROWSING_DIRECTORY,
    STATE_KEY,
    STATE_SELECTING_SESSION,
    STATE_SELECTING_WINDOW,
    UNBOUND_WINDOWS_KEY,
    build_directory_browser,
    build_session_picker,
    build_window_picker,
    clear_browse_state,
    get_browse_state,
    set_browse_state,
)
from .handlers.cleanup import clear_topic_state
from .handlers.history import send_history
from .handlers.interactive_ui import (
    INTERACTIVE_TOOL_NAMES,
    clear_interactive_msg,
    get_interactive_msg_id,
    get_interactive_window,
    handle_interactive_ui,
    send_ui_key,
)
from .handlers.message_queue import (
    clear_status_msg_info,
    drain_queues,
    enqueue_content_message,
    enqueue_status_update,
    get_message_queue,
    shutdown_workers,
)
from .handlers.message_sender import (
    edit_with_fallback,
    safe_edit,
    safe_reply,
    safe_send,
    send_with_fallback,
)
from .handlers.response_builder import build_response_parts
from .handlers.maintenance import maintenance_loop
from .handlers.status_polling import status_poll_loop
from .screenshot import text_to_image
from .session import session_manager
from .session_monitor import NewMessage, SessionMonitor
from .terminal_parser import extract_bash_output, is_interactive_ui
from .tmux_manager import tmux_manager
from .transcribe import close_client as close_transcribe_client
from .transcribe import transcribe_voice
from .utils import ccbot_dir, supervise_loop

logger = logging.getLogger(__name__)


def _resolve_browser_start_path() -> str:
    """Return the directory the new-session browser should open at.

    Uses ``config.default_dir`` if set and pointing to an existing
    directory; otherwise falls back to the bot process's cwd.
    """
    pinned = config.default_dir
    if pinned:
        candidate = Path(pinned).expanduser()
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.is_dir():
            return str(resolved)
        logger.warning(
            "CCBOT_DEFAULT_DIR=%r is not a directory; falling back to cwd",
            pinned,
        )
    return str(Path.cwd())


# Session monitor instance
session_monitor: SessionMonitor | None = None

# Status polling task
_status_poll_task: asyncio.Task | None = None
_maintenance_task: asyncio.Task | None = None

# Claude Code commands shown in bot menu (forwarded via tmux)
CC_COMMANDS: dict[str, str] = {
    "clear": "↗ Clear conversation history",
    "compact": "↗ Compact conversation context",
    "cost": "↗ Show token/cost usage",
    "help": "↗ Show Claude Code help",
    "memory": "↗ Edit CLAUDE.md",
    "model": "↗ Switch AI model",
    "context": "↗ Show context window usage",
}


def is_user_allowed(user_id: int | None) -> bool:
    return user_id is not None and config.is_user_allowed(user_id)


def _get_thread_id(update: Update) -> int | None:
    """Extract thread_id from an update, returning None if not in a named topic."""
    msg = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if msg is None:
        return None
    tid = getattr(msg, "message_thread_id", None)
    if tid is None or tid == 1:
        return None
    return tid


# --- Command handlers ---


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        if update.message:
            await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    thread_id = _get_thread_id(update)
    if thread_id is not None:
        clear_browse_state(context.user_data, thread_id)

    if update.message:
        await safe_reply(
            update.message,
            "🤖 *Claude Code Monitor*\n\n"
            "Each topic is a session. Create a new topic to start.",
        )


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show message history for the active session or bound thread."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    wid = session_manager.resolve_window_for_thread(user.id, thread_id)
    if not wid:
        await safe_reply(update.message, "❌ No session bound to this topic.")
        return

    await send_history(update.message, wid)


async def screenshot_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Capture the current tmux pane and send it as an image."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    wid = session_manager.resolve_window_for_thread(user.id, thread_id)
    if not wid:
        await safe_reply(update.message, "❌ No session bound to this topic.")
        return

    w = await tmux_manager.find_window_by_id(wid)
    if not w:
        display = session_manager.get_display_name(wid)
        await safe_reply(update.message, f"❌ Window '{display}' no longer exists.")
        return

    text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not text:
        await safe_reply(update.message, "❌ Failed to capture pane content.")
        return

    png_bytes = await text_to_image(text, with_ansi=True)
    keyboard = _build_screenshot_keyboard(wid)
    await update.message.reply_document(
        document=io.BytesIO(png_bytes),
        filename="screenshot.png",
        reply_markup=keyboard,
    )


async def unbind_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unbind this topic from its Claude session without killing the window."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        await safe_reply(update.message, "❌ This command only works in a topic.")
        return

    wid = session_manager.get_window_for_thread(user.id, thread_id)
    if not wid:
        await safe_reply(update.message, "❌ No session bound to this topic.")
        return

    display = session_manager.get_display_name(wid)
    session_manager.unbind_thread(user.id, thread_id)
    await clear_topic_state(user.id, thread_id, context.bot, context.user_data)

    await safe_reply(
        update.message,
        f"✅ Topic unbound from window '{display}'.\n"
        "The Claude session is still running in tmux.\n"
        "Send a message to bind to a new session.",
    )


async def kill_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Kill this topic's Claude session and delete the Telegram topic.

    Unlike /unbind (which detaches the topic but leaves tmux running),
    this actually kills the tmux window, then deletes the forum topic
    itself. Mirrors the teardown sequence in topic_closed_handler, plus
    the topic deletion that a native Telegram "close+delete" would do.
    """
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        await safe_reply(update.message, "❌ This command only works in a topic.")
        return

    wid = session_manager.get_window_for_thread(user.id, thread_id)
    if not wid:
        await safe_reply(update.message, "❌ No session bound to this topic.")
        return

    display = session_manager.get_display_name(wid)

    w = await tmux_manager.find_window_by_id(wid)
    if w:
        await tmux_manager.kill_window(w.window_id)
        logger.info(
            "Killed window %s via /kill (user=%d, thread=%d)",
            display,
            user.id,
            thread_id,
        )
    else:
        logger.info(
            "/kill: window %s already gone (user=%d, thread=%d)",
            display,
            user.id,
            thread_id,
        )

    session_manager.unbind_thread(user.id, thread_id)
    await clear_topic_state(user.id, thread_id, context.bot, context.user_data)

    resolved_chat = session_manager.resolve_chat_id(user.id, thread_id)
    try:
        await context.bot.delete_forum_topic(
            chat_id=resolved_chat, message_thread_id=thread_id
        )
    except Exception as e:
        # Bots need "Manage Topics" rights to delete a forum topic; without
        # them the teardown above already happened, so just tell the user
        # to finish the job manually instead of leaving them guessing.
        logger.warning("Failed to delete forum topic (thread=%d): %s", thread_id, e)
        await safe_reply(
            update.message,
            f"✅ Killed session '{display}'.\n"
            "⚠️ Could not delete this topic automatically "
            "(bot may lack the 'Manage Topics' right) — "
            "please delete it manually.",
        )


async def esc_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send Escape key to interrupt Claude."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    wid = session_manager.resolve_window_for_thread(user.id, thread_id)
    if not wid:
        await safe_reply(update.message, "❌ No session bound to this topic.")
        return

    w = await tmux_manager.find_window_by_id(wid)
    if not w:
        display = session_manager.get_display_name(wid)
        await safe_reply(update.message, f"❌ Window '{display}' no longer exists.")
        return

    # Send Escape control character (no enter)
    await tmux_manager.send_keys(w.window_id, "\x1b", enter=False)
    await safe_reply(update.message, "⎋ Sent Escape")


async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Restart this topic's Claude session in place (respawn-pane, --resume).

    Proceeds immediately — the user picks a clean point. The same @window_id is
    reused (no rebind / no session_map churn) and the session resumes on the
    current default model, so this also clears a stale model pin.
    """
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        await safe_reply(update.message, "❌ /restart only works inside a topic.")
        return
    wid = session_manager.resolve_window_for_thread(user.id, thread_id)
    if not wid:
        await safe_reply(update.message, "❌ No session bound to this topic.")
        return

    w = await tmux_manager.find_window_by_id(wid)
    if not w:
        display = session_manager.get_display_name(wid)
        await safe_reply(update.message, f"❌ Window '{display}' no longer exists.")
        return

    await safe_reply(update.message, "♻️ Restarting this session…")
    from .update_watcher import restart_topic_in_place

    await restart_topic_in_place(context.bot, user.id, thread_id, w.window_id)


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch Claude Code usage stats from TUI and send to Telegram."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    wid = session_manager.resolve_window_for_thread(user.id, thread_id)
    if not wid:
        await safe_reply(update.message, "No session bound to this topic.")
        return

    w = await tmux_manager.find_window_by_id(wid)
    if not w:
        await safe_reply(update.message, f"Window '{wid}' no longer exists.")
        return

    # Send /usage command to Claude Code TUI
    await tmux_manager.send_keys(w.window_id, "/usage")
    # Wait for the modal to render
    await asyncio.sleep(2.0)
    # Capture the pane content
    pane_text = await tmux_manager.capture_pane(w.window_id)
    # Dismiss the modal
    await tmux_manager.send_keys(w.window_id, "Escape", enter=False, literal=False)

    if not pane_text:
        await safe_reply(update.message, "Failed to capture usage info.")
        return

    # Try to parse structured usage info
    from .terminal_parser import parse_usage_output

    usage = parse_usage_output(pane_text)
    if usage and usage.parsed_lines:
        text = "\n".join(usage.parsed_lines)
        await safe_reply(update.message, f"```\n{text}\n```")
    else:
        # Fallback: send raw pane capture trimmed
        trimmed = pane_text.strip()
        if len(trimmed) > 3000:
            trimmed = trimmed[:3000] + "\n... (truncated)"
        await safe_reply(update.message, f"```\n{trimmed}\n```")


# --- Screenshot keyboard with quick control keys ---

# key_id → (tmux_key, enter, literal)
_KEYS_SEND_MAP: dict[str, tuple[str, bool, bool]] = {
    "up": ("Up", False, False),
    "dn": ("Down", False, False),
    "lt": ("Left", False, False),
    "rt": ("Right", False, False),
    "esc": ("Escape", False, False),
    "ent": ("Enter", False, False),
    "spc": ("Space", False, False),
    "tab": ("Tab", False, False),
    "cc": ("C-c", False, False),
}

# key_id → display label (shown in callback answer toast)
_KEY_LABELS: dict[str, str] = {
    "up": "↑",
    "dn": "↓",
    "lt": "←",
    "rt": "→",
    "esc": "⎋ Esc",
    "ent": "⏎ Enter",
    "spc": "␣ Space",
    "tab": "⇥ Tab",
    "cc": "^C",
}


def _build_screenshot_keyboard(window_id: str) -> InlineKeyboardMarkup:
    """Build inline keyboard for screenshot: control keys + refresh."""

    def btn(label: str, key_id: str) -> InlineKeyboardButton:
        return InlineKeyboardButton(
            label,
            callback_data=f"{CB_KEYS_PREFIX}{key_id}:{window_id}"[:64],
        )

    return InlineKeyboardMarkup(
        [
            [btn("␣ Space", "spc"), btn("↑", "up"), btn("⇥ Tab", "tab")],
            [btn("←", "lt"), btn("↓", "dn"), btn("→", "rt")],
            [btn("⎋ Esc", "esc"), btn("^C", "cc"), btn("⏎ Enter", "ent")],
            [
                InlineKeyboardButton(
                    "🔄 Refresh",
                    callback_data=f"{CB_SCREENSHOT_REFRESH}{window_id}"[:64],
                )
            ],
        ]
    )


async def topic_closed_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle topic closure — kill the associated tmux window and clean up state."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        return

    wid = session_manager.get_window_for_thread(user.id, thread_id)
    if wid:
        display = session_manager.get_display_name(wid)
        w = await tmux_manager.find_window_by_id(wid)
        if w:
            await tmux_manager.kill_window(w.window_id)
            logger.info(
                "Topic closed: killed window %s (user=%d, thread=%d)",
                display,
                user.id,
                thread_id,
            )
        else:
            logger.info(
                "Topic closed: window %s already gone (user=%d, thread=%d)",
                display,
                user.id,
                thread_id,
            )
        session_manager.unbind_thread(user.id, thread_id)
        # Clean up all memory state for this topic
        await clear_topic_state(user.id, thread_id, context.bot, context.user_data)
    else:
        logger.debug(
            "Topic closed: no binding (user=%d, thread=%d)", user.id, thread_id
        )


async def topic_edited_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle topic rename — sync new name to tmux window and internal state."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return

    msg = update.message
    if not msg or not msg.forum_topic_edited:
        return

    new_name = msg.forum_topic_edited.name
    if new_name is None:
        # Icon-only change, no rename needed
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        return

    wid = session_manager.get_window_for_thread(user.id, thread_id)
    if not wid:
        logger.debug(
            "Topic edited: no binding (user=%d, thread=%d)", user.id, thread_id
        )
        return

    # Reject renames that would corrupt display-name-based lookups (RC3/f36):
    # the reserved main-window sentinel is skipped by list_windows, so this
    # window would vanish from resolve_stale_ids/find_window_by_id forever;
    # a name collision with another live window makes resolve_stale_ids'
    # name-keyed remap ambiguous after the next tmux server restart.
    if new_name == config.tmux_main_window_name:
        logger.warning(
            "Topic edited: rejecting rename to reserved name '%s' "
            "(window=%s, user=%d, thread=%d)",
            new_name,
            wid,
            user.id,
            thread_id,
        )
        await safe_reply(
            msg,
            f"❌ '{new_name}' is a reserved name and can't be used here.",
        )
        return

    collision = await tmux_manager.find_window_by_name(new_name)
    if collision and collision.window_id != wid:
        logger.warning(
            "Topic edited: rejecting rename to '%s' (already used by window %s) "
            "(window=%s, user=%d, thread=%d)",
            new_name,
            collision.window_id,
            wid,
            user.id,
            thread_id,
        )
        await safe_reply(
            msg,
            f"❌ Another window is already named '{new_name}'. Pick a different name.",
        )
        return

    old_name = session_manager.get_display_name(wid)
    await tmux_manager.rename_window(wid, new_name)
    session_manager.update_display_name(wid, new_name)
    logger.info(
        "Topic renamed: '%s' -> '%s' (window=%s, user=%d, thread=%d)",
        old_name,
        new_name,
        wid,
        user.id,
        thread_id,
    )


def _strip_bot_mention(cmd_text: str) -> str:
    """Strip a trailing "@botname" mention from the command token only.

    Telegram appends "@botname" to the command TOKEN itself in group chats
    (e.g. "/clear@my_bot"), never to its arguments. Splitting the whole
    string on "@" would also truncate arguments that legitimately contain
    "@" (emails, "@decorators", etc.), so only the first whitespace-
    delimited token is desentineled and the remainder is passed through
    untouched.
    """
    parts = cmd_text.split(maxsplit=1)
    if not parts:
        return cmd_text
    token = re.sub(r"@\w+$", "", parts[0])
    if len(parts) == 1:
        return token
    return f"{token} {parts[1]}"


async def forward_command_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Forward any non-bot command as a slash command to the active Claude Code session."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)

    # Capture group chat_id for supergroup forum topic routing.
    # Required: Telegram Bot API needs group chat_id (not user_id) to send
    # messages with message_thread_id. Do NOT remove — see session.py docs.
    chat = update.effective_chat
    if chat and chat.type in ("group", "supergroup"):
        session_manager.set_group_chat_id(user.id, thread_id, chat.id)

    cmd_text = update.message.text or ""
    # The full text is already a slash command like "/clear" or "/compact foo".
    cc_slash = _strip_bot_mention(cmd_text)
    wid = session_manager.resolve_window_for_thread(user.id, thread_id)
    if not wid:
        await safe_reply(update.message, "❌ No session bound to this topic.")
        return

    w = await tmux_manager.find_window_by_id(wid)
    if not w:
        display = session_manager.get_display_name(wid)
        await safe_reply(update.message, f"❌ Window '{display}' no longer exists.")
        return

    display = session_manager.get_display_name(wid)
    logger.info(
        "Forwarding command %s to window %s (user=%d)", cc_slash, display, user.id
    )
    success, message = await session_manager.send_to_window(wid, cc_slash)
    if success:
        await safe_reply(update.message, f"⚡ [{display}] Sent: {cc_slash}")
        # If /clear command was sent, clear the session association
        # so we can detect the new session after first message
        if cc_slash.strip().lower() == "/clear":
            logger.info("Clearing session for window %s after /clear", display)
            session_manager.clear_window_session(wid)

        # Interactive commands (e.g. /model) render a terminal-based UI
        # with no JSONL tool_use entry.  The status poller already detects
        # interactive UIs every 1s (status_polling.py), so no
        # proactive detection needed here — the poller handles it.
    else:
        await safe_reply(update.message, f"❌ {message}")


async def unsupported_content_handler(
    update: Update,
    _context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Reply to non-text messages (stickers, video, etc.)."""
    if not update.message:
        return
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    logger.debug("Unsupported content from user %d", user.id)
    await safe_reply(
        update.message,
        "⚠ Only text, image, and voice messages are supported. Stickers, video, and other media cannot be forwarded to Claude Code.",
    )


# --- Image directory for incoming images ---
_IMAGES_DIR = ccbot_dir() / "images"
_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Image formats Claude can read, by MIME type -> saved file extension. A
# photo is always JPEG (Telegram re-encodes it); an image sent as a file
# keeps its own format, so anything else (e.g. an iPhone's HEIC) is refused.
_READABLE_IMAGE_TYPES = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

# Telegram's Bot API only lets bots download files up to 20 MB.
_BOT_DOWNLOAD_LIMIT = 20 * 1024 * 1024


async def _media_blocked_by_ui(
    bot: Bot,
    message: Message,
    user_id: int,
    window_id: str,
    thread_id: int | None,
) -> bool:
    """Refuse to forward media into a window with a pending interactive UI.

    text_handler deliberately still sends text when a UI is on screen — a
    short reply may legitimately BE the menu answer (e.g. "1", "yes"). Photo
    captions and voice transcripts can never be a menu answer — they're a
    file path / free-form transcript — so unlike text, blocking here is
    unambiguously correct rather than a design trade-off.
    """
    pane_text = await tmux_manager.capture_pane(window_id)
    if not pane_text or not is_interactive_ui(pane_text):
        return False
    await handle_interactive_ui(bot, user_id, window_id, thread_id)
    await safe_reply(
        message,
        "⚠️ Claude is waiting on an interactive prompt in this topic — "
        "answer it first (dialog above), then resend.",
    )
    return True


async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle images sent by the user — a photo, or an image sent as a file —
    by downloading it and asking Claude Code to open it."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        if update.message:
            await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    if not update.message:
        return
    if update.message.photo:
        # Highest resolution of the sizes Telegram generated.
        image: PhotoSize | Document = update.message.photo[-1]
        extension = "jpg"
    elif update.message.document:
        image = update.message.document
        mime_type = image.mime_type or ""
        if mime_type not in _READABLE_IMAGE_TYPES:
            await safe_reply(
                update.message,
                f"⚠ Claude Code can't read {mime_type or 'this'} images "
                "(PNG, JPEG, GIF and WebP only). Send it as a photo instead — "
                "Telegram converts it to JPEG.",
            )
            return
        extension = _READABLE_IMAGE_TYPES[mime_type]
    else:
        return
    if image.file_size and image.file_size > _BOT_DOWNLOAD_LIMIT:
        await safe_reply(
            update.message,
            "⚠ This image is over Telegram's 20 MB download limit for bots. "
            "Send it as a photo instead.",
        )
        return

    chat = update.message.chat
    thread_id = _get_thread_id(update)
    if chat.type in ("group", "supergroup") and thread_id is not None:
        session_manager.set_group_chat_id(user.id, thread_id, chat.id)

    # Must be in a named topic
    if thread_id is None:
        await safe_reply(
            update.message,
            "❌ Please use a named topic. Create a new topic to start a session.",
        )
        return

    wid = session_manager.get_window_for_thread(user.id, thread_id)
    if wid is None:
        await safe_reply(
            update.message,
            "❌ No session bound to this topic. Send a text message first to create one.",
        )
        return

    w = await tmux_manager.find_window_by_id(wid)
    if not w:
        display = session_manager.get_display_name(wid)
        session_manager.unbind_thread(user.id, thread_id)
        await safe_reply(
            update.message,
            f"❌ Window '{display}' no longer exists. Binding removed.\n"
            "Send a message to start a new session.",
        )
        return

    if await _media_blocked_by_ui(context.bot, update.message, user.id, wid, thread_id):
        return

    # Save to ~/.ccbot/images/<timestamp>_<file_unique_id>.<extension>
    tg_file = await image.get_file()
    filename = f"{int(time.time())}_{image.file_unique_id}.{extension}"
    file_path = _IMAGES_DIR / filename
    await tg_file.download_to_drive(file_path)

    # Phrase it as the user's own instruction to open this one file: the
    # session reads a bare "(image attached: <path>)" as a description, and
    # may stop to ask before reading a file outside its working directory.
    caption = update.message.caption or ""
    if caption:
        text_to_send = f"{caption}\n\n(Open the image I sent: {file_path})"
    else:
        text_to_send = f"Open and look at the image I sent: {file_path}"

    clear_status_msg_info(user.id, thread_id)

    success, message = await session_manager.send_to_window(wid, text_to_send)
    if not success:
        await safe_reply(update.message, f"❌ {message}")
        return

    # Confirm to user
    await safe_reply(update.message, "📷 Image sent to Claude Code.")


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages: transcribe via OpenAI and forward text to Claude Code."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        if update.message:
            await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    if not update.message or not update.message.voice:
        return

    if not config.openai_api_key:
        await safe_reply(
            update.message,
            "⚠ Voice transcription requires an OpenAI API key.\n"
            "Set `OPENAI_API_KEY` in your `.env` file and restart the bot.",
        )
        return

    chat = update.message.chat
    thread_id = _get_thread_id(update)
    if chat.type in ("group", "supergroup") and thread_id is not None:
        session_manager.set_group_chat_id(user.id, thread_id, chat.id)

    if thread_id is None:
        await safe_reply(
            update.message,
            "❌ Please use a named topic. Create a new topic to start a session.",
        )
        return

    wid = session_manager.get_window_for_thread(user.id, thread_id)
    if wid is None:
        await safe_reply(
            update.message,
            "❌ No session bound to this topic. Send a text message first to create one.",
        )
        return

    w = await tmux_manager.find_window_by_id(wid)
    if not w:
        display = session_manager.get_display_name(wid)
        session_manager.unbind_thread(user.id, thread_id)
        await safe_reply(
            update.message,
            f"❌ Window '{display}' no longer exists. Binding removed.\n"
            "Send a message to start a new session.",
        )
        return

    if await _media_blocked_by_ui(context.bot, update.message, user.id, wid, thread_id):
        return

    # Download voice as in-memory bytes
    voice_file = await update.message.voice.get_file()
    ogg_data = bytes(await voice_file.download_as_bytearray())

    # Transcribe
    try:
        text = await transcribe_voice(ogg_data)
    except ValueError as e:
        await safe_reply(update.message, f"⚠ {e}")
        return
    except Exception as e:
        logger.error("Voice transcription failed: %s", e)
        await safe_reply(update.message, f"⚠ Transcription failed: {e}")
        return

    clear_status_msg_info(user.id, thread_id)

    success, message = await session_manager.send_to_window(wid, text)
    if not success:
        await safe_reply(update.message, f"❌ {message}")
        return

    await safe_reply(update.message, f'🎤 "{text}"')


# Active bash capture tasks: (user_id, thread_id) → asyncio.Task
_bash_capture_tasks: dict[tuple[int, int], asyncio.Task[None]] = {}


def _cancel_bash_capture(user_id: int, thread_id: int) -> None:
    """Cancel any running bash capture for this topic."""
    key = (user_id, thread_id)
    task = _bash_capture_tasks.pop(key, None)
    if task and not task.done():
        task.cancel()


async def _capture_bash_output(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
    command: str,
) -> None:
    """Background task: capture ``!`` bash command output from tmux pane.

    Sends the first captured output as a new message, then edits it
    in-place as more output appears.  Stops after 30 s or when cancelled
    (e.g. user sends a new message, which pushes content down).
    """
    try:
        # Wait for the command to start producing output
        await asyncio.sleep(2.0)

        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        msg_id: int | None = None
        last_output: str = ""

        for _ in range(30):
            raw = await tmux_manager.capture_pane(window_id)
            if raw is None:
                return

            output = extract_bash_output(raw, command)
            if not output:
                await asyncio.sleep(1.0)
                continue

            # Skip edit if nothing changed
            if output == last_output:
                await asyncio.sleep(1.0)
                continue

            last_output = output

            # Truncate to fit Telegram's 4096-char limit. This is a live
            # terminal window (only the tail is ever meaningful while the
            # command keeps running), not stored-content truncation.
            if len(output) > 3800:
                output = "… " + output[-3800:]

            if msg_id is None:
                # First capture — send a new message
                sent = await send_with_fallback(
                    bot,
                    chat_id,
                    output,
                    message_thread_id=thread_id,
                )
                if sent:
                    msg_id = sent.message_id
            else:
                # Subsequent captures — edit in place, MarkdownV2 falling
                # back to plain text.
                try:
                    await edit_with_fallback(bot, chat_id, msg_id, output)
                except RetryAfter as e:
                    retry_secs = (
                        e.retry_after
                        if isinstance(e.retry_after, (int, float))
                        else e.retry_after.total_seconds()
                    )
                    await asyncio.sleep(retry_secs)
                    try:
                        await edit_with_fallback(bot, chat_id, msg_id, output)
                    except RetryAfter as e2:
                        logger.warning(
                            "Bash-output edit rate-limited twice for chat "
                            "%s; skipping this tick: %s",
                            chat_id,
                            e2,
                        )
                    except Exception as e2:
                        logger.warning(
                            "Bash-output edit failed after retry for chat %s: %s",
                            chat_id,
                            e2,
                        )

            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        return
    finally:
        _bash_capture_tasks.pop((user_id, thread_id), None)


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        if update.message:
            await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    if not update.message or not update.message.text:
        return

    thread_id = _get_thread_id(update)

    # Capture group chat_id for supergroup forum topic routing.
    # Required: Telegram Bot API needs group chat_id (not user_id) to send
    # messages with message_thread_id. Do NOT remove — see session.py docs.
    chat = update.effective_chat
    if chat and chat.type in ("group", "supergroup"):
        session_manager.set_group_chat_id(user.id, thread_id, chat.id)

    text = update.message.text

    # Directory-browser/window-picker/session-picker state is per-thread
    # (RC13 / f31 / f33): a flow in progress in another topic is stored
    # under its own thread_id and can never be seen or clobbered here.
    browse_state = (
        get_browse_state(context.user_data, thread_id) if thread_id is not None else {}
    )
    current_state = browse_state.get(STATE_KEY)

    if current_state == STATE_SELECTING_WINDOW:
        await safe_reply(
            update.message,
            "Please use the window picker above, or tap Cancel.",
        )
        return

    if current_state == STATE_BROWSING_DIRECTORY:
        await safe_reply(
            update.message,
            "Please use the directory browser above, or tap Cancel.",
        )
        return

    if current_state == STATE_SELECTING_SESSION:
        await safe_reply(
            update.message,
            "Please use the session picker above, or tap Cancel.",
        )
        return

    # Must be in a named topic
    if thread_id is None:
        await safe_reply(
            update.message,
            "❌ Please use a named topic. Create a new topic to start a session.",
        )
        return

    wid = session_manager.get_window_for_thread(user.id, thread_id)
    if wid is None:
        # Unbound topic — check for unbound windows first
        all_windows = await tmux_manager.list_windows()
        bound_ids = {wid for _, _, wid in session_manager.iter_thread_bindings()}
        unbound = [
            (w.window_id, w.window_name, w.cwd)
            for w in all_windows
            if w.window_id not in bound_ids
        ]
        logger.debug(
            "Window picker check: all=%s, bound=%s, unbound=%s",
            [w.window_name for w in all_windows],
            bound_ids,
            [name for _, name, _ in unbound],
        )

        if unbound:
            # Show window picker
            logger.info(
                "Unbound topic: showing window picker (%d unbound windows, user=%d, thread=%d)",
                len(unbound),
                user.id,
                thread_id,
            )
            msg_text, keyboard, win_ids = build_window_picker(unbound)
            set_browse_state(
                context.user_data,
                thread_id,
                {
                    STATE_KEY: STATE_SELECTING_WINDOW,
                    UNBOUND_WINDOWS_KEY: win_ids,
                    PENDING_TEXT_KEY: text,
                },
            )
            await safe_reply(update.message, msg_text, reply_markup=keyboard)
            return

        # No unbound windows — show directory browser to create a new session
        logger.info(
            "Unbound topic: showing directory browser (user=%d, thread=%d)",
            user.id,
            thread_id,
        )
        start_path = _resolve_browser_start_path()
        msg_text, keyboard, subdirs = build_directory_browser(start_path)
        set_browse_state(
            context.user_data,
            thread_id,
            {
                STATE_KEY: STATE_BROWSING_DIRECTORY,
                BROWSE_PATH_KEY: start_path,
                BROWSE_PAGE_KEY: 0,
                BROWSE_DIRS_KEY: subdirs,
                PENDING_TEXT_KEY: text,
            },
        )
        await safe_reply(update.message, msg_text, reply_markup=keyboard)
        return

    # Bound topic — forward to bound window
    w = await tmux_manager.find_window_by_id(wid)
    if not w:
        display = session_manager.get_display_name(wid)
        logger.info(
            "Stale binding: window %s gone, unbinding (user=%d, thread=%d)",
            display,
            user.id,
            thread_id,
        )
        session_manager.unbind_thread(user.id, thread_id)
        await safe_reply(
            update.message,
            f"❌ Window '{display}' no longer exists. Binding removed.\n"
            "Send a message to start a new session.",
        )
        return

    await enqueue_status_update(context.bot, user.id, wid, None, thread_id=thread_id)

    # Cancel any running bash capture — new message pushes pane content down
    _cancel_bash_capture(user.id, thread_id)

    # Check for pending interactive UI before sending text.
    # This catches UIs (permission prompts, etc.) that status polling might have missed.
    pane_text = await tmux_manager.capture_pane(w.window_id)
    if pane_text and is_interactive_ui(pane_text):
        # UI detected — show it to user, then send text (acts as Enter)
        logger.info(
            "Detected pending interactive UI before sending text (user=%d, thread=%s)",
            user.id,
            thread_id,
        )
        await handle_interactive_ui(context.bot, user.id, wid, thread_id)
        # Small delay to let UI render in Telegram before text arrives
        await asyncio.sleep(0.3)

    success, message = await session_manager.send_to_window(wid, text)
    if not success:
        await safe_reply(update.message, f"❌ {message}")
        return

    # Start background capture for ! bash command output
    if text.startswith("!") and len(text) > 1:
        bash_cmd = text[1:]  # strip leading "!"
        task = asyncio.create_task(
            _capture_bash_output(context.bot, user.id, thread_id, wid, bash_cmd)
        )
        _bash_capture_tasks[(user.id, thread_id)] = task

    # If in interactive mode, refresh the UI after sending text
    interactive_window = get_interactive_window(user.id, thread_id)
    if interactive_window and interactive_window == wid:
        await asyncio.sleep(0.2)
        await handle_interactive_ui(context.bot, user.id, wid, thread_id)


# --- Window creation helper ---


def _bind_outcome_message(message: str, hook_ok: bool, resumed: bool) -> str:
    """Build the user-facing text for a just-created, topic-bound window.

    ``message`` is the tmux window-creation confirmation (e.g. "Created
    window 'foo' at /path"). ``hook_ok`` reports whether Claude Code's
    SessionStart hook registered the window in session_map within the
    timeout. ``resumed`` distinguishes a `--resume` window from a fresh one.

    Resume windows get their WindowState.session_id manually pinned by the
    caller even when the hook times out, so routing works either way and the
    normal "Resumed" text stays truthful regardless of ``hook_ok``. Fresh
    windows have no such fallback: if the hook never registers,
    WindowState.session_id stays empty forever and every Claude reply is
    silently dropped at routing, so that path gets an honest warning instead
    of a false "success" message (f65 / RC29).
    """
    if not hook_ok and not resumed:
        return (
            f"⚠️ {message}\n\n"
            "Window created, but Claude session tracking did not register "
            "(SessionStart hook missing or failed). You can send messages, "
            "but replies may not reach this topic. Fix: run `ccbot hook "
            "--install`, then use /restart here."
        )
    status = "Resumed" if resumed else "Created"
    return f"✅ {message}\n\n{status}. Send messages here."


async def _create_and_bind_window(
    query: object,
    context: ContextTypes.DEFAULT_TYPE,
    user: object,
    selected_path: str,
    pending_thread_id: int | None,
    resume_session_id: str | None = None,
) -> None:
    """Create a tmux window, bind it to a topic, and forward pending text.

    Shared by CB_DIR_CONFIRM (no sessions), CB_SESSION_NEW, and CB_SESSION_SELECT.
    """
    from telegram import CallbackQuery, User

    assert isinstance(query, CallbackQuery)
    assert isinstance(user, User)

    # Capture this topic's pending first message (if any) before dropping
    # its browse/picker state — this is the only remaining use of it.
    pending_text = (
        get_browse_state(context.user_data, pending_thread_id).get(PENDING_TEXT_KEY)
        if pending_thread_id is not None
        else None
    )
    if pending_thread_id is not None:
        clear_browse_state(context.user_data, pending_thread_id)

    success, message, created_wname, created_wid = await tmux_manager.create_window(
        selected_path, resume_session_id=resume_session_id
    )
    if success:
        logger.info(
            "Window created: %s (id=%s) at %s (user=%d, thread=%s, resume=%s)",
            created_wname,
            created_wid,
            selected_path,
            user.id,
            pending_thread_id,
            resume_session_id,
        )
        # Wait for Claude Code's SessionStart hook to register in session_map.
        # Resume sessions take longer to start (loading session state), so use
        # a longer timeout to avoid silently dropping messages.
        hook_timeout = 15.0 if resume_session_id else 5.0
        hook_ok = await session_manager.wait_for_session_map_entry(
            created_wid, timeout=hook_timeout
        )
        if not hook_ok and not resume_session_id:
            # No resume-override fallback exists for fresh windows: if the
            # hook never registers, WindowState.session_id stays empty
            # forever and every Claude reply is silently dropped at
            # routing (outbound sends still work). Surface it loudly.
            logger.warning(
                "SessionStart hook did not register fresh window %s "
                "(cwd=%s) within %.1fs; Claude replies will not route to "
                "its topic until 'ccbot hook --install' + /restart fix it",
                created_wid,
                selected_path,
                hook_timeout,
            )

        # --resume creates a new session_id in the hook, but messages continue
        # writing to the resumed session's JSONL file. Override window_state to
        # track the original session_id so the monitor can route messages back.
        if resume_session_id:
            ws = session_manager.get_window_state(created_wid)
            if not hook_ok:
                # Hook timed out — manually populate window_state so the
                # monitor can still route messages back to this topic. Pin
                # over "" so ANY future hook entry with a non-empty, different
                # session_id unpins (the hook-sync loop must not revert this).
                logger.warning(
                    "Hook timed out for resume window %s, "
                    "manually setting session_id=%s cwd=%s",
                    created_wid,
                    resume_session_id,
                    selected_path,
                )
                ws.session_id = resume_session_id
                ws.cwd = str(selected_path)
                ws.window_name = created_wname
                ws.pinned_over = ""
                session_manager._save_state()
            elif ws.session_id != resume_session_id:
                logger.info(
                    "Resume override: window %s session_id %s -> %s",
                    created_wid,
                    ws.session_id,
                    resume_session_id,
                )
                # Pin over the hook-reported sid being outranked so the
                # hook-sync loop doesn't revert this override on its next poll.
                ws.pinned_over = ws.session_id
                ws.session_id = resume_session_id
                session_manager._save_state()

        # Pin the version that's running in this window so future turn-ends
        # can detect a binary upgrade just for THIS window. Per-window means
        # an upgrade in one session doesn't silence the signal for others.
        from .update_watcher import current_claude_version

        launch_version = await current_claude_version()
        if launch_version:
            session_manager.set_claude_launch_version(created_wid, launch_version)

        if pending_thread_id is not None:
            # Thread bind flow: bind thread to newly created window
            bound = session_manager.bind_thread(
                user.id, pending_thread_id, created_wid, window_name=created_wname
            )
            if not bound:
                # Freshly created window already bound to another topic
                # (should not happen in practice, but the owner refuses to
                # guess — see RC3/RC4).
                await safe_edit(
                    query,
                    "❌ That window is already bound to another topic.",
                )
                await query.answer("Already bound")
                return

            # Rename the topic to match the window name
            resolved_chat = session_manager.resolve_chat_id(user.id, pending_thread_id)
            try:
                await context.bot.edit_forum_topic(
                    chat_id=resolved_chat,
                    message_thread_id=pending_thread_id,
                    name=created_wname,
                )
            except Exception as e:
                logger.debug(f"Failed to rename topic: {e}")

            await safe_edit(
                query,
                _bind_outcome_message(
                    message, hook_ok=hook_ok, resumed=bool(resume_session_id)
                ),
            )

            # Send pending text if any
            if pending_text:
                logger.debug(
                    "Forwarding pending text to window %s (len=%d)",
                    created_wname,
                    len(pending_text),
                )
                send_ok, send_msg = await session_manager.send_to_window(
                    created_wid,
                    pending_text,
                )
                if not send_ok:
                    logger.warning("Failed to forward pending text: %s", send_msg)
                    await safe_send(
                        context.bot,
                        resolved_chat,
                        f"❌ Failed to send pending message: {send_msg}",
                        message_thread_id=pending_thread_id,
                    )
        else:
            # Should not happen in topic-only mode, but handle gracefully
            await safe_edit(query, f"✅ {message}")
    else:
        await safe_edit(query, f"❌ {message}")
    await query.answer("Created" if success else "Failed")


# --- Callback query handler ---


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        await query.answer("Not authorized")
        return

    data = query.data

    # Capture group chat_id for supergroup forum topic routing.
    # Required: Telegram Bot API needs group chat_id (not user_id) to send
    # messages with message_thread_id. Do NOT remove — see session.py docs.
    cb_thread_id = _get_thread_id(update)
    chat = update.effective_chat
    if chat and chat.type in ("group", "supergroup"):
        session_manager.set_group_chat_id(user.id, cb_thread_id, chat.id)

    # History: older/newer pagination
    # Format: hp:<page>:<window_id>:<start>:<end> or hn:<page>:<window_id>:<start>:<end>
    if data.startswith(CB_HISTORY_PREV) or data.startswith(CB_HISTORY_NEXT):
        prefix_len = len(CB_HISTORY_PREV)  # same length for both
        rest = data[prefix_len:]
        try:
            parts = rest.split(":")
            if len(parts) < 4:
                # Old format without byte range: page:window_id
                offset_str, window_id = rest.split(":", 1)
                start_byte, end_byte = 0, 0
            else:
                # New format: page:window_id:start:end (window_id may contain colons)
                offset_str = parts[0]
                start_byte = int(parts[-2])
                end_byte = int(parts[-1])
                window_id = ":".join(parts[1:-2])
            offset = int(offset_str)
        except (ValueError, IndexError):
            await query.answer("Invalid data")
            return

        w = await tmux_manager.find_window_by_id(window_id)
        if w:
            await send_history(
                query,
                window_id,
                offset=offset,
                edit=True,
                start_byte=start_byte,
                end_byte=end_byte,
                # Don't pass user_id for pagination - offset update only on initial view
                # This prevents offset from going backwards if new messages arrive while paging
            )
        else:
            await safe_edit(query, "Window no longer exists.")
        await query.answer("Page updated")

    # Directory browser handlers
    elif data.startswith(CB_DIR_SELECT):
        # State is resolved from this callback's own topic — a stale button
        # from a superseded browse in another topic can't read/clobber it.
        dir_thread_id = _get_thread_id(update)
        # callback_data contains index, not dir name (to avoid 64-byte limit)
        try:
            idx = int(data[len(CB_DIR_SELECT) :])
        except ValueError:
            await query.answer("Invalid data")
            return

        # Look up dir name from this topic's cached subdirs
        browse_state = get_browse_state(context.user_data, dir_thread_id)
        cached_dirs: list[str] = browse_state.get(BROWSE_DIRS_KEY, [])
        if idx < 0 or idx >= len(cached_dirs):
            await query.answer(
                "Directory list changed, please refresh", show_alert=True
            )
            return
        subdir_name = cached_dirs[idx]

        default_path = str(Path.cwd())
        current_path = browse_state.get(BROWSE_PATH_KEY, default_path)
        new_path = (Path(current_path) / subdir_name).resolve()

        if not new_path.exists() or not new_path.is_dir():
            await query.answer("Directory not found", show_alert=True)
            return

        new_path_str = str(new_path)
        msg_text, keyboard, subdirs = build_directory_browser(new_path_str)
        set_browse_state(
            context.user_data,
            dir_thread_id,
            {
                BROWSE_PATH_KEY: new_path_str,
                BROWSE_PAGE_KEY: 0,
                BROWSE_DIRS_KEY: subdirs,
            },
        )
        await safe_edit(query, msg_text, reply_markup=keyboard)
        await query.answer()

    elif data == CB_DIR_UP:
        dir_thread_id = _get_thread_id(update)
        browse_state = get_browse_state(context.user_data, dir_thread_id)
        default_path = str(Path.cwd())
        current_path = browse_state.get(BROWSE_PATH_KEY, default_path)
        current = Path(current_path).resolve()
        parent = current.parent
        # No restriction - allow navigating anywhere

        parent_path = str(parent)
        msg_text, keyboard, subdirs = build_directory_browser(parent_path)
        set_browse_state(
            context.user_data,
            dir_thread_id,
            {
                BROWSE_PATH_KEY: parent_path,
                BROWSE_PAGE_KEY: 0,
                BROWSE_DIRS_KEY: subdirs,
            },
        )
        await safe_edit(query, msg_text, reply_markup=keyboard)
        await query.answer()

    elif data.startswith(CB_DIR_PAGE):
        dir_thread_id = _get_thread_id(update)
        try:
            pg = int(data[len(CB_DIR_PAGE) :])
        except ValueError:
            await query.answer("Invalid data")
            return
        browse_state = get_browse_state(context.user_data, dir_thread_id)
        default_path = str(Path.cwd())
        current_path = browse_state.get(BROWSE_PATH_KEY, default_path)

        msg_text, keyboard, subdirs = build_directory_browser(current_path, pg)
        set_browse_state(
            context.user_data,
            dir_thread_id,
            {BROWSE_PAGE_KEY: pg, BROWSE_DIRS_KEY: subdirs},
        )
        await safe_edit(query, msg_text, reply_markup=keyboard)
        await query.answer()

    elif data == CB_DIR_CONFIRM:
        # This topic's own thread_id doubles as the thread-bind flow's
        # pending_thread_id — the browse state lives under this same key.
        pending_thread_id = _get_thread_id(update)
        browse_state = get_browse_state(context.user_data, pending_thread_id)
        default_path = str(Path.cwd())
        selected_path = browse_state.get(BROWSE_PATH_KEY, default_path)

        # Check for existing sessions in this directory
        sessions = await session_manager.list_sessions_for_directory(selected_path)
        if sessions:
            # Show session picker — store state for later. Any pending first
            # message for this topic stays untouched in its own entry.
            set_browse_state(
                context.user_data,
                pending_thread_id,
                {
                    STATE_KEY: STATE_SELECTING_SESSION,
                    SESSIONS_KEY: sessions,
                    SELECTED_PATH_KEY: selected_path,
                },
            )
            text, keyboard = build_session_picker(sessions)
            await safe_edit(query, text, reply_markup=keyboard)
            await query.answer()
            return

        # No existing sessions — create new window directly
        await _create_and_bind_window(
            query, context, user, selected_path, pending_thread_id
        )

    elif data == CB_DIR_CANCEL:
        clear_browse_state(context.user_data, _get_thread_id(update))
        await safe_edit(query, "Cancelled")
        await query.answer("Cancelled")

    # Session picker: resume existing session
    elif data.startswith(CB_SESSION_SELECT):
        pending_tid = _get_thread_id(update)
        try:
            idx = int(data[len(CB_SESSION_SELECT) :])
        except ValueError:
            await query.answer("Invalid data")
            return

        browse_state = get_browse_state(context.user_data, pending_tid)
        cached_sessions = browse_state.get(SESSIONS_KEY, [])
        if idx < 0 or idx >= len(cached_sessions):
            await query.answer("Session not found")
            return

        session = cached_sessions[idx]
        selected_path = browse_state.get(SELECTED_PATH_KEY, str(Path.cwd()))

        await _create_and_bind_window(
            query,
            context,
            user,
            selected_path,
            pending_tid,
            resume_session_id=session.session_id,
        )

    elif data == CB_SESSION_NEW:
        pending_tid = _get_thread_id(update)
        browse_state = get_browse_state(context.user_data, pending_tid)
        selected_path = browse_state.get(SELECTED_PATH_KEY, str(Path.cwd()))

        await _create_and_bind_window(query, context, user, selected_path, pending_tid)

    elif data == CB_SESSION_CANCEL:
        clear_browse_state(context.user_data, _get_thread_id(update))
        await safe_edit(query, "Cancelled")
        await query.answer("Cancelled")

    # Window picker: bind existing window
    elif data.startswith(CB_WIN_BIND):
        thread_id = _get_thread_id(update)
        if thread_id is None:
            await query.answer("Not in a topic", show_alert=True)
            return

        try:
            idx = int(data[len(CB_WIN_BIND) :])
        except ValueError:
            await query.answer("Invalid data")
            return

        browse_state = get_browse_state(context.user_data, thread_id)
        cached_windows: list[str] = browse_state.get(UNBOUND_WINDOWS_KEY, [])
        if idx < 0 or idx >= len(cached_windows):
            await query.answer("Window list changed, please retry", show_alert=True)
            return
        selected_wid = cached_windows[idx]

        # Verify window still exists
        w = await tmux_manager.find_window_by_id(selected_wid)
        if not w:
            display = session_manager.get_display_name(selected_wid)
            await query.answer(f"Window '{display}' no longer exists", show_alert=True)
            return

        display = w.window_name
        pending_text = browse_state.get(PENDING_TEXT_KEY)
        clear_browse_state(context.user_data, thread_id)
        bound = session_manager.bind_thread(
            user.id, thread_id, selected_wid, window_name=display
        )
        if not bound:
            # Someone else bound this window between the picker snapshot and
            # this confirmation (or a concurrent picker did) — refuse the
            # double-bind rather than cross-wire two topics onto one window
            # (RC4/f46).
            await safe_edit(
                query,
                "❌ That window is already bound to another topic.",
            )
            await query.answer("Already bound", show_alert=True)
            return

        # Rename the topic to match the window name
        resolved_chat = session_manager.resolve_chat_id(user.id, thread_id)
        try:
            await context.bot.edit_forum_topic(
                chat_id=resolved_chat,
                message_thread_id=thread_id,
                name=display,
            )
        except Exception as e:
            logger.debug(f"Failed to rename topic: {e}")

        await safe_edit(
            query,
            f"✅ Bound to window `{display}`",
        )

        # Forward pending text if any
        if pending_text:
            send_ok, send_msg = await session_manager.send_to_window(
                selected_wid, pending_text
            )
            if not send_ok:
                logger.warning("Failed to forward pending text: %s", send_msg)
                await safe_send(
                    context.bot,
                    resolved_chat,
                    f"❌ Failed to send pending message: {send_msg}",
                    message_thread_id=thread_id,
                )
        await query.answer("Bound")

    # Window picker: new session → transition to directory browser
    elif data == CB_WIN_NEW:
        thread_id = _get_thread_id(update)
        # Transition to the directory browser; any pending first message for
        # this topic stays untouched in its own entry.
        start_path = _resolve_browser_start_path()
        msg_text, keyboard, subdirs = build_directory_browser(start_path)
        set_browse_state(
            context.user_data,
            thread_id,
            {
                STATE_KEY: STATE_BROWSING_DIRECTORY,
                BROWSE_PATH_KEY: start_path,
                BROWSE_PAGE_KEY: 0,
                BROWSE_DIRS_KEY: subdirs,
            },
        )
        await safe_edit(query, msg_text, reply_markup=keyboard)
        await query.answer()

    # Window picker: cancel
    elif data == CB_WIN_CANCEL:
        clear_browse_state(context.user_data, _get_thread_id(update))
        await safe_edit(query, "Cancelled")
        await query.answer("Cancelled")

    # Screenshot: Refresh
    elif data.startswith(CB_SCREENSHOT_REFRESH):
        window_id = data[len(CB_SCREENSHOT_REFRESH) :]
        w = await tmux_manager.find_window_by_id(window_id)
        if not w:
            await query.answer("Window no longer exists", show_alert=True)
            return

        text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
        if not text:
            await query.answer("Failed to capture pane", show_alert=True)
            return

        png_bytes = await text_to_image(text, with_ansi=True)
        keyboard = _build_screenshot_keyboard(window_id)
        try:
            await query.edit_message_media(
                media=InputMediaDocument(
                    media=io.BytesIO(png_bytes), filename="screenshot.png"
                ),
                reply_markup=keyboard,
            )
            await query.answer("Refreshed")
        except Exception as e:
            logger.error(f"Failed to refresh screenshot: {e}")
            await query.answer("Failed to refresh", show_alert=True)

    elif data == "noop":
        await query.answer()

    # Interactive UI: Up arrow
    elif data.startswith(CB_ASK_UP):
        window_id = data[len(CB_ASK_UP) :]
        thread_id = _get_thread_id(update)
        status = await send_ui_key(context.bot, user.id, window_id, "Up", thread_id)
        await query.answer(None if status == "Up" else status)

    # Interactive UI: Down arrow
    elif data.startswith(CB_ASK_DOWN):
        window_id = data[len(CB_ASK_DOWN) :]
        thread_id = _get_thread_id(update)
        status = await send_ui_key(context.bot, user.id, window_id, "Down", thread_id)
        await query.answer(None if status == "Down" else status)

    # Interactive UI: Left arrow
    elif data.startswith(CB_ASK_LEFT):
        window_id = data[len(CB_ASK_LEFT) :]
        thread_id = _get_thread_id(update)
        status = await send_ui_key(context.bot, user.id, window_id, "Left", thread_id)
        await query.answer(None if status == "Left" else status)

    # Interactive UI: Right arrow
    elif data.startswith(CB_ASK_RIGHT):
        window_id = data[len(CB_ASK_RIGHT) :]
        thread_id = _get_thread_id(update)
        status = await send_ui_key(context.bot, user.id, window_id, "Right", thread_id)
        await query.answer(None if status == "Right" else status)

    # Interactive UI: Escape
    elif data.startswith(CB_ASK_ESC):
        window_id = data[len(CB_ASK_ESC) :]
        thread_id = _get_thread_id(update)
        status = await send_ui_key(
            context.bot, user.id, window_id, "Escape", thread_id, clear_on_send=True
        )
        await query.answer("⎋ Esc" if status == "Escape" else status)

    # Interactive UI: Enter
    elif data.startswith(CB_ASK_ENTER):
        window_id = data[len(CB_ASK_ENTER) :]
        thread_id = _get_thread_id(update)
        status = await send_ui_key(context.bot, user.id, window_id, "Enter", thread_id)
        await query.answer("⏎ Enter" if status == "Enter" else status)

    # Interactive UI: Space
    elif data.startswith(CB_ASK_SPACE):
        window_id = data[len(CB_ASK_SPACE) :]
        thread_id = _get_thread_id(update)
        status = await send_ui_key(context.bot, user.id, window_id, "Space", thread_id)
        await query.answer("␣ Space" if status == "Space" else status)

    # Interactive UI: Tab
    elif data.startswith(CB_ASK_TAB):
        window_id = data[len(CB_ASK_TAB) :]
        thread_id = _get_thread_id(update)
        status = await send_ui_key(context.bot, user.id, window_id, "Tab", thread_id)
        await query.answer("⇥ Tab" if status == "Tab" else status)

    # Interactive UI: refresh display
    elif data.startswith(CB_ASK_REFRESH):
        window_id = data[len(CB_ASK_REFRESH) :]
        thread_id = _get_thread_id(update)
        await handle_interactive_ui(context.bot, user.id, window_id, thread_id)
        await query.answer("🔄")

    # Divergence notice: re-point a window's session_map entry (human-approved)
    elif data.startswith(CB_REPOINT):
        rest = data[len(CB_REPOINT) :]
        window_id, _, new_sid = rest.partition(":")
        if not window_id or not new_sid:
            await query.answer("Invalid data")
            return
        ok = await session_manager.repoint_window_session(window_id, new_sid)
        if ok:
            await query.answer("Re-pointed")
            try:
                await query.edit_message_text(
                    f"✅ This window now tracks session {new_sid[:8]}… — "
                    "new messages will flow again shortly."
                )
            except Exception:
                pass  # Original notice may be old or already edited
        else:
            await query.answer("No session_map entry for this window", show_alert=True)

    # Screenshot quick keys: send key to tmux window
    elif data.startswith(CB_KEYS_PREFIX):
        rest = data[len(CB_KEYS_PREFIX) :]
        colon_idx = rest.find(":")
        if colon_idx < 0:
            await query.answer("Invalid data")
            return
        key_id = rest[:colon_idx]
        window_id = rest[colon_idx + 1 :]

        key_info = _KEYS_SEND_MAP.get(key_id)
        if not key_info:
            await query.answer("Unknown key")
            return

        tmux_key, enter, literal = key_info
        w = await tmux_manager.find_window_by_id(window_id)
        if not w:
            await query.answer("Window not found", show_alert=True)
            return

        await tmux_manager.send_keys(
            w.window_id, tmux_key, enter=enter, literal=literal
        )
        await query.answer(_KEY_LABELS.get(key_id, key_id))

        # Refresh screenshot after key press
        await asyncio.sleep(0.5)
        text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
        if text:
            png_bytes = await text_to_image(text, with_ansi=True)
            keyboard = _build_screenshot_keyboard(window_id)
            try:
                await query.edit_message_media(
                    media=InputMediaDocument(
                        media=io.BytesIO(png_bytes),
                        filename="screenshot.png",
                    ),
                    reply_markup=keyboard,
                )
            except Exception:
                pass  # Screenshot unchanged or message too old


# --- Streaming response / notifications ---


async def handle_new_message(msg: NewMessage, bot: Bot) -> None:
    """Handle a new assistant message — enqueue for sequential processing.

    Messages are queued per-user to ensure status messages always appear last.
    Routes via thread_bindings to deliver to the correct topic.
    """
    status = "complete" if msg.is_complete else "streaming"
    logger.info(
        f"handle_new_message [{status}]: session={msg.session_id}, "
        f"text_len={len(msg.text)}"
    )

    # Find users whose thread-bound window matches this session
    active_users = await session_manager.find_users_for_session(msg.session_id)

    if not active_users:
        logger.info(f"No active users for session {msg.session_id}")
        return

    for user_id, wid, thread_id in active_users:
        # Pane-as-source: AskUserQuestion/ExitPlanMode UIs are delivered by
        # status_polling (which captures the rendered pane) → message_queue
        # worker. The JSONL `tool_use` entry is just a notification that the
        # tool was invoked; we suppress it here so it doesn't appear as a
        # regular content message, and advance the read offset so a restart
        # doesn't reprocess it.
        if msg.tool_name in INTERACTIVE_TOOL_NAMES and msg.content_type == "tool_use":
            session = await session_manager.resolve_session_for_window(wid)
            if session and session.file_path:
                try:
                    file_size = Path(session.file_path).stat().st_size
                    session_manager.update_user_window_offset(user_id, wid, file_size)
                except OSError:
                    pass
            continue  # Pane is the source — JSONL entry suppressed.

        # Any non-interactive message means the interaction is complete — delete the UI message
        if get_interactive_msg_id(user_id, thread_id):
            await clear_interactive_msg(user_id, bot, thread_id)

        # Skip tool call notifications when CCBOT_SHOW_TOOL_CALLS=false
        if not config.show_tool_calls and msg.content_type in (
            "tool_use",
            "tool_result",
        ):
            continue

        parts = build_response_parts(
            msg.text,
            msg.is_complete,
            msg.content_type,
            msg.role,
        )

        if msg.is_complete:
            # Enqueue content message task
            # Note: tool_result editing is handled inside _process_content_task
            # to ensure sequential processing with tool_use message sending
            await enqueue_content_message(
                bot=bot,
                user_id=user_id,
                window_id=wid,
                parts=parts,
                tool_use_id=msg.tool_use_id,
                content_type=msg.content_type,
                text=msg.text,
                thread_id=thread_id,
                image_data=msg.image_data,
                role=msg.role,
            )

            # Update user's read offset to current file position
            # This marks these messages as "read" for this user
            session = await session_manager.resolve_session_for_window(wid)
            if session and session.file_path:
                try:
                    file_size = Path(session.file_path).stat().st_size
                    session_manager.update_user_window_offset(user_id, wid, file_size)
                except OSError:
                    pass


# --- App lifecycle ---


def _prefill_limiter(limiter: AsyncLimiter) -> None:
    """Make an aiolimiter bucket start "full" immediately, surviving a restart.

    Telegram's server-side flood counters persist across bot restarts, so a
    freshly-constructed (empty) local bucket would let ccbot burst back up
    to ``max_rate`` requests the moment it reconnects -- against a
    server-side counter that never reset. Setting ``_level`` to ``max_rate``
    alone is not enough: aiolimiter's ``_leak()`` drains the bucket based on
    elapsed *loop* time since ``_last_check``, which defaults to ``0.0`` at
    construction. The very first capacity check after pre-fill would then
    see an "elapsed" of (current loop time - 0.0) -- an enormous number --
    and instantly zero the level back out, silently undoing the pre-fill
    regardless of how soon the first request actually arrives. Stamping
    ``_last_check`` with the current loop time (the same clock ``_leak()``
    reads via ``self._loop.time()``) closes that gap.
    """
    if not (hasattr(limiter, "_level") and hasattr(limiter, "_last_check")):
        logger.warning(
            "aiolimiter.AsyncLimiter is missing expected _level/_last_check "
            "attributes (library version mismatch?); skipping rate limiter "
            "pre-fill"
        )
        return
    try:
        now = limiter._loop.time()
    except AttributeError:
        logger.warning(
            "aiolimiter.AsyncLimiter is missing the expected _loop clock "
            "(library version mismatch?); skipping rate limiter pre-fill"
        )
        return
    limiter._level = limiter.max_rate
    limiter._last_check = now


async def post_init(application: Application) -> None:
    global session_monitor, _status_poll_task

    await application.bot.delete_my_commands()

    bot_commands = [
        BotCommand("start", "Show welcome message"),
        BotCommand("history", "Message history for this topic"),
        BotCommand("screenshot", "Terminal screenshot with control keys"),
        BotCommand("esc", "Send Escape to interrupt Claude"),
        BotCommand("kill", "Kill session and delete topic"),
        BotCommand("unbind", "Unbind topic from session (keeps window running)"),
        BotCommand("usage", "Show Claude Code usage remaining"),
    ]
    # Add Claude Code slash commands
    for cmd_name, desc in CC_COMMANDS.items():
        bot_commands.append(BotCommand(cmd_name, desc))

    await application.bot.set_my_commands(bot_commands)

    # Re-resolve stale window IDs from persisted state against live tmux windows
    await session_manager.resolve_stale_ids()

    # Pre-fill global rate limiter bucket on restart.
    # AsyncLimiter starts at _level=0 (full burst capacity), but Telegram's
    # server-side counter persists across bot restarts. _prefill_limiter()
    # forces the bucket to start "full" so capacity drains in naturally (~1s)
    # instead of being immediately reset by aiolimiter's leak calculation --
    # see its docstring for why the clock reference must be stamped too.
    # AIORateLimiter has no per-private-chat limiter, so max_retries is the
    # primary protection (retry + pause all concurrent requests on 429).
    rate_limiter = application.bot.rate_limiter
    if rate_limiter and rate_limiter._base_limiter:
        _prefill_limiter(rate_limiter._base_limiter)
        logger.info("Pre-filled global rate limiter bucket")
        # Also pre-fill per-group limiters for known chat IDs.
        # Without this, the group limiter allows a burst of 20 requests on restart,
        # which can exceed Telegram's persisted server-side per-group counter.
        if hasattr(rate_limiter, "_group_limiters"):
            group_rate = getattr(rate_limiter, "_group_max_rate", 20)
            group_period = getattr(rate_limiter, "_group_time_period", 60)
            seen_chat_ids: set[int] = set()
            for chat_id in session_manager.group_chat_ids.values():
                if chat_id < 0 and chat_id not in seen_chat_ids:
                    seen_chat_ids.add(chat_id)
                    limiter = AsyncLimiter(group_rate, group_period)
                    _prefill_limiter(limiter)
                    rate_limiter._group_limiters[chat_id] = limiter
            if seen_chat_ids:
                logger.info(
                    "Pre-filled %d group rate limiter bucket(s)", len(seen_chat_ids)
                )

    monitor = SessionMonitor()

    async def message_callback(msg: NewMessage) -> None:
        await handle_new_message(msg, application.bot)

    async def turn_end_callback(session_id: str) -> None:
        from .update_watcher import maybe_notify_update_or_failure

        active_users = await session_manager.find_users_for_session(session_id)
        for user_id, window_id, thread_id in active_users:
            queue = get_message_queue(user_id, thread_id)
            if queue:
                await queue.join()
            await maybe_notify_update_or_failure(
                application.bot, user_id, thread_id, window_id
            )

    monitor.set_message_callback(message_callback)
    monitor.set_turn_end_callback(turn_end_callback)
    monitor.start()
    session_monitor = monitor
    logger.info("Session monitor started")

    # Start status polling task (supervised: restarts on crash/unexpected exit)
    _status_poll_task = asyncio.create_task(
        supervise_loop("status polling", lambda: status_poll_loop(application.bot))
    )
    logger.info("Status polling task started")

    # Start local-state maintenance task (session_map hygiene), supervised
    global _maintenance_task
    _maintenance_task = asyncio.create_task(
        supervise_loop("maintenance", lambda: maintenance_loop(application.bot))
    )
    logger.info("Maintenance task started")


async def post_shutdown(application: Application) -> None:
    global _status_poll_task, _maintenance_task

    # Stop status polling
    if _status_poll_task:
        _status_poll_task.cancel()
        try:
            await _status_poll_task
        except asyncio.CancelledError:
            pass
        _status_poll_task = None
        logger.info("Status polling stopped")

    # Stop maintenance
    if _maintenance_task:
        _maintenance_task.cancel()
        try:
            await _maintenance_task
        except asyncio.CancelledError:
            pass
        _maintenance_task = None
        logger.info("Maintenance stopped")

    # Order matters: stop producers, flush queues, THEN cancel workers.
    # 1. The session monitor enqueues already-read-but-undelivered messages.
    #    Offsets now advance only on delivery ACK (see session_monitor's
    #    delivery contract), so nothing is lost if we skip this — but
    #    draining first avoids needlessly redelivering them next start.
    if session_monitor:
        await session_monitor.drain_callbacks()
        session_monitor.stop()
        logger.info("Session monitor stopped")

    # 2. Let the still-running workers actually deliver everything enqueued
    #    (including the just-drained messages) before we cancel them — a bare
    #    shutdown_workers() cancels mid-flight and the queued sends are dropped.
    await drain_queues()

    # 3. Now tear the workers down.
    await shutdown_workers()

    await close_transcribe_client()


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch exceptions escaping any handler: log them and notify the user.

    PTB advances the getUpdates offset before dispatching, so an update whose
    handler dies is never redelivered — without this, the message is dropped
    with no trace ("silence never means delivered"). The notice is plain text
    sent directly (no MarkdownV2 conversion) and best-effort: if Telegram is
    unreachable the loud log is the fallback.
    """
    logger.error(
        "Handler exception while processing update %s", update, exc_info=context.error
    )
    message = update.effective_message if isinstance(update, Update) else None
    if message is None:
        return
    try:
        await message.reply_text(
            "⚠️ Error while handling this message — it may not have reached "
            "Claude. If no response follows, please resend it."
        )
    except Exception:
        logger.exception("Failed to send error notice to user")


def create_bot() -> Application:
    application = (
        Application.builder()
        .token(config.telegram_bot_token)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_error_handler(error_handler)

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("screenshot", screenshot_command))
    application.add_handler(CommandHandler("esc", esc_command))
    application.add_handler(CommandHandler("restart", restart_command))
    application.add_handler(CommandHandler("kill", kill_command))
    application.add_handler(CommandHandler("unbind", unbind_command))
    application.add_handler(CommandHandler("usage", usage_command))
    application.add_handler(CallbackQueryHandler(callback_handler))
    # Topic closed event — auto-kill associated window
    application.add_handler(
        MessageHandler(
            filters.StatusUpdate.FORUM_TOPIC_CLOSED,
            topic_closed_handler,
        )
    )
    # Topic edited event — sync renamed topic to tmux window
    application.add_handler(
        MessageHandler(
            filters.StatusUpdate.FORUM_TOPIC_EDITED,
            topic_edited_handler,
        )
    )
    # Forward any other /command to Claude Code
    application.add_handler(MessageHandler(filters.COMMAND, forward_command_handler))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler)
    )
    # Images (photos, or image files): download and ask Claude Code to open
    application.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_handler)
    )
    # Voice: transcribe via OpenAI and forward text to Claude Code
    application.add_handler(MessageHandler(filters.VOICE, voice_handler))
    # Catch-all: non-text content (stickers, video, etc.)
    application.add_handler(
        MessageHandler(
            ~filters.COMMAND & ~filters.TEXT & ~filters.StatusUpdate.ALL,
            unsupported_content_handler,
        )
    )

    return application
