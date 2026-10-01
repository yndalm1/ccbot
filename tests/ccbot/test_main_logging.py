"""Tests for configure_logging — the size-capped rotating log file.

All logging goes to `ccbot.log` in the config dir. It rotates at
LOG_MAX_BYTES and keeps LOG_BACKUP_COUNT backups, so total log size on disk
stays within 100 MB however long the service runs.
"""

import logging
from logging.handlers import RotatingFileHandler
from unittest.mock import patch

import pytest

from ccbot import main


@pytest.fixture
def root_logger():
    """Restore the root logger's handlers and level after each test."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield root
    for handler in root.handlers:
        if handler not in saved_handlers:
            handler.close()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_cap_is_100_mb_total():
    assert main.LOG_MAX_BYTES * (main.LOG_BACKUP_COUNT + 1) == 100 * 1024 * 1024


def test_logs_go_to_rotating_file_in_config_dir(tmp_path, root_logger):
    with patch.object(main.sys.stderr, "isatty", return_value=False):
        main.configure_logging(tmp_path / "cfg")

    [handler] = root_logger.handlers
    assert isinstance(handler, RotatingFileHandler)
    assert handler.baseFilename == str(tmp_path / "cfg" / main.LOG_FILE_NAME)
    assert handler.maxBytes == main.LOG_MAX_BYTES
    assert handler.backupCount == main.LOG_BACKUP_COUNT


def test_terminal_run_also_echoes_to_stderr(tmp_path, root_logger):
    with patch.object(main.sys.stderr, "isatty", return_value=True):
        main.configure_logging(tmp_path)

    kinds = {type(h) for h in root_logger.handlers}
    assert kinds == {RotatingFileHandler, logging.StreamHandler}


def test_rotation_keeps_disk_use_within_cap(tmp_path, root_logger):
    max_bytes = 2_000
    with (
        patch.object(main, "LOG_MAX_BYTES", max_bytes),
        patch.object(main.sys.stderr, "isatty", return_value=False),
    ):
        main.configure_logging(tmp_path)

    log = logging.getLogger("ccbot.test_rotation")
    for i in range(500):
        log.warning("line %d — 日本語 %s", i, "x" * 40)

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [main.LOG_FILE_NAME, f"{main.LOG_FILE_NAME}.1"]
    total = sum(p.stat().st_size for p in tmp_path.iterdir())
    assert total <= max_bytes * (main.LOG_BACKUP_COUNT + 1)
