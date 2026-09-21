"""Run-log construction shared by full and preview pipeline entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


# Send messages to the GUI and log file. If writing fails, warn once
# and continue with GUI logging.
def make_logger(gui_log_func: Callable[[str], None], logfile_path: Path) -> Callable[[str], None]:
    log_write_failed = False

    def logger(message: str) -> None:
        nonlocal log_write_failed
        text = str(message)
        gui_log_func(text)
        if log_write_failed:
            return
        try:
            with open(logfile_path, "a", encoding="utf-8") as log_file:
                log_file.write(text + "\n")
        except (OSError, UnicodeError) as exc:
            log_write_failed = True
            gui_log_func(f"[WARN] Run log could not be written to {logfile_path}: {exc}")

    return logger
