"""Prepare source and packaged GUI runtimes before the main window starts."""

from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

from cellonaut.system.subprocesses import hidden_window_kwargs
from cellonaut.version import format_app_version

_crash_log_handle = None
CRASH_LOG_MAX_BYTES = 5 * 1024 * 1024


def rotate_crash_log(crash_log_path: Path, *, max_bytes: int = CRASH_LOG_MAX_BYTES) -> Path | None:
    """Keep one recoverable previous log instead of allowing unlimited growth."""
    try:
        if not crash_log_path.exists() or crash_log_path.stat().st_size <= max_bytes:
            return None
        previous_path = crash_log_path.with_name("crash.previous.log")
        previous_path.unlink(missing_ok=True)
        crash_log_path.replace(previous_path)
        return previous_path
    except OSError:
        return None


# Find the Python executable in the project's .venv folder.
def _project_venv_python(project_root: Path) -> Path | None:
    candidate = project_root / ".venv" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else None


# Relaunch only source checkouts; frozen applications already carry their selected interpreter and packages.
def relaunch_with_project_venv() -> None:
    if getattr(sys, "frozen", False):
        return
    if not sys.argv or str(sys.argv[0]).startswith("-"):
        return

    project_root = Path(__file__).resolve().parents[2]
    venv_python = _project_venv_python(project_root)
    if venv_python is None:
        return
    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except OSError:
        return

    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("__PYVENV_LAUNCHER__", None)
    subprocess.Popen(
        [str(venv_python), *sys.argv],
        cwd=str(project_root),
        env=env,
        close_fds=True,
        **hidden_window_kwargs(),
    )
    raise SystemExit(0)


# Frozen GUI failures may never reach a terminal, so retain Python and native crash details in user data.
def install_crash_logging() -> None:
    global _crash_log_handle
    try:
        from cellonaut.resources import USER_APP_DIR, ensure_user_app_dirs

        ensure_user_app_dirs()
        crash_log_path = USER_APP_DIR / "crash.log"
        rotate_crash_log(crash_log_path)
        _crash_log_handle = open(crash_log_path, "a", encoding="utf-8", buffering=1)
        _crash_log_handle.write("\n===== Cellonaut session start =====\n")
        _crash_log_handle.write(f"Started UTC: {datetime.now(timezone.utc).isoformat()}\n")
        _crash_log_handle.write(f"App: {format_app_version()}\n")
        _crash_log_handle.write(f"Python: {sys.version}\n")
        faulthandler.enable(file=_crash_log_handle, all_threads=True)

        previous_excepthook = sys.excepthook

        # Preserve the normal interpreter hook after recording the same uncaught exception on disk.
        def log_excepthook(exc_type, exc_value, exc_tb):
            try:
                traceback.print_exception(exc_type, exc_value, exc_tb, file=_crash_log_handle)
            finally:
                previous_excepthook(exc_type, exc_value, exc_tb)

        sys.excepthook = log_excepthook

        previous_thread_hook = getattr(threading, "excepthook", None)

        # Worker-thread exceptions bypass sys.excepthook, so mirror them through threading's hook as well.
        def log_thread_excepthook(args):
            try:
                traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=_crash_log_handle)
            finally:
                if previous_thread_hook is not None:
                    previous_thread_hook(args)

        threading.excepthook = log_thread_excepthook
    except Exception:
        _crash_log_handle = None
