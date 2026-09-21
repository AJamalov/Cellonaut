"""Shared subprocess options for desktop application helpers."""

from __future__ import annotations

import subprocess
import sys
from typing import Any


def hidden_window_kwargs() -> dict[str, Any]:
    """Prevent console helpers from flashing a window on Windows."""
    if not sys.platform.startswith("win"):
        return {}

    kwargs: dict[str, Any] = {
        "creationflags": int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    }
    startupinfo_type = getattr(subprocess, "STARTUPINFO", None)
    if startupinfo_type is not None:
        startupinfo = startupinfo_type()
        startupinfo.dwFlags |= int(getattr(subprocess, "STARTF_USESHOWWINDOW", 0))
        startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
        kwargs["startupinfo"] = startupinfo
    return kwargs
