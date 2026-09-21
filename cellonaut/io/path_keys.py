"""Stable path comparison keys shared by GUI-independent services."""

from __future__ import annotations

import ntpath
from pathlib import Path


def input_path_key(value: str | Path) -> str:
    """Normalize pasted Windows slash and case variants for comparisons."""
    text = str(value or "").strip()
    if not text:
        return ""
    return ntpath.normcase(ntpath.normpath(text))
