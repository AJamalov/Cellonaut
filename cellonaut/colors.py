"""Shared validation helpers for display colors."""

from __future__ import annotations


def normalize_rgb_hex(value: object, *, fallback: str = "") -> str:
    """Return canonical ``#RRGGBB`` text or the requested fallback."""
    text = str(value or "").strip().upper()
    if len(text) == 7 and text.startswith("#"):
        try:
            int(text[1:], 16)
        except ValueError:
            pass
        else:
            return text
    return fallback
