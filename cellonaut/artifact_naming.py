"""Portable, collision-aware names for generated artifacts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def portable_component(value: object, *, fallback: str = "", strip: bool = False) -> str:
    """Replace filesystem punctuation while preserving readable Unicode letters."""
    text = str(value or "")
    if strip:
        text = text.strip()
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
    return safe or fallback


def collision_resistant_ascii_component(value: object, *, fallback: str = "Artifact") -> str:
    """Return an ASCII filename component and hash labels changed by normalization."""
    text = str(value or "")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or fallback
    if safe != text:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}_{digest}"
    return safe


def relative_artifact_stem(path: Path) -> str:
    """Encode a relative path as one portable artifact stem."""
    parts = Path(path).as_posix().split("/")
    return "__".join(portable_component(part) for part in parts)
