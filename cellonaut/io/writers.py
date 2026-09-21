"""Filesystem-safe writers shared by pipeline and export modules."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

WriteResult = TypeVar("WriteResult")


def filesystem_path(path: Path) -> str:
    """Return a third-party-safe path, adding Windows long-path syntax when needed."""
    resolved = Path(path)
    if os.name != "nt":
        return str(resolved)

    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return text

    absolute = str(resolved.absolute())
    if len(absolute) < 240:
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute.lstrip("\\")
    return "\\\\?\\" + absolute


def ensure_parent_dir(path: Path) -> None:
    """Create an output parent using the same long-path behavior as all writers."""
    os.makedirs(filesystem_path(Path(path).parent), exist_ok=True)


def temporary_output_path(path: Path) -> Path:
    """Return a unique sibling path that retains the format-selecting suffix."""
    path = Path(path)
    return path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")


# Publish exclusive outputs through a hard link so another process cannot create
# the destination between an existence check and the final operation.
def _publish_temporary_output(temp_path: Path, path: Path, *, exclusive: bool = False) -> None:
    if exclusive:
        os.link(filesystem_path(temp_path), filesystem_path(path))
        return
    os.replace(filesystem_path(temp_path), filesystem_path(path))


# Ignore cleanup errors so they do not hide the original write failure.
def _remove_temporary_output(temp_path: Path) -> None:
    try:
        os.unlink(filesystem_path(temp_path))
    except OSError:
        pass


# Always attempt temporary-file cleanup, including after cancellation.
def _write_atomically(
    path: Path,
    write_temporary: Callable[[Path], WriteResult],
    *,
    exclusive: bool = False,
) -> WriteResult:
    path = Path(path)
    ensure_parent_dir(path)
    temp_path = temporary_output_path(path)
    try:
        result = write_temporary(temp_path)
        _publish_temporary_output(temp_path, path, exclusive=exclusive)
        return result
    finally:
        _remove_temporary_output(temp_path)


def write_dataframe_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Write a spreadsheet-friendly CSV atomically; append modes are unsupported."""
    path = Path(path)
    if not df.columns.is_unique:
        duplicates = list(dict.fromkeys(str(column) for column in df.columns[df.columns.duplicated()]))
        raise ValueError(f"CSV columns must be unique: {', '.join(duplicates)}")
    ensure_parent_dir(path)
    csv_kwargs = dict(kwargs)
    encoding = csv_kwargs.pop("encoding", "utf-8-sig")
    mode = csv_kwargs.pop("mode", "w")
    newline = csv_kwargs.pop("newline", "")

    if mode not in {"w", "x"}:
        raise ValueError("Atomic CSV writes support only mode='w' or mode='x'.")
    # Pass pandas an open handle so it never has to interpret extended Windows paths itself.
    def write_temporary(temp_path: Path) -> None:
        with open(filesystem_path(temp_path), "x", encoding=encoding, newline=newline) as handle:
            df.to_csv(handle, **csv_kwargs)

    _write_atomically(path, write_temporary, exclusive=mode == "x")


def write_tiff(path: Path, data: np.ndarray, *, exclusive: bool = False, **kwargs: Any) -> None:
    """Write a TIFF atomically, defaulting to an uncompressed portable output."""
    path = Path(path)
    kwargs.setdefault("compression", None)

    def write_temporary(temp_path: Path) -> None:
        tifffile.imwrite(filesystem_path(temp_path), data, **kwargs)

    _write_atomically(path, write_temporary, exclusive=exclusive)


def save_imagej_tiff(path: Path, image: Any, file_saver_class: Any) -> bool:
    """Save an ImageJ TIFF atomically and honor FileSaver's boolean failure result."""
    path = Path(path)
    ensure_parent_dir(path)
    temp_path = temporary_output_path(path)
    try:
        saved = bool(file_saver_class(image).saveAsTiff(filesystem_path(temp_path)))
        if not saved:
            _remove_temporary_output(temp_path)
            return False
        _publish_temporary_output(temp_path, path)
        return True
    finally:
        _remove_temporary_output(temp_path)


def save_imagej_png(path: Path, image: Any, file_saver_class: Any) -> bool:
    """Save an ImageJ PNG atomically without exposing a partial flattened overlay."""
    path = Path(path)
    ensure_parent_dir(path)
    temp_path = temporary_output_path(path)
    try:
        saved = bool(file_saver_class(image).saveAsPng(filesystem_path(temp_path)))
        if not saved:
            return False
        _publish_temporary_output(temp_path, path)
        return True
    finally:
        _remove_temporary_output(temp_path)


def save_qt_image(path: Path, image: Any, file_format: str) -> bool:
    """Save a Qt-compatible image atomically without importing Qt in this module."""
    path = Path(path)
    ensure_parent_dir(path)
    temp_path = temporary_output_path(path)
    try:
        saved = bool(image.save(filesystem_path(temp_path), file_format))
        if not saved:
            return False
        _publish_temporary_output(temp_path, path)
        return True
    finally:
        _remove_temporary_output(temp_path)


def save_pil_image(image: Image.Image, path: Path, **kwargs: Any) -> None:
    """Save a Pillow image atomically while preserving suffix-based format selection."""
    path = Path(path)

    # Preserve Pillow's ordinary save API while the shared wrapper owns publication and cleanup.
    def write_temporary(temp_path: Path) -> None:
        image.save(filesystem_path(temp_path), **kwargs)

    _write_atomically(path, write_temporary)


def write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write a manifest or sidecar atomically so metadata is never left truncated."""
    path = Path(path)

    # Open exclusively so a UUID collision cannot silently reuse an unrelated temporary file.
    def write_temporary(temp_path: Path) -> None:
        with open(filesystem_path(temp_path), "x", encoding=encoding) as handle:
            handle.write(text)

    _write_atomically(path, write_temporary)
