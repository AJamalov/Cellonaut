from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from cellonaut.io import writers


def test_atomic_csv_write_preserves_existing_file_on_failure(
    monkeypatch,
    tmp_path: Path,
):
    output = tmp_path / "measurements.csv"
    output.write_text("original", encoding="utf-8")

    def fail_to_csv(self, handle, **_kwargs):
        handle.write("partial")
        raise OSError("interrupted")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    with pytest.raises(OSError, match="interrupted"):
        writers.write_dataframe_csv(pd.DataFrame([{"value": 1}]), output)

    assert output.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob("*.tmp.csv"))


def test_atomic_text_write_replaces_existing_file(tmp_path: Path):
    output = tmp_path / "summary.json"
    output.write_text("old", encoding="utf-8")

    writers.write_text(output, "new")

    assert output.read_text(encoding="utf-8") == "new"


def test_exclusive_csv_write_does_not_replace_file_created_during_write(monkeypatch, tmp_path: Path):
    output = tmp_path / "measurements.csv"

    def create_competing_output(self, handle, **_kwargs):
        handle.write("candidate")
        output.write_text("created elsewhere", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_csv", create_competing_output)

    with pytest.raises(FileExistsError):
        writers.write_dataframe_csv(pd.DataFrame([{"value": 1}]), output, mode="x")

    assert output.read_text(encoding="utf-8") == "created elsewhere"
    assert not list(tmp_path.glob("*.tmp.csv"))


def test_atomic_csv_writer_rejects_non_atomic_append_mode(tmp_path: Path):
    with pytest.raises(ValueError, match="mode='w' or mode='x'"):
        writers.write_dataframe_csv(pd.DataFrame([{"value": 1}]), tmp_path / "data.csv", mode="a")


def test_atomic_tiff_write_preserves_existing_file_on_failure(
    monkeypatch,
    tmp_path: Path,
):
    output = tmp_path / "mask.tif"
    output.write_bytes(b"original")

    def fail_imwrite(path, _data, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(writers.tifffile, "imwrite", fail_imwrite)

    with pytest.raises(OSError, match="interrupted"):
        writers.write_tiff(output, np.zeros((2, 2), dtype=np.uint8))

    assert output.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp.tif"))


def test_imagej_false_return_preserves_existing_file_and_removes_partial_output(tmp_path: Path):
    output = tmp_path / "skeleton.tif"
    output.write_bytes(b"original")

    class FailingFileSaver:
        def __init__(self, _image):
            pass

        def saveAsTiff(self, path):
            Path(path).write_bytes(b"partial")
            return False

    saved = writers.save_imagej_tiff(output, object(), FailingFileSaver)

    assert saved is False
    assert output.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp.tif"))


def test_imagej_png_false_return_preserves_existing_file_and_removes_partial_output(tmp_path: Path):
    output = tmp_path / "overlay.png"
    output.write_bytes(b"original")

    class FailingFileSaver:
        def __init__(self, _image):
            pass

        def saveAsPng(self, path):
            Path(path).write_bytes(b"partial")
            return False

    saved = writers.save_imagej_png(output, object(), FailingFileSaver)

    assert saved is False
    assert output.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp.png"))


def test_exclusive_tiff_write_preserves_existing_file(tmp_path: Path):
    output = tmp_path / "converted.ome.tif"
    output.write_bytes(b"original")

    with pytest.raises(FileExistsError):
        writers.write_tiff(output, np.zeros((2, 2), dtype=np.uint8), exclusive=True)

    assert output.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp.tif"))


def test_qt_false_return_preserves_existing_file_and_removes_partial_output(tmp_path: Path):
    output = tmp_path / "snapshot.png"
    output.write_bytes(b"original")

    class FailingQtImage:
        def save(self, path, _file_format):
            Path(path).write_bytes(b"partial")
            return False

    saved = writers.save_qt_image(output, FailingQtImage(), "PNG")

    assert saved is False
    assert output.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp.png"))
