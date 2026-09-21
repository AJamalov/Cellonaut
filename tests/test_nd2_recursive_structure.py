from __future__ import annotations

from pathlib import Path

import pytest

from cellonaut.io import nd2_import
from cellonaut.io.nd2_import import (
    ND2ImportConfig,
    list_nd2_files,
    nd2_output_path,
    validate_nd2_output_plan,
)


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real nd2, only a path fixture")


def test_list_nd2_files_recurses_and_sorts_by_relative_path(tmp_path: Path):
    touch(tmp_path / "BatchB" / "sample_002.nd2")
    touch(tmp_path / "BatchA" / "sample_001.nd2")
    touch(tmp_path / "BatchA" / "notes.txt")

    nd2_files = list_nd2_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in nd2_files] == [
        "BatchA/sample_001.nd2",
        "BatchB/sample_002.nd2",
    ]


def test_list_nd2_files_supports_input_folder_with_only_nested_nd2_files(tmp_path: Path):
    touch(tmp_path / "ExperimentA" / "raw" / "sample_001.nd2")

    nd2_files = list_nd2_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in nd2_files] == [
        "ExperimentA/raw/sample_001.nd2",
    ]


def test_list_nd2_files_ignores_revisited_directory_aliases(tmp_path: Path, monkeypatch):
    child = tmp_path / "child"
    nd2_path = child / "sample.nd2"
    touch(nd2_path)
    real_safe_iterdir = nd2_import.safe_iterdir

    def cyclic_iterdir(folder: Path) -> list[Path]:
        if folder == child:
            return [nd2_path, tmp_path]
        return real_safe_iterdir(folder)

    monkeypatch.setattr(nd2_import, "safe_iterdir", cyclic_iterdir)

    assert list_nd2_files(tmp_path) == [nd2_path]


def test_nd2_output_path_preserves_relative_parent_by_default(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "converted"
    nd2_path = source / "Experiment1" / "sample_001.nd2"
    touch(nd2_path)
    cfg = ND2ImportConfig(
        source_dir=source,
        output_dir=output,
        channel_map={"GFP": 0},
    )

    assert nd2_output_path(cfg, nd2_path, "sample_001") == output / "Experiment1" / "sample_001.ome.tif"


def test_nd2_output_path_can_flatten_when_requested(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "converted"
    nd2_path = source / "Experiment1" / "sample_001.nd2"
    touch(nd2_path)
    cfg = ND2ImportConfig(
        source_dir=source,
        output_dir=output,
        channel_map={"GFP": 0},
        preserve_subfolders=False,
    )

    assert nd2_output_path(cfg, nd2_path, "sample_001") == output / "sample_001.ome.tif"


def test_nd2_import_config_defaults_to_max_projection(tmp_path: Path):
    cfg = ND2ImportConfig(
        source_dir=tmp_path,
        output_dir=tmp_path / "converted",
        channel_map={"GFP": 0},
    )

    assert cfg.z_mode == "max_projection"


def test_nd2_output_plan_rejects_normalized_name_collisions(tmp_path: Path):
    source = tmp_path / "source"
    first = source / "sample_001.nd2"
    second = source / "sample_002.nd2"
    touch(first)
    touch(second)
    cfg = ND2ImportConfig(
        source_dir=source,
        output_dir=tmp_path / "converted",
        channel_map={"GFP": 0},
        strip_after_last_underscore=True,
    )

    with pytest.raises(ValueError, match="same TIFF output"):
        validate_nd2_output_plan(cfg, [first, second])


def test_nd2_output_plan_allows_existing_converted_tiff(tmp_path: Path):
    source = tmp_path / "source"
    nd2_path = source / "sample_001.nd2"
    touch(nd2_path)
    output = tmp_path / "converted"
    existing = output / "sample_001.ome.tif"
    touch(existing)
    cfg = ND2ImportConfig(source_dir=source, output_dir=output, channel_map={"GFP": 0})

    validate_nd2_output_plan(cfg, [nd2_path])
    assert existing.read_bytes() == b"not a real nd2, only a path fixture"


@pytest.mark.parametrize("race", [False, True])
def test_nd2_export_numbers_existing_files_without_overwriting(tmp_path, monkeypatch, race):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np
    import tifffile

    source = tmp_path / "input"
    nd2_path = source / "sample.nd2"
    touch(nd2_path)
    output = tmp_path / "output"
    output.mkdir()
    original = output / "sample.ome.tif"
    original.write_bytes(b"original")
    second = output / "sample_2.ome.tif"
    second.write_bytes(b"second")
    cfg = ND2ImportConfig(source_dir=source, output_dir=output, channel_map={"GFP": 0})
    pixels = np.arange(20, dtype=np.uint16).reshape(4, 5)
    monkeypatch.setattr(nd2_import, "require_nd2", lambda: None)
    monkeypatch.setattr(nd2_import, "nd2", SimpleNamespace(ND2File=lambda _path: nullcontext(SimpleNamespace(sizes={}))))
    monkeypatch.setattr(nd2_import, "get_nd2_channel_names", lambda _f: ["GFP"])
    monkeypatch.setattr(nd2_import, "get_nd2_channel_colors", lambda *_args: ["#00FF00"])
    monkeypatch.setattr(nd2_import, "get_nd2_pixel_sizes_um", lambda *_args: (None, None))
    monkeypatch.setattr(nd2_import, "read_first_nd2_position", lambda _f: pixels)
    monkeypatch.setattr(nd2_import, "_select_channel_plane", lambda **_kwargs: pixels)
    real_writer = nd2_import.write_ome_channel_stack
    raced = output / "sample_3.ome.tif"
    def write_stack(path, data, **kwargs):
        if race and path == raced:
            path.write_bytes(b"another writer")
            raise FileExistsError(path)
        real_writer(path, data, **kwargs)
    monkeypatch.setattr(nd2_import, "write_ome_channel_stack", write_stack)

    exported = nd2_import.export_nd2_file(nd2_path, cfg)

    expected = output / ("sample_4.ome.tif" if race else "sample_3.ome.tif")
    assert exported == [expected]
    np.testing.assert_array_equal(tifffile.imread(expected).squeeze(), pixels)
    assert original.read_bytes() == b"original"
    assert second.read_bytes() == b"second"
    if race:
        assert raced.read_bytes() == b"another writer"
