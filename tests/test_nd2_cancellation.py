from __future__ import annotations

import json
import pytest

from cellonaut.exceptions import PipelineCancelled
from cellonaut.io import nd2_import


# Cancellation must bypass per-file recovery or the next ND2 file would start
# after the user had already asked the conversion to stop.
def test_folder_conversion_does_not_count_cancellation_as_file_failure(tmp_path, monkeypatch):
    cfg = nd2_import.ND2ImportConfig(
        source_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        channel_map={"Channel 1": 0},
    )
    nd2_path = cfg.source_dir / "sample.nd2"
    messages: list[str] = []

    monkeypatch.setattr(nd2_import, "require_nd2", lambda: None)
    monkeypatch.setattr(nd2_import, "list_nd2_files", lambda _source_dir: [nd2_path])
    monkeypatch.setattr(
        nd2_import,
        "export_nd2_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PipelineCancelled("cancelled")),
    )

    with pytest.raises(PipelineCancelled, match="cancelled"):
        nd2_import.convert_nd2_folder(cfg, log_func=messages.append)

    assert "[ND2] Import cancelled by user." in messages
    assert not any("[ND2][ERROR]" in message for message in messages)
    state = json.loads((cfg.output_dir / "ConversionState.json").read_text(encoding="utf-8"))
    assert state["state"] == "cancelled"


def test_folder_conversion_marks_partial_conversion_outputs(tmp_path, monkeypatch):
    cfg = nd2_import.ND2ImportConfig(
        source_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        channel_map={"Channel 1": 0},
    )
    nd2_paths = [cfg.source_dir / "good.nd2", cfg.source_dir / "bad.nd2"]

    monkeypatch.setattr(nd2_import, "require_nd2", lambda: None)
    monkeypatch.setattr(nd2_import, "list_nd2_files", lambda _source_dir: nd2_paths)

    def export_file(path, *_args, **_kwargs):
        if path.name == "bad.nd2":
            raise OSError("corrupt source")

    monkeypatch.setattr(nd2_import, "export_nd2_file", export_file)

    summary = nd2_import.convert_nd2_folder(cfg)

    assert summary == {"total": 2, "processed": 1, "failed": 1}
    state = json.loads((cfg.output_dir / "ConversionState.json").read_text(encoding="utf-8"))
    assert state["state"] == "completed_with_errors"
