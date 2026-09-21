from __future__ import annotations

import json
from pathlib import Path
import shutil

from cellonaut.release_checks import release_documents
from cellonaut.release_checks.offline_assets import MODEL_NOTICE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_METADATA = {
    "cpsam": {
        "bytes": 1_233_587_898,
        "sha256": "e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2",
    },
    "cpsam_v2": {
        "bytes": 1_233_586_851,
        "sha256": "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667",
    },
}


def copy_release_documents(destination: Path) -> None:
    for relative_path in release_documents.REQUIRED_DOCUMENTS:
        source = PROJECT_ROOT / relative_path
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def write_offline_metadata(offline_root: Path, models: dict[str, dict[str, object]]) -> None:
    offline_root.mkdir(parents=True)
    (offline_root / "OFFLINE_ASSETS.json").write_text(
        json.dumps({"offline": True, "fiji": "Fiji.app", "cellpose_models": models}),
        encoding="utf-8",
    )
    model_root = offline_root / "cellpose_models"
    model_root.mkdir()
    (model_root / "MODEL_NOTICE.txt").write_text(MODEL_NOTICE, encoding="utf-8")


def test_current_release_documents_are_synchronized():
    assert release_documents.validate_release_documents(PROJECT_ROOT) == []


def test_release_document_validation_detects_version_drift(tmp_path: Path):
    copy_release_documents(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(citation.read_text(encoding="utf-8").replace("version: 1.0.0", "version: 9.9.9"), encoding="utf-8")

    problems = release_documents.validate_release_documents(tmp_path)

    assert "CITATION.cff version does not match 1.0.0" in problems


def test_release_document_validation_detects_missing_download_name(tmp_path: Path):
    copy_release_documents(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("Cellonaut-1.0.0-windows.exe", "Cellonaut-windows.exe"),
        encoding="utf-8",
    )

    problems = release_documents.validate_release_documents(tmp_path)

    assert "README.md does not name release artifact Cellonaut-1.0.0-windows.exe" in problems


def test_release_document_validation_detects_missing_support_file(tmp_path: Path):
    copy_release_documents(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "Cellonaut-1.0.0-windows.sha256",
            "the Windows checksum manifest",
        ),
        encoding="utf-8",
    )

    problems = release_documents.validate_release_documents(tmp_path)

    assert (
        "README.md does not name release support file "
        "Cellonaut-1.0.0-windows.sha256"
    ) in problems


def test_release_document_validation_rejects_deferred_platform_downloads(tmp_path: Path):
    copy_release_documents(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\nDownload Cellonaut-1.0.0-linux-x86_64-cpu.tar.gz.\n",
        encoding="utf-8",
    )

    problems = release_documents.validate_release_documents(tmp_path)

    assert "README.md advertises deferred platform artifact marker '-linux-'" in problems


def test_release_document_validation_rejects_deferred_platform_mentions(tmp_path: Path):
    copy_release_documents(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\nA macOS package is planned.\n",
        encoding="utf-8",
    )

    problems = release_documents.validate_release_documents(tmp_path)

    assert "README.md mentions deferred platform 'macOS'" in problems


def test_release_document_validation_matches_prepared_model_metadata(tmp_path: Path):
    copy_release_documents(tmp_path)
    offline_root = tmp_path / "offline"
    write_offline_metadata(offline_root, MODEL_METADATA)

    assert release_documents.validate_release_documents(tmp_path, offline_root) == []

    manifest_path = offline_root / "OFFLINE_ASSETS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cellpose_models"]["cpsam"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    problems = release_documents.validate_release_documents(tmp_path, offline_root)

    assert "BUNDLED_COMPONENTS.md SHA-256 does not match cpsam" in problems


def test_release_document_cli_reports_success(capsys):
    assert release_documents.main(["--project-root", str(PROJECT_ROOT)]) == 0
    assert "synchronized" in capsys.readouterr().out
