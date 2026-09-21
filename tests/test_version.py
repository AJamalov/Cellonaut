from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cellonaut import __version__
from cellonaut.version import format_app_version


def test_package_metadata_reads_authoritative_version_module():
    project_root = Path(__file__).resolve().parents[1]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'dynamic = ["version"]' in pyproject
    assert 'version = { attr = "cellonaut.version.__version__" }' in pyproject


def test_citation_uses_application_version():
    project_root = Path(__file__).resolve().parents[1]
    citation = (project_root / "CITATION.cff").read_text(encoding="utf-8")

    assert f"version: {__version__}" in citation
    assert "date-released:" not in citation


def test_release_facing_documents_use_application_version():
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    bundled_components = (project_root / "BUNDLED_COMPONENTS.md").read_text(encoding="utf-8")

    assert f"Current release: **Cellonaut {__version__}**" in readme
    assert f"Cellonaut {__version__} Windows release" in bundled_components


def test_module_cli_reports_version_without_starting_gui():
    completed = subprocess.run(
        [sys.executable, "-m", "cellonaut", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == format_app_version()
    assert __version__ in completed.stdout
