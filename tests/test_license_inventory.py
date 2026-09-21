from __future__ import annotations

from pathlib import Path
from importlib.metadata import PackagePath

import pytest

from cellonaut.release_checks import license_inventory


def test_python_license_discovery_has_a_bundled_fallback():
    licenses = license_inventory._python_license_files()

    assert license_inventory.PYTHON_LICENSE_FALLBACK in licenses
    assert license_inventory.PYTHON_LICENSE_FALLBACK.read_text(encoding="utf-8").startswith(
        "A. HISTORY OF THE SOFTWARE"
    )


def test_license_destination_names_shorten_deep_paths_without_collisions():
    prefix = "torch.dist-info/licenses/" + "third_party/" * 12
    first = license_inventory._license_destination_name(PackagePath(prefix + "one/LICENSE.txt"))
    second = license_inventory._license_destination_name(PackagePath(prefix + "two/LICENSE.txt"))

    assert len(first) <= 96
    assert first.endswith("LICENSE.txt")
    assert first != second


def test_license_inventory_collects_dependency_and_python_licenses(tmp_path, monkeypatch):
    python_license = tmp_path / "PYTHON_LICENSE.txt"
    python_license.write_text("Python license for test\n", encoding="utf-8")
    monkeypatch.setattr(license_inventory, "_python_license_files", lambda: [python_license])

    output_dir = tmp_path / "THIRD_PARTY_LICENSES"
    entries = license_inventory.write_license_inventory(output_dir, root_names=["packaging"])

    assert [entry.name.lower() for entry in entries] == ["packaging"]
    assert entries[0].copied_license_files
    assert (output_dir / "DEPENDENCIES.md").is_file()
    assert "packaging" in (output_dir / "DEPENDENCIES.md").read_text(encoding="utf-8").lower()
    assert (output_dir / "Python" / "PYTHON_LICENSE.txt").read_text(encoding="utf-8") == "Python license for test\n"


def test_license_inventory_rejects_an_unexpected_output_folder(tmp_path):
    with pytest.raises(ValueError, match="THIRD_PARTY_LICENSES"):
        license_inventory.write_license_inventory(Path(tmp_path) / "licenses", root_names=["packaging"])


def test_license_inventory_supplies_a_license_override(tmp_path, monkeypatch):
    python_license = tmp_path / "PYTHON_LICENSE.txt"
    python_license.write_text("Python license for test\n", encoding="utf-8")
    monkeypatch.setattr(license_inventory, "_python_license_files", lambda: [python_license])
    override = license_inventory.LICENSE_OVERRIDES["antlr4-python3-runtime"]
    monkeypatch.setitem(license_inventory.LICENSE_OVERRIDES, "packaging", override)
    monkeypatch.setattr(license_inventory, "_distribution_license_files", lambda _distribution: [])

    output_dir = tmp_path / "THIRD_PARTY_LICENSES"
    entries = license_inventory.write_license_inventory(output_dir, root_names=["packaging"])

    packaging_entry = next(entry for entry in entries if entry.name == "packaging")
    assert packaging_entry.copied_license_files == ("ANTLR4_LICENSE.txt",)
    package_dir = next(output_dir.glob("packaging-*"))
    assert "BSD 3-clause license" in (package_dir / "ANTLR4_LICENSE.txt").read_text(encoding="utf-8")


def test_qt_inventory_selects_open_source_gpl_terms(tmp_path, monkeypatch):
    python_license = tmp_path / "PYTHON_LICENSE.txt"
    python_license.write_text("Python license for test\n", encoding="utf-8")
    monkeypatch.setattr(license_inventory, "_python_license_files", lambda: [python_license])

    output_dir = tmp_path / "THIRD_PARTY_LICENSES"
    entries = license_inventory.write_license_inventory(output_dir, root_names=["pyside6"])
    qt_entries = [
        entry
        for entry in entries
        if entry.name.lower().replace("_", "-") in license_inventory.QT_OPEN_SOURCE_DISTRIBUTIONS
    ]

    assert {entry.name.lower().replace("_", "-") for entry in qt_entries} == (
        license_inventory.QT_OPEN_SOURCE_DISTRIBUTIONS
    )
    for entry in qt_entries:
        assert entry.distribution_basis.startswith("GPL-3.0-only")
        assert license_inventory.QT_GPL_LICENSE in entry.copied_license_files
        assert license_inventory.QT_OPEN_SOURCE_NOTICE in entry.copied_license_files
        assert all("commercial" not in filename.casefold() for filename in entry.copied_license_files)

    report = (output_dir / "DEPENDENCIES.md").read_text(encoding="utf-8")
    assert "open-source option selected by Cellonaut" in report


def test_cli_fails_when_a_dependency_has_no_license_file(tmp_path, monkeypatch):
    entry = license_inventory.LicenseInventoryEntry(
        name="missing-license",
        version="1.0",
        license_summary="Not declared in package metadata",
        distribution_basis="Not declared in package metadata",
        source_url="",
        copied_license_files=(),
    )
    monkeypatch.setattr(license_inventory, "write_license_inventory", lambda _output: [entry])

    assert license_inventory.main(["--output", str(tmp_path / "THIRD_PARTY_LICENSES")]) == 1
