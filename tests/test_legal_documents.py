from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_project_file(name: str) -> str:
    return (PROJECT_ROOT / name).read_text(encoding="utf-8")


def test_release_legal_and_citation_documents_are_present():
    for name in (
        "LICENSE",
        "COPYRIGHT",
        "THIRD_PARTY_NOTICES.md",
        "BUNDLED_COMPONENTS.md",
        "CITATIONS.md",
        "CITATION.cff",
        "SOURCE_AVAILABILITY.md",
    ):
        assert (PROJECT_ROOT / name).is_file()


def test_third_party_notice_matches_runtime_license_policy():
    notices = read_project_file("THIRD_PARTY_NOTICES.md")

    assert "GPL-3.0-only open-source option" in notices
    assert "No commercial license is claimed" in notices
    assert "THIRD_PARTY_LICENSES/DEPENDENCIES.md" in notices
    assert "DINOv3 and DINO-based" in notices


def test_public_documentation_stays_end_user_focused():
    readme = read_project_file("README.md")

    for maintainer_text in (
        "## Run From Source",
        "requirements-dev.txt",
        "python -m pytest",
        "release checklist",
    ):
        assert maintainer_text not in readme

    for removed_section in (
        "### Verify Windows downloads",
        "## First Analysis",
        "## Important Limitations",
        "## Version 1.0.1 Highlights",
    ):
        assert removed_section not in readme

    assert "**Tutorial**" in readme
    assert "**Help** tab" in readme
    assert "64-bit Windows only" in readme
    assert "macOS" not in readme
    assert "Linux" not in readme
    assert "-macos-" not in readme
    assert "-linux-" not in readme

    for removed_guide in (
        "dependency_security.md",
        "linux_release_validation.md",
        "macos_release_validation.md",
        "release_checklist.md",
        "windows_release_validation.md",
    ):
        assert not (PROJECT_ROOT / "docs" / removed_guide).exists()


def test_source_notice_identifies_release_location_and_copyleft_sources():
    notice = read_project_file("SOURCE_AVAILABILITY.md")

    assert "https://github.com/AJamalov/Cellonaut/releases" in notice
    assert "Qt/PySide" in notice
    assert "fastremap" in notice
    assert "fill-voids" in notice
    assert "THIRD_PARTY_LICENSES/DEPENDENCIES.md" in notice
    assert "installer build generates" in notice
    assert "not a generated file stored in this source checkout" in notice
    assert "BUNDLED_COMPONENTS.md" in notice
    assert "ImageScience is not" in notice
    assert "redistributed" in notice
    assert "Cellonaut-1.0.1-corresponding-sources.zip" in notice
    assert "corresponding_sources prepare" in notice
    for component in ("Qt", "PySide", "Fiji", "Bio-Formats", "Trainable Weka", "fastremap", "fill-voids"):
        assert component in notice


def test_bundled_component_inventory_records_versions_sources_and_release_blocker():
    inventory = read_project_file("BUNDLED_COMPONENTS.md")

    for component, version in (
        ("Fiji", "2.18.1-SNAPSHOT"),
        ("Azul Zulu OpenJDK", "21.0.7"),
        ("Bio-Formats", "8.5.0"),
        ("Trainable Weka Segmentation", "4.0.0"),
        ("ImageScience", "Optional; not bundled"),
    ):
        assert component in inventory
        assert version in inventory
    assert "DINO-based models are not distributed" in inventory
    assert "7c61431b5fbb078f3296754bd15d9f51b320f837" in inventory
    assert "e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2" in inventory
    assert "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667" in inventory
    assert "c691dc761719086b49a9f927e77d744ae4e5a816" in inventory
    assert "877c317e4e396381dc76e56c1539b24947f71dce" in inventory
    assert "a55f593c08ea3fd94e860a7cd1878f58ed1bff1b" in inventory
    assert "must not contain `imagescience.jar`" in inventory.lower()


def test_scientific_citations_include_workflow_specific_identifiers():
    citations = read_project_file("CITATIONS.md")

    for identifier in (
        "10.1101/2025.04.28.651001",
        "10.1038/s41592-020-01018-x",
        "10.1038/s41592-022-01663-4",
        "10.1038/s41592-025-02595-5",
        "10.1038/nmeth.2019",
        "10.1038/nmeth.2089",
        "10.1038/s41592-022-01655-4",
        "10.1186/s12859-017-1934-z",
        "10.1083/jcb.201004104",
        "10.1093/bioinformatics/btx180",
    ):
        assert identifier in citations


def test_cellpose_model_count_and_names_are_consistent_in_release_documents():
    documents = "\n".join(
        read_project_file(name)
        for name in (
            "README.md",
            "CITATIONS.md",
            "THIRD_PARTY_NOTICES.md",
            "BUNDLED_COMPONENTS.md",
        )
    )

    assert "four built-in Cellpose" not in documents
    assert "four built-in model weights" not in documents
    assert "`cpsam`" in documents
    assert "`cpsam_v2`" in documents
