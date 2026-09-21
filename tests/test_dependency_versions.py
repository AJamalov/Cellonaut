from __future__ import annotations

from cellonaut.dependency_versions import collect_key_package_versions


def test_collect_key_package_versions_includes_runtime_stack():
    versions = collect_key_package_versions()

    assert "PySide6" in versions
    assert "numpy" in versions
    assert "cellpose" in versions
    assert "torch" in versions
