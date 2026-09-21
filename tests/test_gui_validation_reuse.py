from __future__ import annotations

from types import SimpleNamespace

import pytest

from cellonaut.gui import validation
from cellonaut.gui.validation import CellonautGuiValidationMixin


class CheckBoxStub:
    def __init__(self, checked: bool):
        self.checked = checked

    def isChecked(self):
        return self.checked


class ValidationHarness(CellonautGuiValidationMixin):
    def __init__(self, classifier: str, reuse: bool):
        self.classifier = classifier
        self.reuse_existing_masks_checkbox = CheckBoxStub(reuse)

    def get_active_image_definitions(self):
        return [{"classifier": self.classifier}]


class MatrixWarningHarness(CellonautGuiValidationMixin):
    def __init__(self, image_definitions):
        self.image_definitions = image_definitions
        self.measurement_options = {}

    def get_active_image_definitions(self):
        return self.image_definitions


def test_missing_classifier_is_allowed_when_reusing_masks(tmp_path):
    gui = ValidationHarness(str(tmp_path / "missing.model"), reuse=True)

    assert gui.validate_classifier_paths() is True


def test_missing_classifier_is_blocked_without_mask_reuse(tmp_path):
    gui = ValidationHarness(str(tmp_path / "missing.model"), reuse=False)

    assert gui.validate_classifier_paths() is False


def test_combined_mask_does_not_emit_missing_classifier_warning():
    gui = MatrixWarningHarness(
        [
            {
                "name": "POI",
                "mask_relationships": {"Hmg2+BFP": True},
            },
            {
                "name": "Hmg2+BFP",
                "mask_source_mode": "Combined masks",
                "combined_mask_sources": ["Hmg2", "BFP"],
            },
        ]
    )

    assert gui.get_analysis_matrix_warnings() == []


def test_cellpose_only_measurement_does_not_require_another_mask_relationship():
    gui = MatrixWarningHarness(
        [
            {
                "name": "GFP",
                "analysis_cell_segmentation_enabled": True,
                "mask_relationships": {},
            }
        ]
    )

    assert gui.get_analysis_matrix_warnings() == []


def test_mask_inside_cell_choice_warns_without_assigned_mask():
    gui = MatrixWarningHarness(
        [{"name": "GFP", "analysis_cell_segmentation_enabled": True, "mask_relationships": {}}]
    )
    gui.measurement_options = {"positive_area_in_cell": True}

    assert "no measurement mask is assigned" in " ".join(gui.get_analysis_matrix_warnings())


def test_whole_cell_choice_requires_cellpose_but_not_an_assigned_mask():
    gui = MatrixWarningHarness(
        [{"name": "GFP", "analysis_cell_segmentation_enabled": False, "mask_relationships": {}}]
    )
    gui.measurement_options = {"cell_median": True}

    assert "Cellpose masks are disabled" in " ".join(gui.get_analysis_matrix_warnings())

    gui.image_definitions[0]["analysis_cell_segmentation_enabled"] = True
    assert gui.get_analysis_matrix_warnings() == []


class MaskReportHarness(CellonautGuiValidationMixin):
    def __init__(self, cfg):
        self.cfg = cfg

    def collect_config(self):
        return self.cfg


@pytest.mark.parametrize("samples, checked, unchecked, success", [
    ([], 0, 0, False),
    ([{"sample": "blocked", "ready": False, "missing_required": ["Missing TIFF"]}], 0, 1, False),
    ([{"sample": "ready", "ready": True, "reuse_mask_warnings": []}], 1, 0, True),
    ([{"sample": "ready", "ready": True, "reuse_mask_warnings": []},
      {"sample": "blocked", "ready": False, "missing_required": ["Missing TIFF"]}], 1, 1, False),
    ([{"sample": "missing-mask", "ready": True, "reuse_mask_warnings": ["Missing mask"]}], 1, 0, False),
])
def test_mask_reuse_report_distinguishes_checked_and_blocked_samples(monkeypatch, samples, checked, unchecked, success):
    cfg = SimpleNamespace(reuse_existing_masks=True, mask_source_dir="saved-run", output_dir="output")
    gui = MaskReportHarness(cfg)
    reports = []
    monkeypatch.setattr(validation, "analyze_sample_readiness", lambda _cfg: {"samples": samples})
    monkeypatch.setattr(validation.QMessageBox, "information", lambda _parent, _title, text: reports.append(text))

    gui.show_mask_source_report()

    assert f"Samples checked: {checked}" in reports[0]
    assert f"Samples not checked: {unchecked}" in reports[0]
    assert ("All configured imported/reusable masks were found" in reports[0]) is success
    expected = sum(sample.get("ready", False) and sample.get("reuse_mask_warnings") == [] for sample in samples)
    assert f"Samples with all expected masks: {expected}" in reports[0]
    if unchecked:
        assert "Missing TIFF" in reports[0]


def test_mask_reuse_report_requires_reuse_to_be_enabled(monkeypatch):
    gui = MaskReportHarness(SimpleNamespace(reuse_existing_masks=False))
    reports = []
    monkeypatch.setattr(validation.QMessageBox, "information", lambda _parent, _title, text: reports.append(text))
    monkeypatch.setattr(validation, "analyze_sample_readiness", lambda _cfg: pytest.fail("Must not scan with reuse off"))

    gui.show_mask_source_report()

    assert "Enable Reuse existing masks" in reports[0]
