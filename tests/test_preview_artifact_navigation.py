from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cellonaut.gui.state import PreviewState  # noqa: E402
from cellonaut.gui.preview import CellonautGuiPreviewMixin
from cellonaut.gui.preview_artifacts import PREVIEW_ARTIFACT_SELECTOR_LABELS


class TextStub:
    def __init__(self, value: str):
        self.value = value

    def get(self) -> str:
        return self.value


class LabelStub:
    def __init__(self):
        self.value = ""
        self.tooltip = ""

    def setText(self, value: str):
        self.value = value

    def text(self) -> str:
        return self.value

    def setToolTip(self, value: str):
        self.tooltip = value


class ButtonStub:
    def __init__(self):
        self.enabled = None
        self.checked = None

    def setEnabled(self, value: bool):
        self.enabled = bool(value)

    def blockSignals(self, _value: bool):
        pass

    def setChecked(self, value: bool):
        self.checked = bool(value)


class ArtifactSelectorStub:
    def __init__(self):
        self.items = list(PREVIEW_ARTIFACT_SELECTOR_LABELS)
        self.current_index = 0

    def blockSignals(self, _value: bool):
        pass

    def count(self) -> int:
        return len(self.items)

    def itemData(self, index: int) -> str:
        return self.items[index]

    def setCurrentIndex(self, index: int):
        self.current_index = int(index)

    def currentData(self) -> str:
        return self.items[self.current_index]


class PreviewArtifactHarness(CellonautGuiPreviewMixin):
    def __init__(self, output_dir: Path):
        self.preview_state = PreviewState()
        self.output_dir = TextStub(str(output_dir))
        self.preview_info_label = LabelStub()
        self.preview_sample_nav_label = LabelStub()
        self.preview_artifact_nav_label = LabelStub()
        self.preview_file_title_label = LabelStub()
        self.preview_prev_sample_button = ButtonStub()
        self.preview_next_sample_button = ButtonStub()
        self.preview_prev_artifact_button = ButtonStub()
        self.preview_next_artifact_button = ButtonStub()
        self.preview_artifact_selector = ArtifactSelectorStub()
        self.image_definitions = [
            {"name": "GFP", "folder": "GFP"},
            {"name": "RFP", "folder": "RFP"},
            {"name": "Hmg2 mask", "folder": "Hmg2 mask"},
        ]
        self.preview_state.file_path = ""
        self.opened_paths = []
        self.fit_count = 0

    def preview_file(self, file_path: str):
        self.opened_paths.append(file_path)
        self.preview_state.file_path = file_path
        self.set_preview_file_title(file_path)
        self.preview_info_label.setText(f"File: {Path(file_path).name}")

    def fit_preview_image(self):
        self.fit_count += 1


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def test_overlay_preview_button_groups_known_channel_outputs_by_sample(tmp_path: Path):
    results = tmp_path / "Results"
    touch(results / "Overlays" / "TIFF Overlays" / "sample_a_GFP_combined_overlay.tif")
    touch(results / "Overlays" / "TIFF Overlays" / "sample_a_RFP_combined_overlay.tif")
    touch(results / "Overlays" / "TIFF Overlays" / "sample_b_GFP_combined_overlay.tif")
    preview = PreviewArtifactHarness(tmp_path)

    preview.select_preview_artifact_view("overlay")

    assert [entry["sample"] for entry in preview.preview_state.artifact_entries] == ["sample_a", "sample_b"]
    assert preview.preview_sample_nav_label.text() == "Sample: 1 / 2"
    assert preview.preview_artifact_nav_label.text() == "Overlay: 1 / 2"
    assert preview.preview_artifact_selector.currentData() == "overlay"
    assert preview.preview_prev_sample_button.enabled is False
    assert preview.preview_next_sample_button.enabled is True
    assert preview.preview_prev_artifact_button.enabled is False
    assert preview.preview_next_artifact_button.enabled is True
    assert preview.opened_paths[-1].endswith("sample_a_GFP_combined_overlay.tif")
    assert preview.preview_file_title_label.text() == "sample_a_GFP_combined_overlay.tif"
    assert "Overlay: 1 / 2" in preview.preview_info_label.text()


def test_preview_sample_arrows_step_within_selected_view(tmp_path: Path):
    results = tmp_path / "Results"
    touch(results / "Overlays" / "TIFF Overlays" / "sample_a_GFP_combined_overlay.tif")
    touch(results / "Overlays" / "TIFF Overlays" / "sample_b_GFP_combined_overlay.tif")
    preview = PreviewArtifactHarness(tmp_path)
    preview.select_preview_artifact_view("overlay")

    preview.step_preview_artifact_sample(1)

    assert preview.preview_sample_nav_label.text() == "Sample: 2 / 2"
    assert preview.preview_prev_sample_button.enabled is True
    assert preview.preview_next_sample_button.enabled is False
    assert preview.opened_paths[-1].endswith("sample_b_GFP_combined_overlay.tif")

    preview.step_preview_artifact_sample(-1)

    assert preview.preview_sample_nav_label.text() == "Sample: 1 / 2"
    assert preview.opened_paths[-1].endswith("sample_a_GFP_combined_overlay.tif")


def test_montage_preview_button_uses_processing_montage_sample_folders(tmp_path: Path):
    results = tmp_path / "Results"
    touch(results / "Processing Montages" / "sample_a" / "sample_a_GFP_segmentation_montage.png")
    touch(results / "Processing Montages" / "sample_b" / "sample_b_GFP_segmentation_montage.png")
    preview = PreviewArtifactHarness(tmp_path)

    preview.select_preview_artifact_view("montage")

    assert [entry["sample"] for entry in preview.preview_state.artifact_entries] == ["sample_a", "sample_b"]
    assert preview.preview_sample_nav_label.text() == "Sample: 1 / 2"
    assert preview.preview_artifact_nav_label.text() == "Montage: 1 / 1"
    assert preview.preview_artifact_selector.currentData() == "montage"
    assert preview.opened_paths[-1].endswith("sample_a_GFP_segmentation_montage.png")


def test_artifact_arrows_step_between_multiple_montages_for_current_sample(tmp_path: Path):
    results = tmp_path / "Results"
    touch(results / "Processing Montages" / "sample_a" / "sample_a_GFP_segmentation_montage.png")
    touch(results / "Processing Montages" / "sample_a" / "sample_a_GFP_measurement_montage.png")
    touch(results / "Processing Montages" / "sample_b" / "sample_b_GFP_segmentation_montage.png")
    preview = PreviewArtifactHarness(tmp_path)
    preview.select_preview_artifact_view("montage")

    preview.step_preview_artifact_file(1)

    assert preview.preview_sample_nav_label.text() == "Sample: 1 / 2"
    assert preview.preview_artifact_nav_label.text() == "Montage: 2 / 2"
    assert preview.preview_prev_artifact_button.enabled is True
    assert preview.preview_next_artifact_button.enabled is False
    assert preview.opened_paths[-1].endswith("sample_a_GFP_segmentation_montage.png")

    preview.step_preview_artifact_sample(1)

    assert preview.preview_sample_nav_label.text() == "Sample: 2 / 2"
    assert preview.preview_artifact_nav_label.text() == "Montage: 1 / 1"
    assert preview.opened_paths[-1].endswith("sample_b_GFP_segmentation_montage.png")


def test_artifact_arrows_sync_to_currently_open_montage_before_stepping(tmp_path: Path):
    results = tmp_path / "Results"
    touch(results / "Processing Montages" / "sample_1" / "sample_1_GFP_measurement_montage.png")
    touch(results / "Processing Montages" / "sample_1" / "sample_1_GFP_segmentation_montage.png")
    sample_3_first = results / "Processing Montages" / "sample_3" / "sample_3_GFP_measurement_montage.png"
    sample_3_second = results / "Processing Montages" / "sample_3" / "sample_3_GFP_segmentation_montage.png"
    touch(sample_3_first)
    touch(sample_3_second)
    preview = PreviewArtifactHarness(tmp_path)
    preview.select_preview_artifact_view("montage")

    preview.preview_state.file_path = str(sample_3_first)
    preview.preview_state.artifact_index = 0
    preview.preview_state.artifact_file_index = 0

    preview.step_preview_artifact_file(1)

    assert preview.preview_sample_nav_label.text() == "Sample: 2 / 2"
    assert preview.preview_artifact_nav_label.text() == "Montage: 2 / 2"
    assert preview.opened_paths[-1] == str(sample_3_second)


def test_preview_artifact_selector_browses_png_previews(tmp_path: Path):
    results = tmp_path / "Results"
    touch(results / "Overlays" / "PNG" / "sample_a_GFP_qc.png")
    touch(results / "Cells" / "PNG" / "sample_b_qc_overlay.png")
    preview = PreviewArtifactHarness(tmp_path)

    preview.select_preview_artifact_view("png")

    assert [entry["sample"] for entry in preview.preview_state.artifact_entries] == ["sample_a", "sample_b"]
    assert preview.preview_artifact_selector.currentData() == "png"
    assert preview.preview_artifact_nav_label.text() == "PNG: 1 / 1"
    assert preview.opened_paths[-1].endswith("sample_a_GFP_qc.png")


def test_preview_artifact_selector_browses_snapshots_in_one_group(tmp_path: Path):
    results = tmp_path / "Results"
    first = results / "Image Preview Tools" / "Snapshots" / "sample_a_snapshot.png"
    second = results / "Image Preview Tools" / "Snapshots" / "sample_b_layer_montage.png"
    touch(first)
    touch(second)
    preview = PreviewArtifactHarness(tmp_path)

    preview.select_preview_artifact_view("snapshot")

    assert [entry["sample"] for entry in preview.preview_state.artifact_entries] == ["Snapshots"]
    assert preview.preview_artifact_selector.currentData() == "snapshot"
    assert preview.preview_artifact_nav_label.text() == "Snapshot: 1 / 2"
    assert preview.opened_paths[-1] == str(first)
    assert preview.fit_count == 1

    preview.step_preview_artifact_file(1)
    assert preview.preview_artifact_nav_label.text() == "Snapshot: 2 / 2"
    assert preview.opened_paths[-1] == str(second)
    assert preview.fit_count == 2


def test_preview_artifact_selector_browses_mask_outputs_by_sample(tmp_path: Path):
    results = tmp_path / "Results"
    touch(results / "Masks" / "ProbabilityMaps" / "sample_a_Hmg2 mask_prob_classifier.tif")
    touch(results / "Masks" / "MaskImages" / "sample_a_Hmg2 mask_class1_thr_classifier.tif")
    touch(results / "Masks" / "BinaryMasks" / "sample_a_Hmg2_mask_binary.tif")
    preview = PreviewArtifactHarness(tmp_path)

    preview.select_preview_artifact_view("probability")
    assert [entry["sample"] for entry in preview.preview_state.artifact_entries] == ["sample_a"]
    assert preview.preview_artifact_nav_label.text() == "Weka probability map: 1 / 1"

    preview.select_preview_artifact_view("mask_image")
    assert [entry["sample"] for entry in preview.preview_state.artifact_entries] == ["sample_a"]
    assert preview.preview_artifact_nav_label.text() == "Weka threshold mask: 1 / 1"

    preview.select_preview_artifact_view("binary_mask")
    assert [entry["sample"] for entry in preview.preview_state.artifact_entries] == ["sample_a"]
    assert preview.preview_artifact_nav_label.text() == "Final binary mask: 1 / 1"
