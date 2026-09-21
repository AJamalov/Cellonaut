from __future__ import annotations

from html import escape

from cellonaut.config.defaults import (
    CELLPOSE_MODEL_OPTIONS,
    DEFAULT_CELLPOSE_MODEL_TYPE,
    IMAGE_PROCESSING_STEP_DEFINITIONS,
    MASK_PROCESSING_STEP_DEFINITIONS,
    MEASUREMENT_METADATA,
)
from cellonaut.gui.help_content import HELP_PAGES


def help_text() -> str:
    return "\n".join(title + "\n" + html for title, html in HELP_PAGES)


def test_help_content_has_complete_task_focused_manual_structure():
    assert [title for title, _html in HELP_PAGES] == [
        "Introduction",
        "Input & ND2",
        "Channels & Masks",
        "Processing",
        "Mask Processing",
        "Trainable Weka Segmentation",
        "Cellpose",
        "Measurements",
        "Image Preview Tools",
        "Files & Results",
        "Presets & Settings",
        "Troubleshooting",
        "About",
    ]


def test_help_content_covers_every_configured_operation_and_measurement():
    text = help_text()
    for definition in [*IMAGE_PROCESSING_STEP_DEFINITIONS, *MASK_PROCESSING_STEP_DEFINITIONS]:
        assert f"<b>{definition['label']}</b>" in text
    for metadata in MEASUREMENT_METADATA.values():
        assert f"<b>{escape(str(metadata['label']))}" in text


def test_help_content_covers_major_gui_workflows_and_utilities():
    text = help_text()
    for expected in (
        "Import ND2 files...",
        "Check Setup",
        "Preview One Sample",
        "Run Pipeline",
        "Reuse existing masks",
        "Check existing masks",
        "Compare Runs",
        "Preview",
        "Export filtered CSV",
        "Layer montage PNG",
        "Open Cellonaut Data Folder",
        "Open Fiji Folder",
        "Export Selected Preset",
        "final file or folder",
        "Import Preset",
        "destination free space",
        "Critically low destinations",
        "RunState.json",
        "RunSummary.txt",
        "Click a file once to open it in Image Preview",
        "Recent images",
        "Mask Adjustments",
        "Cell Groups",
        "Weka threshold masks",
        "final binary masks",
        "Composite",
        "X/Y shifts",
        "combined dataset table",
        "<b>Automatic</b>",
        "<b>CPU only</b>",
        "compatible NVIDIA GPU and driver",
    ):
        assert expected in text


def test_help_content_does_not_duplicate_the_guided_tutorial():
    titles = [title for title, _html in HELP_PAGES]

    assert "Guided tutorial" not in titles

def test_help_content_preserves_scientific_and_legal_limits():
    text = help_text()
    for expected in (
        "Multiple timepoints and XY positions use the first one",
        "RGB",
        "output folder <b>outside it</b>",
        "ImageScience must be installed separately",
        "GPL-3.0-or-later",
        "SOURCE_AVAILABILITY.md",
        "THIRD_PARTY_NOTICES.md",
        "DINO-based models and the DINOv3 runtime are not distributed",
    ):
        assert expected in text


def test_help_content_lists_supported_cellpose_models_and_avoids_removed_models():
    text = help_text()
    assert DEFAULT_CELLPOSE_MODEL_TYPE in CELLPOSE_MODEL_OPTIONS
    for model_name in CELLPOSE_MODEL_OPTIONS:
        assert f"<code>{model_name}</code>" in text
    assert "<code>cpdino</code>" not in text
    assert "<code>cpdino-vitb</code>" not in text
