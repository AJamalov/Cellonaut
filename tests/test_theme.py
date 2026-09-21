from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

from cellonaut.config.defaults import DEFAULT_THEME
from cellonaut.gui.theme import THEMES, apply_theme, build_app_stylesheet
from cellonaut.gui.widgets import MatrixToggleButton


pytestmark = pytest.mark.gui


# Narrow PySide's general QCoreApplication return type once so individual tests
# can use QApplication-only palette and stylesheet methods without casts.
def get_application() -> QApplication:
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return instance
    return QApplication([])


def test_muted_checkbox_indicator_has_explicit_disabled_style():
    stylesheet = build_app_stylesheet(THEMES["dark_blue"])

    assert 'QCheckBox[mutedState="true"]::indicator' in stylesheet
    assert "QCheckBox::indicator:disabled" in stylesheet


def test_supported_light_and_dark_themes_are_available():
    assert list(THEMES) == ["light_blue", "dark_blue", "dark_teal", "dark_purple"]
    assert DEFAULT_THEME == "dark_blue"
    assert THEMES["light_blue"]["mode"] == "light"
    assert all(THEMES[name]["mode"] == "dark" for name in ("dark_blue", "dark_teal", "dark_purple"))


def test_matrix_toggle_button_styles_live_in_theme():
    stylesheet = build_app_stylesheet(THEMES["dark_blue"])

    assert 'QPushButton[matrixToggle="true"]' in stylesheet
    assert 'QPushButton[matrixToggle="true"][matrixToggleState="on"]' in stylesheet
    assert 'QPushButton[matrixToggle="true"][matrixToggleState="off"]' in stylesheet
    assert 'QPushButton[matrixToggle="true"][matrixToggleState="unavailable"]' in stylesheet


def test_processing_editor_controls_use_shared_theme_roles():
    stylesheet = build_app_stylesheet(THEMES["dark_teal"])

    assert 'QToolButton[processingRole="step_type"]' in stylesheet
    assert 'QToolButton[processingRole="mask_step_type"]' in stylesheet
    assert 'QPushButton[processingRole="remove"]' in stylesheet
    assert 'QLineEdit[processingRole="blocked_value"]:disabled' in stylesheet
    assert THEMES["dark_teal"]["inactive_surface"] in stylesheet
    assert 'QTableWidget[scientificTable="true"]' in stylesheet
    assert 'QTableWidget[scientificTable="true"]::item' in stylesheet
    assert "gridline-color: transparent" in stylesheet


def test_processing_step_menus_have_a_brighter_interactive_surface():
    theme = THEMES["dark_blue"]
    stylesheet = build_app_stylesheet(theme)

    assert 'QToolButton[processingRole="step_type"],' in stylesheet
    assert 'QToolButton[processingRole="mask_step_type"] {' in stylesheet
    assert theme["interactive_surface"] in stylesheet
    assert theme["interactive_surface_hover"] in stylesheet
    assert 'QToolButton[processingRole="mask_step_type"]:hover' in stylesheet


def test_every_custom_theme_supplies_semantic_state_colors():
    for theme_name, theme in THEMES.items():
        stylesheet = build_app_stylesheet(theme)

        assert theme["accent"] in stylesheet, theme_name
        assert theme["warning"] in stylesheet, theme_name
        assert theme["inactive_surface"] in stylesheet, theme_name
        assert theme["success_outline"] in stylesheet, theme_name
        assert theme["danger_outline"] in stylesheet, theme_name


def test_main_workspace_uses_stable_non_splitter_styling():
    stylesheet = build_app_stylesheet(THEMES["dark_blue"])

    assert "QSplitter#MainWorkspaceSplitter::handle" not in stylesheet


def test_application_shell_has_header_brand_and_footer_styles():
    stylesheet = build_app_stylesheet(THEMES["dark_blue"])

    assert "QFrame#ApplicationHeader" in stylesheet
    assert "QLabel#ApplicationBrandName" in stylesheet
    assert "QLabel#ApplicationBrandIcon" in stylesheet
    assert "QFrame#ApplicationStatusBar" in stylesheet


def test_pipeline_accordion_cards_have_header_and_summary_styles():
    stylesheet = build_app_stylesheet(THEMES["dark_blue"])

    assert 'QPushButton[uiRole="pipelineSectionHeader"]' in stylesheet
    assert 'QLabel[uiRole="pipelineStepBadge"]' in stylesheet
    assert 'QLabel[uiRole="pipelineSectionSummary"]' in stylesheet
    assert 'QWidget[uiRole="pipelineSectionContent"]' in stylesheet
    assert "QFrame#PipelineEditorHeader" in stylesheet
    assert 'QLabel[uiRole="pipelineEditorTitle"]' in stylesheet
    assert 'QWidget[uiRole="panelNotes"]' in stylesheet
    assert 'QLabel[uiRole="panelNotesLabel"]' in stylesheet
    assert "QToolButton#PanelNotesToggleButton" in stylesheet
    assert "QFrame#PipelineOverviewHeader" in stylesheet
    assert 'QLabel[uiRole="pipelineOverviewTitle"]' in stylesheet
    assert 'QLabel[uiRole="workspaceTitle"]' in stylesheet
    assert 'QWidget[uiRole="previewToolbarGroup"]' in stylesheet
    assert 'QLabel[uiRole="previewLayerGroup"]' in stylesheet
    assert 'QWidget[uiRole="previewLayerOpacity"]' in stylesheet
    assert "QTabWidget#PreviewToolsTabs::pane" in stylesheet
    assert 'QGroupBox[previewToolSection="true"]' in stylesheet
    assert 'QLabel[uiRole="previewToolEmptyState"]' in stylesheet
    assert "QLabel#PipelineSpinnerLabel" in stylesheet
    assert 'QLabel#PipelineSpinnerLabel[status="warning"]' in stylesheet
    assert 'QLabel[status="warning"]' in stylesheet


def test_preview_inspector_has_a_dedicated_card_surface():
    stylesheet = build_app_stylesheet(THEMES["dark_blue"])

    assert "QFrame#PreviewInspector" in stylesheet
    assert "QFrame#PreviewToolStrip" in stylesheet
    assert "QFrame#PreviewBottomToolbar" in stylesheet
    assert "QListWidget#PreviewLayerList::item:selected" in stylesheet
    assert "QLabel#PreviewZoomLabel" in stylesheet


def test_modern_navigation_focus_and_scrollbar_styles_are_present():
    stylesheet = build_app_stylesheet(THEMES["dark_blue"])

    assert "QFrame#PrimaryNavigationRail" in stylesheet
    assert 'QPushButton[uiRole="primaryNavigation"]:checked' in stylesheet
    assert "QStackedWidget#LeftContentStack" in stylesheet
    assert "QPushButton:focus" in stylesheet
    assert "QLineEdit:focus" in stylesheet
    assert "QScrollBar::handle:vertical" in stylesheet


def test_desktop_surfaces_use_restrained_corner_radii_and_flat_navigation():
    stylesheet = build_app_stylesheet(THEMES["light_blue"])

    assert stylesheet.count("border-radius: 10px") == 1  # Semantic status pill only.
    assert "border-radius: 8px" not in stylesheet
    assert "border-radius: 7px" not in stylesheet
    assert "QFrame#PrimaryNavigationRail" in stylesheet
    assert "border-right: 1px solid" in stylesheet
    assert 'QWidget[uiRole="pipelineSectionContent"][editorVisible="true"]' in stylesheet
    assert 'QLabel[uiRole="pipelineSectionSummary"]' in stylesheet
    assert "font-size: 8.5pt" in stylesheet


def test_theme_name_is_normalized_and_unknown_names_use_default():
    app = get_application()
    original_style = app.style().objectName()
    original_palette = QPalette(app.palette())
    original_stylesheet = app.styleSheet()
    original_font = app.font()
    try:
        apply_theme(app, "  DARK_TEAL  ")
        assert app.styleSheet() == build_app_stylesheet(THEMES["dark_teal"])
        assert app.font().family()
        assert app.font().pointSize() == 10

        apply_theme(app, "not-a-theme")
        assert app.styleSheet() == build_app_stylesheet(THEMES[DEFAULT_THEME])
    finally:
        app.setStyle(original_style)
        app.setPalette(original_palette)
        app.setStyleSheet(original_stylesheet)
        app.setFont(original_font)


def test_matrix_toggle_button_updates_theme_state_properties():
    app = get_application()
    button = MatrixToggleButton(False)
    try:
        assert button.property("matrixToggle") == "true"
        assert button.property("matrixToggleState") == "off"
        assert button.text() == "OFF"

        button.setChecked(True)

        assert button.property("matrixToggleState") == "on"
        assert button.text() == "ON"

        button.set_unavailable()

        assert button.property("matrixToggleState") == "unavailable"
        assert button.text() == "N/A"
    finally:
        button.deleteLater()
        app.processEvents()
