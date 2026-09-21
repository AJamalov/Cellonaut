"""Central stylesheet builders for Cellonaut's Qt interface.

Keeping visual tokens here makes the main GUI code focus on behavior while the
release UI can stay consistent throughout the Windows application.
"""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from cellonaut.config.defaults import DEFAULT_THEME
from cellonaut.gui.ui_tokens import CONTROLS, TYPOGRAPHY

Theme = dict[str, str]

_DARK_THEME_BASE: Theme = {
    "mode": "dark",
    "window": "#252526",
    "surface": "#2b2b2e",
    "surface_alt": "#313135",
    "interactive_surface": "#4a4a52",
    "interactive_surface_hover": "#585862",
    "base": "#1e1e1e",
    "text": "#f3f3f3",
    "muted": "#c8c8c8",
    "border": "#3b3d43",
    "danger": "#d96c6c",
    "danger_hover": "#e27d7d",
    "danger_surface": "rgba(217, 108, 108, 0.3)",
    "danger_outline": "#f0a0a0",
    "success": "#4caf50",
    "success_hover": "#5bc460",
    "success_surface": "rgba(76, 175, 80, 0.3)",
    "success_outline": "#9ee6a4",
    "warning": "#d8a657",
    "warning_surface": "rgba(216, 166, 87, 0.3)",
    "inactive_surface": "#202126",
    "inactive_border": "#33363d",
    "inactive_text": "#a8adb7",
}

THEMES: dict[str, Theme] = {
    "light_blue": {
        "mode": "light",
        "window": "#f4f6f9",
        "surface": "#ffffff",
        "surface_alt": "#f7f9fc",
        "interactive_surface": "#edf3fa",
        "interactive_surface_hover": "#e2ebf6",
        "base": "#ffffff",
        "text": "#182230",
        "muted": "#5f6b7a",
        "border": "#d7dee8",
        "accent": "#1769d2",
        "accent_hover": "#0f5dbd",
        "accent_surface": "rgba(23, 105, 210, 0.12)",
        "danger": "#c43d3d",
        "danger_hover": "#ae3030",
        "danger_surface": "rgba(196, 61, 61, 0.12)",
        "danger_outline": "#a92f2f",
        "success": "#24823c",
        "success_hover": "#1d7133",
        "success_surface": "rgba(36, 130, 60, 0.12)",
        "success_outline": "#247338",
        "warning": "#9a6114",
        "warning_surface": "rgba(154, 97, 20, 0.13)",
        "inactive_surface": "#edf0f4",
        "inactive_border": "#dce1e8",
        "inactive_text": "#7b8796",
    },
    "dark_blue": {
        **_DARK_THEME_BASE,
        "accent": "#2a82da",
        "accent_hover": "#3d92e8",
        "accent_surface": "rgba(42, 130, 218, 0.18)",
    },
    "dark_teal": {
        **_DARK_THEME_BASE,
        "accent": "#1fa3a3",
        "accent_hover": "#2ab8b8",
        "accent_surface": "rgba(31, 163, 163, 0.18)",
    },
    "dark_purple": {
        **_DARK_THEME_BASE,
        "accent": "#8b5cf6",
        "accent_hover": "#9a6cff",
        "accent_surface": "rgba(139, 92, 246, 0.18)",
    },
}


# Assemble independent style sections so each UI area can evolve without turning
# the application stylesheet into one difficult-to-review block.
def build_app_stylesheet(theme: Mapping[str, str]) -> str:
    return "\n".join(
        [
            build_base_stylesheet(theme),
            build_panel_stylesheet(theme),
            build_button_stylesheet(theme),
            build_input_stylesheet(theme),
            build_tab_stylesheet(theme),
            build_status_stylesheet(theme),
            build_table_stylesheet(theme),
            build_help_stylesheet(theme),
            build_structure_stylesheet(theme),
        ]
    )


# Application-wide defaults remain deliberately small so native widget behavior is
# retained unless Cellonaut needs a consistent cross-platform color.
def build_base_stylesheet(theme: Mapping[str, str]) -> str:
    return f"""
    QMainWindow {{
        background-color: {theme['window']};
        color: {theme['text']};
    }}

    QWidget {{
        color: {theme['text']};
        font-size: {TYPOGRAPHY.body_pt}pt;
    }}

    QWidget:disabled {{
        color: {theme['inactive_text']};
    }}

    QToolTip {{
        background-color: {theme['surface']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        padding: 4px;
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}

    QScrollBar::handle:vertical {{
        background: {theme['border']};
        border-radius: 4px;
        min-height: 28px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {theme['muted']};
    }}

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{
        background: transparent;
        border: none;
        height: 0px;
    }}

    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 2px;
    }}

    QScrollBar::handle:horizontal {{
        background: {theme['border']};
        border-radius: 4px;
        min-width: 28px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: {theme['muted']};
    }}

    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal,
    QScrollBar::sub-page:horizontal {{
        background: transparent;
        border: none;
        width: 0px;
    }}
    """


# Use shared style properties so panels and dialogs do not need separate styles.
def build_panel_stylesheet(theme: Mapping[str, str]) -> str:
    return f"""
    QFrame[uiRole="card"],
    QFrame[card="true"] {{
        border: 1px solid {theme['border']};
        border-radius: 5px;
        background-color: {theme['surface']};
    }}

    QFrame[uiRole="toolbar"] {{
        border: 1px solid {theme['border']};
        border-radius: 5px;
        background-color: {theme['surface']};
    }}

    QFrame[uiRole="flatPanel"] {{
        border: none;
        background-color: transparent;
    }}

    QFrame[previewToolPage="true"] {{
        border: none;
        background-color: transparent;
    }}

    QFrame#PreviewToolsContext {{
        border: 1px solid {theme['border']};
        border-radius: {CONTROLS.radius}px;
        background-color: {theme['surface_alt']};
    }}

    QGroupBox[previewToolSection="true"],
    QGroupBox[previewToolFooter="true"] {{
        border: none;
        border-top: 1px solid {theme['border']};
        border-radius: 0px;
        margin-top: 14px;
        padding-top: 8px;
        background-color: transparent;
    }}

    QGroupBox[previewToolSection="true"]::title,
    QGroupBox[previewToolFooter="true"]::title {{
        subcontrol-origin: margin;
        left: 0px;
        padding: 0 8px 0 0;
        color: {theme['text']};
        background-color: {theme['surface']};
    }}

    QGroupBox {{
        border: 1px solid {theme['border']};
        border-radius: 5px;
        margin-top: 10px;
        padding-top: 8px;
        background-color: {theme['surface']};
        font-weight: 600;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        background-color: {theme['surface']};
    }}

    QWidget[uiRole="formLabelPanel"],
    QLabel[uiRole="formLabel"],
    QLabel[uiRole="plainText"],
    QLabel[uiRole="taskText"],
    QLabel[uiRole="mutedLabel"],
    QLabel[uiRole="statusText"] {{
        background: transparent;
        border: none;
    }}

    QLabel[uiRole="formLabel"] {{
        color: {theme['text']};
    }}

    QLabel[uiRole="mutedLabel"] {{
        color: {theme['muted']};
        font-size: {TYPOGRAPHY.caption_pt}pt;
    }}

    QLabel[uiRole="sectionTitle"] {{
        font-size: {TYPOGRAPHY.workspace_title_pt}pt;
        font-weight: 600;
    }}

    QLabel[uiRole="previewToolContext"] {{
        color: {theme['muted']};
        background-color: {theme['surface_alt']};
        border: none;
        border-radius: {CONTROLS.radius}px;
        padding: 6px 8px;
        font-size: {TYPOGRAPHY.caption_pt}pt;
    }}

    QLabel[uiRole="previewToolSubheading"] {{
        color: {theme['muted']};
        background: transparent;
        border: none;
        font-size: {TYPOGRAPHY.caption_pt}pt;
        font-weight: {TYPOGRAPHY.semibold_weight};
    }}

    QLabel[uiRole="previewToolEmptyState"] {{
        color: {theme['muted']};
        background-color: {theme['base']};
        border: none;
        border-radius: {CONTROLS.radius}px;
        padding: 12px;
        font-size: {TYPOGRAPHY.caption_pt}pt;
    }}

    QLabel[uiRole="dialogHeader"] {{
        font-size: 18pt;
        font-weight: 800;
    }}

    QLabel[uiRole="cardTitle"] {{
        font-size: 12pt;
        font-weight: 800;
    }}

    QLabel[uiRole="statusPill"] {{
        border-radius: 10px;
        padding: 5px 10px;
        font-weight: 700;
    }}

    QLabel[uiRole="statusPill"][statusKind="success"] {{
        color: {theme['success']};
        background-color: {theme['success_surface']};
    }}

    QLabel[uiRole="statusPill"][statusKind="warning"] {{
        color: {theme['warning']};
        background-color: {theme['warning_surface']};
    }}

    QLabel[uiRole="statusPill"][statusKind="danger"] {{
        color: {theme['danger']};
        background-color: {theme['danger_surface']};
    }}

    QLabel[uiRole="statusPill"][statusKind="neutral"] {{
        color: {theme['accent']};
        background-color: {theme['accent_surface']};
    }}

    QLabel[messageState="ok"] {{
        color: {theme['success']};
        font-weight: 700;
    }}

    QLabel[messageState="warning"] {{
        color: {theme['warning']};
        font-weight: 700;
    }}
    """


# Use consistent button colors across tables, dialogs, and forms.
def build_button_stylesheet(theme: Mapping[str, str]) -> str:
    return f"""
    QPushButton {{
        background-color: {theme['surface_alt']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
        padding: 4px 12px;
        min-height: 24px;
    }}

    QPushButton[uiRole="compactFormButton"] {{
        padding: 3px 7px;
        min-height: 20px;
        font-size: 8.5pt;
        border-radius: 4px;
    }}

    QPushButton[uiRole="matrixActionButton"] {{
        padding: 3px 8px;
        min-height: 22px;
        border-radius: 4px;
    }}

    QPushButton[uiRole="compactFormButton"]:hover {{
        background-color: {theme['surface']};
    }}

    QPushButton:hover {{
        background-color: {theme['surface']};
        border-color: {theme['muted']};
    }}

    QPushButton[uiRole="processingActionButton"] {{
        padding: 0;
        min-width: 22px;
        max-width: 22px;
        min-height: 22px;
        max-height: 22px;
        border-radius: 3px;
    }}

    QPushButton[uiRole="processingEnableButton"] {{
        padding: 0;
        min-width: 18px;
        max-width: 18px;
        min-height: 18px;
        max-height: 18px;
        border-radius: 3px;
    }}

    QPushButton:focus {{
        border: 2px solid {theme['accent']};
    }}

    QPushButton:disabled {{
        background-color: {theme['inactive_surface']};
        border-color: {theme['inactive_border']};
        color: {theme['inactive_text']};
    }}

    QPushButton[mutedState="true"] {{
        background-color: {theme['base']};
        border-color: {theme['border']};
        color: {theme['muted']};
    }}

    QPushButton[mutedState="true"]:hover {{
        background-color: {theme['base']};
    }}

    QPushButton[role="primary"] {{
        background-color: {theme['accent']};
        border: 1px solid {theme['accent']};
        color: white;
        font-weight: 600;
    }}

    QPushButton[role="primary"]:hover {{
        background-color: {theme['accent_hover']};
    }}

    QPushButton[role="danger"] {{
        background-color: {theme['danger']};
        border: 1px solid {theme['danger']};
        color: white;
        font-weight: 600;
    }}

    QPushButton[role="danger"]:hover {{
        background-color: {theme['danger_hover']};
    }}

    QPushButton[role="primary"]:disabled,
    QPushButton[role="danger"]:disabled {{
        background-color: {theme['inactive_surface']};
        border-color: {theme['inactive_border']};
        color: {theme['inactive_text']};
    }}

    QPushButton[matrixToggle="true"] {{
        border-radius: 4px;
        padding: 5px 10px;
        color: white;
        font-weight: 700;
    }}

    QPushButton[matrixToggle="true"][matrixToggleState="on"] {{
        background-color: {theme['success_surface']};
        border: 2px solid {theme['success_outline']};
    }}

    QPushButton[matrixToggle="true"][matrixToggleState="on"]:hover {{
        background-color: {theme['success_hover']};
    }}

    QPushButton[matrixToggle="true"][matrixToggleState="off"] {{
        background-color: {theme['danger_surface']};
        border: 2px solid {theme['danger_outline']};
    }}

    QPushButton[matrixToggle="true"][matrixToggleState="off"]:hover {{
        background-color: {theme['danger_hover']};
    }}

    QPushButton[matrixToggle="true"][matrixToggleState="unavailable"] {{
        background-color: {theme['inactive_surface']};
        border: 1px solid {theme['inactive_border']};
        color: {theme['inactive_text']};
    }}
    """


# Style validation and disabled states at the input level so every form reports
# problems consistently without individual widgets choosing colors.
def build_input_stylesheet(theme: Mapping[str, str]) -> str:
    return f"""
    QLineEdit,
    QComboBox,
    QPlainTextEdit,
    QTextEdit,
    QTableWidget,
    QListWidget,
    QTreeView {{
        background-color: {theme['base']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
        padding: 4px 8px;
        selection-background-color: {theme['accent']};
        selection-color: white;
    }}

    QLineEdit:focus,
    QComboBox:focus,
    QPlainTextEdit:focus,
    QTextEdit:focus,
    QTableWidget:focus,
    QListWidget:focus,
    QTreeView:focus {{
        border: 2px solid {theme['accent']};
    }}

    QLineEdit,
    QComboBox {{
        min-height: 24px;
    }}

    QLineEdit[validationState="invalid"],
    QComboBox[validationState="invalid"] {{
        border: 1px solid {theme['danger']};
        background-color: {theme['danger_surface']};
    }}

    QLineEdit[validationState="warning"],
    QComboBox[validationState="warning"] {{
        border: 1px solid {theme['warning']};
        background-color: {theme['warning_surface']};
    }}

    QLineEdit[mutedState="true"],
    QComboBox[mutedState="true"] {{
        color: {theme['muted']};
        background-color: {theme['surface']};
        border-color: {theme['border']};
    }}

    QCheckBox[mutedState="true"] {{
        color: {theme['muted']};
        background-color: transparent;
    }}

    QCheckBox[mutedState="true"]::indicator,
    QCheckBox::indicator:disabled {{
        width: 13px;
        height: 13px;
        border: 1px solid {theme['surface_alt']};
        background-color: {theme['surface']};
    }}

    QCheckBox[mutedState="true"]::indicator:checked,
    QCheckBox::indicator:checked:disabled {{
        border-color: {theme['border']};
        background-color: {theme['surface_alt']};
    }}

    QMenu {{
        background-color: {theme['base']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
    }}

    QMenu::item {{
        padding: 5px 28px 5px 12px;
    }}

    QMenu::item:selected {{
        background-color: {theme['accent']};
        color: white;
    }}
    """


# Adjust spacing for each navigation area while keeping selection styling consistent.
def build_tab_stylesheet(theme: Mapping[str, str]) -> str:
    return f"""
    /* Conservative default tabs */
    QTabWidget::pane {{
        border: 1px solid {theme['border']};
        background-color: {theme['surface']};
    }}

    QTabBar::tab {{
        background: {theme['surface_alt']};
        border: 1px solid {theme['border']};
        padding: 5px 10px;
        margin-right: 2px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
    }}

    QTabBar::tab:selected {{
        background: {theme['accent']};
        color: white;
        font-weight: 600;
    }}

    QTabWidget#PreviewToolsTabs::pane {{
        border: none;
        border-top: 1px solid {theme['border']};
        background-color: {theme['surface']};
        top: -1px;
    }}

    QTabWidget#PreviewToolsTabs QTabBar::tab {{
        min-height: 20px;
        padding: 7px 14px;
        margin-right: 4px;
        border-color: transparent;
        border-bottom: 2px solid transparent;
        border-radius: 0px;
        background-color: transparent;
        color: {theme['muted']};
    }}

    QTabWidget#PreviewToolsTabs QTabBar::tab:hover {{
        background-color: {theme['surface_alt']};
        color: {theme['text']};
    }}

    QTabWidget#PreviewToolsTabs QTabBar::tab:selected {{
        background-color: transparent;
        color: {theme['text']};
        border-bottom-color: {theme['accent']};
        font-weight: {TYPOGRAPHY.semibold_weight};
    }}

    /* Persistent image workspace */
    QWidget#MainImageWorkspace {{
        border: none;
        background-color: {theme['surface']};
    }}

    QFrame#PreviewInspector {{
        background-color: {theme['surface_alt']};
        border: none;
        border-left: 1px solid {theme['border']};
    }}

    QFrame#PreviewToolStrip {{
        background-color: {theme['surface']};
        border: none;
        border-right: 1px solid {theme['border']};
    }}

    QLabel[uiRole="previewToolCategory"] {{
        color: {theme['muted']};
        background-color: transparent;
        border: none;
        font-size: {TYPOGRAPHY.caption_pt}pt;
        font-weight: {TYPOGRAPHY.medium_weight};
    }}

    QFrame#PreviewHeaderToolbar {{
        background-color: transparent;
        border: none;
    }}

    QWidget[uiRole="previewToolbarGroup"] {{
        background-color: transparent;
        border: none;
    }}

    QWidget[uiRole="previewLayerRow"] {{
        background-color: transparent;
        border: none;
    }}

    QListWidget#PreviewLayerList {{
        background-color: transparent;
        border: none;
        padding: 0px;
        outline: none;
    }}

    QListWidget#PreviewLayerList::item {{
        border: none;
        border-radius: {CONTROLS.radius}px;
    }}

    QListWidget#PreviewLayerList::item:selected {{
        background-color: {theme['accent_surface']};
        color: {theme['text']};
    }}

    QLabel[uiRole="previewLayerGroup"] {{
        color: {theme['muted']};
        background-color: transparent;
        border: none;
        font-size: {TYPOGRAPHY.caption_pt}pt;
        font-weight: {TYPOGRAPHY.bold_weight};
    }}

    QLabel[uiRole="previewLayerName"] {{
        color: {theme['text']};
        background-color: transparent;
        border: none;
    }}

    QWidget[uiRole="previewLayerOpacity"],
    QLabel[uiRole="previewLayerOpacityLabel"] {{
        color: {theme['inactive_text']};
        background-color: transparent;
        border: none;
        font-size: {TYPOGRAPHY.caption_pt}pt;
    }}

    QFrame#PreviewBottomToolbar {{
        background-color: transparent;
        border-top: 1px solid {theme['border']};
    }}

    QLabel#PreviewZoomLabel {{
        color: {theme['text']};
        background-color: {theme['base']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
        padding: 4px 8px;
    }}

    QWidget[pipelineSection="true"] {{
        background-color: transparent;
    }}

    QFrame#PipelineOverviewHeader {{
        background-color: transparent;
        border: none;
    }}

    QLabel[uiRole="pipelineOverviewTitle"] {{
        color: {theme['text']};
        background: transparent;
        border: none;
        font-size: {TYPOGRAPHY.body_pt}pt;
        font-weight: {TYPOGRAPHY.bold_weight};
    }}

    QLabel[uiRole="workspaceTitle"] {{
        color: {theme['text']};
        background: transparent;
        border: none;
        font-size: {TYPOGRAPHY.workspace_title_pt}pt;
        font-weight: {TYPOGRAPHY.bold_weight};
    }}

    QPushButton[uiRole="pipelineSectionHeader"] {{
        background-color: {theme['surface_alt']};
        border: 1px solid {theme['inactive_border']};
        border-radius: {CONTROLS.radius}px;
        padding: 0px;
        text-align: left;
    }}

    QPushButton[uiRole="pipelineSectionHeader"]:hover {{
        background-color: {theme['surface_alt']};
        border-color: {theme['border']};
    }}

    QPushButton[uiRole="pipelineSectionHeader"][expanded="true"] {{
        background-color: {theme['accent_surface']};
        border-color: transparent;
    }}

    QLabel[uiRole="pipelineStepBadge"] {{
        color: white;
        background-color: {theme['accent']};
        border: none;
        border-radius: 12px;
        font-weight: {TYPOGRAPHY.bold_weight};
    }}

    QLabel[uiRole="pipelineSectionTitle"] {{
        color: {theme['text']};
        background: transparent;
        border: none;
        font-size: {TYPOGRAPHY.label_pt}pt;
        font-weight: {TYPOGRAPHY.bold_weight};
    }}

    QLabel[uiRole="pipelineSectionSummary"],
    QLabel[uiRole="pipelineSectionChevron"] {{
        color: {theme['muted']};
        background: transparent;
        border: none;
    }}

    QLabel[uiRole="pipelineSectionSummary"] {{
        font-size: {TYPOGRAPHY.caption_pt}pt;
        color: {theme['inactive_text']};
    }}

    QWidget[uiRole="pipelineSectionContent"] {{
        background-color: {theme['surface']};
        border: 1px solid {theme['border']};
        border-top: none;
        border-bottom-left-radius: 5px;
        border-bottom-right-radius: 5px;
    }}

    QWidget[uiRole="pipelineSectionContent"][editorVisible="true"] {{
        border-top: 1px solid {theme['border']};
        border-radius: {CONTROLS.radius}px;
    }}

    QFrame#PipelineEditorHeader {{
        background-color: {theme['surface']};
        border: 1px solid {theme['border']};
        border-radius: {CONTROLS.radius}px;
    }}

    QLabel[uiRole="pipelineEditorTitle"] {{
        color: {theme['text']};
        background: transparent;
        border: none;
        font-size: {TYPOGRAPHY.section_title_pt}pt;
        font-weight: {TYPOGRAPHY.bold_weight};
    }}

    QLabel[uiRole="pipelineEditorDescription"] {{
        color: {theme['muted']};
        background: transparent;
        border: none;
    }}

    QWidget[uiRole="panelNotes"],
    QLabel[uiRole="panelNotesLabel"] {{
        background: transparent;
        border: none;
    }}

    QLabel[uiRole="panelNotesLabel"] {{
        color: {theme['text']};
        font-weight: {TYPOGRAPHY.semibold_weight};
    }}

    QGroupBox[embeddedPipelineSection="true"] {{
        border: none;
        margin-top: 0px;
        padding-top: 0px;
        background-color: transparent;
        font-weight: 400;
    }}

    /* Persistent vertical navigation and its stacked content surface. */
    QFrame#PrimaryNavigationRail {{
        background-color: {theme['surface']};
        border: none;
        border-right: 1px solid {theme['border']};
        border-radius: 0px;
    }}

    QPushButton[uiRole="primaryNavigation"] {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 6px 6px;
        min-height: 28px;
        color: {theme['muted']};
        text-align: left;
        font-size: {TYPOGRAPHY.body_pt}pt;
    }}

    QPushButton[uiRole="primaryNavigation"]:hover {{
        background-color: {theme['surface_alt']};
        border-color: transparent;
        color: {theme['text']};
    }}

    QPushButton[uiRole="primaryNavigation"]:checked {{
        background-color: {theme['accent_surface']};
        border-color: transparent;
        color: {theme['accent']};
        font-weight: {TYPOGRAPHY.bold_weight};
    }}

    QPushButton#GuidedTutorialButton[attention="true"] {{
        background-color: rgba(255, 212, 59, 0.18);
        border: 1px solid #FFD43B;
        color: #FFD43B;
        font-weight: {TYPOGRAPHY.bold_weight};
    }}

    QStackedWidget#LeftContentStack {{
        background-color: {theme['surface']};
        border: none;
    }}
    """


# Long-running task feedback remains visually stable because status labels and
# progress bars update frequently during processing.
def build_status_stylesheet(theme: Mapping[str, str]) -> str:
    return f"""
    QFrame#ApplicationHeader {{
        border-bottom: 1px solid {theme['border']};
        background-color: {theme['surface']};
    }}

    QLabel#ApplicationBrandName {{
        color: {theme['text']};
        font-size: {TYPOGRAPHY.brand_pt}pt;
        font-weight: {TYPOGRAPHY.semibold_weight};
        background: transparent;
        border: none;
    }}

    QLabel#ApplicationBrandIcon {{
        background: transparent;
        border: none;
        padding: 0px;
    }}

    QLabel#ApplicationBrandVersion {{
        color: {theme['muted']};
        font-size: 8.5pt;
        background: transparent;
        border: none;
    }}

    QToolButton {{
        background-color: {theme['base']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
        padding: 4px 8px;
        min-height: 24px;
        font-weight: 700;
    }}

    QToolButton:hover,
    QToolButton:checked {{
        background-color: {theme['surface_alt']};
        border-color: {theme['accent']};
    }}

    QToolButton#PanelNotesToggleButton {{
        padding: 0px;
        margin: 0px;
        min-width: 16px;
        max-width: 16px;
        min-height: 14px;
        max-height: 14px;
        border-radius: 3px;
        font-weight: 400;
    }}

    QFrame#ApplicationStatusBar {{
        border-top: 1px solid {theme['border']};
        background-color: {theme['surface']};
    }}

    QLabel[uiRole="statusText"] {{
        background: transparent;
        border: none;
    }}

    QLabel#CurrentTaskLabel {{
        background: transparent;
        border: none;
        color: {theme['muted']};
    }}

    QLabel#PipelineSpinnerLabel {{
        background: transparent;
        border: none;
        font-size: 10pt;
    }}

    QLabel#PipelineSpinnerLabel[status="idle"],
    QLabel#PipelineSpinnerLabel[status="done"] {{
        color: {theme['success']};
    }}

    QLabel#PipelineSpinnerLabel[status="running"] {{
        color: {theme['accent']};
    }}

    QLabel#PipelineSpinnerLabel[status="warning"] {{
        color: {theme['warning']};
    }}

    QLabel#PipelineSpinnerLabel[status="error"] {{
        color: {theme['danger']};
    }}

    QLabel#CurrentTaskTitleLabel {{
        background: transparent;
        border: none;
        color: {theme['text']};
        font-weight: 500;
    }}

    QLabel[status="running"] {{
        color: {theme['accent']};
        font-weight: 700;
        background: transparent;
    }}

    QLabel[status="done"] {{
        color: {theme['success']};
        font-weight: 700;
        background: transparent;
    }}

    QLabel[status="error"] {{
        color: {theme['danger']};
        font-weight: 700;
        background: transparent;
    }}

    QLabel[status="idle"] {{
        color: {theme['muted']};
        font-weight: 700;
        background: transparent;
    }}

    QProgressBar#PipelineProgressBar {{
        border: 1px solid {theme['border']};
        border-radius: 4px;
        background-color: {theme['base']};
        max-height: 8px;
    }}

    QProgressBar#PipelineProgressBar::chunk {{
        background-color: {theme['accent']};
        border-radius: 3px;
    }}
    """


# Use dynamic row properties for processing tables so disabling a step changes
# presentation without discarding its stored values.
def build_table_stylesheet(theme: Mapping[str, str]) -> str:
    return f"""
    QTableWidget[scientificTable="true"] {{
        gridline-color: transparent;
        alternate-background-color: {theme['surface_alt']};
        selection-background-color: {theme['accent_surface']};
        selection-color: {theme['text']};
    }}

    QLabel[status="warning"] {{
        color: {theme['warning']};
        font-weight: 700;
        background: transparent;
    }}

    QTableWidget[scientificTable="true"]::item {{
        border: none;
        padding: 2px 5px;
    }}

    QHeaderView::section {{
        background-color: {theme['surface_alt']};
        border: none;
        border-right: 1px solid {theme['border']};
        border-bottom: 1px solid {theme['border']};
        padding: 3px 6px;
        font-weight: 600;
    }}

    QWidget[uiRole="tableHeaderCell"] {{
        background-color: {theme['surface_alt']};
        border: none;
        border-right: 1px solid {theme['border']};
        border-bottom: 1px solid {theme['border']};
    }}

    QWidget[uiRole="matrixActionCell"] {{
        background-color: transparent;
    }}

    QWidget[uiRole="matrixActionCell"][mutedState="true"] {{
        background-color: {theme['inactive_surface']};
    }}

    QLabel[uiRole="tableHeaderCellLabel"] {{
        background-color: transparent;
        font-weight: 600;
        padding: 4px;
    }}

    QLabel[processingRole="drag_handle"] {{
        color: {theme['muted']};
        font-weight: 700;
    }}

    QLabel[processingRole="warning"] {{
        color: {theme['warning']};
        font-weight: 700;
    }}

    QLabel[uiRole="previewFilterSwatch"] {{
        background-color: {theme['danger']};
        border: 1px solid {theme['danger_outline']};
        border-radius: 4px;
    }}

    QToolButton[processingRole="step_type"],
    QToolButton[processingRole="mask_step_type"] {{
        background-color: {theme['surface_alt']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
        padding: 2px 8px;
        color: {theme['text']};
        text-align: left;
    }}

    QToolButton[processingRole="step_type"][mutedState="true"],
    QToolButton[processingRole="mask_step_type"][mutedState="true"] {{
        color: {theme['inactive_text']};
    }}

    QToolButton[processingRole="step_type"]::menu-button,
    QToolButton[processingRole="mask_step_type"]::menu-button {{
        width: 30px;
        border-left: 1px solid {theme['border']};
        background-color: {theme['surface']};
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
    }}

    QPushButton[processingRole="remove"] {{
        background-color: {theme['danger']};
        border: 1px solid {theme['danger_outline']};
        color: white;
        font-weight: 700;
        min-width: 22px;
        max-width: 22px;
        min-height: 22px;
        max-height: 22px;
        padding: 0px;
    }}

    QPushButton[processingRole="remove"]:hover {{
        background-color: {theme['danger_hover']};
    }}

    QLineEdit[processingRole="blocked_value"]:disabled {{
        background-color: {theme['inactive_surface']};
        border: 1px solid {theme['inactive_border']};
        border-radius: 5px;
        color: {theme['inactive_text']};
    }}

    QWidget[uiRole="tableHeaderCell"] QComboBox {{
        background-color: {theme['surface_alt']};
        border: 0;
        border-radius: 0;
    }}
    QWidget[processingRowState="inactive"],
    QLabel[processingRowState="inactive"],
    QLineEdit[processingRowState="inactive"],
    QComboBox[processingRowState="inactive"],
    QToolButton[processingRowState="inactive"],
    QPushButton[processingRowState="inactive"] {{
        background-color: {theme['inactive_surface']};
        border-color: {theme['inactive_border']};
        color: {theme['inactive_text']};
    }}

    QLineEdit[processingRowState="inactive"] {{
        background-color: {theme['inactive_surface']};
        border: 1px solid {theme['inactive_border']};
        color: {theme['inactive_text']};
    }}

    /* Keep processing-step menus visually prominent. These rules follow
       inactive-row styling so empty "Choose step" menus stay clearly clickable. */
    QToolButton[processingRole="step_type"],
    QToolButton[processingRole="mask_step_type"] {{
        background-color: {theme['interactive_surface']};
        border-color: {theme['muted']};
        color: {theme['text']};
        font-weight: 600;
    }}

    QToolButton[processingRole="step_type"]:hover,
    QToolButton[processingRole="mask_step_type"]:hover {{
        background-color: {theme['interactive_surface_hover']};
        border-color: {theme['text']};
    }}

    QToolButton[processingRole="step_type"]::menu-button,
    QToolButton[processingRole="mask_step_type"]::menu-button {{
        background-color: {theme['interactive_surface_hover']};
        border-left-color: {theme['muted']};
    }}

    QToolButton[processingRole="step_type"]:disabled,
    QToolButton[processingRole="mask_step_type"]:disabled {{
        background-color: {theme['inactive_surface']};
        border-color: {theme['inactive_border']};
        color: {theme['inactive_text']};
    }}

    QPushButton[processingRole="enabled"] {{
        border-radius: 4px;
        padding: 0px;
    }}

    QPushButton[processingRole="enabled"][enabledState="on"] {{
        background-color: {theme['success']};
        border: 2px solid {theme['success_outline']};
    }}

    QPushButton[processingRole="enabled"][enabledState="off"] {{
        background-color: {theme['danger']};
        border: 2px solid {theme['danger_outline']};
    }}
    """


# Help markers remain subtle and theme-aware so they are discoverable without
# competing with the controls they explain.
def build_help_stylesheet(theme: Mapping[str, str]) -> str:
    return f"""
    QLabel#HelpMarker {{
        color: {theme['accent']};
        border: 1px solid {theme['accent']};
        border-radius: 9px;
        font-weight: 700;
        background-color: transparent;
    }}

    QLabel#HelpMarker:hover {{
        background-color: {theme['surface_alt']};
    }}
    """


# Reuse semantic accent and success colors for dataset-structure choices so
# selection and successful detection remain visually distinct.
def build_structure_stylesheet(theme: Mapping[str, str]) -> str:
    return f"""
    QPushButton[structureCard="true"] {{
        text-align: left;
        background-color: {theme['surface_alt']};
        border: 1px solid {theme['border']};
        border-radius: 5px;
        padding: 10px;
        font-weight: 500;
    }}

    QPushButton[structureCard="true"]:hover {{
        border: 1px solid {theme['accent']};
        background-color: {theme['surface']};
    }}

    QPushButton[structureCard="true"][selected="true"] {{
        border: 2px solid {theme['accent']};
        background-color: {theme['surface']};
    }}

    QPushButton[structureCard="true"][detected="true"] {{
        border: 2px solid {theme['success']};
        background-color: {theme['surface']};
    }}

    QLabel[structurePreview="true"] {{
        border: 1px solid {theme['border']};
        border-radius: 4px;
        background-color: {theme['base']};
        padding: 8px;
        font-family: Consolas, monospace;
    }}
    """


# Apply palette and stylesheet together so switching themes cannot leave colors
# from the previous mode on widgets that rely on the Qt palette.
def apply_palette_from_theme(app: QApplication, theme_name: str) -> None:
    resolved_name = theme_name if theme_name in THEMES else DEFAULT_THEME
    theme = THEMES[resolved_name]
    app.setProperty("cellonautTheme", resolved_name)

    app.setStyle("Fusion")
    app_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    app_font.setPointSize(10)
    app.setFont(app_font)
    palette = QPalette()

    palette.setColor(QPalette.ColorRole.Window, QColor(theme["window"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(theme["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(theme["base"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(theme["surface"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(theme["text"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(theme["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(theme["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(theme["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Link, QColor(theme["accent"]))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(theme["accent_hover"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(theme["inactive_text"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(theme["inactive_text"]))

    app.setPalette(palette)
    app.setStyleSheet(build_app_stylesheet(theme))


# Fall back to the default theme when a saved theme name is invalid.
def apply_theme(app: QApplication, theme: str) -> None:
    apply_palette_from_theme(app, (theme or DEFAULT_THEME).strip().lower())
