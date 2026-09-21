from cellonaut.gui.ui_tokens import CONTROLS, SPACING, TYPOGRAPHY, WORKSPACE


def test_ui_tokens_define_one_consistent_desktop_scale():
    assert (SPACING.xs, SPACING.sm, SPACING.md) == (4, 8, 12)
    assert 4 <= CONTROLS.radius <= 6
    assert WORKSPACE.sidebar_width == 120
    assert WORKSPACE.header_height == 60
    assert WORKSPACE.pipeline_row_height == 72
    assert WORKSPACE.preview_tool_strip_width == 64
    assert WORKSPACE.preview_tool_button_size == 36
    assert WORKSPACE.preview_bottom_bar_height == 42
    assert WORKSPACE.inspector_min_width < WORKSPACE.inspector_max_width
    assert WORKSPACE.inspector_min_width == 190
    assert WORKSPACE.inspector_max_width == 220
    assert WORKSPACE.status_bar_height == 28
    assert TYPOGRAPHY.caption_pt < TYPOGRAPHY.body_pt < TYPOGRAPHY.section_title_pt
