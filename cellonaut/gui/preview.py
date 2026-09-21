"""Composition root for interactive preview behavior.

Preview responsibilities live in focused loading, rendering, workflow, layer,
artifact, filter, adjustment, normalization, and snapshot mixins. The composed
class provides the complete preview behavior used by the main window.
"""

import tifffile as tifffile

from cellonaut.gui.preview_artifacts import CellonautGuiPreviewArtifactsMixin
from cellonaut.gui.preview_filter import CellonautGuiPreviewFilterMixin
from cellonaut.gui.preview_layers import CellonautGuiPreviewLayersMixin
from cellonaut.gui.preview_loading import CellonautGuiPreviewLoadingMixin
from cellonaut.gui.preview_mask_adjustment import CellonautGuiPreviewMaskAdjustmentMixin
from cellonaut.gui.preview_normalization import CellonautGuiPreviewNormalizationMixin
from cellonaut.gui.preview_rendering import CellonautGuiPreviewRenderingMixin
from cellonaut.gui.preview_snapshot import CellonautGuiPreviewSnapshotMixin
from cellonaut.gui.preview_workflow import CellonautGuiPreviewWorkflowMixin


class CellonautGuiPreviewMixin(
    CellonautGuiPreviewArtifactsMixin,
    CellonautGuiPreviewLayersMixin,
    CellonautGuiPreviewSnapshotMixin,
    CellonautGuiPreviewNormalizationMixin,
    CellonautGuiPreviewMaskAdjustmentMixin,
    CellonautGuiPreviewFilterMixin,
    CellonautGuiPreviewRenderingMixin,
    CellonautGuiPreviewLoadingMixin,
    CellonautGuiPreviewWorkflowMixin,
):
    """Compose the complete preview API used by the main window and tests."""
