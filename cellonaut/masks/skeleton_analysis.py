"""Fiji Analyze Skeleton measurements for a copy of the current recipe mask."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jpype.types import JBoolean

from cellonaut.io.imagej_runtime import get_java_classes, jimport
from cellonaut.io.writers import save_imagej_tiff


# Convert Java arrays explicitly because JPype numeric wrappers are not pandas-safe values.
def _sum_java_values(values: Any) -> int:
    if values is None:
        return 0
    return sum(int(value) for value in values)


# Work on a duplicate because Fiji's Skeletonize command modifies its input image in place.
def analyze_mask_skeleton(
    binary_mask: Any,
    skeleton_path: Path | None = None,
) -> dict[str, int]:
    classes = get_java_classes()
    duplicator = classes["Duplicator"]
    skeleton = duplicator().run(binary_mask)
    if skeleton is None:
        raise RuntimeError("Fiji could not duplicate the mask for skeleton analysis.")

    try:
        classes["IJ"].run(skeleton, "Skeletonize", "")
        if skeleton_path is not None and not save_imagej_tiff(
            skeleton_path,
            skeleton,
            classes["FileSaver"],
        ):
            raise RuntimeError(f"Fiji could not save the skeleton image: {skeleton_path}")
        try:
            analyze_skeleton_type = jimport("sc.fiji.analyzeSkeleton.AnalyzeSkeleton_")
        except Exception as exc:
            raise RuntimeError(
                "Fiji Analyze Skeleton is unavailable. Update Fiji and ensure "
                "the Analyze Skeleton plugin is installed."
            ) from exc

        analyzer = analyze_skeleton_type()
        analyzer.setup("", skeleton)
        result = analyzer.run(
            analyze_skeleton_type.NONE,
            JBoolean(False),
            JBoolean(False),
            None,
            JBoolean(True),
            JBoolean(False),
        )
        if result is None:
            return {
                "SkeletonCount": 0,
                "BranchCount": 0,
                "EndpointCount": 0,
                "JunctionCount": 0,
            }
        branches = result.getBranches()
        return {
            "SkeletonCount": len(branches) if branches is not None else 0,
            "BranchCount": _sum_java_values(branches),
            "EndpointCount": _sum_java_values(result.getEndPoints()),
            "JunctionCount": _sum_java_values(result.getJunctions()),
        }
    finally:
        skeleton.close()
