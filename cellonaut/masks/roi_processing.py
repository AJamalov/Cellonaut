"""Weka ROI creation, reuse, combination, and mask post-processing.

This module bridges Fiji/ImageJ objects with Cellonaut's Python pipeline. It
turns Weka probability maps into ROIs, combines them when requested, applies
mask processing steps, and prepares masks for overlays and measurements.
"""

from __future__ import annotations

from pathlib import Path

from cellonaut.results.artifacts import record_artifact, ArtifactResolver, ArtifactMetadataError
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from cellonaut.config.defaults import (
    IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    MASK_PROCESSING_STEP_ANALYZE_PARTICLES,
    MASK_PROCESSING_STEP_ANALYZE_SKELETON,
    MASK_PROCESSING_STEP_BINARY_CLOSE,
    MASK_PROCESSING_STEP_BINARY_DILATE,
    MASK_PROCESSING_STEP_BINARY_ERODE,
    MASK_PROCESSING_STEP_BINARY_FILL_HOLES,
    MASK_PROCESSING_STEP_BINARY_OPEN,
    MASK_PROCESSING_STEP_BINARY_OUTLINE,
    MASK_PROCESSING_STEP_BINARY_SKELETONIZE,
    MASK_PROCESSING_STEP_BINARY_WATERSHED,
    MASK_PROCESSING_STEP_TRANSLATE,
)
from cellonaut.io.image_io import imageplus_to_numpy_2d
from cellonaut.io.imagej_runtime import (
    format_weka_runtime_error,
    get_java_classes,
    load_weka_classifier,
    jimport,
)
from cellonaut.config.processing_steps import parse_binary_settings, parse_particle_settings, parse_translate_offsets
from cellonaut.io.writers import save_imagej_tiff
from cellonaut.masks.skeleton_analysis import analyze_mask_skeleton
from cellonaut.measurement.math import base_name_no_ext
from cellonaut.pipeline.cancellation import check_cancel
from cellonaut.pipeline.image_processing import (
    apply_imagej_processing_recipe_copy,
    imagej_processing_recipe_signature,
)
from cellonaut.pipeline.roi_defs import class_roi_key, mask_reference_image_keys, parse_probability_class_indices


# Disk loading must distinguish a valid zero-pixel mask from an unreadable file.
_EMPTY_THRESHOLD_MASK = object()


# Process a duplicate so background correction cannot alter another measurement path.
def subtract_background_copy(img, radius: float, options: str = ""):
    classes = get_java_classes()
    IJ = classes["IJ"]
    Duplicator = classes["Duplicator"]

    if img is None:
        return None
    dup = Duplicator().run(img)
    safe_radius = max(0.0, radius)
    try:
        IJ.run(dup, "Subtract Background...", f"rolling={safe_radius}{options}")
    except Exception:
        try:
            dup.close()
        except Exception:
            pass
        raise
    return dup


# Fiji versions expose a few measurement constants under different names.
def _measurement_flag(Measurements, *names: str) -> int:
    for name in names:
        value = getattr(Measurements, name, None)
        if value is not None:
            return int(value)
    return 0


# Missing Fiji result columns become NaN so table generation can continue predictably.
def _result_value(table, *labels: str) -> float:
    for label in labels:
        try:
            return float(table.getValue(label, 0))
        except Exception:
            continue
    return float("nan")


# Request all supported statistics once to avoid repeatedly changing Fiji's global settings.
def measure_roi_stats(img, roi) -> Dict[str, float]:
    classes = get_java_classes()
    Analyzer = classes["Analyzer"]
    ResultsTable = classes["ResultsTable"]
    Measurements = classes["Measurements"]
    Calibration = classes["Calibration"]

    if img is None or roi is None:
        return {}

    tmp = ResultsTable()
    img.setRoi(roi.clone())

    flags = 0
    for names in [
        ("AREA",),
        ("MEAN",),
        ("STD_DEV",),
        ("MODE",),
        ("MIN_MAX",),
        ("CENTROID",),
        ("CENTER_OF_MASS",),
        ("PERIMETER",),
        ("RECT",),
        ("ELLIPSE",),
        ("FERET",),
        ("INTEGRATED_DENSITY",),
        ("MEDIAN",),
        ("SKEWNESS",),
        ("KURTOSIS",),
    ]:
        flags |= _measurement_flag(Measurements, *names)

    # The UI and exports promise pixel units, while OME-TIFF calibration can
    # otherwise make Fiji report physical units beside pixel-based Cellpose data.
    original_calibration = img.getCalibration()
    pixel_calibration = Calibration()
    pixel_calibration.setUnit("pixel")
    img.setCalibration(pixel_calibration)
    analyzer = Analyzer(img, flags, tmp)
    try:
        analyzer.measure()
    finally:
        img.setCalibration(original_calibration)
        img.killRoi()

    raw_intden = _result_value(tmp, "RawIntDen", "IntDen")
    intden = _result_value(tmp, "IntDen", "RawIntDen")
    return {
        "Area": _result_value(tmp, "Area"),
        "Mean": _result_value(tmp, "Mean"),
        "StdDev": _result_value(tmp, "StdDev"),
        "Mode": _result_value(tmp, "Mode"),
        "Min": _result_value(tmp, "Min"),
        "Max": _result_value(tmp, "Max"),
        "CentroidX": _result_value(tmp, "X"),
        "CentroidY": _result_value(tmp, "Y"),
        "CenterOfMassX": _result_value(tmp, "XM"),
        "CenterOfMassY": _result_value(tmp, "YM"),
        "Perimeter": _result_value(tmp, "Perim."),
        "BoundingRectX": _result_value(tmp, "BX"),
        "BoundingRectY": _result_value(tmp, "BY"),
        "BoundingRectWidth": _result_value(tmp, "Width"),
        "BoundingRectHeight": _result_value(tmp, "Height"),
        "EllipseMajor": _result_value(tmp, "Major"),
        "EllipseMinor": _result_value(tmp, "Minor"),
        "EllipseAngle": _result_value(tmp, "Angle"),
        "Feret": _result_value(tmp, "Feret"),
        "FeretX": _result_value(tmp, "FeretX"),
        "FeretY": _result_value(tmp, "FeretY"),
        "FeretAngle": _result_value(tmp, "FeretAngle"),
        "MinFeret": _result_value(tmp, "MinFeret"),
        "RawIntDen": raw_intden,
        "IntDen": intden,
        "Median": _result_value(tmp, "Median"),
        "Skewness": _result_value(tmp, "Skew"),
        "Kurtosis": _result_value(tmp, "Kurt"),
    }


# A small tuple API remains available for callers that need only the core intensity measurements.
def measure_stats(img, roi) -> Tuple[float, float, float]:
    stats = measure_roi_stats(img, roi)
    area = stats.get("Area", float("nan"))
    mean = stats.get("Mean", float("nan"))
    int_den = stats.get("RawIntDen", stats.get("IntDen", float("nan")))
    return area, mean, int_den


# Rasterize against a reference image so exported masks always match source dimensions.
def create_mask_from_roi(ref_img, roi, title: str):
    classes = get_java_classes()
    ByteProcessor = classes["ByteProcessor"]
    ImagePlus = classes["ImagePlus"]

    w = ref_img.getWidth()
    h = ref_img.getHeight()
    bp = ByteProcessor(w, h)

    if roi is not None:
        bp.setColor(255)
        bp.fill(roi)

    return ImagePlus(title, bp)


# Try both bridge spellings because Python cannot call Java's reserved-word method directly everywhere.
def union_shape_rois(base_roi, added_roi):
    for method_name in ("or_", "or"):
        method = getattr(base_roi, method_name, None)
        if callable(method):
            return method(added_roi)

    try:
        method = base_roi.getClass().getMethod("or", added_roi.getClass())
        return method.invoke(base_roi, added_roi)
    except Exception:
        pass

    raise AttributeError("ImageJ ShapeRoi union method is unavailable through this Python-Java bridge.")


# Threshold selected Weka class slices separately so each class can be reused on a later run.
def build_roi(
    result,
    out_path: Path,
    result_id: str,
    tag: str,
    clf_tag: str,
    threshold_method: str,
    probability_class_index: Any,
    log_func: Callable[[str], None],
    *,
    target_key: str = "",
):
    classes = get_java_classes()
    IJ = classes["IJ"]
    ImagePlus = classes["ImagePlus"]
    FileSaver = classes["FileSaver"]
    ShapeRoi = classes["ShapeRoi"]

    out_path.mkdir(parents=True, exist_ok=True)
    n_slices = result.getStackSize()
    class_indices = parse_probability_class_indices(probability_class_index)
    method = threshold_method.strip() if threshold_method else "Default"

    invalid_indices = [index for index in class_indices if index > n_slices]
    if invalid_indices:
        requested = ", ".join(str(index) for index in invalid_indices)
        raise ValueError(
            f"Weka class number(s) {requested} are unavailable for {tag}. "
            f"The classifier returned {n_slices} probability class(es); choose 1-{n_slices}."
        )

    rois_by_class: Dict[int, Any] = {}
    used_indices: List[int] = []

    # Weka writes one probability slice per class. Cellonaut thresholds only
    # the user-selected class slices, then unions them later if needed.
    for requested_index in class_indices:
        class_index = max(1, int(requested_index))
        used_indices.append(class_index)
        ip = result.getStack().getProcessor(class_index).duplicate()
        prob = ImagePlus(f"{result_id}_{tag}_class{class_index}_prob", ip)

        try:
            IJ.run(prob, "16-bit", "")
            prob.getProcessor().setAutoThreshold(method + " dark")
            IJ.run(prob, "Create Selection", "")
            roi = prob.getRoi()

            shape_roi = ShapeRoi(roi) if roi is not None else None
            threshold_path = out_path / f"{result_id}_{tag}_class{class_index}_thr_{clf_tag}.tif"
            saved_mask = create_mask_from_roi(
                prob,
                shape_roi,
                f"{result_id}_{tag}_class{class_index}_mask",
            )
            try:
                if not save_imagej_tiff(
                    threshold_path,
                    saved_mask,
                    FileSaver,
                ):
                    raise OSError(f"Could not save Weka threshold mask: {threshold_path}")
                record_artifact(threshold_path, sample=result_id, target=target_key or tag, label=tag, kind="threshold", mask=str(class_index), classifier=clf_tag)
            finally:
                try:
                    saved_mask.close()
                except Exception:
                    pass

            rois_by_class[class_index] = shape_roi
        finally:
            try:
                prob.close()
            except Exception:
                pass

    class_label = ",".join(str(i) for i in used_indices) if used_indices else str(probability_class_index)
    if not rois_by_class:
        log_func(f"[{result_id}] ROI creation FAILED for {tag} classes {class_label}")
    else:
        log_func(f"[{result_id}] ROI creation OK for {tag} classes {class_label}")

    return rois_by_class


# Use the same filenames when saving and finding reusable masks.
def threshold_mask_path(out_path: Path, result_id: str, tag: str, clf_tag: str, class_index: int, *, target_key: str = "") -> Path:
    resolver = ArtifactResolver.load(out_path)
    if resolver is not None:
        matches = [record for record in resolver.matching(sample=result_id, target=target_key or None, kind="threshold") if record.get("mask") == str(class_index) and record.get("classifier") == clf_tag and (target_key or record.get("label") == tag)]
        if len(matches) != 1:
            raise ArtifactMetadataError(f"No unique recorded Weka threshold mask for {result_id} / {tag} / class {class_index} / {clf_tag}.")
        return resolver.path(matches[0])
    # Historical exact classifier/class pairing remains the legacy contract.
    return out_path / f"{result_id}_{tag}_class{int(class_index)}_thr_{clf_tag}.tif"


# Inspect pixel values because bit depth alone does not prove that an image is a mask.
# An unreadable histogram remains unknown so grayscale data is never reused as a binary mask.
def _is_binary_mask_image(imp) -> bool | None:
    try:
        histogram = imp.getProcessor().getHistogram()
        populated_values = [index for index, count in enumerate(histogram) if int(count) > 0]
        maximum = len(histogram) - 1
        return bool(populated_values) and set(populated_values).issubset({0, maximum})
    except Exception:
        return None


# Recreate an ROI from disk so Weka can be skipped when its class mask already exists.
def roi_from_existing_threshold_mask(
    mask_path: Path,
    log_func: Callable[[str], None],
):
    classes = get_java_classes()
    IJ = classes["IJ"]
    ShapeRoi = classes["ShapeRoi"]

    if not mask_path.exists():
        return None

    imp = IJ.openImage(str(mask_path))
    if imp is None:
        log_func(f"[WARN] Could not open existing threshold mask: {mask_path}")
        return None

    try:
        binary_mask = _is_binary_mask_image(imp)
        if binary_mask is None:
            log_func(f"[WARN] Could not verify existing threshold mask as binary; cannot reuse: {mask_path}")
            return None
        if binary_mask:
            try:
                IJ.setThreshold(imp, 1, 65535)
            except Exception as exc:
                log_func(
                    f"[WARN] Could not prepare existing threshold mask for reuse: {mask_path} "
                    f"({type(exc).__name__}: {exc})"
                )
                return None
        else:
            log_func(f"[WARN] Existing threshold mask is not binary and cannot be reused: {mask_path}")
            return None
        IJ.run(imp, "Create Selection", "")
        roi = imp.getRoi()
        if roi is None:
            histogram = imp.getProcessor().getHistogram()
            if not any(int(count) > 0 for count in histogram[1:]):
                return _EMPTY_THRESHOLD_MASK
            log_func(f"[WARN] Existing threshold mask has foreground but no selectable ROI: {mask_path}")
            return None
        return ShapeRoi(roi)
    finally:
        try:
            imp.close()
        except Exception:
            pass


# Load classes independently because a partially complete prior run can still save work.
def load_existing_class_rois(
    threshold_out_path: Path,
    result_id: str,
    tag: str,
    clf_tag: str,
    probability_class_index: Any,
    log_func: Callable[[str], None],
    *,
    target_key: str = "",
) -> Dict[int, Any]:
    rois_by_class: Dict[int, Any] = {}
    for class_index in parse_probability_class_indices(probability_class_index):
        try:
            mask_path = threshold_mask_path(threshold_out_path, result_id, tag, clf_tag, class_index, target_key=target_key)
        except ArtifactMetadataError as exc:
            log_func(f"[WARN] {exc}")
            continue
        roi = roi_from_existing_threshold_mask(
            mask_path,
            log_func,
        )
        if roi is not None:
            rois_by_class[int(class_index)] = None if roi is _EMPTY_THRESHOLD_MASK else roi
            log_func(f"[{result_id}] Reused existing Weka mask: {mask_path}")
    return rois_by_class


# Build the selection in Fiji so adjusted masks use the same ROI representation as Weka masks.
def roi_from_mask_array(mask: np.ndarray):
    classes = get_java_classes()
    IJ = classes["IJ"]
    ShapeRoi = classes["ShapeRoi"]
    ImagePlus = classes["ImagePlus"]
    ByteProcessor = classes["ByteProcessor"]

    binary = np.asarray(mask) > 0
    height, width = binary.shape
    processor = ByteProcessor(width, height)
    byte_pixels = np.where(binary.ravel(), -1, 0).astype(np.int8)
    try:
        pixels = np.asarray(processor.getPixels())
        if pixels.shape == byte_pixels.shape and pixels.flags.writeable:
            pixels[:] = byte_pixels
        else:
            processor.setPixels(byte_pixels)
    except Exception:
        for y in range(height):
            row = binary[y]
            for x, active in enumerate(row):
                if active:
                    processor.set(x, y, 255)
    imp = ImagePlus("Cellonaut mask", processor)
    try:
        IJ.setThreshold(imp, 1, 255)
        IJ.run(imp, "Create Selection", "")
        roi = imp.getRoi()
        return ShapeRoi(roi) if roi is not None else None
    finally:
        imp.close()


FIJI_BINARY_COMMANDS = {
    MASK_PROCESSING_STEP_BINARY_ERODE: "Erode",
    MASK_PROCESSING_STEP_BINARY_DILATE: "Dilate",
    MASK_PROCESSING_STEP_BINARY_OPEN: "Open",
    MASK_PROCESSING_STEP_BINARY_CLOSE: "Close-",
    MASK_PROCESSING_STEP_BINARY_FILL_HOLES: "Fill Holes",
    MASK_PROCESSING_STEP_BINARY_WATERSHED: "Watershed",
    MASK_PROCESSING_STEP_BINARY_OUTLINE: "Outline",
    MASK_PROCESSING_STEP_BINARY_SKELETONIZE: "Skeletonize",
}


def apply_fiji_binary_step(roi, ref_img, step_type: str, params: dict):
    """Run the actual Fiji binary command on Cellonaut's white-on-black mask."""
    if roi is None or ref_img is None:
        return roi
    classes = get_java_classes()
    IJ = classes["IJ"]
    prefs = jimport("ij.Prefs")
    mask = create_mask_from_roi(ref_img, roi, "Cellonaut Fiji binary step")
    prior_black = bool(prefs.blackBackground)
    prior_pad = bool(prefs.padEdges)
    configured_options = False
    try:
        if step_type in {
            MASK_PROCESSING_STEP_BINARY_ERODE,
            MASK_PROCESSING_STEP_BINARY_DILATE,
            MASK_PROCESSING_STEP_BINARY_OPEN,
            MASK_PROCESSING_STEP_BINARY_CLOSE,
        }:
            iterations, count, pad_edges = parse_binary_settings(params.get("binary_settings", "1,1,false"))
            option_text = f"iterations={iterations} count={count} black" + (" pad" if pad_edges else "")
            IJ.run(mask, "Options...", option_text)
            configured_options = True
        else:
            prefs.blackBackground = True
        IJ.run(mask, FIJI_BINARY_COMMANDS[step_type], "")
        return roi_from_mask_array(imageplus_to_numpy_2d(mask))
    finally:
        try:
            if configured_options:
                IJ.run(mask, "Options...", "iterations=1 count=1 black")
        finally:
            prefs.blackBackground = prior_black
            prefs.padEdges = prior_pad
            mask.close()


def apply_fiji_translate_step(roi, ref_img, params: dict):
    """Translate the binary mask with Fiji's Image > Transform command."""
    if roi is None or ref_img is None:
        return roi
    x_offset, y_offset = parse_translate_offsets(params.get("translate_offsets", "0,0"))
    mask = create_mask_from_roi(ref_img, roi, "Cellonaut Fiji Translate")
    try:
        get_java_classes()["IJ"].run(
            mask, "Translate...", f"x={x_offset:g} y={y_offset:g} interpolation=None"
        )
        return roi_from_mask_array(imageplus_to_numpy_2d(mask))
    finally:
        mask.close()


def apply_fiji_analyze_particles_step(roi, ref_img, params: dict):
    """Use ImageJ's particle analyzer and retain its accepted mask pixels."""
    if roi is None or ref_img is None:
        return roi
    min_size, max_size, min_circ, max_circ, exclude_edges, include_holes = parse_particle_settings(
        params.get("particle_settings", "0-Infinity,0-1,false,false")
    )
    particle_analyzer = jimport("ij.plugin.filter.ParticleAnalyzer")
    results_table = jimport("ij.measure.ResultsTable")
    flags = int(particle_analyzer.SHOW_MASKS)
    if exclude_edges:
        flags |= int(particle_analyzer.EXCLUDE_EDGE_PARTICLES)
    if include_holes:
        flags |= int(particle_analyzer.INCLUDE_HOLES)
    mask = create_mask_from_roi(ref_img, roi, "Cellonaut Fiji Analyze Particles")
    output = None
    try:
        analyzer = particle_analyzer(flags, 0, results_table(), min_size, max_size, min_circ, max_circ)
        analyzer.setHideOutputImage(True)
        if not analyzer.analyze(mask):
            raise RuntimeError("Fiji Analyze Particles could not process the binary mask.")
        output = analyzer.getOutputImage()
        if output is None:
            raise RuntimeError("Fiji Analyze Particles did not return a mask image.")
        # Show=Masks uses an inverted LUT; raw foreground pixels are still 255.
        accepted = np.asarray(imageplus_to_numpy_2d(output)) > 0
        return roi_from_mask_array(accepted)
    finally:
        if output is not None:
            output.close()
        mask.close()


# Threshold selection belongs to the mask source, so recipe row order cannot change it.
def mask_threshold_method_for_image(image_def: Any, cfg: Any) -> str:
    return str(getattr(image_def, "threshold_method", None) or getattr(cfg, "threshold_method", "Default") or "Default")


# Execute mask rows in the order displayed in the settings table.
def apply_ordered_mask_processing_recipe(
    roi,
    img,
    image_def: Any,
    *,
    result_id: str,
    roi_label: str,
    log_func: Callable[[str], None],
    skeleton_metrics: Optional[Dict[str, int]] = None,
    skeleton_out_path: Optional[Path] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
):
    for step in list(getattr(image_def, "mask_processing_steps", []) or []):
        check_cancel(should_cancel)
        if roi is None or not isinstance(step, dict) or not bool(step.get("enabled", True)):
            continue
        step_type = str(step.get("type", "") or "")
        params = dict(step.get("params", {}) or {})
        if step_type == MASK_PROCESSING_STEP_TRANSLATE:
            if img is None:
                raise RuntimeError(f"Fiji Translate requires the source image for {roi_label}.")
            roi = apply_fiji_translate_step(roi, img, params)
            log_func(f"[{result_id}] Fiji Translate for {roi_label}")
        elif step_type in FIJI_BINARY_COMMANDS:
            if img is None:
                raise RuntimeError(f"Fiji Binary {FIJI_BINARY_COMMANDS[step_type]} requires the source image for {roi_label}.")
            roi = apply_fiji_binary_step(roi, img, step_type, params)
            log_func(f"[{result_id}] Fiji Binary {FIJI_BINARY_COMMANDS[step_type]} for {roi_label}")
        elif step_type == MASK_PROCESSING_STEP_ANALYZE_PARTICLES:
            if img is None:
                raise RuntimeError(f"Fiji Analyze Particles requires the source image for {roi_label}.")
            roi = apply_fiji_analyze_particles_step(roi, img, params)
            log_func(f"[{result_id}] Fiji Analyze Particles for {roi_label}")
        elif step_type == MASK_PROCESSING_STEP_ANALYZE_SKELETON:
            if img is None:
                raise RuntimeError(
                    f"Skeleton analysis requires the source image for {roi_label}, but it was not opened."
                )
            mask = create_mask_from_roi(img, roi, f"{result_id}_{roi_label}_skeleton_input")
            try:
                skeleton_dir = skeleton_out_path or Path(".")
                skeleton_path = skeleton_dir / f"{result_id}_{roi_label}_skeleton.tif"
                metrics = analyze_mask_skeleton(mask, skeleton_path=skeleton_path)
            finally:
                mask.close()
            for metric_name, value in metrics.items():
                column = f"{roi_label}_MaskSkeleton_{metric_name}"
                if skeleton_metrics is not None:
                    skeleton_metrics[column] = int(value)
            log_func(
                f"[{result_id}] Analyze Skeleton for {roi_label}: "
                f"junctions={metrics['JunctionCount']}, "
                f"branches={metrics['BranchCount']} | saved={skeleton_path}"
            )
    return roi


def _shape_roi_operation(base_roi: Any, added_roi: Any, operation: str) -> Any:
    if operation == "OR":
        return union_shape_rois(base_roi, added_roi)
    method_name = {"OR": "or", "AND": "and", "XOR": "xor"}[operation]
    for bridge_name in (f"{method_name}_", method_name):
        method = getattr(base_roi, bridge_name, None)
        if callable(method):
            return method(added_roi)
    try:
        method = base_roi.getClass().getMethod(method_name, added_roi.getClass())
        return method.invoke(base_roi, added_roi)
    except Exception as exc:
        raise AttributeError(f"ImageJ ShapeRoi {operation} method is unavailable through this Python-Java bridge.") from exc


def _combined_roi_from_sources(roi_map: Dict[str, Any], source_keys: List[str], operation: str = "OR"):
    if operation not in {"OR", "AND", "XOR"}:
        raise ValueError(f"Unsupported combined mask operation: {operation}")
    source_rois = []
    for source_key in source_keys:
        direct_roi = roi_map.get(source_key)
        class_rois = [direct_roi] if direct_roi is not None else []
        class_prefix = f"{source_key}__class"
        class_rois.extend(roi for key, roi in roi_map.items() if key.startswith(class_prefix) and roi is not None)
        if not class_rois and operation == "AND":
            return None
        source_roi = class_rois[0].clone() if class_rois else None
        for class_roi in class_rois[1:]:
            source_roi = union_shape_rois(source_roi, class_roi)
        if source_roi is not None:
            source_rois.append(source_roi)
    combined_roi = source_rois[0].clone() if source_rois else None
    for source_roi in source_rois[1:]:
        combined_roi = _shape_roi_operation(combined_roi, source_roi, operation)
    return combined_roi


def _generate_missing_class_rois(
    *,
    img: Any,
    img_def: Any,
    cfg: Any,
    missing_classes: List[int],
    segs: Dict[str, Any],
    classifier_cache: Dict[Tuple[str, str, str], Any],
    file_map: Optional[Dict[str, Optional[Path]]],
    probability_out_path: Path,
    threshold_out_path: Path,
    result_id: str,
    clf_tag: str,
    log_func: Callable[[str], None],
    should_cancel: Optional[Callable[[], bool]],
) -> tuple[Any, Dict[int, Any], bool]:
    if not missing_classes:
        return None, {}, False
    if img is None:
        return None, {}, True

    measure_img, applied_steps = apply_imagej_processing_recipe_copy(
        img, img_def, IMAGE_PROCESSING_SCOPE_SEGMENTATION
    )
    if applied_steps:
        log_func(f"[{result_id}] Applied segmentation processing for {img_def.label}: {', '.join(applied_steps)}")
    source_path = (file_map or {}).get(img_def.key)
    # Opened images identify the resolved channel/Z plane, including inferred
    # channels. Shared planes reuse the same object throughout this sample.
    cache_source = f"image-object:{id(img)}"
    cache_key = (
        cache_source,
        str(img_def.model_path),
        imagej_processing_recipe_signature(img_def, IMAGE_PROCESSING_SCOPE_SEGMENTATION),
    )
    cached_result = classifier_cache.get(cache_key)
    if cached_result is None:
        if img_def.model_path is None or not Path(img_def.model_path).exists():
            raise FileNotFoundError(
                f"Reusable masks were missing for {img_def.label}, "
                f"and the classifier file is unavailable: {img_def.model_path}"
            )
        if img_def.key not in segs:
            classes = get_java_classes()
            seg = classes["WekaSegmentation"]()
            load_weka_classifier(seg, img_def.model_path)
            segs[img_def.key] = seg
            log_func(f"[{result_id}] Loaded classifier for missing masks: {img_def.model_path}")
        seg = segs[img_def.key]
        log_func(
            f"[{result_id}] Running classifier for {img_def.label} with model tag {clf_tag} "
            f"on image: {source_path or '(already-open image object)'}"
        )
        try:
            cached_result = seg.applyClassifier(measure_img, 0, True)
        except Exception as exc:
            raise format_weka_runtime_error(exc, mask_label=img_def.label) from exc
        if cached_result is None:
            raise RuntimeError(f"Weka returned no probability image for {img_def.label}.")
        classifier_cache[cache_key] = cached_result
        check_cancel(should_cancel)
        probability_out_path.mkdir(parents=True, exist_ok=True)
        classes = get_java_classes()
        probability_path = probability_out_path / f"{result_id}_{img_def.label}_prob_{clf_tag}.tif"
        if not save_imagej_tiff(probability_path, cached_result, classes["FileSaver"]):
            raise OSError(f"Could not save Weka probability map: {probability_path}")
        record_artifact(probability_path, sample=result_id, target=img_def.key, label=img_def.label, kind="probability")
    else:
        log_func(f"[{result_id}] Reusing classifier result for {img_def.label} with model tag {clf_tag}")

    check_cancel(should_cancel)
    generated = build_roi(
        cached_result,
        threshold_out_path,
        result_id,
        img_def.label,
        clf_tag,
        mask_threshold_method_for_image(img_def, cfg),
        ",".join(str(class_index) for class_index in missing_classes),
        log_func,
        target_key=img_def.key,
    )
    return measure_img, generated, False


# Prepare every ROI in dependency order while caching shared Weka probability maps.
def prepare_rois_for_defs(
    image_map,
    cfg: Any,
    out_path: Path,
    result_id: str,
    roi_defs: List[Any],
    segs: Dict[str, Any],
    log_func: Callable[[str], None],
    file_map: Optional[Dict[str, Optional[Path]]] = None,
    skeleton_metrics: Optional[Dict[str, int]] = None,
    skeleton_out_path: Optional[Path] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
):
    """Return ROI and prepared-image maps for the requested masks, writing artifacts.

    roi_defs must put combined masks after their sources. ROI values may be None
    for valid empty masks; absent keys are not equivalent. Reuse saved classes
    when requested and regenerate only missing classes. The caller owns returned
    prepared images, which can alias input images and must be closed only once.
    Cached probability images are local to this call and closed on exit.
    """
    roi_map: Dict[str, Any] = {}
    roi_measure_imgs: Dict[str, Any] = {}

    if roi_defs:
        out_path.mkdir(parents=True, exist_ok=True)
    probability_out_path = out_path / "ProbabilityMaps"
    threshold_out_path = out_path / "MaskImages"
    mask_source_dir = getattr(cfg, "mask_source_dir", None)
    if mask_source_dir is not None:
        source_results = Path(mask_source_dir) / "Results"
        current_masks_root = Path(cfg.output_dir) / "Results" / "Masks"
        try:
            relative_mask_dir = out_path.relative_to(current_masks_root)
        except ValueError:
            relative_mask_dir = Path()
        source_masks_dir = source_results / "Masks" / relative_mask_dir
        source_threshold_out_path = source_masks_dir / "MaskImages"

    else:
        source_threshold_out_path = threshold_out_path

    classifier_cache: Dict[Tuple[str, str, str], Any] = {}

    try:
        for img_def in roi_defs:
            check_cancel(should_cancel)
            combined_source_keys = list(getattr(img_def, "combined_mask_source_keys", []) or [])
            if combined_source_keys:
                for source_key in combined_source_keys:
                    if source_key not in roi_map and not any(key.startswith(f"{source_key}__class") for key in roi_map):
                        raise ValueError(f"Combined mask {img_def.label} has an unavailable source: {source_key}")
                operation = str(getattr(img_def, "combined_mask_operation", "OR") or "OR")
                combined_roi = _combined_roi_from_sources(roi_map, combined_source_keys, operation)
                reference_img = next(
                    (image_map[key] for key in mask_reference_image_keys(combined_source_keys, cfg.images)
                     if image_map.get(key) is not None),
                    None,
                )
                combined_roi = apply_ordered_mask_processing_recipe(
                    combined_roi,
                    reference_img,
                    img_def,
                    result_id=result_id,
                    roi_label=img_def.label,
                    log_func=log_func,
                    skeleton_metrics=skeleton_metrics,
                    skeleton_out_path=skeleton_out_path or out_path / "Skeletons",
                    should_cancel=should_cancel,
                )
                roi_map[img_def.key] = combined_roi
                roi_measure_imgs[img_def.key] = None
                if combined_roi is None:
                    log_func(f"[{result_id}] Combined mask {img_def.label} is empty.")
                else:
                    log_func(
                        f"[{result_id}] Combined mask {img_def.label} created "
                        f"from {len(combined_source_keys)} source masks ({operation})."
                    )
                continue

            probability_class_index = (
                img_def.probability_class_index
                if img_def.probability_class_index is not None
                else cfg.probability_class_index
            )
            selected_classes = parse_probability_class_indices(probability_class_index)
            clf_tag = base_name_no_ext(img_def.model_path)

            class_rois: Dict[int, Any] = {}
            img = image_map.get(img_def.key)
            if getattr(cfg, "reuse_existing_masks", False):
                class_rois = load_existing_class_rois(
                    source_threshold_out_path,
                    result_id,
                    img_def.label,
                    clf_tag,
                    probability_class_index,
                    log_func,
                    target_key=img_def.key,
                )

            missing_classes = [class_index for class_index in selected_classes if class_index not in class_rois]
            if getattr(cfg, "reuse_existing_masks", False):
                if missing_classes:
                    log_func(
                        f"[{result_id}] Missing reusable Weka classes for "
                        f"{img_def.label}: {missing_classes}; regenerating only "
                        "those classes."
                    )
                else:
                    log_func(
                        f"[{result_id}] Reused all requested Weka classes for "
                        f"{img_def.label} from {source_threshold_out_path}."
                    )

            measure_img, generated_class_rois, missing_image = _generate_missing_class_rois(
                img=img,
                img_def=img_def,
                cfg=cfg,
                missing_classes=missing_classes,
                segs=segs,
                classifier_cache=classifier_cache,
                file_map=file_map,
                probability_out_path=probability_out_path,
                threshold_out_path=threshold_out_path,
                result_id=result_id,
                clf_tag=clf_tag,
                log_func=log_func,
                should_cancel=should_cancel,
            )
            if measure_img is not None:
                roi_measure_imgs[img_def.key] = measure_img
            if missing_image:
                raise ValueError(f"Missing image for {img_def.label}; cannot generate missing masks.")
            class_rois.update(generated_class_rois)
            split_by_class = len(selected_classes) > 1

            for class_index in selected_classes:
                if class_index not in class_rois:
                    raise RuntimeError(f"No mask result for {img_def.label}, class {class_index}.")
                roi = class_rois.get(class_index)
                roi_label = f"{img_def.label}_Class{class_index}" if split_by_class else img_def.label
                roi_key = class_roi_key(img_def.key, class_index) if split_by_class else img_def.key
                roi = apply_ordered_mask_processing_recipe(
                    roi,
                    img,
                    img_def,
                    result_id=result_id,
                    roi_label=roi_label,
                    log_func=log_func,
                    skeleton_metrics=skeleton_metrics,
                    skeleton_out_path=skeleton_out_path or out_path / "Skeletons",
                    should_cancel=should_cancel,
                )

                roi_map[roi_key] = roi
    finally:
        for result in classifier_cache.values():
            try:
                result.close()
            except Exception:
                pass

    return roi_map, roi_measure_imgs
