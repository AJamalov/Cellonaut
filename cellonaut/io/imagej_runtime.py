"""Fiji/PyImageJ runtime startup, validation, and Weka error helpers.

Cellonaut uses Fiji for Weka segmentation and some ImageJ measurements. This
module centralizes JVM setup and dependency checks so the pipeline can report
clear errors when a Fiji installation is incomplete.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Callable, Optional

from cellonaut.io.fiji_installation import (
    BIOP_IMAGE_LOADER_SERVICE,
    IMAGE_SCIENCE_CLASS,
    MASTODON_PLUGIN_API,
    add_clean_runtime_dependencies,
    fiji_contains_class,
    find_jar_containing_class,
    requires_clean_fiji_runtime,
)
from cellonaut.io.java_runtime import (
    configure_java_caches,
    configure_java_home,
    ensure_stdio,
    find_java_home_from_jvm_dll,
    is_supported_java_home,
    is_valid_java_home,
    java_major_version,
)

__all__ = [
    "BIOP_IMAGE_LOADER_SERVICE",
    "FIJI_MAVEN_ENDPOINT",
    "IMAGE_SCIENCE_CLASS",
    "MASTODON_PLUGIN_API",
    "add_clean_runtime_dependencies",
    "configure_java_caches",
    "configure_java_home",
    "ensure_stdio",
    "fiji_contains_class",
    "find_jar_containing_class",
    "find_java_home_from_jvm_dll",
    "format_weka_runtime_error",
    "get_ij",
    "get_java_classes",
    "get_python_modules",
    "is_supported_java_home",
    "is_valid_java_home",
    "inspect_weka_classifier",
    "java_major_version",
    "jimport",
    "load_weka_classifier",
    "requires_clean_fiji_runtime",
    "suppress_java_stderr",
]

FIJI_MAVEN_ENDPOINT = "sc.fiji:fiji:2.17.0"


_imagej_mod = None
_sj_mod = None


DEFAULT_JAVA_HEAP = "6g"


# PyInstaller can miss package metadata even when imports work in development,
# so imports are centralized here to turn packaging mistakes into clear errors.
def get_python_modules():
    global _imagej_mod, _sj_mod
    if _imagej_mod is None or _sj_mod is None:
        try:
            import imagej as imagej_mod
            import scyjava as sj_mod
        except Exception as exc:
            msg = str(exc)
            if "No package metadata was found for scyjava" in msg:
                raise RuntimeError(
                    "PyImageJ/scyjava could not start because the packaged app is missing the "
                    "scyjava package metadata. Rebuild the installer with PyInstaller "
                    "copy_metadata('scyjava') and related package metadata included."
                ) from exc
            if "No package metadata was found for imagej" in msg:
                raise RuntimeError(
                    "PyImageJ could not start because the packaged app is missing the imagej "
                    "package metadata. Rebuild the installer with PyInstaller "
                    "copy_metadata('imagej') included."
                ) from exc
            raise

        _imagej_mod = imagej_mod
        _sj_mod = sj_mod
    return _imagej_mod, _sj_mod


# Fiji prints noisy Java loader diagnostics during normal startup; suppressing them
# keeps the GUI log focused while still restoring streams for real failures.
def _silence_java_during_imagej_init(sj_mod, init_func):
    streams = {}

    # Defer stream replacement until the JVM exists; doing it earlier has no Java System object to modify.
    def silence_streams():
        System = sj_mod.jimport("java.lang.System")
        OutputStream = sj_mod.jimport("java.io.OutputStream")
        PrintStream = sj_mod.jimport("java.io.PrintStream")
        streams["System"] = System
        streams["out"] = System.out
        streams["err"] = System.err
        streams["sink"] = PrintStream(OutputStream.nullOutputStream())
        System.setOut(streams["sink"])
        System.setErr(streams["sink"])

    sj_mod.when_jvm_starts(silence_streams)
    try:
        return init_func()
    finally:
        System = streams.get("System")
        if System is not None:
            System.setOut(streams["out"])
            System.setErr(streams["err"])
            streams["sink"].close()


# Weka often reports recoverable Java warnings on stderr, so classifier calls
# silence stderr locally and promote actual failures to Python exceptions.
@contextmanager
def suppress_java_stderr():
    _, sj_mod = get_python_modules()
    System = sj_mod.jimport("java.lang.System")
    OutputStream = sj_mod.jimport("java.io.OutputStream")
    PrintStream = sj_mod.jimport("java.io.PrintStream")
    original = System.err
    sink = PrintStream(OutputStream.nullOutputStream())
    System.setErr(sink)
    try:
        yield
    finally:
        System.setErr(original)
        sink.close()


# Weka signals a rejected model with False instead of raising, which would
# otherwise look like a successful load to the rest of the pipeline.
def load_weka_classifier(segmentation, model_path: Path | str) -> None:
    with suppress_java_stderr():
        loaded = segmentation.loadClassifier(str(model_path))
    if not loaded:
        raise RuntimeError(f"Weka could not load classifier: {model_path}")


def inspect_weka_classifier(
    model_path: Path | str,
    fiji_app_path: Path | str | None = None,
) -> list[str]:
    """Load a TWS model and return its class labels in probability-map order."""

    get_ij(Path(fiji_app_path) if fiji_app_path else None, log_func=lambda _message: None)
    segmentation = get_java_classes()["WekaSegmentation"]()
    load_weka_classifier(segmentation, model_path)
    class_count = int(segmentation.getNumOfClasses())
    if class_count < 1:
        raise RuntimeError(f"Weka classifier contains no classes: {model_path}")
    return [str(segmentation.getClassLabel(index)) for index in range(class_count)]


# Best-effort third-party fallback: Java plugin exceptions do not have stable
# Python types/codes. App-owned setup errors use SetupErrorCode instead.
def format_weka_runtime_error(exc: BaseException, *, mask_label: str) -> RuntimeError:
    message = str(exc)
    if "imagescience/image/Image" in message or "imagescience.image.Image" in message:
        return RuntimeError(
            f"Weka could not generate the mask for {mask_label} because Cellonaut's "
            "Fiji runtime is missing ImageScience. Install ImageScience in the app's "
            "Fiji directory, restart Cellonaut, and run the preview again."
        )
    if "OutOfMemoryError" in message or "Java heap space" in message:
        return RuntimeError(
            f"Weka ran out of Java memory while generating the mask for {mask_label}. "
            "Close other heavy applications and try again. If this keeps happening, "
            "increase CELLONAUT_JAVA_HEAP before launching Cellonaut."
        )
    if "NullPointerException" in message and "classifiedSlices" in message:
        return RuntimeError(
            f"Weka could not generate the mask for {mask_label}. Fiji returned an empty "
            "classifier result, which usually means a Fiji plugin dependency is missing "
            "or Java ran out of memory. Check that ImageScience is installed in the "
            "app's Fiji directory, then try again."
        )
    return RuntimeError(f"Weka could not generate the mask for {mask_label}: {message}")


_ij = None
_JAVA = {}

_IMGLYB_ENDPOINT_PREFIX = "net.imglib2:imglib2-imglyb:"
_LOCAL_IMGLYB_JARS = ("imglib2-imglyb-1.1.0.jar", "imglib2-unsafe-1.0.0.jar")


def _configure_local_imglyb(sj_mod, fiji_app_path: Path, *, offline: bool) -> None:
    """Use bundled ImgLyb jars so local Fiji startup never needs Maven offline."""
    if not offline:
        return
    jar_paths = [fiji_app_path / "jars" / name for name in _LOCAL_IMGLYB_JARS]
    missing = [path.name for path in jar_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "The bundled Fiji installation is missing its PyImageJ bridge: "
            + ", ".join(missing)
            + ". Reinstall Cellonaut from the complete official offline package."
        )
    for path in jar_paths:
        sj_mod.config.add_classpath(str(path))
    endpoints = getattr(sj_mod.config, "endpoints", None)
    if endpoints is not None:
        endpoints[:] = [item for item in endpoints if not item.startswith(_IMGLYB_ENDPOINT_PREFIX)]


def _init_with_local_java_version_hint(imagej_mod, init_func, java_home: Path | None, *, offline: bool):
    """Work around PyImageJ 1.8 leaving its Java major version uninitialized offline."""
    guess_java_version = getattr(imagej_mod, "_guess_java_version", None)
    major_version = java_major_version(java_home) if offline and java_home is not None else None
    if not callable(guess_java_version) or major_version is None:
        return init_func()

    imagej_mod._guess_java_version = lambda: major_version
    try:
        return init_func()
    finally:
        imagej_mod._guess_java_version = guess_java_version


# PyImageJ owns a single JVM per process, so startup choices have to be made
# once and cached before any Java class is imported.
def get_ij(fiji_app_path: Optional[Path] = None, log_func: Callable[[str], None] = print):
    global _ij
    if _ij is None:
        ensure_stdio()
        configure_java_caches()
        java_home = configure_java_home(fiji_app_path=fiji_app_path, log_func=log_func)

        imagej_mod, sj_mod = get_python_modules()
        offline_release = os.environ.get("CELLONAUT_OFFLINE", "").strip() == "1"
        sj_mod.config.set_java_constraints(fetch="never" if offline_release else "auto")
        sj_mod.config.add_option("-Djava.awt.headless=true")
        java_heap = os.environ.get("CELLONAUT_JAVA_HEAP", DEFAULT_JAVA_HEAP).strip()
        if java_heap:
            sj_mod.config.add_option(f"-Xmx{java_heap}")

        selected_fiji = Path(fiji_app_path) if fiji_app_path is not None else None
        use_clean_runtime = selected_fiji is not None and requires_clean_fiji_runtime(selected_fiji)
        if offline_release and selected_fiji is None:
            raise RuntimeError("The offline Cellonaut installation is missing its bundled Fiji folder. Reinstall Cellonaut.")
        if offline_release and use_clean_runtime:
            raise RuntimeError(
                "The bundled Fiji installation is incomplete and would require an internet download. "
                "Reinstall Cellonaut from the complete official offline package."
            )
        if selected_fiji is not None:
            add_clean_runtime_dependencies(sj_mod, selected_fiji)
            _configure_local_imglyb(sj_mod, selected_fiji, offline=offline_release)
            if not fiji_contains_class(selected_fiji, IMAGE_SCIENCE_CLASS):
                log_func(
                    "[WARN] ImageScience was not found in Cellonaut's Fiji runtime. "
                    "Some Weka classifiers require it and may fail until ImageScience is installed."
                )
        init_target = str(selected_fiji) if selected_fiji is not None and not use_clean_runtime else FIJI_MAVEN_ENDPOINT
        log_func("Starting Fiji runtime. The first initialization can take a minute...")
        _ij = _silence_java_during_imagej_init(
            sj_mod,
            lambda: _init_with_local_java_version_hint(
                imagej_mod,
                lambda: imagej_mod.init(init_target, mode="headless"),
                java_home,
                offline=offline_release,
            ),
        )

        if selected_fiji is not None and fiji_contains_class(selected_fiji, IMAGE_SCIENCE_CLASS):
            try:
                sj_mod.jimport("imagescience.image.Image")
            except Exception as exc:
                raise RuntimeError(
                    "The Fiji runtime could not load ImageScience, which is required "
                    "by the selected Weka classifiers. Install ImageScience in the app's "
                    "Fiji directory and restart Cellonaut."
                ) from exc

        log_func("Fiji initialized successfully.")
    return _ij


# Java class lookups are cached because repeated scyjava imports are slow and
# can trigger extra JVM chatter during batch runs.
def jimport(name: str):
    if name not in _JAVA:
        _, sj_mod = get_python_modules()
        get_ij()
        _JAVA[name] = sj_mod.jimport(name)
    return _JAVA[name]


# Import the stable ImageJ surface together so downstream modules do not repeat Java class-name strings.
def get_java_classes():
    return {
        "IJ": jimport("ij.IJ"),
        "ImagePlus": jimport("ij.ImagePlus"),
        "ImageStack": jimport("ij.ImageStack"),
        "CompositeImage": jimport("ij.CompositeImage"),
        "LUT": jimport("ij.process.LUT"),
        "Color": jimport("java.awt.Color"),
        "Font": jimport("java.awt.Font"),
        "Overlay": jimport("ij.gui.Overlay"),
        "TextRoi": jimport("ij.gui.TextRoi"),
        "ShapeRoi": jimport("ij.gui.ShapeRoi"),
        "ByteProcessor": jimport("ij.process.ByteProcessor"),
        "FileSaver": jimport("ij.io.FileSaver"),
        "Duplicator": jimport("ij.plugin.Duplicator"),
        "HyperStackConverter": jimport("ij.plugin.HyperStackConverter"),
        "Analyzer": jimport("ij.plugin.filter.Analyzer"),
        "ResultsTable": jimport("ij.measure.ResultsTable"),
        "Measurements": jimport("ij.measure.Measurements"),
        "Calibration": jimport("ij.measure.Calibration"),
        "WekaSegmentation": jimport("trainableSegmentation.WekaSegmentation"),
        "PolygonRoi": jimport("ij.gui.PolygonRoi"),
    }
