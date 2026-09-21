from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

import cellonaut.io.fiji_installation as fiji_installation
import cellonaut.io.imagej_runtime as imagej_runtime
from cellonaut.io.fiji_installation import (
    BIO_FORMATS_READER_CLASS,
    WEKA_SEGMENTATION_CLASS,
    find_jars_containing_classes,
    looks_like_fiji_installation,
    scan_fiji_installation,
)
from cellonaut.io.imagej_runtime import (
    BIOP_IMAGE_LOADER_SERVICE,
    IMAGE_SCIENCE_CLASS,
    MASTODON_PLUGIN_API,
    add_clean_runtime_dependencies,
    configure_java_caches,
    configure_java_home,
    fiji_contains_class,
    find_jar_containing_class,
    find_java_home_from_jvm_dll,
    format_weka_runtime_error,
    is_supported_java_home,
    is_valid_java_home,
    inspect_weka_classifier,
    load_weka_classifier,
    requires_clean_fiji_runtime,
)


def make_java_home(root: Path) -> Path:
    (root / "bin" / "server").mkdir(parents=True)
    (root / "bin" / "java.exe").write_text("", encoding="utf-8")
    (root / "bin" / "server" / "jvm.dll").write_text("", encoding="utf-8")
    return root


def test_is_valid_java_home_accepts_standard_layout(tmp_path):
    java_home = make_java_home(tmp_path / "jdk")

    assert is_valid_java_home(java_home)


def test_find_java_home_from_jvm_dll_detects_nested_jdk(tmp_path):
    java_home = make_java_home(tmp_path / "Fiji.app" / "java")

    assert find_java_home_from_jvm_dll(tmp_path / "Fiji.app") == java_home


def test_configure_java_home_prefers_existing_valid_home(tmp_path, monkeypatch):
    java_home = make_java_home(tmp_path / "existing-jdk")
    messages = []
    monkeypatch.setenv("JAVA_HOME", str(java_home))

    selected = configure_java_home(log_func=messages.append)

    assert selected == java_home
    assert Path(os.environ["JAVA_HOME"]) == java_home
    assert messages == []


def test_configure_java_home_uses_selected_fiji_folder(tmp_path, monkeypatch):
    fiji_path = tmp_path / "Fiji.app"
    java_home = make_java_home(fiji_path / "java")
    messages = []
    monkeypatch.delenv("JAVA_HOME", raising=False)

    selected = configure_java_home(fiji_path, log_func=messages.append)

    assert selected == java_home
    assert Path(os.environ["JAVA_HOME"]) == java_home
    assert Path(os.environ["PATH"].split(os.pathsep)[0]) == java_home / "bin"
    assert messages == []


def test_supported_java_home_rejects_known_java_8(tmp_path, monkeypatch):
    java_home = make_java_home(tmp_path / "jdk8")
    monkeypatch.setattr("cellonaut.io.java_runtime.java_major_version", lambda _path: 8)

    assert not is_supported_java_home(java_home)


def test_configure_java_caches_uses_project_local_directory(monkeypatch):
    monkeypatch.delenv("CJDK_CACHE_DIR", raising=False)

    cache_root = configure_java_caches()

    assert Path(os.environ["CJDK_CACHE_DIR"]).parent == cache_root


def make_portable_java_home(root: Path) -> Path:
    (root / "bin" / "server").mkdir(parents=True)
    (root / "bin" / "java.exe").write_text("", encoding="utf-8")
    (root / "bin" / "server" / "jvm.dll").write_text("", encoding="utf-8")
    return root


def write_jar(path: Path, *class_files: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for class_file in class_files:
            archive.writestr(class_file, b"")


def test_fiji_contains_class_reads_jar_entries(tmp_path):
    fiji_path = tmp_path / "Fiji.app"
    write_jar(fiji_path / "jars" / "plugin.jar", BIOP_IMAGE_LOADER_SERVICE)

    assert fiji_contains_class(fiji_path, BIOP_IMAGE_LOADER_SERVICE)
    assert not fiji_contains_class(fiji_path, MASTODON_PLUGIN_API)


def test_find_jar_containing_class_returns_matching_jar(tmp_path):
    fiji_path = tmp_path / "Fiji.app"
    image_science_jar = fiji_path / "plugins" / "imagescience.jar"
    write_jar(image_science_jar, IMAGE_SCIENCE_CLASS)

    assert find_jar_containing_class(fiji_path, IMAGE_SCIENCE_CLASS) == image_science_jar


def test_find_jars_containing_classes_reports_each_requested_component(tmp_path):
    fiji_path = tmp_path / "Fiji.app"
    core_jar = fiji_path / "plugins" / "core.jar"
    write_jar(core_jar, IMAGE_SCIENCE_CLASS, WEKA_SEGMENTATION_CLASS)

    found = find_jars_containing_classes(
        fiji_path,
        (IMAGE_SCIENCE_CLASS, WEKA_SEGMENTATION_CLASS, BIO_FORMATS_READER_CLASS),
    )

    assert found[IMAGE_SCIENCE_CLASS] == core_jar
    assert found[WEKA_SEGMENTATION_CLASS] == core_jar
    assert found[BIO_FORMATS_READER_CLASS] is None


def test_clean_runtime_adds_image_science_to_classpath(tmp_path):
    class Config:
        def __init__(self):
            self.paths = []

        def add_classpath(self, path):
            self.paths.append(path)

    class ScyJava:
        def __init__(self):
            self.config = Config()

    fiji_path = tmp_path / "Fiji.app"
    image_science_jar = fiji_path / "plugins" / "imagescience.jar"
    write_jar(image_science_jar, IMAGE_SCIENCE_CLASS)
    sj_mod = ScyJava()

    add_clean_runtime_dependencies(sj_mod, fiji_path)

    assert sj_mod.config.paths == [str(image_science_jar)]


def test_local_fiji_adds_image_science_to_classpath_before_init(tmp_path, monkeypatch):
    class Config:
        def __init__(self):
            self.paths = []
            self.options = []

        def set_java_constraints(self, **_kwargs):
            pass

        def add_option(self, option):
            self.options.append(option)

        def add_classpath(self, path):
            self.paths.append(path)

    class ScyJava:
        def __init__(self):
            self.config = Config()
            self.imports = []

        def jimport(self, name):
            self.imports.append(name)
            return object()

    class ImageJ:
        def __init__(self):
            self.init_calls = []

        def init(self, *args, **kwargs):
            self.init_calls.append((args, kwargs))
            return object()

    fiji_path = tmp_path / "Fiji.app"
    image_science_jar = fiji_path / "jars" / "imagescience.jar"
    write_jar(image_science_jar, IMAGE_SCIENCE_CLASS)
    sj_mod = ScyJava()
    imagej_mod = ImageJ()

    monkeypatch.setattr(imagej_runtime, "_ij", None)
    imagej_runtime._JAVA.clear()
    monkeypatch.setattr(imagej_runtime, "ensure_stdio", lambda: None)
    monkeypatch.setattr(imagej_runtime, "configure_java_caches", lambda: tmp_path)
    monkeypatch.setattr(imagej_runtime, "configure_java_home", lambda fiji_app_path, log_func: None)
    monkeypatch.setattr(imagej_runtime, "get_python_modules", lambda: (imagej_mod, sj_mod))
    monkeypatch.setattr(imagej_runtime, "requires_clean_fiji_runtime", lambda _path: False)
    monkeypatch.setattr(imagej_runtime, "_silence_java_during_imagej_init", lambda _sj, init_func: init_func())

    imagej_runtime.get_ij(fiji_path, log_func=lambda _message: None)

    assert sj_mod.config.paths == [str(image_science_jar)]
    assert "-Xmx6g" in sj_mod.config.options
    assert imagej_mod.init_calls[0][0] == (str(fiji_path),)
    assert "imagescience.image.Image" in sj_mod.imports


def test_offline_local_fiji_uses_bundled_imglyb_without_maven(tmp_path):
    class Config:
        def __init__(self):
            self.paths: list[str] = []
            self.endpoints: list[str] = [
                "net.imglib2:imglib2-imglyb:1.1.0",
                "example:kept:1",
            ]

        def add_classpath(self, path: str) -> None:
            self.paths.append(path)

    class ScyJava:
        def __init__(self) -> None:
            self.config = Config()

    fiji_path = tmp_path / "Fiji.app"
    expected = []
    for name in imagej_runtime._LOCAL_IMGLYB_JARS:
        path = fiji_path / "jars" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jar")
        expected.append(str(path))
    sj_mod = ScyJava()

    imagej_runtime._configure_local_imglyb(sj_mod, fiji_path, offline=True)

    assert sj_mod.config.paths == expected
    assert sj_mod.config.endpoints == ["example:kept:1"]


def test_offline_local_fiji_rejects_missing_imglyb_bridge(tmp_path):
    sj_mod = type("ScyJava", (), {"config": object()})()

    with pytest.raises(RuntimeError, match="missing its PyImageJ bridge"):
        imagej_runtime._configure_local_imglyb(sj_mod, tmp_path / "Fiji.app", offline=True)


def test_partial_biop_install_uses_clean_runtime(tmp_path):
    fiji_path = tmp_path / "Fiji.app"
    write_jar(fiji_path / "jars" / "biop.jar", BIOP_IMAGE_LOADER_SERVICE)

    assert requires_clean_fiji_runtime(fiji_path)


def test_offline_imagej_runtime_disables_java_downloads(monkeypatch, tmp_path):
    constraints = []

    class Config:
        def set_java_constraints(self, **kwargs):
            constraints.append(kwargs)

        def add_option(self, _option):
            pass

    class ScyJava:
        config = Config()

    monkeypatch.setenv("CELLONAUT_OFFLINE", "1")
    monkeypatch.setattr(imagej_runtime, "_ij", None)
    monkeypatch.setattr(imagej_runtime, "ensure_stdio", lambda: None)
    monkeypatch.setattr(imagej_runtime, "configure_java_caches", lambda: tmp_path)
    monkeypatch.setattr(imagej_runtime, "configure_java_home", lambda **_kwargs: None)
    monkeypatch.setattr(imagej_runtime, "get_python_modules", lambda: (object(), ScyJava()))

    with pytest.raises(RuntimeError, match="missing its bundled Fiji"):
        imagej_runtime.get_ij(None, log_func=lambda _message: None)

    assert constraints == [{"fetch": "never"}]


def test_offline_imagej_init_uses_bundled_java_version_hint(monkeypatch, tmp_path):
    def original_guess():
        return None

    class ImageJ:
        _guess_java_version = staticmethod(original_guess)

    observed = []
    java_home = tmp_path / "java"
    monkeypatch.setattr(imagej_runtime, "java_major_version", lambda path: 21 if path == java_home else None)

    result = imagej_runtime._init_with_local_java_version_hint(
        ImageJ,
        lambda: observed.append(ImageJ._guess_java_version()) or "initialized",
        java_home,
        offline=True,
    )

    assert result == "initialized"
    assert observed == [21]
    assert ImageJ._guess_java_version is original_guess


def test_complete_biop_install_keeps_local_runtime(tmp_path):
    fiji_path = tmp_path / "Fiji.app"
    write_jar(
        fiji_path / "jars" / "plugins.jar",
        BIOP_IMAGE_LOADER_SERVICE,
        MASTODON_PLUGIN_API,
    )

    assert not requires_clean_fiji_runtime(fiji_path)


def test_load_weka_classifier_raises_when_model_is_rejected(monkeypatch, tmp_path):
    class Segmentation:
        def loadClassifier(self, _path):
            return False

    class NoopContext:
        def __enter__(self):
            return None

        def __exit__(self, _exc_type, _exc_value, _traceback):
            return False

    monkeypatch.setattr(
        "cellonaut.io.imagej_runtime.suppress_java_stderr",
        lambda: NoopContext(),
    )
    model_path = tmp_path / "classifier.model"

    try:
        load_weka_classifier(Segmentation(), model_path)
    except RuntimeError as exc:
        assert str(model_path) in str(exc)
    else:
        raise AssertionError("Rejected Weka classifiers must raise RuntimeError")


def test_inspect_weka_classifier_returns_numbered_class_order(monkeypatch, tmp_path):
    class NoopContext:
        def __enter__(self):
            return None

        def __exit__(self, _exc_type, _exc_value, _traceback):
            return False

    class Segmentation:
        def loadClassifier(self, _path):
            return True

        def getNumOfClasses(self):
            return 3

        def getClassLabel(self, index):
            return ["background", "cell", "bud"][index]

    monkeypatch.setattr(imagej_runtime, "get_java_classes", lambda: {"WekaSegmentation": Segmentation})
    monkeypatch.setattr(imagej_runtime, "get_ij", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(imagej_runtime, "suppress_java_stderr", lambda: NoopContext())
    model_path = tmp_path / "classifier.model"

    assert inspect_weka_classifier(model_path, tmp_path / "Fiji.app") == ["background", "cell", "bud"]


def test_format_weka_runtime_error_explains_missing_imagescience():
    error = format_weka_runtime_error(
        RuntimeError("java.lang.NoClassDefFoundError: imagescience/image/Image"),
        mask_label="Hmg2 Mask",
    )

    assert "Hmg2 Mask" in str(error)
    assert "missing ImageScience" in str(error)


def test_format_weka_runtime_error_explains_java_heap():
    error = format_weka_runtime_error(
        RuntimeError("java.lang.OutOfMemoryError: Java heap space"),
        mask_label="BFP mask",
    )

    assert "BFP mask" in str(error)
    assert "CELLONAUT_JAVA_HEAP" in str(error)


def test_format_weka_runtime_error_explains_empty_classifier_result():
    error = format_weka_runtime_error(
        RuntimeError(
            'java.lang.NullPointerException: Cannot invoke "ij.ImagePlus.getStack()" because "classifiedSlices[i]" is null'
        ),
        mask_label="Hmg2 Mask",
    )

    assert "ImageScience" in str(error)
    assert "Java ran out of memory" in str(error)


def test_scan_fiji_installation_reports_ready_components(tmp_path):
    fiji_path = tmp_path / "Fiji.app"
    (fiji_path / "plugins").mkdir(parents=True)
    (fiji_path / "jars").mkdir(parents=True)
    (fiji_path / "macros").mkdir()
    (fiji_path / "ImageJ-win64.exe").write_text("", encoding="utf-8")
    make_portable_java_home(fiji_path / "java")
    write_jar(
        fiji_path / "plugins" / "core.jar",
        BIO_FORMATS_READER_CLASS,
        WEKA_SEGMENTATION_CLASS,
        IMAGE_SCIENCE_CLASS,
    )

    status = scan_fiji_installation(fiji_path)

    assert looks_like_fiji_installation(fiji_path)
    assert status.ready
    assert [component.ok for component in status.components]


def test_scan_fiji_installation_opens_each_jar_once(tmp_path, monkeypatch):
    fiji_path = tmp_path / "Fiji.app"
    (fiji_path / "plugins").mkdir(parents=True)
    (fiji_path / "jars").mkdir(parents=True)
    (fiji_path / "macros").mkdir()
    (fiji_path / "ImageJ-win64.exe").write_text("", encoding="utf-8")
    make_portable_java_home(fiji_path / "java")
    write_jar(
        fiji_path / "plugins" / "core.jar",
        BIO_FORMATS_READER_CLASS,
        WEKA_SEGMENTATION_CLASS,
        IMAGE_SCIENCE_CLASS,
    )
    real_zip_file = fiji_installation.zipfile.ZipFile
    opened_paths = []

    def tracked_zip_file(path, *args, **kwargs):
        jar_path = Path(path)
        if jar_path.is_relative_to(fiji_path):
            opened_paths.append(jar_path)
        return real_zip_file(path, *args, **kwargs)

    monkeypatch.setattr(fiji_installation.zipfile, "ZipFile", tracked_zip_file)

    status = scan_fiji_installation(fiji_path)

    assert status.ready
    assert opened_paths == [fiji_path / "plugins" / "core.jar"]


def test_scan_fiji_installation_stops_between_jars_when_cancelled(tmp_path):
    fiji_path = tmp_path / "Fiji.app"
    (fiji_path / "plugins").mkdir(parents=True)
    (fiji_path / "jars").mkdir()
    for index in range(2):
        write_jar(fiji_path / "plugins" / f"unused-{index}.jar", f"example/Unused{index}.class")
    checks = 0

    def cancel_after_first_jar():
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(InterruptedError, match="cancelled"):
        scan_fiji_installation(fiji_path, cancel_requested=cancel_after_first_jar)

    assert checks >= 3


def test_scan_fiji_installation_reports_missing_components_only_for_fiji_folder(tmp_path):
    plain_folder = tmp_path / "plain"
    plain_folder.mkdir()
    assert not scan_fiji_installation(plain_folder).looks_like_fiji

    fiji_path = tmp_path / "Fiji.app"
    (fiji_path / "plugins").mkdir(parents=True)
    (fiji_path / "jars").mkdir(parents=True)
    (fiji_path / "macros").mkdir()
    (fiji_path / "fiji").write_text("", encoding="utf-8")

    status = scan_fiji_installation(fiji_path)

    assert status.looks_like_fiji
    assert {component.key for component in status.missing_required} >= {"bioformats", "weka"}
    assert "imagescience" not in {component.key for component in status.missing_required}
    image_science = next(component for component in status.components if component.key == "imagescience")
    assert image_science.required is False
