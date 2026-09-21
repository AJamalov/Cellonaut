"""Run the spec's filters without building or importing native Qt frameworks."""

import ast
from pathlib import Path

import pytest


@pytest.fixture
def filters():
    spec = Path(__file__).resolve().parents[1] / "release_tools/pyinstaller/Cellonaut.spec"
    tree = ast.parse(spec.read_text(encoding="utf-8"))
    names = {"unused_qt_binary_keys", "unused_qt_plugin_keys", "_binary_key",
             "_is_unused_qt_binary", "_is_unused_qt_data"}
    nodes: list[ast.stmt] = [node for node in tree.body if
                             isinstance(node, ast.FunctionDef) and node.name in names or
                             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(spec), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "name",
    ["Qt6QmlModels.dll", "Qt6Qml.dll", "Qt6QmlWorkerScript.dll", "Qt6Pdf.dll",
     "Qt6QmlMeta.dll", "Qt6VirtualKeyboard.dll", "Qt6Quick.dll"],
)
def test_excluded_windows_qt_binaries_are_removed(filters, name):
    item = (f"PySide6/{name}", "source", "BINARY")
    assert filters["_is_unused_qt_binary"](item)
    assert filters["_is_unused_qt_data"](item)


@pytest.mark.parametrize(
    "path",
    ["PySide6/Qt6Core.dll", "PySide6/Qt6Gui.dll", "PySide6/Qt6Widgets.dll",
     "PySide6/plugins/platforms/qwindows.dll"],
)
def test_required_windows_qt_binaries_survive(filters, path):
    assert not filters["_is_unused_qt_binary"]((path, "source", "BINARY"))
    assert not filters["_is_unused_qt_data"]((path, "source", "DATA"))
