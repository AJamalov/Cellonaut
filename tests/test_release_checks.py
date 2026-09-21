from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cellonaut.release_checks import smoke


def test_import_modules_reports_successes_and_failures(monkeypatch, capsys):
    imported: list[str] = []

    def fake_import_module(module_name: str):
        imported.append(module_name)
        if module_name == "missing.module":
            raise RuntimeError("not bundled")
        return object()

    monkeypatch.setattr(smoke.importlib, "import_module", fake_import_module)

    failures = smoke.import_modules(["ok.module", "missing.module"])

    assert imported == ["ok.module", "missing.module"]
    assert failures == ["missing.module: not bundled"]
    output = capsys.readouterr().out
    assert "OK import ok.module" in output
    assert "FAIL import missing.module: not bundled" in output


def test_smoke_main_can_skip_heavy_imports(monkeypatch, capsys):
    calls: list[list[str]] = []

    def fake_import_modules(module_names: list[str]) -> list[str]:
        calls.append(module_names)
        return []

    monkeypatch.setattr(smoke, "import_modules", fake_import_modules)

    exit_code = smoke.main(["--skip-heavy"])

    assert exit_code == 0
    assert calls == [smoke.BASE_IMPORTS]
    assert "Packaging smoke test passed." in capsys.readouterr().out


def test_smoke_main_returns_failure_when_imports_fail(monkeypatch, capsys):
    def fake_import_modules(module_names: list[str]) -> list[str]:
        return ["broken.module: import error"] if module_names is smoke.BASE_IMPORTS else []

    monkeypatch.setattr(smoke, "import_modules", fake_import_modules)

    exit_code = smoke.main(["--skip-heavy"])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "Packaging smoke test failed:" in output
    assert "broken.module: import error" in output


def test_windows_cuda_check_reports_missing_gpu_without_inference():
    torch_stub = SimpleNamespace(
        version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )

    assert smoke.validate_windows_cuda_inference(windows_release=True, torch_module=torch_stub) == (
        "no_compatible_gpu",
        [],
    )


def test_windows_cuda_check_rejects_cpu_only_torch():
    torch_stub = SimpleNamespace(version=SimpleNamespace(cuda=None))

    status, failures = smoke.validate_windows_cuda_inference(windows_release=True, torch_module=torch_stub)

    assert status == "failed"
    assert "expected CUDA 12.6" in failures[0]


def test_windows_cuda_check_runs_model_on_gpu(monkeypatch):
    calls = []

    class FakeModel:
        device = SimpleNamespace(type="cuda")

        def eval(self, image, **kwargs):
            calls.append((image.shape, kwargs))
            return np.zeros(image.shape, dtype=np.int32), None, None

    torch_stub = SimpleNamespace(
        version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=lambda: calls.append("synchronized")),
    )
    monkeypatch.setattr("cellonaut.cell_segmentation.core.clear_cellpose_model_cache", lambda: calls.append("cleared"))

    status, failures = smoke.validate_windows_cuda_inference(
        windows_release=True,
        torch_module=torch_stub,
        model_factory=FakeModel,
    )

    assert (status, failures) == ("passed", [])
    assert calls[0][0] == (96, 96)
    assert calls[1:] == ["synchronized", "cleared"]
