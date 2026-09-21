from __future__ import annotations

import sys
from types import SimpleNamespace


from cellonaut.system import gpu_detection
from cellonaut.system.gpu_detection import describe_cellpose_backend


class _CudaStub:
    def __init__(self, available=False, names=()):
        self._available = available
        self._names = tuple(names)

    def is_available(self):
        return self._available

    def device_count(self):
        return len(self._names)

    def get_device_name(self, index):
        return self._names[index]


def _torch_stub(cuda_available=False, cuda_names=(), cuda_version=""):
    return SimpleNamespace(
        cuda=_CudaStub(cuda_available, cuda_names),
        version=SimpleNamespace(cuda=cuda_version),
    )


def test_describe_backend_reports_cuda_when_available(monkeypatch):
    monkeypatch.setattr(gpu_detection.platform, "system", lambda: "Windows")

    backend = describe_cellpose_backend(
        torch_module=_torch_stub(True, ["NVIDIA RTX 4070"], "12.8"),
        gpu_names=["NVIDIA RTX 4070"],
    )

    assert backend.status == "OK"
    assert "CUDA acceleration is available with CUDA 12.8" in backend.detail
    assert backend.device_names == ("NVIDIA RTX 4070",)


def test_describe_backend_reports_installer_cpu_choice(monkeypatch):
    monkeypatch.setattr(gpu_detection.platform, "system", lambda: "Windows")
    monkeypatch.setenv(gpu_detection.CELLPOSE_BACKEND_ENV, "cpu")

    backend = describe_cellpose_backend(
        torch_module=_torch_stub(True, ["NVIDIA RTX 4070"], "12.6"),
        gpu_names=["NVIDIA RTX 4070"],
        acceleration_allowed=False,
    )

    assert backend.status == "OK"
    assert "CPU-only mode was selected during installation" in backend.detail
    assert "Cellpose will run on CPU" in backend.detail


def test_describe_backend_distinguishes_analysis_cpu_choice(monkeypatch):
    monkeypatch.setattr(gpu_detection.platform, "system", lambda: "Windows")
    monkeypatch.setenv(gpu_detection.CELLPOSE_BACKEND_ENV, "auto")

    backend = describe_cellpose_backend(
        torch_module=_torch_stub(True, ["NVIDIA RTX 4070"], "12.6"),
        gpu_names=["NVIDIA RTX 4070"],
        acceleration_allowed=False,
    )

    assert "disabled for this analysis" in backend.detail
    assert "selected during installation" not in backend.detail


def test_describe_backend_reports_automatic_cpu_fallback(monkeypatch):
    monkeypatch.setattr(gpu_detection.platform, "system", lambda: "Windows")
    monkeypatch.setenv(gpu_detection.CELLPOSE_BACKEND_ENV, "auto")

    backend = describe_cellpose_backend(
        torch_module=_torch_stub(cuda_available=False),
        gpu_names=["NVIDIA RTX 4070"],
        acceleration_allowed=False,
        acceleration_requested=True,
    )

    assert "Automatic mode is using CPU" in backend.detail
    assert "disabled for this analysis" not in backend.detail


def test_installed_backend_preference_uses_environment(monkeypatch):
    monkeypatch.setenv(gpu_detection.CELLPOSE_BACKEND_ENV, "CPU")

    assert gpu_detection.installed_cellpose_backend_preference() == "cpu"


def test_installed_backend_preference_reads_frozen_windows_marker(monkeypatch, tmp_path):
    executable = tmp_path / "Cellonaut.exe"
    executable.write_bytes(b"")
    (tmp_path / gpu_detection.CELLPOSE_BACKEND_MARKER).write_text("cpu\n", encoding="ascii")
    monkeypatch.setenv(gpu_detection.CELLPOSE_BACKEND_ENV, "auto")
    monkeypatch.setattr(gpu_detection.sys, "platform", "win32")
    monkeypatch.setattr(gpu_detection.sys, "frozen", True, raising=False)
    monkeypatch.setattr(gpu_detection.sys, "executable", str(executable))

    assert gpu_detection.installed_cellpose_backend_preference() == "cpu"


def test_windows_automatic_backend_requires_available_cuda():
    assert gpu_detection.cellpose_acceleration_enabled(
        True,
        torch_module=_torch_stub(cuda_available=True),
        system_name="Windows",
        backend_preference="auto",
    )
    assert not gpu_detection.cellpose_acceleration_enabled(
        True,
        torch_module=_torch_stub(cuda_available=False),
        system_name="Windows",
        backend_preference="auto",
    )
    assert not gpu_detection.cellpose_acceleration_enabled(
        True,
        torch_module=_torch_stub(cuda_available=True),
        system_name="Windows",
        backend_preference="cpu",
    )
    assert not gpu_detection.cellpose_acceleration_enabled(
        True, torch_module=_torch_stub(cuda_available=False), system_name="Unsupported"
    )


def test_describe_backend_warns_for_amd_on_windows(monkeypatch):
    monkeypatch.setattr(gpu_detection.platform, "system", lambda: "Windows")

    backend = describe_cellpose_backend(
        torch_module=_torch_stub(),
        gpu_names=["AMD Radeon RX 7800 XT"],
    )

    assert backend.status == "WARNING"
    assert "AMD graphics detected" in backend.detail
    assert "does not support ROCm acceleration on Windows" in backend.detail
    assert "CPU" in backend.detail


def test_windows_backend_warning_explains_automatic_cpu_fallback(monkeypatch):
    monkeypatch.setattr(gpu_detection.platform, "system", lambda: "Windows")

    backend = describe_cellpose_backend(
        torch_module=_torch_stub(cuda_available=False),
        gpu_names=["NVIDIA RTX 4070"],
    )

    assert backend.status == "WARNING"
    assert "Automatic mode will use CPU" in backend.detail
    assert "Install the CUDA profile" not in backend.detail


def test_describe_backend_reports_cpu_when_no_gpu_known(monkeypatch):
    monkeypatch.setattr(gpu_detection.platform, "system", lambda: "Windows")

    backend = describe_cellpose_backend(
        torch_module=_torch_stub(),
        gpu_names=[],
    )

    assert backend.status == "OK"
    assert "Cellpose will run on CPU" in backend.detail


def test_windows_gpu_probe_is_hidden_and_cached(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="NVIDIA RTX 4070\n")

    gpu_detection._windows_gpu_names.cache_clear()
    monkeypatch.setattr(gpu_detection.subprocess, "run", fake_run)

    assert gpu_detection._windows_gpu_names() == ("NVIDIA RTX 4070",)
    assert gpu_detection._windows_gpu_names() == ("NVIDIA RTX 4070",)
    assert len(calls) == 1
    if sys.platform.startswith("win"):
        assert calls[0][1]["creationflags"] == gpu_detection.subprocess.CREATE_NO_WINDOW  # pyright: ignore[reportAttributeAccessIssue]
        assert calls[0][1]["startupinfo"].dwFlags & gpu_detection.subprocess.STARTF_USESHOWWINDOW  # pyright: ignore[reportAttributeAccessIssue]

    gpu_detection._windows_gpu_names.cache_clear()
