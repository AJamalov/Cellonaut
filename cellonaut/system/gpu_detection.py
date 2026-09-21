"""Detect the local compute backend available for Cellpose/PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

from cellonaut.system.subprocesses import hidden_window_kwargs


CELLPOSE_BACKEND_ENV = "CELLONAUT_CELLPOSE_BACKEND"
CELLPOSE_BACKEND_MARKER = "CELLPOSE_BACKEND.txt"
CELLPOSE_BACKEND_AUTO = "auto"
CELLPOSE_BACKEND_CPU = "cpu"


@dataclass(frozen=True)
class ComputeBackendStatus:
    """User-facing summary of the best Cellpose compute backend Cellonaut can use."""

    status: str
    label: str
    detail: str
    device_names: tuple[str, ...] = ()


def installed_cellpose_backend_preference() -> str:
    """Return the Windows installer choice, defaulting safely to automatic selection."""
    if sys.platform.startswith("win") and getattr(sys, "frozen", False):
        marker = Path(sys.executable).resolve().parent / CELLPOSE_BACKEND_MARKER
        try:
            configured = marker.read_text(encoding="ascii").strip().casefold()
        except OSError:
            configured = ""
        if configured in {CELLPOSE_BACKEND_AUTO, CELLPOSE_BACKEND_CPU}:
            return configured
    configured = os.environ.get(CELLPOSE_BACKEND_ENV, "").strip().casefold()
    if configured in {CELLPOSE_BACKEND_AUTO, CELLPOSE_BACKEND_CPU}:
        return configured
    return CELLPOSE_BACKEND_AUTO


def cellpose_acceleration_enabled(
    requested: bool,
    *,
    torch_module=None,
    system_name: str | None = None,
    backend_preference: str | None = None,
) -> bool:
    """Resolve pipeline intent against the installed CPU/automatic backend choice."""
    if not requested:
        return False
    system = (system_name or platform.system()).lower()
    if system != "windows":
        return False
    preference = backend_preference or installed_cellpose_backend_preference()
    if preference == CELLPOSE_BACKEND_CPU:
        return False
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception:
            return False
    try:
        return bool(torch_module.cuda.is_available())
    except Exception:
        return False


# Normalize and deduplicate OS output so user messages remain stable across detection methods.
def _normalize_device_names(names: Iterable[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for name in names:
        text = " ".join(str(name or "").strip().split())
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return tuple(cleaned)


# Vendor matching uses product-family tokens because OS tools do not expose one common vendor field.
def _device_vendor(names: Iterable[str]) -> str:
    joined = " ".join(names).casefold()
    if any(token in joined for token in ("nvidia", "geforce", "quadro", "tesla", "rtx", "gtx")):
        return "nvidia"
    if any(token in joined for token in ("amd", "advanced micro devices", "radeon", "firepro")):
        return "amd"
    if any(token in joined for token in ("apple", "m1", "m2", "m3", "m4", "m5")):
        return "apple"
    if any(token in joined for token in ("intel", "arc", "iris", "uhd graphics")):
        return "intel"
    return "unknown"


# Prefer CIM output on Windows, with registry detection available when PowerShell is restricted.
@lru_cache(maxsize=1)
def _windows_gpu_names() -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            **hidden_window_kwargs(),
        )
    except Exception:
        completed = None
    if completed is not None and completed.returncode == 0:
        names = _normalize_device_names(completed.stdout.splitlines())
        if names:
            return names
    return _windows_registry_gpu_names()


# Read PCI instances directly so packaged builds can still identify GPUs without CIM access.
def _windows_registry_gpu_names() -> tuple[str, ...]:
    try:
        import winreg as winreg_module
    except Exception:
        return ()
    winreg: Any = winreg_module

    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Enum\PCI")
    except OSError:
        return ()

    names: list[str] = []
    vendor_tokens = ("amd", "radeon", "nvidia", "geforce", "quadro", "rtx", "gtx", "intel", "arc")
    graphics_tokens = ("radeon", "nvidia", "geforce", "quadro", "rtx", "gtx", "graphics", "display", "arc")
    skip_tokens = ("audio", "bridge", "controller", "smbus", "usb", "sata", "host", "ethernet", "nvme")

    # Registry descriptions often include a driver prefix separated by a semicolon.
    def parse_device_desc(value: str) -> str:
        text = str(value or "").strip()
        if ";" in text:
            text = text.rsplit(";", 1)[-1].strip()
        return text

    # Filter PCI devices carefully because vendor names also appear on audio and bus controllers.
    def read_instance(parent, instance_name: str) -> None:
        try:
            instance = winreg.OpenKey(parent, instance_name)
        except OSError:
            return
        try:
            desc, _ = winreg.QueryValueEx(instance, "DeviceDesc")
        except OSError:
            return
        name = parse_device_desc(desc)
        lowered = name.casefold()
        if not any(token in lowered for token in vendor_tokens):
            return
        if not any(token in lowered for token in graphics_tokens):
            return
        if any(token in lowered for token in skip_tokens):
            return
        names.append(name)

    try:
        index = 0
        while True:
            try:
                device_name = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            try:
                device = winreg.OpenKey(root, device_name)
            except OSError:
                continue
            instance_index = 0
            while True:
                try:
                    instance_name = winreg.EnumKey(device, instance_index)
                except OSError:
                    break
                instance_index += 1
                read_instance(device, instance_name)
    finally:
        try:
            winreg.CloseKey(root)
        except Exception:
            pass

    return _normalize_device_names(names)


# Cellonaut currently ships only for Windows, so unsupported hosts do not run
# platform-specific probing commands.
def detect_gpu_names() -> tuple[str, ...]:
    return _windows_gpu_names() if platform.system().lower() == "windows" else ()


# Trust PyTorch runtime availability over hardware detection when deciding whether acceleration works.
def _torch_accelerator_status(torch_module) -> tuple[bool, str, tuple[str, ...], str]:
    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception:
        return False, "", (), ""
    if not cuda_available:
        return False, "", (), ""
    names = []
    try:
        count = int(torch_module.cuda.device_count())
    except Exception:
        count = 0
    for index in range(max(count, 1)):
        try:
            names.append(str(torch_module.cuda.get_device_name(index)))
        except Exception:
            pass
    version = getattr(torch_module, "version", object())
    cuda_version = str(getattr(version, "cuda", "") or "")
    return True, "CUDA", _normalize_device_names(names), cuda_version


# Explain both detected hardware and usable runtime so CPU fallback is never mistaken for GPU use.
def describe_cellpose_backend(
    torch_module=None,
    gpu_names: Iterable[str] | None = None,
    *,
    acceleration_allowed: bool | None = None,
    acceleration_requested: bool | None = None,
) -> ComputeBackendStatus:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception as exc:
            names = detect_gpu_names() if gpu_names is None else _normalize_device_names(gpu_names)
            device_text = "; ".join(names) if names else "No graphics adapter details detected"
            return ComputeBackendStatus(
                "WARNING",
                "Cellpose backend",
                f"PyTorch could not be imported, so Cellpose will not run until the installation is fixed. Detected hardware: {device_text}. Original error: {exc}",
                names,
            )

    names = detect_gpu_names() if gpu_names is None else _normalize_device_names(gpu_names)
    system = platform.system().lower()
    installer_cpu_mode = system == "windows" and installed_cellpose_backend_preference() == CELLPOSE_BACKEND_CPU
    if acceleration_allowed is None:
        acceleration_allowed = not installer_cpu_mode
    if not acceleration_allowed:
        if installer_cpu_mode:
            detail = "CPU-only mode was selected during installation. Cellpose will run on CPU."
        elif system == "windows" and acceleration_requested:
            detail = "CUDA is unavailable, so Automatic mode is using CPU for Cellpose."
        else:
            detail = "GPU acceleration is disabled for this analysis. Cellpose will run on CPU."
        return ComputeBackendStatus(
            "OK",
            "Cellpose backend",
            detail,
            names,
        )
    accelerator_available, accelerator_name, accelerator_devices, accelerator_version = _torch_accelerator_status(
        torch_module
    )
    if accelerator_available:
        devices = accelerator_devices or names
        device_text = "; ".join(devices) if devices else f"{accelerator_name} device"
        version_text = f" with {accelerator_name} {accelerator_version}" if accelerator_version else ""
        return ComputeBackendStatus(
            "OK",
            "Cellpose backend",
            f"{accelerator_name} acceleration is available{version_text}: {device_text}.",
            devices,
        )

    vendor = _device_vendor(names)
    device_text = "; ".join(names) if names else "No dedicated GPU detected"

    if vendor == "amd" and system == "windows":
        return ComputeBackendStatus(
            "WARNING",
            "Cellpose backend",
            f"AMD graphics detected ({device_text}), but Cellpose does not support ROCm acceleration on Windows. This Cellonaut install will run Cellpose on CPU.",
            names,
        )

    if vendor == "nvidia":
        return ComputeBackendStatus(
            "WARNING",
            "Cellpose backend",
            f"NVIDIA graphics detected ({device_text}), but CUDA is not available to PyTorch. "
            "Automatic mode will use CPU; update the NVIDIA driver if CUDA should be available.",
            names,
        )

    return ComputeBackendStatus(
        "OK",
        "Cellpose backend",
        f"Cellpose will run on CPU. Detected hardware: {device_text}.",
        names,
    )
