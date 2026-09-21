"""Validate the CUDA packages used by the Windows release build."""

from __future__ import annotations


EXPECTED_CUDA_VERSION = "12.6"
EXPECTED_BUILD_TAG = "+cu126"


def validate_versions(torch_version: str, torchvision_version: str, cuda_version: str | None) -> str | None:
    """Return an error when the installed packages do not match the release profile."""
    if (
        cuda_version == EXPECTED_CUDA_VERSION
        and EXPECTED_BUILD_TAG in torch_version
        and EXPECTED_BUILD_TAG in torchvision_version
    ):
        return None
    return (
        f"Expected CUDA {EXPECTED_CUDA_VERSION} PyTorch and torchvision, "
        f"got torch={torch_version}, torchvision={torchvision_version}, CUDA={cuda_version}"
    )


def main() -> int:
    import torch
    import torchvision

    cuda_version = torch.version.cuda
    problem = validate_versions(torch.__version__, torchvision.__version__, cuda_version)
    if problem:
        print(f"ERROR: {problem}")
        return 1
    print(
        f"CUDA runtime bundled: torch={torch.__version__}, "
        f"torchvision={torchvision.__version__}, CUDA={cuda_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
