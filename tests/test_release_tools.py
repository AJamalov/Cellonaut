from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_windows_release_scripts_include_user_facing_artifacts():
    build_script = PROJECT_ROOT / "release_tools" / "scripts" / "build_release.ps1"
    installer_script = PROJECT_ROOT / "release_tools" / "scripts" / "build_installer.bat"
    installer_spec = PROJECT_ROOT / "release_tools" / "installer" / "CellonautInstaller.iss"

    build_text = build_script.read_text(encoding="utf-8")
    installer_text = installer_script.read_text(encoding="utf-8")
    spec_text = installer_spec.read_text(encoding="utf-8")

    assert "README_WINDOWS.txt" in build_text
    assert "dist\\Cellonaut\\Cellonaut.exe" in build_text
    assert '[ValidateSet("standard", "cuda126", "current")]' in build_text
    assert '[string]$Profile = "cuda126"' in build_text
    assert "cellonaut.release_checks.cuda_runtime" in build_text
    assert "windows-standard" in build_text
    assert "windows-cuda126" in build_text
    assert "THIRD_PARTY_LICENSES" in build_text
    assert "BUNDLED_COMPONENTS.md" in build_text
    assert "COPYRIGHT" in build_text
    assert "CITATIONS.md" in build_text
    assert "SOURCE_AVAILABILITY.md" in build_text
    assert "pip_audit" in build_text
    assert "cellonaut.release_checks.python_runtime" in build_text
    assert '"pip==26.2.1"' in build_text
    assert '"setuptools==83.0.0"' in build_text
    assert '"wheel==0.47.0"' in build_text
    assert build_text.index('"pip==26.2.1"') < build_text.index('"pip_audit"')
    assert "SkipAudit" in build_text
    assert "release_checks.dependency_lock" in build_text
    assert "pylock." in build_text
    assert "pip==26.2.1" in build_text
    assert "requirements-packaging.txt" not in build_text
    assert "BUILD_PROFILE.txt" in build_text
    assert build_text.index('Set-Content -Path $ProfileMarker') < build_text.index('$PackagedRuntimeResult =')
    assert "BUILD_INFO.json" in build_text
    assert "--release-smoke" in build_text
    assert "--release-runtime-check" in build_text
    assert "cellonaut.release_checks.packaged_results" in build_text
    assert "--require-cuda-check" in build_text
    assert "assert data[" not in build_text
    assert "release_checks.offline_assets" in build_text
    assert "release_checks.release_documents" in build_text
    assert "Preparing complete offline runtime assets" in build_text
    assert "release_tools\\templates\\README_WINDOWS.txt" in build_text
    assert "build_provenance verify" in installer_text
    assert "cellonaut.release_checks.python_runtime" in installer_text
    assert "release_checks.checksum" in installer_text
    assert "--input-pattern" in installer_text
    assert "%OUTPUT_STEM%.sha256" in installer_text
    assert "%LocalAppData%\\Programs\\Inno Setup 6\\ISCC.exe" in installer_text
    assert 'set "OUTPUT_STEM=Cellonaut-%CELLONAUT_VERSION%-windows"' in installer_text
    assert 'set "OUTPUT_NAME=%OUTPUT_STEM%.exe"' in installer_text
    assert 'if /I not "%CELLONAUT_PROFILE%"=="cuda126"' in installer_text
    assert "#define ProjectRoot ExtractFileDir(ExtractFileDir(ExtractFileDir(SourcePath)))" in spec_text
    assert "README_WINDOWS.txt" in spec_text
    assert "Windows quick start" in spec_text
    assert "Bundled components" in spec_text
    assert "Fiji (optional plugins)" in spec_text
    assert "DiskSpanning=yes" in spec_text
    assert "DiskSliceSize=" in spec_text
    assert "ArchitecturesAllowed=x64compatible" in spec_text
    assert "ArchitecturesInstallIn64BitMode=x64compatible" in spec_text
    assert '[InstallDelete]' in spec_text
    assert 'Type: filesandordirs; Name: "{app}\\_internal"' in spec_text
    assert 'Name: "cellposeauto"' in spec_text
    assert 'Name: "cellposecpu"' in spec_text
    assert "CELLPOSE_BACKEND.txt" in spec_text
    assert "WizardIsTaskSelected('cellposecpu')" in spec_text


def test_windows_runtime_requirement_profiles_are_explicit():
    project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.13"' in project
    assert "facebookresearch/dinov3" not in project
    assert "windows-cuda126" in project
    assert "linux-cpu" not in project
    for deferred_path in (
        "requirements-linux-cpu.txt",
        "requirements-macos.txt",
        ".github/workflows/linux-build.yml",
        ".github/workflows/macos-build.yml",
        "release_tools/scripts/build_release_linux.sh",
        "release_tools/scripts/build_release_macos.sh",
    ):
        assert not (PROJECT_ROOT / deferred_path).exists()


def test_test_workflow_uses_supported_python():
    workflow = PROJECT_ROOT / ".github" / "workflows" / "tests.yml"
    text = workflow.read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)
    jobs = document["jobs"]
    steps = {step["name"]: step for step in jobs["windows-tests"]["steps"]}

    assert set(jobs) == {"windows-tests"}
    assert jobs["windows-tests"]["runs-on"] == "windows-latest"
    assert steps["Set up Python"]["with"]["python-version-file"] == ".python-version"
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12.10"
    assert "release_checks.release_documents --project-root ." in text
    assert "python -m pip install -e ." in text
    assert "requirements-dev.txt" in text
    assert "requirements.txt" not in text
    assert "requirements-linux-cpu.txt" not in text
    assert "requirements-macos.txt" not in text
    assert "python -m ruff check ." in text
    assert "python -m mypy" in text
    assert "python -m pyright" in text
    assert "--cov=cellonaut --cov-branch --cov-fail-under=70" in text
    assert "runs-on: macos-15" not in text
    assert "runs-on: ubuntu-22.04" not in text
    assert text.splitlines().count("          python -m pip install --upgrade pip==26.2.1") == 1


def test_mutation_workflow_targets_deterministic_core_modules():
    workflow = PROJECT_ROOT / ".github" / "workflows" / "mutation-tests.yml"
    text = workflow.read_text(encoding="utf-8")
    project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements-mutation.txt").read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)
    job = document["jobs"]["focused-mutations"]
    steps = {step["name"]: step for step in job["steps"]}

    assert document["name"] == "Diagnostic Mutation Report"
    assert job["name"] == "Report deterministic core mutations"
    assert steps["Set up Python"]["with"]["python-version-file"] == ".python-version"
    assert "pip==26.2.1" in steps["Install runtime and mutation-test dependencies"]["run"]
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "ubuntu-22.04" in text
    assert "timeout-minutes: 60" in text
    assert "mutmut run" in text
    assert "mutmut run || mutation_status=$?" in text
    assert "mutmut results" in text
    assert 'exit "$mutation_status"' in text
    assert "[tool.mutmut]" in project
    assert 'mutate_only_covered_lines = true' in project
    assert "use_setproctitle = false" in project
    assert '"cellonaut/measurement/math.py"' in project
    assert '"cellonaut/masks/cell_qc.py"' in project
    assert '"cellonaut/pipeline/summary.py"' in project
    assert '"cellonaut/pipeline/validation.py"' in project
    assert "mutmut==3.7.0" in requirements
    assert not (PROJECT_ROOT / "requirements.txt").exists()


def test_windows_release_build_powershell_has_valid_syntax():
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")

    script = PROJECT_ROOT / "release_tools" / "scripts" / "build_release.ps1"
    escaped_script = str(script).replace("'", "''")
    command = (
        "$tokens = $null; $errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{escaped_script}', "
        "[ref]$tokens, [ref]$errors) > $null; "
        "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )

    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_local_cleanup_includes_mutation_and_parallel_coverage_caches():
    cleanup_script = PROJECT_ROOT / "release_tools" / "scripts" / "clean_local_artifacts.ps1"
    source = cleanup_script.read_text(encoding="utf-8")

    assert '"mutants"' in source
    assert '".coverage.*"' in source
    assert "Assert-SafeCleanupTarget" in source


def test_pyinstaller_spec_is_windows_only():
    spec = PROJECT_ROOT / "release_tools" / "pyinstaller" / "Cellonaut.spec"
    source = spec.read_text(encoding="utf-8")

    compile(source, str(spec), "exec")
    assert 'if not sys.platform.startswith("win"):' in source
    assert "currently built for Windows only" in source
    assert "is_macos" not in source
    assert "is_linux" not in source
    assert "BUNDLE(" not in source
    assert "icon=str(icon_file) if icon_file.exists()" in source
    assert "collect_data_files(pkg, include_py_files=True)" in source
    assert "\"triton\"" in source
    assert "\"pytorch-triton-rocm\"" in source
    assert "dinov3" not in source.casefold()
    assert "copy_metadata(pkg)" in source
    assert "offline_assets_dir" in source
    assert '(str(offline_assets_dir / "cellpose_models"), "offline/cellpose_models")' in source
    assert '(str(offline_assets_dir / "Fiji.app"), "offline/Fiji.app")' in source


def test_pyinstaller_spec_does_not_bulk_collect_pyside6():
    spec = PROJECT_ROOT / "release_tools" / "pyinstaller" / "Cellonaut.spec"
    source = spec.read_text(encoding="utf-8")

    assert 'collect_data_files("PySide6")' not in source
    assert 'collect_dynamic_libs("PySide6")' not in source
    assert '"PySide6",' not in source
    assert '"shiboken6",' not in source
    assert '"PySide6.QtWebEngineCore"' in source
    assert '"PySide6.QtQml"' in source
    assert '"qtvirtualkeyboardplugin"' in source
    assert '"qpdf"' in source
    assert '"qt6quick"' in source
    assert '"qtquick"' not in source
    assert '".dylib"' not in source
    assert '".so"' not in source
    assert '"/translations/"' in source
    assert "analysis.binaries = [" in source
    assert "if not _is_unused_qt_binary(item)" in source
    assert "analysis.datas = [" in source
    assert "if not _is_unused_qt_data(item)" in source
