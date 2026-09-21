[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$IncludeBuildOutputs,
    [switch]$IncludeInstallerArtifacts,
    [switch]$IncludeReleaseReports
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $ProjectRoot
$ProjectRootPrefix = [System.IO.Path]::GetFullPath($ProjectRoot.Path).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar

function Assert-SafeCleanupTarget {
    param([string]$TargetPath)

    $FullPath = [System.IO.Path]::GetFullPath($TargetPath)
    if (-not $FullPath.StartsWith($ProjectRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove outside workspace: $FullPath"
    }

    # Reject junctions and symlinks in any ancestor, not only at the final target.
    $CurrentPath = $FullPath
    while ($CurrentPath.StartsWith($ProjectRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        $Item = Get-Item -LiteralPath $CurrentPath -Force
        if ($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to remove through a link: $($Item.FullName)"
        }
        $CurrentPath = Split-Path -Parent $CurrentPath
    }
}

# Keep the default list limited to reproducible caches; build outputs and reports
# require explicit switches because they may still be under manual review.
$RelativePaths = @(
    ".pytest_cache",
    "pytest-of-*",
    "pytest-cache-files-*",
    ".*-pytest*",
    ".*_test_run_*",
    ".test-tmp-*",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
    ".coverage",
    ".coverage.*",
    ".local/pytest-*",
    "cellonaut.egg-info",
    "mutants",
    "pip-build-tracker-*",
    "pip-install-*",
    "pip-ephem-wheel-cache-*",
    "torchinductor_*"
)

if ($IncludeBuildOutputs) {
    $RelativePaths += @("build", "dist", "build_tools")
}
if ($IncludeInstallerArtifacts) {
    $RelativePaths += "installer_dist"
}

if ($IncludeReleaseReports) {
    $RelativePaths += @("release_reports")
}

$PythonCacheDirs = @()
foreach ($SourceDir in @("cellonaut", "tests", "release_tools")) {
    $SourcePath = Join-Path $ProjectRoot $SourceDir
    $PythonCacheDirs += Get-ChildItem -LiteralPath $SourcePath -Recurse -Force -Directory -Filter "__pycache__"
}
$RelativePaths += "__pycache__"
$CleanupFailures = @()
# Resolve and verify every recursive target before deletion so wildcard matches
# and symlinks cannot carry cleanup outside the repository.
foreach ($CacheDir in $PythonCacheDirs) {
    Assert-SafeCleanupTarget $CacheDir.FullName

    if ($PSCmdlet.ShouldProcess($CacheDir.FullName, "Remove Python bytecode cache")) {
        try {
            Remove-Item -LiteralPath $CacheDir.FullName -Recurse -Force -ErrorAction Stop
        } catch {
            $CleanupFailures += $CacheDir.FullName
            Write-Warning "Could not remove $($CacheDir.FullName): $($_.Exception.Message)"
        }
    }
}
foreach ($RelativePath in $RelativePaths) {
    $Path = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $Path)) {
        continue
    }

    $ResolvedPaths = @(Resolve-Path $Path)
    foreach ($Resolved in $ResolvedPaths) {
        Assert-SafeCleanupTarget $Resolved.Path

        if ($PSCmdlet.ShouldProcess($Resolved.Path, "Remove local generated artifact")) {
            try {
                Remove-Item -LiteralPath $Resolved.Path -Recurse -Force -ErrorAction Stop
            } catch {
                $CleanupFailures += $Resolved.Path
                Write-Warning "Could not remove $($Resolved.Path): $($_.Exception.Message)"
            }
        }
    }
}
if ($CleanupFailures.Count -gt 0) {
    throw "Could not remove $($CleanupFailures.Count) cleanup targets; see warnings above."
}
