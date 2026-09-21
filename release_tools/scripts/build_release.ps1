param(
    [string]$PythonExe = "",
    [switch]$SkipInstall,
    [switch]$SkipTests,
    [switch]$SkipAudit,
    [switch]$SkipHeavyImports,
    [switch]$Clean,
    [switch]$AllowDirty,
    [ValidateSet("standard", "cuda126", "current")]
    [string]$Profile = "cuda126"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $ProjectRoot

$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not $PythonExe) {
    if (Test-Path -LiteralPath $ProjectPython) {
        $PythonExe = (Resolve-Path -LiteralPath $ProjectPython).Path
    } else {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
    }
}

$ReportDir = Join-Path $ProjectRoot "release_reports"
$ReportPath = Join-Path $ReportDir ("build_report_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

# Mirror numbered stages to the terminal and report so a failed build remains
# understandable after its interactive output is gone.
function Write-Step {
    param([string]$Message)
    Write-Output ""
    Write-Output $Message
    Add-Content -Path $ReportPath -Value ""
    Add-Content -Path $ReportPath -Value $Message
}

# Temporarily relax PowerShell's native-command handling so output can pass
# through Tee-Object while the real process exit code remains authoritative.
function Invoke-LoggedCommand {
    param([string[]]$Command)
    Add-Content -Path $ReportPath -Value ("> " + ($Command -join " "))

    $PreviousErrorActionPreference = $ErrorActionPreference
    $NativePreferenceExists = Test-Path Variable:\PSNativeCommandUseErrorActionPreference
    if ($NativePreferenceExists) {
        $PreviousNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }

    try {
        $ErrorActionPreference = "Continue"
        & $Command[0] @($Command | Select-Object -Skip 1) 2>&1 |
            ForEach-Object { "$_" } |
            Tee-Object -FilePath $ReportPath -Append
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
        if ($NativePreferenceExists) {
            $PSNativeCommandUseErrorActionPreference = $PreviousNativePreference
        }
    }

    if ($ExitCode -ne 0) {
        throw "Command failed: $($Command -join ' ')"
    }
}

Write-Step "[1/10] Cellonaut release build"
Add-Content -Path $ReportPath -Value ("Started: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Add-Content -Path $ReportPath -Value ("Project: {0}" -f $ProjectRoot)
Add-Content -Path $ReportPath -Value ("Install profile: {0}" -f $Profile)

Write-Step "[2/10] Checking Python"
Write-Output "Python executable: $PythonExe"
Add-Content -Path $ReportPath -Value ("Python executable: {0}" -f $PythonExe)
Invoke-LoggedCommand @($PythonExe, "--version")
Invoke-LoggedCommand @(
    $PythonExe, "-m", "cellonaut.release_checks.python_runtime",
    "--project-root", $ProjectRoot
)
# Pin the lock producer/installer itself so dependency resolution semantics do not drift.
Invoke-LoggedCommand @(
    $PythonExe, "-m", "pip", "install", "--upgrade",
    "pip==26.2.1", "setuptools==83.0.0", "wheel==0.47.0"
)
$AppVersion = (& $PythonExe "-c" "from cellonaut.version import __version__; print(__version__)").Trim()
if ($LASTEXITCODE -ne 0 -or -not $AppVersion) {
    throw "Could not read the Cellonaut version."
}
Add-Content -Path $ReportPath -Value ("Cellonaut version: {0}" -f $AppVersion)
Write-Output "Cellonaut version: $AppVersion"

$BuildInfoStaging = Join-Path $ReportDir ("build_info_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
$BuildInfoArgs = @(
    $PythonExe,
    "-m",
    "cellonaut.release_checks.build_provenance",
    "write",
    "--output",
    $BuildInfoStaging,
    "--project-root",
    $ProjectRoot,
    "--profile",
    $Profile
)
if ($AllowDirty) {
    $BuildInfoArgs += "--allow-dirty"
}
Invoke-LoggedCommand $BuildInfoArgs

$DependencyLock = ""
if (-not $SkipInstall) {
    Write-Step "[3/10] Locking and installing Windows release dependencies"
    $LockProfile = switch ($Profile) {
        "standard" { "windows-standard" }
        "cuda126" { "windows-cuda126" }
        "current" { throw "The current profile is for development only. Use -SkipInstall or select standard/cuda126." }
    }
    $DependencyLockDir = Join-Path $ReportDir "dependency_locks"
    Invoke-LoggedCommand @(
        $PythonExe, "-m", "cellonaut.release_checks.dependency_lock", $LockProfile,
        "--output-dir", $DependencyLockDir, "--project-root", $ProjectRoot, "--python", $PythonExe
    )
    $DependencyLock = Join-Path $DependencyLockDir ("pylock.{0}.toml" -f $LockProfile)
    Invoke-LoggedCommand @($PythonExe, "-m", "pip", "install", "-r", $DependencyLock)
    Invoke-LoggedCommand @($PythonExe, "-m", "pip", "install", "-e", ".", "--no-deps")
    Invoke-LoggedCommand @($PythonExe, "-m", "pip", "cache", "purge")
} else {
    Write-Step "[3/10] Skipping dependency install"
}

if ($Profile -eq "cuda126") {
    Invoke-LoggedCommand @(
        $PythonExe,
        "-m",
        "cellonaut.release_checks.cuda_runtime"
    )
}

Write-Step "[4/10] Preparing complete offline runtime assets"
$OfflineAssetsRoot = Join-Path $ProjectRoot ".local\release_assets"
Invoke-LoggedCommand @(
    $PythonExe,
    "-m",
    "cellonaut.release_checks.offline_assets",
    "prepare",
    "--output",
    $OfflineAssetsRoot
)
Invoke-LoggedCommand @(
    $PythonExe,
    "-m",
    "cellonaut.release_checks.release_documents",
    "--project-root",
    $ProjectRoot,
    "--offline-assets",
    (Join-Path $OfflineAssetsRoot "offline")
)

Write-Step "[5/10] Running packaging smoke test"
$SmokeArgs = @($PythonExe, "-m", "cellonaut.release_checks.smoke")
if ($SkipHeavyImports) {
    $SmokeArgs += "--skip-heavy"
}
Invoke-LoggedCommand $SmokeArgs

if (-not $SkipAudit) {
    Write-Step "[6/10] Auditing installed dependencies"
    Invoke-LoggedCommand @($PythonExe, "-m", "pip_audit", "--local", "--skip-editable", "--progress-spinner", "off")
} else {
    Write-Step "[6/10] Skipping dependency audit"
}

if (-not $SkipTests) {
    Write-Step "[7/10] Running test suite"
    Invoke-LoggedCommand @($PythonExe, "-m", "pytest")
} else {
    Write-Step "[7/10] Skipping tests"
}

if ($Clean) {
    Write-Step "[8/10] Cleaning build/dist folders"
    if (Test-Path "build") {
        Remove-Item -LiteralPath "build" -Recurse -Force
    }
    if (Test-Path "dist") {
        Remove-Item -LiteralPath "dist" -Recurse -Force
    }
} else {
    Write-Step "[8/10] Keeping existing build/dist folders"
}

Write-Step "[9/10] Building Cellonaut with PyInstaller"
Invoke-LoggedCommand @($PythonExe, "-m", "PyInstaller", "release_tools\pyinstaller\Cellonaut.spec", "--noconfirm", "--clean")

Write-Step "[10/10] Checking packaged output"
$ExePath = Join-Path $ProjectRoot "dist\Cellonaut\Cellonaut.exe"
if (-not (Test-Path $ExePath)) {
    throw "Expected executable was not created: $ExePath"
}

$InternalDir = Join-Path $ProjectRoot "dist\Cellonaut\_internal"
if (Test-Path $InternalDir) {
    Get-ChildItem -Path $InternalDir -Filter "icu*.dll" -ErrorAction SilentlyContinue | Remove-Item -Force
}

$DistDir = Join-Path $ProjectRoot "dist\Cellonaut"
$ProfileMarker = Join-Path $DistDir "BUILD_PROFILE.txt"
Set-Content -Path $ProfileMarker -Value $Profile -Encoding ASCII
$PackagedSmokeResult = Join-Path $env:TEMP ("cellonaut_packaged_smoke_{0}.json" -f [guid]::NewGuid())
try {
    $SmokeProcess = Start-Process -FilePath $ExePath -ArgumentList @("--release-smoke", $PackagedSmokeResult) -WindowStyle Hidden -Wait -PassThru
    if ($SmokeProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $PackagedSmokeResult)) {
        throw "The packaged executable failed its runtime smoke test."
    }
    Invoke-LoggedCommand @(
        $PythonExe,
        "-m", "cellonaut.release_checks.packaged_results",
        "--input", $PackagedSmokeResult,
        "--app-version", $AppVersion
    )
} finally {
    Remove-Item -LiteralPath $PackagedSmokeResult -Force -ErrorAction SilentlyContinue
}

$PackagedRuntimeResult = Join-Path $env:TEMP ("cellonaut_packaged_runtime_{0}.json" -f [guid]::NewGuid())
try {
    $RuntimeProcess = Start-Process -FilePath $ExePath -ArgumentList @("--release-runtime-check", $PackagedRuntimeResult) -WindowStyle Hidden -Wait -PassThru
    if ($RuntimeProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $PackagedRuntimeResult)) {
        throw "The packaged executable failed its Fiji/Cellpose runtime check."
    }
    $RuntimeValidationArgs = @(
        $PythonExe,
        "-m", "cellonaut.release_checks.packaged_results",
        "--input", $PackagedRuntimeResult,
        "--app-version", $AppVersion
    )
    if ($Profile -eq "cuda126") {
        $RuntimeValidationArgs += "--require-cuda-check"
    }
    Invoke-LoggedCommand $RuntimeValidationArgs
} finally {
    Remove-Item -LiteralPath $PackagedRuntimeResult -Force -ErrorAction SilentlyContinue
}

Invoke-LoggedCommand @(
    $PythonExe,
    "-m",
    "cellonaut.release_checks.offline_assets",
    "verify",
    "--output",
    $InternalDir
)

$BuildInfoPath = Join-Path $DistDir "BUILD_INFO.json"
Copy-Item -LiteralPath $BuildInfoStaging -Destination $BuildInfoPath -Force
if ($DependencyLock) {
    $PackagedLockDir = Join-Path $DistDir "DEPENDENCY_LOCKS"
    New-Item -ItemType Directory -Force -Path $PackagedLockDir | Out-Null
    Copy-Item -LiteralPath $DependencyLock -Destination $PackagedLockDir -Force
    Copy-Item -LiteralPath $DependencyLock.Replace(".toml", ".metadata.json") -Destination $PackagedLockDir -Force
}
$LicenseBundle = Join-Path $DistDir "THIRD_PARTY_LICENSES"
Invoke-LoggedCommand @(
    $PythonExe,
    "-m",
    "cellonaut.release_checks.license_inventory",
    "--output",
    $LicenseBundle
)
foreach ($Document in @(
    "LICENSE",
    "COPYRIGHT",
    "THIRD_PARTY_NOTICES.md",
    "BUNDLED_COMPONENTS.md",
    "CITATIONS.md",
    "CITATION.cff",
    "SOURCE_AVAILABILITY.md"
)) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $Document) -Destination (Join-Path $DistDir $Document) -Force
}

$ReadmePath = Join-Path $ProjectRoot "dist\Cellonaut\README_WINDOWS.txt"
$CellposeBackendReadme = if ($Profile -eq "cuda126") {
    @"
This app folder includes the CUDA-capable Windows runtime. During installation,
choose Automatic to use a compatible NVIDIA GPU when available, or CPU only to
disable GPU acceleration. Automatic mode falls back to CPU when CUDA is not
available. AMD graphics on Windows use CPU because Cellpose ROCm acceleration
is not supported on Windows.
"@
} else {
    @"
This development build uses the '$Profile' runtime profile. It is not suitable
for the official Windows installer, which requires the CUDA-capable profile.
"@
}
$ReadmeTemplatePath = Join-Path $ProjectRoot "release_tools\templates\README_WINDOWS.txt"
$ReadmeText = Get-Content -LiteralPath $ReadmeTemplatePath -Raw
$ReadmeText = $ReadmeText.Replace("{{APP_VERSION}}", $AppVersion)
$ReadmeText = $ReadmeText.Replace("{{PROFILE}}", $Profile)
$ReadmeText = $ReadmeText.Replace("{{CELLPOSE_BACKEND}}", $CellposeBackendReadme.Trim())
Set-Content -Path $ReadmePath -Value $ReadmeText -Encoding UTF8

Add-Content -Path $ReportPath -Value ("Executable: {0}" -f $ExePath)
Add-Content -Path $ReportPath -Value ("Build profile marker: {0}" -f $ProfileMarker)
Add-Content -Path $ReportPath -Value ("Build provenance: {0}" -f $BuildInfoPath)
Add-Content -Path $ReportPath -Value ("README: {0}" -f $ReadmePath)
Add-Content -Path $ReportPath -Value ("Finished: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

Write-Output ""
Write-Output "Release build complete."
Write-Output "Executable: $ExePath"
Write-Output "Build profile marker: $ProfileMarker"
Write-Output "README: $ReadmePath"
Write-Output "Build report: $ReportPath"
