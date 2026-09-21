@echo off
setlocal enableextensions

cd /d "%~dp0..\.."

echo.
echo [1/7] Checking built Cellonaut app...
if not exist "dist\Cellonaut\Cellonaut.exe" (
    echo ERROR: dist\Cellonaut\Cellonaut.exe was not found.
    echo Build the app first:
    echo     powershell -ExecutionPolicy Bypass -File release_tools\scripts\build_release.ps1 -Clean
    exit /b 1
)

echo.
echo [2/7] Reading Cellonaut version...
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)
echo Python executable: %PYTHON_EXE%
"%PYTHON_EXE%" -m cellonaut.release_checks.python_runtime --project-root "%CD%"
if errorlevel 1 (
    echo ERROR: The installer must be created with the pinned release Python.
    exit /b 1
)
set "CELLONAUT_VERSION="
set "VERSION_FILE=%TEMP%\cellonaut_version_%RANDOM%_%RANDOM%.txt"
"%PYTHON_EXE%" -c "from cellonaut.version import __version__; print(__version__)" > "%VERSION_FILE%"
if errorlevel 1 (
    if exist "%VERSION_FILE%" del "%VERSION_FILE%" >nul 2>nul
    echo ERROR: Could not read the Cellonaut version.
    exit /b 1
)
set /p CELLONAUT_VERSION=<"%VERSION_FILE%"
del "%VERSION_FILE%" >nul 2>nul
if "%CELLONAUT_VERSION%"=="" (
    echo ERROR: Could not read the Cellonaut version.
    exit /b 1
)
echo Cellonaut version: %CELLONAUT_VERSION%

echo.
echo [3/7] Verifying build provenance...
set "BUILD_INFO=dist\Cellonaut\BUILD_INFO.json"
if not exist "%BUILD_INFO%" (
    echo ERROR: %BUILD_INFO% was not found.
    echo Rebuild Cellonaut from a clean source tree before creating the installer.
    exit /b 1
)
"%PYTHON_EXE%" -m cellonaut.release_checks.build_provenance verify --input "%BUILD_INFO%" --project-root "%CD%"
if errorlevel 1 (
    echo ERROR: The packaged app is stale or was not built from the current clean source tree.
    exit /b 1
)

echo.
echo [4/7] Reading Windows build profile...
set "PROFILE_FILE=dist\Cellonaut\BUILD_PROFILE.txt"
if not exist "%PROFILE_FILE%" (
    echo ERROR: %PROFILE_FILE% was not found.
    echo Rebuild Cellonaut before creating the installer.
    exit /b 1
)
set "CELLONAUT_PROFILE="
set /p CELLONAUT_PROFILE=<"%PROFILE_FILE%"
if /I not "%CELLONAUT_PROFILE%"=="cuda126" (
    echo ERROR: The official Windows installer requires the cuda126 build profile.
    echo Rebuild with:
    echo     powershell -ExecutionPolicy Bypass -File release_tools\scripts\build_release.ps1 -Profile cuda126 -Clean
    exit /b 1
)
echo Windows build profile: %CELLONAUT_PROFILE%

echo.
echo [5/7] Checking Inno Setup compiler...
where ISCC.exe >nul 2>nul
if errorlevel 1 (
    if exist "%LocalAppData%\Programs\Inno Setup 6\ISCC.exe" (
        set "ISCC=%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"
    ) else if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" (
        set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    ) else if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" (
        set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
    ) else (
        echo ERROR: ISCC.exe was not found.
        echo Install Inno Setup 6 or add ISCC.exe to PATH.
        exit /b 1
    )
) else (
    set "ISCC=ISCC.exe"
)

echo.
echo [6/7] Building installer...
set "OUTPUT_STEM=Cellonaut-%CELLONAUT_VERSION%-windows"
set "OUTPUT_NAME=%OUTPUT_STEM%.exe"
set "BUILD_OUTPUT_DIR=%CD%\installer_dist\_building_%RANDOM%_%RANDOM%"
set "BUILD_OUTPUT=%BUILD_OUTPUT_DIR%\%OUTPUT_NAME%"

mkdir "%BUILD_OUTPUT_DIR%"

"%ISCC%" /DMyAppVersion=%CELLONAUT_VERSION% /DMyAppProfile=%CELLONAUT_PROFILE% /O"%BUILD_OUTPUT_DIR%" /F"%OUTPUT_STEM%" release_tools\installer\CellonautInstaller.iss
if errorlevel 1 (
    echo.
    echo Installer build failed.
    exit /b 1
)
if not exist "%BUILD_OUTPUT%" (
    echo.
    echo ERROR: Inno Setup finished but did not create the installer.
    exit /b 1
)

echo.
echo [7/7] Generating multipart installer checksums...
"%PYTHON_EXE%" -m cellonaut.release_checks.checksum --input-pattern "%BUILD_OUTPUT_DIR%\%OUTPUT_STEM%*" --output "%BUILD_OUTPUT_DIR%\%OUTPUT_STEM%.sha256"
if errorlevel 1 (
    echo ERROR: Could not generate the installer checksum manifest.
    exit /b 1
)

del /Q "%CD%\installer_dist\%OUTPUT_NAME%" "%CD%\installer_dist\%OUTPUT_STEM%.sha256" >nul 2>nul
for %%F in ("%CD%\installer_dist\%OUTPUT_STEM%-*.bin") do (
    echo(%%~nF| findstr /R /C:"^%OUTPUT_STEM%-[0-9][0-9]*$" >nul
    if not errorlevel 1 del /Q "%%~fF" >nul 2>nul
)
for %%F in ("%BUILD_OUTPUT_DIR%\%OUTPUT_STEM%*") do (
    move /Y "%%~fF" "%CD%\installer_dist\%%~nxF" >nul
    if errorlevel 1 (
        echo ERROR: Could not move %%~nxF to installer_dist.
        exit /b 1
    )
)
rmdir "%BUILD_OUTPUT_DIR%" >nul 2>nul

echo.
echo Installer complete:
echo   %CD%\installer_dist\%OUTPUT_STEM%*
echo Keep the installer EXE and every BIN slice together in the same folder.
echo Verify every installer part with the accompanying %OUTPUT_STEM%.sha256 manifest.
