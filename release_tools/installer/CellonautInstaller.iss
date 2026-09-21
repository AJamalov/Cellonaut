#define MyAppName "Cellonaut"
#ifndef MyAppVersion
  #error MyAppVersion must be supplied by release_tools\scripts\build_installer.bat
#endif
#ifndef MyAppProfile
  #error MyAppProfile must be supplied by release_tools\scripts\build_installer.bat
#endif
#define MyAppPublisher "Cellonaut"
#define MyAppExeName "Cellonaut.exe"
; Resolve the root now so long bundled asset names do not retain \..\.. in source paths.
#define ProjectRoot ExtractFileDir(ExtractFileDir(ExtractFileDir(SourcePath)))

[Setup]
AppId={{9A7F3B1D-8E94-4E0A-A0F4-96FA6DCA7D10}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=1500000000
SlicesPerDisk=1
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
OutputDir={#ProjectRoot}\installer_dist
OutputBaseFilename=Cellonaut-{#MyAppVersion}-windows
SetupIconFile={#ProjectRoot}\cellonaut\data\assets\icon.ico
LicenseFile={#ProjectRoot}\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked
Name: "cellposeauto"; Description: "Automatic (use NVIDIA CUDA when available)"; GroupDescription: "Cellpose processing:"; Flags: exclusive
Name: "cellposecpu"; Description: "CPU only"; GroupDescription: "Cellpose processing:"; Flags: exclusive unchecked

[Files]
Source: "{#ProjectRoot}\dist\Cellonaut\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; PyInstaller's runtime lives here. Remove it before an upgrade so modules or
; native libraries deleted by a newer release cannot survive alongside it.
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Fiji (optional plugins)"; Filename: "{app}\_internal\offline\Fiji.app\fiji-windows-x64.exe"; WorkingDir: "{app}\_internal\offline\Fiji.app"
Name: "{group}\License"; Filename: "{app}\LICENSE"
Name: "{group}\Third-party notices"; Filename: "{app}\THIRD_PARTY_NOTICES.md"
Name: "{group}\Bundled components"; Filename: "{app}\BUNDLED_COMPONENTS.md"
Name: "{group}\Citations"; Filename: "{app}\CITATIONS.md"
Name: "{group}\Source availability"; Filename: "{app}\SOURCE_AVAILABILITY.md"
Name: "{group}\Windows quick start"; Filename: "{app}\README_WINDOWS.txt"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\CELLPOSE_BACKEND.txt"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  CellposeBackend: String;
begin
  if CurStep = ssPostInstall then
  begin
    CellposeBackend := 'auto';
    if WizardIsTaskSelected('cellposecpu') then
      CellposeBackend := 'cpu';
    if not SaveStringToFile(ExpandConstant('{app}\CELLPOSE_BACKEND.txt'), CellposeBackend + #13#10, False) then
      RaiseException('Could not save the selected Cellpose processing mode.');
  end;
end;
