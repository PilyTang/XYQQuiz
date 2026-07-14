param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Version = "0.1.0",
    [string]$Commit = "",
    [string]$OutputDirectory = ".\release",
    [switch]$AllowDevelopmentCommit
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$ProjectPython = Join-Path $ProjectRoot $Python
if (Test-Path -LiteralPath $ProjectPython) {
    $PythonPath = (Resolve-Path -LiteralPath $ProjectPython).Path
}
else {
    $PythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop
    $PythonPath = $PythonCommand.Source
}
$BuildRoot = Join-Path $ProjectRoot "build\portable-release"
$DistRoot = Join-Path $BuildRoot "dist"
$WorkRoot = Join-Path $BuildRoot "work"
$StageRoot = Join-Path $BuildRoot "stage"
$ReleaseRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputDirectory))
$LicensePath = Join-Path $ProjectRoot "LICENSE"
if (-not (Test-Path -LiteralPath $LicensePath)) {
    throw "Build requires the project LICENSE"
}
$LicenseText = Get-Content -Raw -LiteralPath $LicensePath
if (-not $LicenseText.Contains("# PolyForm Noncommercial License 1.0.0") -or
    -not $LicenseText.Contains("https://polyformproject.org/licenses/noncommercial/1.0.0")) {
    throw "LICENSE is not the required PolyForm Noncommercial License 1.0.0 text"
}

if ($AllowDevelopmentCommit) {
    if ([string]::IsNullOrWhiteSpace($Commit)) {
        throw "-Commit is required for development builds"
    }
}
else {
    if ($Commit -notmatch '^[0-9a-fA-F]{40}$') {
        throw "Release builds require a full 40-character Git commit SHA"
    }
    $Git = Get-Command git -ErrorAction Stop
    $Head = (& $Git.Source -C $ProjectRoot rev-parse HEAD | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0 -or $Head -ne $Commit) {
        throw "Requested commit $Commit does not match repository HEAD $Head"
    }
    $Dirty = @(& $Git.Source -C $ProjectRoot status --porcelain --untracked-files=normal)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect Git worktree state"
    }
    if ($Dirty.Count -ne 0) {
        throw "Release builds require a clean Git worktree"
    }
}

function Reset-ProjectDirectory([string]$Path) {
    $Resolved = [System.IO.Path]::GetFullPath($Path)
    if (-not $Resolved.StartsWith($ProjectRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to reset path outside project root: $Resolved"
    }
    if (Test-Path -LiteralPath $Resolved) {
        Remove-Item -LiteralPath $Resolved -Recurse -Force
    }
    New-Item -ItemType Directory -Path $Resolved | Out-Null
}

Reset-ProjectDirectory $DistRoot
Reset-ProjectDirectory $WorkRoot
Reset-ProjectDirectory $StageRoot
if (-not (Test-Path -LiteralPath $ReleaseRoot)) {
    New-Item -ItemType Directory -Path $ReleaseRoot | Out-Null
}

Push-Location $ProjectRoot
try {
    $DetectedVersion = (& $PythonPath -c "import xyq_quiz; print(xyq_quiz.__version__)" | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0) { throw "could not read application version" }
    if ($DetectedVersion -ne $Version) {
        throw "Requested version $Version does not match xyq_quiz.__version__ $DetectedVersion"
    }
    & $PythonPath -m PyInstaller ".\packaging\XYQQuiz.spec" --noconfirm --clean --distpath $DistRoot --workpath $WorkRoot
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

    $PackageName = "XYQQuiz-v$Version-win10-win11-x64"
    $PackageRoot = Join-Path $StageRoot $PackageName
    Move-Item -LiteralPath (Join-Path $DistRoot "XYQQuiz") -Destination $PackageRoot
    $PackageRootFull = [System.IO.Path]::GetFullPath($PackageRoot)
    $WindowsCaptureMetadata = Get-ChildItem -LiteralPath (Join-Path $PackageRoot "_internal") -Directory -Filter "windows_capture-*.dist-info"
    foreach ($MetadataDirectory in $WindowsCaptureMetadata) {
        $SbomDirectory = Join-Path $MetadataDirectory.FullName "sboms"
        if (Test-Path -LiteralPath $SbomDirectory) {
            $ResolvedSbom = (Resolve-Path -LiteralPath $SbomDirectory).Path
            if (-not $ResolvedSbom.StartsWith($PackageRootFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to remove upstream SBOM outside package root: $ResolvedSbom"
            }
            Remove-Item -LiteralPath $ResolvedSbom -Recurse -Force
        }
    }
    Copy-Item -LiteralPath ".\packaging\README.txt" -Destination (Join-Path $PackageRoot "README.txt")
    Copy-Item -LiteralPath $LicensePath -Destination (Join-Path $PackageRoot "LICENSE.txt")
    Copy-Item -LiteralPath ".\THIRD_PARTY_NOTICES.txt" -Destination (Join-Path $PackageRoot "THIRD_PARTY_NOTICES.txt")
    Copy-Item -LiteralPath ".\packaging\一键自检.cmd" -Destination (Join-Path $PackageRoot "一键自检.cmd")
    Copy-Item -LiteralPath ".\config.example.json" -Destination (Join-Path $PackageRoot "_internal\defaults\config.json")

    $ManifestPath = Join-Path $PackageRoot "_internal\build-manifest.json"
    & $PythonPath ".\scripts\generate_build_manifest.py" --package-root $PackageRoot --output $ManifestPath --version $Version --commit $Commit
    if ($LASTEXITCODE -ne 0) { throw "manifest generation failed with exit code $LASTEXITCODE" }

    $ZipPath = Join-Path $ReleaseRoot "$PackageName.zip"
    if (Test-Path -LiteralPath $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
    Compress-Archive -LiteralPath $PackageRoot -DestinationPath $ZipPath -CompressionLevel Optimal
    $Hash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Set-Content -LiteralPath "$ZipPath.sha256" -Value "$Hash  $([System.IO.Path]::GetFileName($ZipPath))" -Encoding ascii

    $ExpandedBytes = (Get-ChildItem -LiteralPath $PackageRoot -File -Recurse | Measure-Object -Property Length -Sum).Sum
    $ArchiveBytes = (Get-Item -LiteralPath $ZipPath).Length
    Write-Host "Package: $ZipPath"
    Write-Host "SHA-256: $Hash"
    Write-Host "Expanded bytes: $ExpandedBytes"
    Write-Host "Archive bytes: $ArchiveBytes"
}
finally {
    Pop-Location
}
