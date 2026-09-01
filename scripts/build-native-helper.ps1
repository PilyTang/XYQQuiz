[CmdletBinding()]
param(
    [string]$Configuration = "Release",
    [string]$Platform = "x64"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectPath = Join-Path $ProjectRoot "native\preview-helper\XYQPreviewHelper.vcxproj"
$ShaderPath = Join-Path $ProjectRoot "native\preview-helper\preview.hlsl"
$ShaderDirectory = Split-Path -Parent $ShaderPath
$ProgramFilesX86 = [Environment]::GetFolderPath('ProgramFilesX86')
$SdkBin = Join-Path $ProgramFilesX86 'Windows Kits\10\bin'
$Sdk = Get-ChildItem -LiteralPath $SdkBin -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^10\.0\.\d+\.\d+$' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'x64\fxc.exe')) } |
    Sort-Object { [version]$_.Name } -Descending |
    Select-Object -First 1
if (-not $Sdk) {
    throw "Windows SDK x64 fxc.exe was not found."
}
$Fxc = Join-Path $Sdk.FullName 'x64\fxc.exe'
$VsWhere = Join-Path $ProgramFilesX86 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $VsWhere -PathType Leaf)) {
    throw "Visual Studio Installer vswhere.exe was not found; install Desktop development with C++."
}
$VisualStudioPaths = @(& $VsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
if ($LASTEXITCODE -ne 0 -or $VisualStudioPaths.Count -eq 0) {
    throw "Visual Studio with x64 C++ build tools was not found."
}
$VisualStudioRoot = $VisualStudioPaths[0].Trim()
$MSBuild = Join-Path $VisualStudioRoot 'MSBuild\Current\Bin\MSBuild.exe'
$Toolset = Get-ChildItem -Path (Join-Path $VisualStudioRoot 'MSBuild\Microsoft\VC\*\Platforms\x64\PlatformToolsets\v*') -Directory |
    Where-Object { $_.Name -match '^v\d+$' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'Toolset.props')) } |
    Sort-Object { [int]$_.Name.Substring(1) } -Descending |
    Select-Object -First 1
if (-not (Test-Path -LiteralPath $MSBuild -PathType Leaf) -or -not $Toolset) {
    throw "MSBuild or the x64 C++ platform toolset was not found in $VisualStudioRoot."
}
Write-Output "Native build: toolset $($Toolset.Name), Windows SDK $($Sdk.Name), MSBuild $MSBuild"
$Shaders = @(
    @{ Entry = "VSMain"; Target = "vs_5_0"; Header = "preview_vs.h"; Variable = "g_preview_vs" },
    @{ Entry = "PSMain"; Target = "ps_5_0"; Header = "preview_ps.h"; Variable = "g_preview_ps" }
)
foreach ($Shader in $Shaders) {
    $ShaderArguments = @(
        "/nologo"
        "/O3"
        "/T"
        $Shader.Target
        "/E"
        $Shader.Entry
        "/Fh"
        (Join-Path $ShaderDirectory $Shader.Header)
        "/Vn"
        $Shader.Variable
        $ShaderPath
    )
    & $Fxc @ShaderArguments
    if ($LASTEXITCODE -ne 0) {
        throw "FXC failed for $($Shader.Entry) with exit code $LASTEXITCODE"
    }
    $HeaderPath = Join-Path $ShaderDirectory $Shader.Header
    $HeaderLines = Get-Content -LiteralPath $HeaderPath |
        ForEach-Object { $_.TrimEnd() }
    Set-Content -LiteralPath $HeaderPath -Value $HeaderLines -Encoding ascii
}
$Arguments = @(
    $ProjectPath
    "/nologo"
    "/m"
    "/t:Build"
    "/p:Configuration=$Configuration"
    "/p:Platform=$Platform"
    "/p:PlatformToolset=$($Toolset.Name)"
    "/p:WindowsTargetPlatformVersion=$($Sdk.Name)"
    "/verbosity:minimal"
)
& $MSBuild @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "MSBuild failed with exit code $LASTEXITCODE"
}

$Output = Join-Path $ProjectRoot "build\native\XYQPreviewHelper.exe"
if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) {
    throw "Native helper output is missing: $Output"
}
Write-Output $Output
