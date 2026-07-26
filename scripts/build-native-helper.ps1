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
$FxcCandidates = @(
    "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe"
)
$Fxc = $FxcCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $Fxc) {
    throw "Windows SDK x64 fxc.exe was not found."
}
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
$MSBuildCandidates = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
    "C:\Program Files\Microsoft Visual Studio\18\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
)
$MSBuild = $MSBuildCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $MSBuild) {
    throw "Visual Studio Build Tools 18 with MSBuild was not found."
}

$Arguments = @(
    $ProjectPath
    "/nologo"
    "/m"
    "/t:Build"
    "/p:Configuration=$Configuration"
    "/p:Platform=$Platform"
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
