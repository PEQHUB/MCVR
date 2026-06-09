param(
    [string]$ShaderDir = 'C:\RadSER\Radiance\src\main\resources\shaders',
    [string]$SpirvDis = '',
    [switch]$RequireSource
)

$ErrorActionPreference = 'Stop'

function Resolve-SpirvDis {
    param([string]$Candidate)

    if ($Candidate -and (Test-Path -LiteralPath $Candidate)) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }

    if ($env:VULKAN_SDK) {
        $sdkTool = Join-Path $env:VULKAN_SDK 'Bin\spirv-dis.exe'
        if (Test-Path -LiteralPath $sdkTool) {
            return (Resolve-Path -LiteralPath $sdkTool).Path
        }
    }

    $cmd = Get-Command spirv-dis.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    throw 'spirv-dis.exe was not found. Install the Vulkan SDK or pass -SpirvDis.'
}

$spirvDisPath = Resolve-SpirvDis -Candidate $SpirvDis
$shaderRoot = Resolve-Path -LiteralPath $ShaderDir
$spvFiles = Get-ChildItem -LiteralPath $shaderRoot.Path -Recurse -Filter '*.spv' | Sort-Object FullName

if (-not $spvFiles) {
    throw "No .spv files were found under $($shaderRoot.Path)"
}

$missing = New-Object System.Collections.Generic.List[string]
$checked = 0

foreach ($spv in $spvFiles) {
    $checked++
    $disassembly = & $spirvDisPath $spv.FullName 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "spirv-dis failed for $($spv.FullName): $disassembly"
    }

    $hasNonSemantic = $disassembly -match 'NonSemantic\.Shader\.DebugInfo'
    $hasFunctionInfo = $disassembly -match '\bDebugFunction\b'
    $hasSource = $disassembly -match '\bDebugSource\b'

    if (-not $hasNonSemantic -or -not $hasFunctionInfo -or ($RequireSource -and -not $hasSource)) {
        $relative = Resolve-Path -LiteralPath $spv.FullName -Relative
        $missing.Add("$relative (NonSemantic=$hasNonSemantic DebugFunction=$hasFunctionInfo DebugSource=$hasSource)")
    }
}

if ($missing.Count -gt 0) {
    Write-Error "Nsight shader debug info check failed. Missing required debug records in $($missing.Count) of $checked shaders:`n$($missing -join "`n")"
    exit 1
}

Write-Host "Nsight shader debug info verified for $checked SPIR-V files under $($shaderRoot.Path)."
