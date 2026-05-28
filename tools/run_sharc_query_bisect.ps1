param(
    [string[]] $Modes = @('stub', 'math', 'hash', 'radiance'),
    [switch] $Launch,
    [switch] $StopExistingGame,
    [switch] $StopAfterProbe,
    [int] $BootTimeoutSec = 240,
    [int] $ProbeFrames = 45,
    [string] $ResultsRoot = 'C:\RadSER\results\sharc_query_bisect'
)

$ErrorActionPreference = 'Stop'

$workspace = 'C:\RadSER'
$mcvr = Join-Path $workspace 'MCVR'
$radiance = Join-Path $workspace 'Radiance'
$minecraft = 'C:\Users\Administrator\AppData\Roaming\PrismLauncher\instances\1.21.4\minecraft'
$mods = Join-Path $minecraft 'mods'
$cmake = 'C:\Program Files\CMake\bin\cmake.exe'
$fxc = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe'
$runId = Get-Date -Format 'yyyyMMdd_HHmmss'
$resultsDir = Join-Path $ResultsRoot $runId
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

$validModes = @('off', 'stub', 'math', 'hash', 'radiance')
foreach ($mode in $Modes) {
    if ($validModes -notcontains $mode) {
        throw "Invalid mode '$mode'. Expected one of: $($validModes -join ', ')"
    }
}

function Invoke-Checked {
    param(
        [string] $Label,
        [scriptblock] $Command
    )
    Write-Host "== $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Get-LatestHsErr {
    Get-ChildItem -LiteralPath $minecraft -Filter 'hs_err_pid*.log' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Invoke-BridgeJson {
    param([string] $Json)
    $raw = powershell.exe -ExecutionPolicy Bypass -File (Join-Path $workspace 'bridge.ps1') -json $Json 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw $raw
    }
    return ($raw | ConvertFrom-Json)
}

function Configure-Build-Deploy {
    param([string] $Mode)

    Invoke-Checked "Configure SHARC query mode '$Mode'" {
        & $cmake -S $mcvr -B (Join-Path $mcvr 'build') -G 'Visual Studio 17 2022' -A x64 `
            "-DJAVA_PROJECT_ROOT_DIR=$radiance" `
            -DUSE_AMD=ON `
            -DMCVR_ENABLE_FFX_UPSCALER=ON `
            -DMCVR_ENABLE_NRD=ON `
            -DMCVR_ENABLE_OMM=ON `
            -DMCVR_ENABLE_SHARC=ON `
            -DMCVR_ENABLE_SHARC_BUFFERS=ON `
            -DMCVR_ENABLE_SHARC_RESOLVE_PASS=ON `
            -DMCVR_ENABLE_SHARC_UPDATE_PASS=ON `
            -DMCVR_ENABLE_SHARC_MAIN_TRACE_QUERY=OFF `
            "-DMCVR_SHARC_MAIN_TRACE_QUERY_MODE=$Mode" `
            "-DSHADERMAKE_FXC_PATH=$fxc"
    }

    Invoke-Checked "Build shaders ($Mode)" {
        & $cmake --build (Join-Path $mcvr 'build') --config Release --target shaders
    }
    Invoke-Checked "Build core ($Mode)" {
        & $cmake --build (Join-Path $mcvr 'build') --config Release --target core
    }
    Invoke-Checked "Install native artifacts ($Mode)" {
        & $cmake --install (Join-Path $mcvr 'build') --config Release --prefix $radiance
    }
    Invoke-Checked "Build Radiance jar ($Mode)" {
        cmd.exe /c "set TEMP=C:\Users\Administrator\AppData\Local\Temp && set TMP=C:\Users\Administrator\AppData\Local\Temp && cd /d C:\RadSER\Radiance && C:\RadSER\Radiance\gradlew.bat clean build 2>&1"
    }

    $jar = Get-ChildItem -LiteralPath (Join-Path $radiance 'build\libs') -Filter 'Radiance-*-windows.jar' |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $jar) {
        throw "No Radiance Windows jar found after build."
    }
    Copy-Item -LiteralPath $jar.FullName -Destination $mods -Force
    return $jar
}

function Invoke-LaunchProbe {
    param([string] $Mode)

    $start = Get-Date
    $hsBefore = Get-LatestHsErr
    $hsBeforeTime = if ($hsBefore) { $hsBefore.LastWriteTime } else { Get-Date '2000-01-01' }
    $launchProc = Start-Process powershell.exe `
        -ArgumentList @('-ExecutionPolicy', 'Bypass', '-File', (Join-Path $workspace 'launch.ps1'), '-wait') `
        -PassThru -WindowStyle Hidden

    $deadline = (Get-Date).AddSeconds($BootTimeoutSec)
    $ping = $null
    $crash = $null
    while ((Get-Date) -lt $deadline) {
        $newHs = Get-LatestHsErr
        if ($newHs -and $newHs.LastWriteTime -gt $hsBeforeTime -and $newHs.LastWriteTime -gt $start.AddSeconds(-5)) {
            $crash = $newHs
            break
        }

        try {
            $ping = Invoke-BridgeJson '{"cmd":"ping"}'
            if ($ping.ok -and $ping.inWorld) {
                break
            }
        } catch {
            $ping = $null
        }
        Start-Sleep -Seconds 2
    }

    if ($crash) {
        $crashText = Get-Content -LiteralPath $crash.FullName -Raw
        return [pscustomobject]@{
            mode = $Mode
            boot = 'crashed'
            probeOk = $false
            crashLog = $crash.FullName
            crashModule = if ($crashText -match 'Problematic frame:\s*#\s*C\s+\[([^\]+]+)') { $Matches[1] } else { $null }
            crashSummary = if ($crashText -match 'EXCEPTION_[^\r\n]+') { $Matches[0] } else { $null }
            probePath = $null
        }
    }

    if (-not ($ping -and $ping.ok -and $ping.inWorld)) {
        return [pscustomobject]@{
            mode = $Mode
            boot = 'timeout'
            probeOk = $false
            crashLog = $null
            crashModule = $null
            crashSummary = $null
            probePath = $null
        }
    }

    $probe = Invoke-BridgeJson ("{`"cmd`":`"sharcProbe`",`"frames`":$ProbeFrames}")
    $probePath = if ($probe.path) { $probe.path } else { $probe.savedPath }

    if ($StopAfterProbe) {
        Get-Process javaw, java -ErrorAction SilentlyContinue | Stop-Process -Force
    }

    return [pscustomobject]@{
        mode = $Mode
        boot = 'ok'
        probeOk = [bool]$probe.ok
        crashLog = $null
        crashModule = $null
        crashSummary = $null
        probePath = $probePath
    }
}

$summaries = @()
if ($Launch) {
    $existing = Get-Process javaw, java -ErrorAction SilentlyContinue
    if ($existing -and -not $StopExistingGame) {
        throw "A Java game process is already running. Re-run with -StopExistingGame for a fully automated launch/probe cycle."
    }
    if ($existing -and $StopExistingGame) {
        $existing | Stop-Process -Force
        Start-Sleep -Seconds 3
    }
}

foreach ($mode in $Modes) {
    $row = [ordered]@{
        mode = $mode
        configured = $false
        built = $false
        deployedJar = $null
        launched = $false
        boot = if ($Launch) { 'not-run' } else { 'build-only' }
        probeOk = $false
        probePath = $null
        crashLog = $null
        crashModule = $null
        crashSummary = $null
        timestamp = (Get-Date).ToString('o')
    }

    try {
        $jar = Configure-Build-Deploy -Mode $mode
        $row.configured = $true
        $row.built = $true
        $row.deployedJar = $jar.FullName

        if ($Launch) {
            $row.launched = $true
            $probeResult = Invoke-LaunchProbe -Mode $mode
            $row.boot = $probeResult.boot
            $row.probeOk = $probeResult.probeOk
            $row.probePath = $probeResult.probePath
            $row.crashLog = $probeResult.crashLog
            $row.crashModule = $probeResult.crashModule
            $row.crashSummary = $probeResult.crashSummary
        }
    } catch {
        $row.boot = 'script-error'
        $row.crashSummary = $_.Exception.Message
        Write-Warning "Mode '$mode' failed: $($_.Exception.Message)"
    }

    $summaries += [pscustomobject]$row
    $summaries | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultsDir 'summary.json') -Encoding UTF8
    $summaries | Export-Csv -LiteralPath (Join-Path $resultsDir 'summary.csv') -NoTypeInformation

    if ($Launch -and -not $StopAfterProbe -and $row.boot -eq 'ok') {
        Write-Host "Mode '$mode' booted and game was left running. Stop it before continuing or use -StopAfterProbe."
        break
    }
}

Write-Host "SHARC query bisect summary: $resultsDir"
$summaries | Format-Table -AutoSize
