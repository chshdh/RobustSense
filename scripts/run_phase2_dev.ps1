param(
    [int]$Fold = 0,
    [int]$Seed = 13
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $ProjectRoot '.venv-dev\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw 'Missing .venv-dev. Run scripts\setup_windows.ps1 first.'
}

$env:MPLCONFIGDIR = Join-Path $ProjectRoot '.cache\matplotlib'
$Models = @(
    'single-phone_acc',
    'single-phone_gyro',
    'single-watch_acc',
    'single-location',
    'single-audio',
    'single-phone_state',
    'linear',
    'mlp',
    'late'
)

Push-Location $ProjectRoot
try {
    & $VenvPython -m robustsense.cli.prepare `
        --config configs/data/extrasensory.yaml `
        --fold $Fold
    if ($LASTEXITCODE -ne 0) { throw 'Phase-2 prepare failed.' }

    foreach ($Model in $Models) {
        & $VenvPython -m robustsense.cli.train `
            --data-config configs/data/extrasensory.yaml `
            --model $Model `
            --fold $Fold `
            --seed $Seed `
            --profile dev `
            --project-root .
        if ($LASTEXITCODE -ne 0) { throw "Phase-2 model failed: $Model" }
    }
}
finally {
    Pop-Location
}
