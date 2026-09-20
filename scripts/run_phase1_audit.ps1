param(
    [int]$Fold = 0,
    [switch]$ProbeOnly
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $ProjectRoot '.venv-dev\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw 'Missing .venv-dev. Run scripts\setup_windows.ps1 first.'
}

$env:MPLCONFIGDIR = Join-Path $ProjectRoot '.cache\matplotlib'
$arguments = @(
    '-m', 'robustsense.cli.audit',
    '--config', 'configs/data/extrasensory.yaml',
    '--fold', $Fold
)
if ($ProbeOnly) {
    $arguments += '--probe-only'
}

Push-Location $ProjectRoot
try {
    & $VenvPython @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Phase-1 audit failed.' }
}
finally {
    Pop-Location
}
