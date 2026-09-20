$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ActivateScript = Join-Path $ProjectRoot '.venv-dev\Scripts\Activate.ps1'

if (-not (Test-Path -LiteralPath $ActivateScript)) {
    throw 'Missing .venv-dev. Run scripts\setup_windows.ps1 first.'
}

. $ActivateScript
$env:MPLCONFIGDIR = Join-Path $ProjectRoot '.cache\matplotlib'
$env:TORCH_HOME = Join-Path $ProjectRoot '.cache\torch'
$env:PYTHONUTF8 = '1'

New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:TORCH_HOME | Out-Null

Write-Host "RobustSense environment active: $ProjectRoot"
python --version

