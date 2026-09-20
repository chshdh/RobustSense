param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-dev\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv-dev. Run scripts\setup_windows.ps1 first."
}

Set-Location -LiteralPath $projectRoot
& $python -m streamlit run app\streamlit_app.py --server.port=$Port --browser.gatherUsageStats=false
exit $LASTEXITCODE
