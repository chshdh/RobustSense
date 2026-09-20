param(
    [switch]$RetryFailed,
    [int]$MaxRuns = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-dev\Scripts\python.exe"
$Arguments = @(
    "-m", "robustsense.cli.sweep",
    "--project-root", $ProjectRoot,
    "--profile", "full",
    "--prepare-folds"
)
if ($RetryFailed) {
    $Arguments += "--retry-failed"
}
if ($MaxRuns -gt 0) {
    $Arguments += @("--max-runs", $MaxRuns)
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Full sweep failed; credible must be complete and failed units require -RetryFailed"
}
