$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-dev\Scripts\python.exe"
$Models = @(
    "single-phone_acc",
    "single-phone_gyro",
    "single-watch_acc",
    "single-location",
    "single-audio",
    "single-phone_state",
    "linear",
    "mlp",
    "late",
    "gated",
    "robust-gated",
    "quality-aware-cls",
    "quality-aware"
)

foreach ($Model in $Models) {
    $RunDir = Join-Path $ProjectRoot "runs\extrasensory-$Model-fold0-seed13-dev"
    & $Python -m robustsense.cli.evaluate --run-dir $RunDir --suite full
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 4 evaluation failed for $Model"
    }
}

& $Python -m robustsense.cli.report `
    --project-root $ProjectRoot `
    --runs-dir runs `
    --output-dir reports `
    --plan configs/evaluation/dev.yaml
if ($LASTEXITCODE -ne 0) {
    throw "Phase 4 report generation failed"
}
