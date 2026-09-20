$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-dev\Scripts\python.exe"
$DataConfig = "configs/data/extrasensory.yaml"
$Models = @("gated", "robust-gated", "quality-aware-cls", "quality-aware")

foreach ($Model in $Models) {
    & $Python -m robustsense.cli.train `
        --project-root $ProjectRoot `
        --data-config $DataConfig `
        --model $Model `
        --fold 0 `
        --seed 13 `
        --profile dev
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 3 dev training failed for $Model"
    }
}
