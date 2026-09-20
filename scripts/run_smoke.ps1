$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $ProjectRoot '.venv-dev\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw 'Missing .venv-dev. Run scripts\setup_windows.ps1 first.'
}

$env:MPLCONFIGDIR = Join-Path $ProjectRoot '.cache\matplotlib'

Push-Location $ProjectRoot
try {
    & $VenvPython -m robustsense.cli.prepare --config configs/data/synthetic.yaml --fold 0
    if ($LASTEXITCODE -ne 0) { throw 'Data preparation smoke test failed.' }

    & $VenvPython -m robustsense.cli.train --model early --fold 0 --seed 13 --profile dev
    if ($LASTEXITCODE -ne 0) { throw 'Training smoke test failed.' }

    & $VenvPython -m robustsense.cli.evaluate `
        --run-dir runs/synthetic-early-fold0-seed13 `
        --suite smoke
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation smoke test failed.' }

    & $VenvPython -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Pytest failed.' }

    & $VenvPython -m ruff check .
    if ($LASTEXITCODE -ne 0) { throw 'Ruff failed.' }
}
finally {
    Pop-Location
}
