param(
    [ValidateSet('cu128', 'cpu')]
    [string]$TorchVariant = 'cu128',
    [string]$PythonExecutable = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $ProjectRoot '.venv-dev\Scripts\python.exe'

Push-Location $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        if ($PythonExecutable) {
            & $PythonExecutable -m venv .venv-dev
        }
        elseif (Get-Command py.exe -ErrorAction SilentlyContinue) {
            & py.exe -3.12 -m venv .venv-dev
        }
        elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
            & python.exe -m venv .venv-dev
        }
        else {
            throw 'Python 3.12 was not found. Install it or pass -PythonExecutable.'
        }
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv-dev.' }
    }

    & $VenvPython -m pip install --upgrade pip wheel
    if ($LASTEXITCODE -ne 0) { throw 'Failed to update pip and wheel.' }

    if ($TorchVariant -eq 'cu128') {
        & $VenvPython -m pip install 'torch==2.11.0+cu128' `
            --index-url https://download.pytorch.org/whl/cu128
    }
    else {
        & $VenvPython -m pip install 'torch==2.11.0' `
            --index-url https://download.pytorch.org/whl/cpu
    }
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install PyTorch.' }

    & $VenvPython -m pip install -e '.[ml,dev,demo,report]' --no-build-isolation
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install RobustSense dependencies.' }

    $env:MPLCONFIGDIR = Join-Path $ProjectRoot '.cache\matplotlib'
    & $VenvPython scripts\verify_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed.' }
}
finally {
    Pop-Location
}

