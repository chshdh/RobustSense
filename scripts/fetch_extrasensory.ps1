$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RawDirectory = Join-Path $ProjectRoot 'data\raw'
$ReceiptPath = Join-Path $ProjectRoot 'configs\data\extrasensory_archives.json'
$Receipt = Get-Content -Raw -LiteralPath $ReceiptPath | ConvertFrom-Json

New-Item -ItemType Directory -Force -Path $RawDirectory | Out-Null

foreach ($Archive in $Receipt.archives) {
    $Target = Join-Path $RawDirectory $Archive.filename
    if (Test-Path -LiteralPath $Target) {
        $Item = Get-Item -LiteralPath $Target
        $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Target).Hash.ToLowerInvariant()
        if ($Item.Length -ne $Archive.bytes -or $Hash -ne $Archive.sha256) {
            throw "Existing archive does not match the local reference receipt: $Target"
        }
        Write-Host "Verified existing archive; skipping download: $($Archive.filename)"
        continue
    }

    $Partial = "$Target.download"
    $CurlArguments = @('-L', '--fail', '--retry', '3')
    if (Test-Path -LiteralPath $Partial) {
        $CurlArguments += @('--continue-at', '-')
    }
    $CurlArguments += @('--output', $Partial, $Archive.source_url)
    & curl.exe @CurlArguments
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $($Archive.filename)" }

    $Item = Get-Item -LiteralPath $Partial
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Partial).Hash.ToLowerInvariant()
    if ($Item.Length -ne $Archive.bytes -or $Hash -ne $Archive.sha256) {
        throw "Downloaded archive does not match the local reference receipt: $Partial"
    }
    Move-Item -LiteralPath $Partial -Destination $Target
    Write-Host "Downloaded and verified: $($Archive.filename)"
}

Write-Host "Reference scope: $($Receipt.checksum_scope)"
