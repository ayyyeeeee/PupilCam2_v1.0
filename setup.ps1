param([switch]$SkipDriver, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'This release requires Windows 10/11 x64 (Intel/AMD), not ARM64 or 32-bit Windows.'
    }
    if ([Environment]::OSVersion.Version.Major -lt 10) { throw 'Windows 10 or 11 is required.' }
    $root = $PSScriptRoot
    $python = Join-Path $root 'runtime\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        if ($CheckOnly) { throw 'Runtime missing. Run 01_install.bat first.' }
        $archive = Join-Path $root 'runtime.zip'
        $expected = (Get-Content -LiteralPath (Join-Path $root 'runtime.sha256') -Raw).Trim()
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) {
            throw 'Runtime archive checksum mismatch. Copy the complete package again.'
        }
        Write-Host 'Unpacking the offline runtime. No system Python or PATH changes.'
        Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $root 'runtime') -Force
    }
    & $python -I (Join-Path $root 'app\launch.py') --selfcheck
    if ($LASTEXITCODE -ne 0) { throw 'Offline self-check failed. Do not continue to driver installation.' }
    if (-not $SkipDriver -and -not $CheckOnly) {
        & (Join-Path $root 'driver_setup.ps1')
        if (-not $?) { throw 'Driver setup was not completed.' }
    }
    Write-Host 'Software is ready. Start with 02_start.bat.' -ForegroundColor Green
    exit 0
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host 'Use a local writable folder. Keep the entire package together.'
    exit 1
}
