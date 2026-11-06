param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
# Exact parent hardware ID only. Never enumerate-and-replace unrelated devices.
$devices = @(Get-PnpDevice -PresentOnly | Where-Object {
    $_.InstanceId -match '^USB\\VID_0C45&PID_64AB\\'
})
$report = @($devices | ForEach-Object {
    $service = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName 'DEVPKEY_Device_Service' -ErrorAction SilentlyContinue).Data
    [PSCustomObject]@{Name=$_.FriendlyName; InstanceId=$_.InstanceId; Status=$_.Status; Service=$service}
})
$report | Format-Table -AutoSize
$ready = $report.Count -eq 2 -and @($report | Where-Object { $_.Service -ne 'libusbK' -or $_.Status -ne 'OK' }).Count -eq 0
if ($ready) {
    Write-Host 'Both parent devices already use libusbK. No driver changes needed.' -ForegroundColor Green
    return
}
if ($CheckOnly) { Write-Host 'Both healthy libusbK parents were not found.'; return }
Write-Host 'Close the tracker and plug in the headset. This step requires administrator approval.'
Write-Host 'The current PC local-generated driver certificate is NOT copied to the new PC.'
Write-Host 'Zadig will create a device-specific driver package/certificate after you approve installation.'
Write-Host 'Select ONLY Pupil Cam2 ID0, USB ID 0C45 64AB, with NO MI/interface suffix.' -ForegroundColor Yellow
Write-Host 'Target driver: libusbK. Do NOT select WinUSB, USB hubs, keyboard, mouse, or Interface 0/1.'
Write-Host 'Install/Replace Driver for each of the two matching parent cameras, then close Zadig.'
Write-Host 'Never disable Windows signature enforcement or Secure Boot for this package.'
$answer = Read-Host 'Type YES to open the driver installer, or Enter to skip'
if ($answer -ne 'YES') { Write-Host 'Driver step skipped. Software installed; camera readiness unverified.'; return }
$exe = Join-Path $PSScriptRoot 'drivers\zadig-2.9.exe'
$signature = Get-AuthenticodeSignature -LiteralPath $exe
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Akeo Consulting') {
    throw 'Zadig signature could not be verified. Do not bypass this check; obtain the official signed file.'
}
# Visible only because this is an interactive driver tool the user must control.
Start-Process -FilePath $exe -WorkingDirectory (Join-Path $PSScriptRoot 'drivers') -Verb RunAs -Wait
Write-Host 'Now run 04_camera_check.bat. Driver installation is not proven until both streams pass.'
