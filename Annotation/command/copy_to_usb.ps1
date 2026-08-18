[CmdletBinding()]
param([string]$DestinationDrive = "")

$ErrorActionPreference = "Stop"
$packageRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

if ($DestinationDrive) {
    $driveId = $DestinationDrive.Trim().TrimEnd("\").ToUpperInvariant()
    if ($driveId -notmatch '^[A-Z]:$' -or -not (Test-Path -LiteralPath "$driveId\")) {
        throw "目标盘符无效，应类似F:。"
    }
} else {
    $candidates = @(Get-CimInstance Win32_LogicalDisk -Filter "DriveType = 2")
    if ($candidates.Count -eq 0) { throw "未检测到可移动磁盘。" }
    if ($candidates.Count -eq 1) {
        $driveId = $candidates[0].DeviceID
    } else {
        for ($i = 0; $i -lt $candidates.Count; $i++) {
            Write-Host "[$($i + 1)] $($candidates[$i].DeviceID) $($candidates[$i].VolumeName)"
        }
        $choice = [int](Read-Host "请选择U盘序号")
        if ($choice -lt 1 -or $choice -gt $candidates.Count) { throw "选择无效。" }
        $driveId = $candidates[$choice - 1].DeviceID
    }
}

$drive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID = '$driveId'"
$sourceBytes = (Get-ChildItem -LiteralPath $packageRoot -Recurse -File | Measure-Object Length -Sum).Sum
if ($drive.FreeSpace -lt $sourceBytes) { throw "U盘剩余空间不足。" }

$destination = Join-Path "$driveId\" "Annotation_Windows"
& robocopy $packageRoot $destination /E /Z /J /R:2 /W:2 /COPY:DAT /DCOPY:DAT /NP
if ($LASTEXITCODE -ge 8) { throw "Robocopy失败，退出码：$LASTEXITCODE" }

Write-Host "复制完成：$destination"
