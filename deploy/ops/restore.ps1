param(
    [Parameter(Mandatory = $true)]
    [string]$BackupId,
    [ValidateSet("voice", "full")]
    [string]$Profile = "voice",
    [ValidatePattern("^[a-z0-9][a-z0-9_-]*$")]
    [string]$ProjectName = "medagent",
    [switch]$ConfirmRestore,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "backup_contract.ps1")
if (-not (Test-BackupId $BackupId)) { throw "invalid BackupId: $BackupId" }
if (-not $ConfirmRestore -and -not $ValidateOnly) {
    throw "Restore overwrites current data; rerun with -ConfirmRestore"
}

function Get-StrictDirectory([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path).TrimEnd("\", "/")
    $item = Get-Item -LiteralPath $full -Force
    if (-not $item.PSIsContainer) { throw "not a directory: $full" }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "reparse-point directories are not allowed: $full"
    }
    return $item.FullName.TrimEnd("\", "/")
}

$repoRoot = Get-StrictDirectory (Join-Path $PSScriptRoot "..\..")
$backupRoot = Get-StrictDirectory (Join-Path $repoRoot "backups")
$backupDir = Get-StrictDirectory (Join-Path $backupRoot $BackupId)
if (-not [string]::Equals(
    (Split-Path -Parent $backupDir),
    $backupRoot,
    [StringComparison]::OrdinalIgnoreCase
)) { throw "BackupId escaped the backup root" }

$manifestPath = Join-Path $backupDir "manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "manifest not found: $manifestPath"
}
$manifestItem = Get-Item -LiteralPath $manifestPath -Force
if (($manifestItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "reparse-point manifests are not allowed"
}
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$allowedVolumes = @(Assert-BackupManifest -Manifest $manifest -BackupDir $backupDir `
    -BackupId $BackupId -Profile $Profile -ProjectName $ProjectName)
if ($ValidateOnly) {
    Write-Host "Backup validation completed: $backupDir"
    return
}

$composeArgs = @("compose", "--project-name", $ProjectName, "--profile", $Profile)
$originalRunning = @(
    (& docker @composeArgs ps --status running --services) |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)
if ($LASTEXITCODE -ne 0) { throw "failed to inspect running services" }

try {
    if ($originalRunning.Count -gt 0) {
        & docker @composeArgs stop @originalRunning
        if ($LASTEXITCODE -ne 0) { throw "failed to stop stack" }
    }

    foreach ($volume in $allowedVolumes) {
        $archive = Join-Path $backupDir "$volume.tgz"
        if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
            throw "required archive missing: $volume.tgz"
        }
        $dockerVolume = "$($ProjectName)_$volume"
        & docker volume create $dockerVolume *> $null
        & docker run --rm --mount "source=$dockerVolume,target=/target" --mount "type=bind,src=$backupDir,target=/backup,readonly" postgres:16.6-alpine sh -ec "find /target -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; tar -C /target -xzf '/backup/$volume.tgz'"
        if ($LASTEXITCODE -ne 0) { throw "failed to restore $dockerVolume" }
    }

    & docker @composeArgs up -d --no-deps postgres
    if ($LASTEXITCODE -ne 0) { throw "failed to start postgres" }
    & docker @composeArgs exec -T -e "RESTORE_ID=$BackupId" postgres sh -ec 'until pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do sleep 1; done; PGPASSWORD="$(cat /run/secrets/postgres_password)" pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists "/backups/$RESTORE_ID/postgres.dump"'
    if ($LASTEXITCODE -ne 0) { throw "pg_restore failed" }
    Write-Host "Restore completed from $backupDir"
}
finally {
    $currentRunning = @(
        (& docker @composeArgs ps --status running --services) |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $extraServices = @($currentRunning | Where-Object { $originalRunning -notcontains $_ })
    if ($extraServices.Count -gt 0) {
        & docker @composeArgs stop @extraServices
    }
    if ($originalRunning.Count -gt 0) {
        & docker @composeArgs up -d --no-deps @originalRunning
    }
}
