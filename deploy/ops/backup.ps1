param(
    [ValidateSet("voice", "full")]
    [string]$Profile = "voice",
    [ValidatePattern("^[a-z0-9][a-z0-9_-]*$")]
    [string]$ProjectName = "medagent",
    [string]$BackupRoot = (Join-Path $PSScriptRoot "..\..\backups")
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "backup_contract.ps1")

function Get-StrictDirectory([string]$Path, [bool]$Create) {
    $full = [IO.Path]::GetFullPath($Path).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    if ($Create -and -not (Test-Path -LiteralPath $full)) {
        [IO.Directory]::CreateDirectory($full) | Out-Null
    }
    $item = Get-Item -LiteralPath $full -Force
    if (-not $item.PSIsContainer) { throw "not a directory: $full" }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "reparse-point directories are not allowed: $full"
    }
    return $item.FullName.TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
}

$repoRoot = Get-StrictDirectory (Join-Path $PSScriptRoot "..\..") $false
$expectedRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot "backups")).TrimEnd("\", "/")
$candidateRoot = [IO.Path]::GetFullPath($BackupRoot).TrimEnd("\", "/")
if (-not [string]::Equals($candidateRoot, $expectedRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "BackupRoot must be exactly $expectedRoot"
}
$resolvedRoot = Get-StrictDirectory $expectedRoot $true

$composeArgs = @("compose", "--project-name", $ProjectName, "--profile", $Profile)
$originalRunning = @(
    (& docker @composeArgs ps --status running --services) |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)
if ($LASTEXITCODE -ne 0) { throw "failed to inspect running services" }
if ($originalRunning -notcontains "postgres") {
    throw "postgres must already be running before backup"
}

$reservation = New-BackupReservation -BackupRoot $resolvedRoot
$backupId = $reservation.Id
$backupDir = $reservation.Directory
if (-not [string]::Equals(
    (Split-Path -Parent $backupDir),
    $resolvedRoot,
    [StringComparison]::OrdinalIgnoreCase
)) { throw "backup directory escaped BackupRoot" }

$writers = @("frontend", "medlive-worker", "medlive-api", "lightrag", "medagent-api")
if ($Profile -eq "full") {
    $writers += @("milvus", "elasticsearch", "neo4j", "etcd", "minio")
}
$stoppedWriters = @($writers | Where-Object { $originalRunning -contains $_ })

try {
    if ($stoppedWriters.Count -gt 0) {
        & docker @composeArgs stop @stoppedWriters
        if ($LASTEXITCODE -ne 0) { throw "failed to stop writers" }
    }
    & docker @composeArgs exec -T -e "BACKUP_ID=$backupId" postgres sh -ec 'PGPASSWORD="$(cat /run/secrets/postgres_password)" pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -f "/backups/$BACKUP_ID/postgres.dump"'
    if ($LASTEXITCODE -ne 0) { throw "pg_dump failed" }

    $volumes = @(Get-BackupVolumes $Profile)
    foreach ($volume in $volumes) {
        $dockerVolume = "$($ProjectName)_$volume"
        & docker volume inspect $dockerVolume *> $null
        if ($LASTEXITCODE -ne 0) { throw "missing volume $dockerVolume" }
        & docker run --rm --mount "source=$dockerVolume,target=/source,readonly" --mount "type=bind,src=$backupDir,target=/backup" postgres:16.6-alpine tar -C /source -czf "/backup/$volume.tgz" .
        if ($LASTEXITCODE -ne 0) { throw "failed to archive $dockerVolume" }
    }

    $files = Get-ChildItem -LiteralPath $backupDir -File | ForEach-Object {
        [ordered]@{
            name = $_.Name
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            bytes = $_.Length
        }
    }
    $manifest = [ordered]@{
        schema = 2
        backup_id = $backupId
        profile = $Profile
        project = $ProjectName
        created_at = (Get-Date).ToUniversalTime().ToString("o")
        running_services_before_backup = @($originalRunning)
        files = @($files)
    }
    $manifest | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $backupDir "manifest.json") -Encoding utf8
    Write-Host "Backup completed: $backupDir"
}
finally {
    if ($stoppedWriters.Count -gt 0) {
        & docker @composeArgs up -d --no-deps @stoppedWriters
    }
}
