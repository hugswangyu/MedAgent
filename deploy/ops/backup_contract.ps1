$script:BackupManifestSchema = 2
$script:BackupIdPattern = '^\d{8}T\d{9}Z-[0-9a-f]{12}$'

function Test-BackupId([string]$BackupId) {
    return $BackupId -cmatch $script:BackupIdPattern
}

function New-BackupReservation(
    [Parameter(Mandatory = $true)][string]$BackupRoot,
    [string]$BackupId
) {
    $root = [IO.Path]::GetFullPath($BackupRoot).TrimEnd("\", "/")
    if ([string]::IsNullOrWhiteSpace($BackupId)) {
        $timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
        $suffix = [Guid]::NewGuid().ToString("N").Substring(0, 12)
        $BackupId = "$timestamp-$suffix"
    }
    if (-not (Test-BackupId $BackupId)) { throw "invalid BackupId: $BackupId" }

    $backupDir = [IO.Path]::GetFullPath((Join-Path $root $BackupId))
    if (-not [string]::Equals(
        (Split-Path -Parent $backupDir),
        $root,
        [StringComparison]::OrdinalIgnoreCase
    )) { throw "BackupId escaped the backup root" }
    $claimPath = [IO.Path]::GetFullPath((Join-Path $root ".$BackupId.reserve"))
    if (-not [string]::Equals(
        (Split-Path -Parent $claimPath),
        $root,
        [StringComparison]::OrdinalIgnoreCase
    )) { throw "backup reservation escaped the backup root" }

    $claim = $null
    $claimOwned = $false
    try {
        try {
            $claim = [IO.File]::Open(
                $claimPath,
                [IO.FileMode]::CreateNew,
                [IO.FileAccess]::Write,
                [IO.FileShare]::None
            )
            $claimOwned = $true
        } catch [IO.IOException] {
            throw "backup reservation already exists: $BackupId"
        }

        if (Test-Path -LiteralPath $backupDir) {
            throw "backup directory already exists: $backupDir"
        }
        [IO.Directory]::CreateDirectory($backupDir) | Out-Null
        $item = Get-Item -LiteralPath $backupDir -Force
        if (-not $item.PSIsContainer) { throw "not a directory: $backupDir" }
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "reparse-point backup directories are not allowed: $backupDir"
        }
        return [PSCustomObject]@{ Id = $BackupId; Directory = $item.FullName.TrimEnd("\", "/") }
    } finally {
        if ($null -ne $claim) { $claim.Dispose() }
        if ($claimOwned -and (Test-Path -LiteralPath $claimPath -PathType Leaf)) {
            Remove-Item -LiteralPath $claimPath -Force
        }
    }
}

function Get-BackupVolumes([string]$Profile) {
    $volumes = @("medagent-data", "medlive-data")
    if ($Profile -eq "full") {
        $volumes += @("milvus-etcd", "milvus-minio", "milvus-data", "elasticsearch-data", "neo4j-data")
    }
    return $volumes
}

function Assert-BackupManifest(
    [Parameter(Mandatory = $true)]$Manifest,
    [Parameter(Mandatory = $true)][string]$BackupDir,
    [Parameter(Mandatory = $true)][string]$BackupId,
    [Parameter(Mandatory = $true)][string]$Profile,
    [Parameter(Mandatory = $true)][string]$ProjectName
) {
    $schemaIsInteger = $Manifest.schema -is [int] -or $Manifest.schema -is [long]
    if (-not $schemaIsInteger -or [int64]$Manifest.schema -ne $script:BackupManifestSchema) {
        throw "unsupported manifest schema: $($Manifest.schema)"
    }
    if ($Manifest.backup_id -cne $BackupId) { throw "manifest BackupId mismatch" }
    if ($Manifest.profile -cne $Profile -or $Manifest.project -cne $ProjectName) {
        throw "manifest target mismatch"
    }

    $volumes = @(Get-BackupVolumes $Profile)
    $expectedPayload = @("postgres.dump") + @($volumes | ForEach-Object { "$_.tgz" })
    $entries = @($Manifest.files)
    if ($entries.Count -ne $expectedPayload.Count) {
        throw "manifest file set is incomplete"
    }

    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($file in $entries) {
        $name = [string]$file.name
        if ($name -notmatch "^[a-z0-9][a-z0-9.-]*$" -or $expectedPayload -cnotcontains $name) {
            throw "unexpected manifest file: $name"
        }
        if (-not $seen.Add($name)) { throw "duplicate manifest file: $name" }
        if ([string]$file.sha256 -cnotmatch "^[0-9a-f]{64}$") {
            throw "invalid checksum metadata: $name"
        }

        $path = [IO.Path]::GetFullPath((Join-Path $BackupDir $name))
        if (-not [string]::Equals(
            (Split-Path -Parent $path),
            $BackupDir,
            [StringComparison]::OrdinalIgnoreCase
        )) { throw "manifest file escaped backup directory" }
        $fileItem = Get-Item -LiteralPath $path -Force
        if (-not $fileItem.PSIsContainer -and
            ($fileItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
            $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        } else {
            throw "backup payload must be a regular file: $name"
        }
        if ($hash -cne $file.sha256) { throw "checksum mismatch: $name" }
        if ($null -eq $file.bytes -or [int64]$file.bytes -ne $fileItem.Length) {
            throw "size mismatch: $name"
        }
    }
    foreach ($name in $expectedPayload) {
        if (-not $seen.Contains($name)) { throw "required manifest file missing: $name" }
    }

    $expectedOnDisk = @($expectedPayload + "manifest.json")
    $actualItems = @(Get-ChildItem -LiteralPath $BackupDir -Force)
    if ($actualItems.Count -ne $expectedOnDisk.Count) {
        throw "backup directory file set does not match manifest"
    }
    foreach ($item in $actualItems) {
        if ($item.PSIsContainer -or
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            $expectedOnDisk -cnotcontains $item.Name) {
            throw "unexpected backup directory entry: $($item.Name)"
        }
    }
    return $volumes
}
