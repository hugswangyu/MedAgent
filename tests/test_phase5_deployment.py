from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from deploy.ops.validate_config import EXAMPLE_SECRETS, SECRET_FILES, production_errors


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "compose.yaml"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def load_compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def run_powershell(arguments: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("PowerShell is required for deployment script behavior tests")
    return subprocess.run(
        [POWERSHELL, "-NoProfile", *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def make_restore_fixture(tmp_path: Path, profile: str) -> tuple[Path, Path, str]:
    ops_dir = tmp_path / "deploy" / "ops"
    ops_dir.mkdir(parents=True)
    for name in ("restore.ps1", "backup_contract.ps1"):
        shutil.copy2(ROOT / "deploy" / "ops" / name, ops_dir / name)

    backup_id = "20260903T010203004Z-abcdef123456"
    backup_dir = tmp_path / "backups" / backup_id
    backup_dir.mkdir(parents=True)
    payloads = {
        "postgres.dump": b"postgres fixture",
        "medagent-data.tgz": b"medagent fixture",
        "medlive-data.tgz": b"medlive fixture",
    }
    if profile == "full":
        payloads.update({
            "milvus-etcd.tgz": b"milvus etcd fixture",
            "milvus-minio.tgz": b"milvus minio fixture",
            "milvus-data.tgz": b"milvus data fixture",
            "elasticsearch-data.tgz": b"elasticsearch fixture",
            "neo4j-data.tgz": b"neo4j fixture",
        })
    entries = []
    for name, content in payloads.items():
        (backup_dir / name).write_bytes(content)
        entries.append({
            "name": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        })
    manifest = {
        "schema": 2,
        "backup_id": backup_id,
        "profile": profile,
        "project": "phase5test",
        "files": entries,
    }
    (backup_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ops_dir / "restore.ps1", backup_dir, backup_id


def validate_fixture(script: Path, backup_id: str, profile: str) -> subprocess.CompletedProcess[str]:
    return run_powershell([
        "-File", str(script),
        "-BackupId", backup_id,
        "-Profile", profile,
        "-ProjectName", "phase5test",
        "-ValidateOnly",
    ])


def test_profiles_cover_core_and_full_medical_services() -> None:
    services = load_compose()["services"]
    core = {
        "config-guard", "postgres", "migrate", "medagent-api", "lightrag",
        "medlive-api", "livekit", "medlive-worker", "frontend",
    }
    full_only = {"etcd", "minio", "milvus", "elasticsearch", "neo4j"}
    assert all(set(services[name]["profiles"]) == {"voice", "full"} for name in core)
    assert all(services[name]["profiles"] == ["full"] for name in full_only)


def test_persistence_healthchecks_and_private_data_services() -> None:
    config = load_compose()
    services = config["services"]
    for name in services:
        if name not in {"config-guard", "migrate"}:
            assert "healthcheck" in services[name]
    for name in ("postgres", "lightrag", "milvus", "elasticsearch", "neo4j", "minio", "etcd"):
        assert "ports" not in services[name]
    required_volumes = {
        "postgres-data", "medagent-data", "medlive-data", "milvus-etcd",
        "milvus-minio", "milvus-data", "elasticsearch-data", "neo4j-data",
    }
    assert required_volumes <= set(config["volumes"])


def test_secrets_and_secure_defaults() -> None:
    config = load_compose()
    assert set(config["secrets"]) >= {
        "jwt_secret", "postgres_password", "internal_api_key",
        "control_plane_key", "livekit_api_key", "livekit_api_secret",
        "lightrag_api_key", "elasticsearch_password",
    }
    registration = config["services"]["medagent-api"]["environment"]["ALLOW_PUBLIC_REGISTRATION"]
    assert registration.endswith(":-false}")
    assert config["services"]["elasticsearch"]["environment"]["xpack.security.enabled"] == "true"


def test_production_rejects_examples_and_default_data_credentials() -> None:
    env = {
        "MEDRAG_ENV": "prod",
        **EXAMPLE_SECRETS,
        "NEO4J_PASSWORD": "phase5-local-neo4j-only",
        "MINIO_ROOT_USER": "phase5-local-minio",
        "MINIO_ROOT_PASSWORD": "phase5-local-minio-password",
    }
    errors = production_errors(env)
    assert len(errors) == len(EXAMPLE_SECRETS) + 3
    assert production_errors({"MEDRAG_ENV": "dev"}) == []


def test_production_guard_reads_compose_secret_files(tmp_path: Path) -> None:
    for name, filename in SECRET_FILES.items():
        (tmp_path / filename).write_text(EXAMPLE_SECRETS[name], encoding="utf-8")
    env = {
        "MEDRAG_ENV": "prod",
        "NEO4J_PASSWORD": "not-a-development-password",
        "MINIO_ROOT_USER": "not-a-development-user",
        "MINIO_ROOT_PASSWORD": "not-a-development-password",
    }
    errors = production_errors(env, tmp_path)
    assert len(errors) == len(EXAMPLE_SECRETS)
    assert all("repository example value" in error for error in errors)


def test_python_images_are_split_and_use_cpu_torch() -> None:
    dockerfile = (ROOT / "deploy" / "Dockerfile.python").read_text(encoding="utf-8")
    assert "AS ops-runtime" in dockerfile
    assert "AS medical-runtime" in dockerfile
    assert "AS live-runtime" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert '".[medical]"' in dockerfile
    assert '".[live]"' in dockerfile
    assert "PIP_DEFAULT_TIMEOUT=300" in dockerfile
    assert "PIP_RETRIES=10" in dockerfile


def test_backup_ids_are_unique_and_existing_directories_are_rejected(tmp_path: Path) -> None:
    contract = ROOT / "deploy" / "ops" / "backup_contract.ps1"
    env = {
        **os.environ,
        "TEST_BACKUP_ROOT": str(tmp_path),
        "TEST_BACKUP_CONTRACT": str(contract),
    }
    command = """
. $env:TEST_BACKUP_CONTRACT
$first = New-BackupReservation -BackupRoot $env:TEST_BACKUP_ROOT
$second = New-BackupReservation -BackupRoot $env:TEST_BACKUP_ROOT
if ($first.Id -eq $second.Id) { throw 'generated duplicate BackupId' }
$rejected = $false
try { New-BackupReservation -BackupRoot $env:TEST_BACKUP_ROOT -BackupId $first.Id }
catch { $rejected = $_.Exception.Message -like 'backup directory already exists:*' }
if (-not $rejected) { throw 'existing backup directory was reused' }
Write-Output $first.Id
Write-Output $second.Id
"""
    result = run_powershell(["-Command", command], env=env)
    assert result.returncode == 0, result.stderr
    ids = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(ids) == 2
    assert ids[0] != ids[1]


def test_backup_reservation_allows_only_one_concurrent_winner(tmp_path: Path) -> None:
    contract = ROOT / "deploy" / "ops" / "backup_contract.ps1"
    backup_id = "20260904T010203004Z-abcdef123456"
    gate = tmp_path / "start.gate"
    env = {
        **os.environ,
        "TEST_BACKUP_ROOT": str(tmp_path),
        "TEST_BACKUP_CONTRACT": str(contract),
        "TEST_BACKUP_ID": backup_id,
        "TEST_BACKUP_GATE": str(gate),
    }
    command = """
. $env:TEST_BACKUP_CONTRACT
while (-not (Test-Path -LiteralPath $env:TEST_BACKUP_GATE -PathType Leaf)) {
    Start-Sleep -Milliseconds 10
}
$reservation = New-BackupReservation -BackupRoot $env:TEST_BACKUP_ROOT -BackupId $env:TEST_BACKUP_ID
Write-Output $reservation.Directory
"""
    processes = [
        subprocess.Popen(
            [POWERSHELL, "-NoProfile", "-Command", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        for _ in range(12)
    ]
    time.sleep(0.25)
    gate.write_text("go", encoding="utf-8")
    results = [process.communicate(timeout=30) + (process.returncode,) for process in processes]

    winners = [result for result in results if result[2] == 0]
    losers = [result for result in results if result[2] != 0]
    assert len(winners) == 1, results
    assert len(losers) == 11, results
    assert (tmp_path / backup_id).is_dir()
    assert not (tmp_path / f".{backup_id}.reserve").exists()


@pytest.mark.parametrize("profile", ["voice", "full"])
def test_restore_manifest_accepts_exact_complete_file_set(tmp_path: Path, profile: str) -> None:
    script, _, backup_id = make_restore_fixture(tmp_path, profile)
    result = validate_fixture(script, backup_id, profile)
    assert result.returncode == 0, result.stderr
    assert "Backup validation completed" in result.stdout


@pytest.mark.parametrize("profile", ["voice", "full"])
@pytest.mark.parametrize("mutation, expected", [
    ("schema", "unsupported manifest schema"),
    ("schema_type", "unsupported manifest schema"),
    ("duplicate", "duplicate manifest file"),
    ("incomplete", "manifest file set is incomplete"),
    ("extra", "backup directory file set does not match manifest"),
])
def test_restore_manifest_rejects_invalid_contract(
    tmp_path: Path,
    profile: str,
    mutation: str,
    expected: str,
) -> None:
    script, backup_dir, backup_id = make_restore_fixture(tmp_path, profile)
    manifest_path = backup_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "schema":
        manifest["schema"] = 3
    elif mutation == "schema_type":
        manifest["schema"] = "2"
    elif mutation == "duplicate":
        manifest["files"][2] = dict(manifest["files"][1])
    elif mutation == "incomplete":
        manifest["files"].pop()
    elif mutation == "extra":
        (backup_dir / "extra.txt").write_text("unexpected", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_fixture(script, backup_id, profile)
    assert result.returncode != 0
    assert expected in (result.stdout + result.stderr)


def test_scripts_reject_backup_root_prefix_and_invalid_backup_id(tmp_path: Path) -> None:
    ops_dir = tmp_path / "deploy" / "ops"
    ops_dir.mkdir(parents=True)
    for name in ("backup.ps1", "restore.ps1", "backup_contract.ps1"):
        shutil.copy2(ROOT / "deploy" / "ops" / name, ops_dir / name)

    bad_root = tmp_path / "backups-evil"
    backup = run_powershell([
        "-File", str(ops_dir / "backup.ps1"),
        "-Profile", "voice",
        "-ProjectName", "phase5test",
        "-BackupRoot", str(bad_root),
    ])
    assert backup.returncode != 0
    assert "BackupRoot must be exactly" in (backup.stdout + backup.stderr)
    assert not bad_root.exists()

    restore = run_powershell([
        "-File", str(ops_dir / "restore.ps1"),
        "-BackupId", "../escape",
        "-Profile", "voice",
        "-ProjectName", "phase5test",
        "-ValidateOnly",
    ])
    assert restore.returncode != 0
    assert "invalid BackupId" in (restore.stdout + restore.stderr)


def test_docker_compose_renders_both_profiles() -> None:
    for profile in ("voice", "full"):
        subprocess.run(
            ["docker", "compose", "--profile", profile, "config", "--quiet"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
