"""Phase 4 repository migration acceptance checks."""

from pathlib import Path

from medcontracts.phase0 import CapabilityEnvelope, Evidence
from medrag.contracts.phase0 import (
    CapabilityEnvelope as LegacyCapabilityEnvelope,
)
from medrag.contracts.phase0 import Evidence as LegacyEvidence


ROOT = Path(__file__).resolve().parents[1]


def test_shared_contracts_have_one_canonical_class_identity():
    assert LegacyCapabilityEnvelope is CapabilityEnvelope
    assert LegacyEvidence is Evidence


def test_medlive_has_no_source_imports_from_old_package_name():
    offenders = []
    for path in (ROOT / "src" / "medlive").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "from liverag" in source or "import liverag" in source:
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_liverag_mit_license_and_source_notice_are_retained():
    license_text = (ROOT / "licenses" / "LiveRAG-MIT.txt").read_text(encoding="utf-8")
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "LiveRAG" in notice
    assert "without importing Git history" in notice


def test_next_frontend_is_root_and_vue_is_only_legacy():
    assert (ROOT / "frontend" / "package.json").is_file()
    assert (ROOT / "frontend" / "app" / "page.tsx").is_file()
    assert (ROOT / "frontend-legacy" / "index.html").is_file()
    assert (ROOT / "frontend" / "public" / "legacy" / "index.html").is_file()
    page = (ROOT / "frontend" / "app" / "page.tsx").read_text(encoding="utf-8")
    assert "UnifiedApp" in page
    assert "frontend/index.html" not in page


def test_voice_token_route_uses_bound_medlive_session():
    route = (ROOT / "frontend" / "app" / "api" / "token" / "route.ts").read_text(
        encoding="utf-8"
    )
    assert "/voice/sessions" in route
    assert "medagent_access_token" not in route
    assert "ALLOW_INSECURE_TOKEN_API" not in route
    assert "kb_id" in route


def test_login_bff_keeps_jwt_out_of_browser_response():
    route = (
        ROOT / "frontend" / "app" / "api" / "auth" / "login" / "route.ts"
    ).read_text(encoding="utf-8")
    browser_payload = route.split("const result = NextResponse.json({", 1)[1].split(
        "});", 1
    )[0]

    assert "access_token" not in browser_payload
    assert "user_id: payload.user_id" in browser_payload
    assert "username: payload.username" in browser_payload
    assert "result.cookies.set(ACCESS_COOKIE, payload.access_token" in route
    assert "httpOnly: true" in route
    assert "new NextResponse(body" not in route


def test_evidence_view_uses_owned_current_voice_session_routes():
    unified = (
        ROOT / "frontend" / "components" / "app" / "unified-app.tsx"
    ).read_text(encoding="utf-8")
    api = (ROOT / "frontend" / "lib" / "liverag-api.ts").read_text(
        encoding="utf-8"
    )
    token_route = (
        ROOT / "frontend" / "app" / "api" / "token" / "route.ts"
    ).read_text(encoding="utf-8")
    turns_route = (
        ROOT / "frontend" / "app" / "api" / "voice" / "current" / "turns" / "route.ts"
    ).read_text(encoding="utf-8")
    rag_route = (
        ROOT
        / "frontend"
        / "app"
        / "api"
        / "voice"
        / "current"
        / "rag-context"
        / "route.ts"
    ).read_text(encoding="utf-8")

    assert "getSessionTurns" not in unified
    assert "/session/turns" not in unified
    assert "getCurrentVoiceSessionTurns" in api
    assert "getCurrentVoiceSessionRagContext" in api
    assert "VOICE_SESSION_COOKIE" in token_route
    assert all(part in turns_route for part in ("'voice'", "'sessions'", "'turns'"))
    assert all(part in rag_route for part in ("'voice'", "'sessions'", "'rag-context'"))


def test_phase4_does_not_add_deployment_or_git_history():
    assert not (ROOT / "frontend" / "docker-compose.yml").exists()
    assert not (ROOT / "frontend" / "Dockerfile").exists()
    assert not (ROOT / "src" / "medlive" / ".git").exists()
