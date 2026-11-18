import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from backend.core.config import (
    DATA_DIR,
    get_watermark_path,
    is_instagram_configured,
    is_tiktok_configured,
    is_youtube_configured,
    save_settings_to_env,
    settings,
)
from backend.modules.discord.bot import discord_bot

os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

router = APIRouter(prefix="/api/settings", tags=["settings"])

_PKCE_FILE = str(DATA_DIR / ".youtube_pkce.json")


class SettingsUpdateRequest(BaseModel):
    ENGINE_ENABLED: Optional[bool] = None
    GAME_QUERY: Optional[str] = None
    FALLBACK_GAME_QUERY: Optional[str] = None
    AUTO_DISCOVER_TWITCH: Optional[bool] = None
    AUTO_DISCOVER_YOUTUBE: Optional[bool] = None
    PREFER_FACECAM_STREAMS: Optional[bool] = None
    PREFER_ENGLISH_STREAMS: Optional[bool] = None
    MAX_CONCURRENT_STREAMS: Optional[int] = None
    STREAM_QUALITY: Optional[str] = None
    DISCOVERY_INTERVAL_SEC: Optional[int] = None

    CHAT_WEIGHT: Optional[float] = None
    VISUAL_WEIGHT: Optional[float] = None
    TRIGGER_SCORE_THRESHOLD: Optional[float] = None
    CLIP_DURATION_SEC: Optional[int] = None
    AUTO_TRIM_ENABLED: Optional[bool] = None
    MIN_CLIP_DURATION_SEC: Optional[int] = None
    POST_TRIGGER_DELAY_SEC: Optional[int] = None
    CLIP_COOLDOWN_SEC: Optional[int] = None
    VISUAL_EVAL_INTERVAL_SEC: Optional[int] = None

    VLM_PROVIDER: Optional[str] = None
    VLM_API_URL: Optional[str] = None
    VLM_API_KEY: Optional[str] = None
    VLM_MODEL_NAME: Optional[str] = None
    LLAMA_SERVER_EXE: Optional[str] = None
    VLM_MODEL_PATH: Optional[str] = None
    VLM_MMPROJ_PATH: Optional[str] = None
    VLM_PORT: Optional[int] = None
    VLM_AUTO_START: Optional[bool] = None

    VIDEO_FORMAT_MODE: Optional[str] = None
    FACECAM_POSITION: Optional[str] = None
    BURN_IN_CAPTIONS: Optional[bool] = None
    ANTI_FINGERPRINT: Optional[bool] = None
    WATERMARK_ALPHA: Optional[float] = None

    DISCORD_WEBHOOK_URL: Optional[str] = None
    DISCORD_BOT_TOKEN: Optional[str] = None
    DISCORD_CHANNEL_ID: Optional[str] = None
    DISCORD_USER_ID: Optional[str] = None
    DISCORD_COMMAND_PREFIX: Optional[str] = None
    AUTO_DELETE_ON_REJECT: Optional[bool] = None

    TWITCH_CLIENT_ID: Optional[str] = None
    TWITCH_CLIENT_SECRET: Optional[str] = None

    YOUTUBE_DEFAULT_PRIVACY: Optional[str] = None
    DEFAULT_YOUTUBE_CAPTION: Optional[str] = None
    TIKTOK_SESSION_ID: Optional[str] = None
    TIKTOK_ACCESS_TOKEN: Optional[str] = None
    TIKTOK_CLIENT_KEY: Optional[str] = None
    TIKTOK_CLIENT_SECRET: Optional[str] = None
    DEFAULT_TIKTOK_CAPTION: Optional[str] = None
    INSTAGRAM_BUSINESS_ID: Optional[str] = None
    PAGE_ACCESS_TOKEN: Optional[str] = None
    REMOTE_HOSTING_URL: Optional[str] = None
    DEFAULT_INSTAGRAM_CAPTION: Optional[str] = None


class ImportAutoPosterRequest(BaseModel):
    autoposter_path: str = r"D:\AutoPoster"


class JsonPayloadRequest(BaseModel):
    content: str


class TestInstagramRequest(BaseModel):
    business_id: Optional[str] = None
    access_token: Optional[str] = None


class TestHostingRequest(BaseModel):
    hosting_url: Optional[str] = None


class TestTikTokRequest(BaseModel):
    access_token: Optional[str] = None
    session_id: Optional[str] = None


@router.get("")
def get_all_settings() -> Dict[str, Any]:
    return {
        "ENGINE_ENABLED": settings.ENGINE_ENABLED,
        "GAME_QUERY": settings.GAME_QUERY,
        "FALLBACK_GAME_QUERY": settings.FALLBACK_GAME_QUERY,
        "AUTO_DISCOVER_TWITCH": settings.AUTO_DISCOVER_TWITCH,
        "AUTO_DISCOVER_YOUTUBE": settings.AUTO_DISCOVER_YOUTUBE,
        "PREFER_FACECAM_STREAMS": settings.PREFER_FACECAM_STREAMS,
        "PREFER_ENGLISH_STREAMS": settings.PREFER_ENGLISH_STREAMS,
        "MAX_CONCURRENT_STREAMS": settings.MAX_CONCURRENT_STREAMS,
        "STREAM_QUALITY": settings.STREAM_QUALITY,
        "DISCOVERY_INTERVAL_SEC": settings.DISCOVERY_INTERVAL_SEC,
        "CHAT_WEIGHT": settings.CHAT_WEIGHT,
        "VISUAL_WEIGHT": settings.VISUAL_WEIGHT,
        "TRIGGER_SCORE_THRESHOLD": settings.TRIGGER_SCORE_THRESHOLD,
        "CLIP_DURATION_SEC": settings.CLIP_DURATION_SEC,
        "AUTO_TRIM_ENABLED": settings.AUTO_TRIM_ENABLED,
        "MIN_CLIP_DURATION_SEC": settings.MIN_CLIP_DURATION_SEC,
        "POST_TRIGGER_DELAY_SEC": settings.POST_TRIGGER_DELAY_SEC,
        "CLIP_COOLDOWN_SEC": settings.CLIP_COOLDOWN_SEC,
        "VISUAL_EVAL_INTERVAL_SEC": settings.VISUAL_EVAL_INTERVAL_SEC,
        "VLM_PROVIDER": settings.VLM_PROVIDER,
        "VLM_API_URL": settings.VLM_API_URL,
        "VLM_API_KEY": settings.VLM_API_KEY,
        "VLM_MODEL_NAME": settings.VLM_MODEL_NAME,
        "LLAMA_SERVER_EXE": settings.LLAMA_SERVER_EXE,
        "VLM_MODEL_PATH": settings.VLM_MODEL_PATH,
        "VLM_MMPROJ_PATH": settings.VLM_MMPROJ_PATH,
        "VLM_PORT": settings.VLM_PORT,
        "VLM_AUTO_START": settings.VLM_AUTO_START,
        "VIDEO_FORMAT_MODE": settings.VIDEO_FORMAT_MODE,
        "FACECAM_POSITION": settings.FACECAM_POSITION,
        "BURN_IN_CAPTIONS": settings.BURN_IN_CAPTIONS,
        "ANTI_FINGERPRINT": settings.ANTI_FINGERPRINT,
        "WATERMARK_ALPHA": settings.WATERMARK_ALPHA,
        "DISCORD_WEBHOOK_URL": settings.DISCORD_WEBHOOK_URL,
        "DISCORD_BOT_TOKEN": settings.DISCORD_BOT_TOKEN,
        "DISCORD_CHANNEL_ID": settings.DISCORD_CHANNEL_ID,
        "DISCORD_USER_ID": settings.DISCORD_USER_ID,
        "DISCORD_COMMAND_PREFIX": settings.DISCORD_COMMAND_PREFIX,
        "AUTO_DELETE_ON_REJECT": settings.AUTO_DELETE_ON_REJECT,
        "TWITCH_CLIENT_ID": settings.TWITCH_CLIENT_ID,
        "TWITCH_CLIENT_SECRET": settings.TWITCH_CLIENT_SECRET,
        "YOUTUBE_DEFAULT_PRIVACY": settings.YOUTUBE_DEFAULT_PRIVACY,
        "DEFAULT_YOUTUBE_CAPTION": settings.DEFAULT_YOUTUBE_CAPTION,
        "TIKTOK_SESSION_ID": settings.TIKTOK_SESSION_ID,
        "TIKTOK_ACCESS_TOKEN": settings.TIKTOK_ACCESS_TOKEN,
        "TIKTOK_CLIENT_KEY": settings.TIKTOK_CLIENT_KEY,
        "TIKTOK_CLIENT_SECRET": settings.TIKTOK_CLIENT_SECRET,
        "DEFAULT_TIKTOK_CAPTION": settings.DEFAULT_TIKTOK_CAPTION,
        "INSTAGRAM_BUSINESS_ID": settings.INSTAGRAM_BUSINESS_ID,
        "PAGE_ACCESS_TOKEN": settings.PAGE_ACCESS_TOKEN,
        "REMOTE_HOSTING_URL": settings.REMOTE_HOSTING_URL,
        "DEFAULT_INSTAGRAM_CAPTION": settings.DEFAULT_INSTAGRAM_CAPTION,
        "status": {
            "youtube_ready": is_youtube_configured(),
            "youtube_has_secret": os.path.isfile(settings.YOUTUBE_CLIENT_SECRETS_FILE),
            "youtube_has_token": os.path.isfile(settings.YOUTUBE_TOKEN_FILE),
            "tiktok_ready": is_tiktok_configured(),
            "tiktok_has_cookies": os.path.isfile(settings.TIKTOK_COOKIES_PATH),
            "instagram_ready": is_instagram_configured(),
            "watermark_exists": get_watermark_path() is not None,
            "discord": discord_bot.get_status(),
            "autoposter_detected": os.path.isdir(r"D:\AutoPoster"),
        },
    }


@router.post("")
async def update_settings(req: SettingsUpdateRequest) -> Dict[str, Any]:
    updates = {k: v for k, v in req.model_dump(exclude_unset=True).items() if v is not None}
    # Auto-route if a user pastes a Discord Webhook URL into the Bot Token box
    tok_val = str(updates.get("DISCORD_BOT_TOKEN", "")).strip()
    if tok_val.startswith("https://") and "/api/webhooks/" in tok_val:
        updates["DISCORD_WEBHOOK_URL"] = tok_val
        updates["DISCORD_BOT_TOKEN"] = ""

    discord_changed = any(
        k in updates for k in ("DISCORD_WEBHOOK_URL", "DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID", "DISCORD_USER_ID")
    )
    stream_sync_needed = any(
        k in updates for k in ("ENGINE_ENABLED", "MAX_CONCURRENT_STREAMS", "PREFER_FACECAM_STREAMS", "PREFER_ENGLISH_STREAMS")
    )
    if updates:
        save_settings_to_env(updates)
    if discord_changed:
        await discord_bot.start_or_restart()
    if stream_sync_needed:
        from backend.modules.streams.manager import stream_manager
        await stream_manager._sync_active_buffers()
    return {"success": True, "updated_fields": list(updates.keys())}


@router.post("/import-autoposter")
def import_from_autoposter(req: ImportAutoPosterRequest) -> Dict[str, Any]:
    ap_root = Path(req.autoposter_path)
    if not ap_root.is_dir():
        return {"success": False, "error": f"AutoPoster directory not found: {req.autoposter_path}"}

    ap_data = ap_root / "storage" / "data"
    copied_files = []
    for fname in ("client_secret.json", "youtube_token.json", "tiktok_cookies.txt", "cookies.txt"):
        src = ap_data / fname
        if src.is_file():
            shutil.copy2(src, DATA_DIR / fname)
            copied_files.append(fname)

    keys_to_migrate = {
        "INSTAGRAM_BUSINESS_ID",
        "PAGE_ACCESS_TOKEN",
        "REMOTE_HOSTING_URL",
        "TIKTOK_ACCESS_TOKEN",
        "TIKTOK_CLIENT_KEY",
        "TIKTOK_CLIENT_SECRET",
        "TIKTOK_SESSION_ID",
        "YOUTUBE_DEFAULT_PRIVACY",
    }
    updates: Dict[str, Any] = {}
    ap_env = ap_root / ".env"
    if ap_env.is_file():
        for line in ap_env.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k in keys_to_migrate and v:
                    updates[k] = v
    if updates:
        save_settings_to_env(updates)
    return {
        "success": True,
        "copied_files": copied_files,
        "imported_keys": list(updates.keys()),
    }


@router.post("/discord/test")
async def test_discord_bot() -> Dict[str, Any]:
    return await discord_bot.send_test_message()


@router.post("/watermark/upload")
async def upload_watermark(file: UploadFile = File(...)) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(settings.WATERMARK_PATH), exist_ok=True)
    content = await file.read()
    with open(settings.WATERMARK_PATH, "wb") as f:
        f.write(content)
    return {"success": True, "size": len(content)}


# ------------------------------------------------------------------
# Instagram & Hosting Relay Endpoints
# ------------------------------------------------------------------
@router.post("/test_instagram")
def test_instagram(req: TestInstagramRequest) -> Dict[str, Any]:
    b_id = req.business_id or settings.INSTAGRAM_BUSINESS_ID
    token = req.access_token or settings.PAGE_ACCESS_TOKEN
    if not b_id or not token:
        return {"valid": False, "error": "Provide Instagram Business ID and Page Access Token."}
    try:
        r = requests.get(
            f"{settings.GRAPH_API_URL}/{b_id}",
            params={"fields": "name,username", "access_token": token},
            timeout=15,
        )
        data = r.json()
        if "error" in data:
            return {"valid": False, "error": data["error"].get("message", "Graph API error")}
        return {"valid": True, "id": data.get("id"), "name": data.get("name"), "username": data.get("username", "")}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


@router.post("/test_hosting")
def test_remote_hosting(req: TestHostingRequest) -> Dict[str, Any]:
    host = (req.hosting_url or settings.REMOTE_HOSTING_URL).rstrip("/")
    if not host:
        return {"valid": False, "error": "No remote hosting URL provided."}
    try:
        r = requests.get(f"{host}/health", timeout=10)
        if r.status_code == 200:
            return {"valid": True, "message": "Hosting relay healthy (HTTP 200)."}
    except Exception:
        pass
    try:
        r = requests.get(f"{host}/get_and_clear_queue", timeout=10)
        if r.status_code == 200:
            return {"valid": True, "message": "Hosting relay reachable (HTTP 200)."}
        return {"valid": False, "error": f"Relay responded HTTP {r.status_code}"}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


# ------------------------------------------------------------------
# TikTok Endpoints
# ------------------------------------------------------------------
@router.post("/test_tiktok")
def test_tiktok(req: TestTikTokRequest) -> Dict[str, Any]:
    token = req.access_token or settings.TIKTOK_ACCESS_TOKEN
    session = req.session_id or settings.TIKTOK_SESSION_ID

    if session:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.tiktok.com/",
            }
            r = requests.get(
                "https://www.tiktok.com/passport/web/account/info/",
                headers=headers,
                cookies={"sessionid": session},
                timeout=12,
            )
            data = r.json()
            if data.get("message") == "success" and data.get("data"):
                acc = data["data"]
                username = acc.get("username") or acc.get("screen_name") or "TikTok Creator"
                return {"valid": True, "mode": "session", "display_name": f"@{username}"}
            desc = data.get("data", {}).get("description") or data.get("message") or "Session expired"
            return {"valid": False, "mode": "session", "error": f"Rejected by TikTok: {desc}"}
        except Exception as exc:
            return {"valid": True, "mode": "session", "display_name": f"Session Saved ({exc})"}

    if token:
        try:
            r = requests.get(
                "https://open.tiktokapis.com/v2/user/info/?fields=open_id,display_name",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
            data = r.json()
            err = data.get("error", {})
            if err.get("code") == "ok":
                u = data.get("data", {}).get("user", {})
                return {"valid": True, "mode": "official_api", "display_name": u.get("display_name", "TikTok Creator")}
            return {"valid": False, "mode": "official_api", "error": err.get("message", "TikTok API error")}
        except Exception as exc:
            return {"valid": False, "mode": "official_api", "error": str(exc)}

    if os.path.isfile(settings.TIKTOK_COOKIES_PATH):
        with open(settings.TIKTOK_COOKIES_PATH, "r", encoding="utf-8", errors="replace") as f:
            ok = "sessionid" in f.read().lower()
        return {"valid": ok, "mode": "cookies_file", "display_name": "tiktok_cookies.txt"}

    return {"valid": False, "error": "No TikTok Session ID, cookies file, or API token configured."}


@router.post("/tiktok/cookies/upload")
async def upload_tiktok_cookies(file: UploadFile = File(...)) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(settings.TIKTOK_COOKIES_PATH), exist_ok=True)
    content = await file.read()
    with open(settings.TIKTOK_COOKIES_PATH, "wb") as f:
        f.write(content)
    return {"success": True, "size": len(content), "filename": file.filename}


# ------------------------------------------------------------------
# YouTube OAuth Endpoints
# ------------------------------------------------------------------
def _get_youtube_channel_info() -> Dict[str, Any]:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    token_path = settings.YOUTUBE_TOKEN_FILE
    if not os.path.isfile(token_path):
        return {"valid": False, "error": "YouTube token not found. Please authorize YouTube."}

    with open(token_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    creds = Credentials.from_authorized_user_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube.readonly",
        ],
    )
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(token_path, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        except Exception as ex:
            return {"valid": False, "error": f"YouTube token expired: {ex}. Re-authorize."}
    elif creds.expired:
        return {"valid": False, "error": "YouTube token expired without refresh token."}

    yt = build("youtube", "v3", credentials=creds)
    res = yt.channels().list(mine=True, part="snippet,statistics").execute()
    items = res.get("items", [])
    if not items:
        return {"valid": True, "channel_title": "Authorized (No YouTube channel)", "channel_id": ""}
    ch = items[0]
    snippet = ch.get("snippet", {})
    stats = ch.get("statistics", {})
    return {
        "valid": True,
        "channel_id": ch.get("id", ""),
        "channel_title": snippet.get("title", ""),
        "subscriber_count": stats.get("subscriberCount", "Hidden"),
    }


def _save_pkce_verifier(state: str, code_verifier: Optional[str]) -> None:
    if not code_verifier:
        return
    try:
        data = {}
        if os.path.exists(_PKCE_FILE):
            with open(_PKCE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        now = time.time()
        data = {k: v for k, v in data.items() if isinstance(v, dict) and (now - v.get("created_at", 0) < 7200)}
        data[state] = {"verifier": code_verifier, "created_at": now}
        data["_latest"] = {"verifier": code_verifier, "state": state, "created_at": now}
        with open(_PKCE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _get_pkce_verifier(state: Optional[str] = None) -> Optional[str]:
    if not os.path.exists(_PKCE_FILE):
        return None
    try:
        with open(_PKCE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if state and state in data:
            return data[state].get("verifier")
        if "_latest" in data:
            return data["_latest"].get("verifier")
    except Exception:
        pass
    return None


@router.post("/youtube/upload_secret")
async def upload_youtube_secret(file: UploadFile = File(...)) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(settings.YOUTUBE_CLIENT_SECRETS_FILE), exist_ok=True)
    content = await file.read()
    with open(settings.YOUTUBE_CLIENT_SECRETS_FILE, "wb") as f:
        f.write(content)
    return {"success": True, "size": len(content), "filename": file.filename}


@router.post("/youtube/save_secret_json")
def save_youtube_secret_json(req: JsonPayloadRequest) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(settings.YOUTUBE_CLIENT_SECRETS_FILE), exist_ok=True)
    with open(settings.YOUTUBE_CLIENT_SECRETS_FILE, "w", encoding="utf-8") as f:
        f.write(req.content.strip())
    return {"success": True, "size": len(req.content)}


@router.get("/youtube/auth_url")
def get_youtube_auth_url() -> Dict[str, Any]:
    if not os.path.isfile(settings.YOUTUBE_CLIENT_SECRETS_FILE):
        return {"ready": False, "error": "Upload Google client_secret.json first."}
    try:
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_secrets_file(
            settings.YOUTUBE_CLIENT_SECRETS_FILE,
            scopes=[
                "https://www.googleapis.com/auth/youtube.upload",
                "https://www.googleapis.com/auth/youtube.readonly",
            ],
            redirect_uri=f"http://localhost:{settings.PORT}/api/settings/youtube/callback",
        )
        auth_url, state = flow.authorization_url(
            access_type="offline",
            prompt="consent",
            include_granted_scopes="true",
        )
        if getattr(flow, "code_verifier", None):
            _save_pkce_verifier(state, flow.code_verifier)
        return {"ready": True, "auth_url": auth_url, "state": state}
    except Exception as exc:
        return {"ready": False, "error": str(exc)}


@router.get("/youtube/callback")
def youtube_oauth_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    if error or not code:
        return HTMLResponse(
            f"<html><body style='background:#0b0f19;color:#ef4444;font-family:sans-serif;padding:2rem;'><h2>OAuth Error: {error or 'Missing code'}</h2></body></html>",
            status_code=400,
        )
    try:
        from google_auth_oauthlib.flow import Flow

        code_verifier = _get_pkce_verifier(state)
        kwargs: Dict[str, Any] = {
            "scopes": [
                "https://www.googleapis.com/auth/youtube.upload",
                "https://www.googleapis.com/auth/youtube.readonly",
            ],
            "redirect_uri": f"http://localhost:{settings.PORT}/api/settings/youtube/callback",
        }
        if state:
            kwargs["state"] = state
        if code_verifier:
            kwargs["code_verifier"] = code_verifier

        flow = Flow.from_client_secrets_file(settings.YOUTUBE_CLIENT_SECRETS_FILE, **kwargs)
        if code_verifier:
            flow.code_verifier = code_verifier
        flow.fetch_token(code=code)
        with open(settings.YOUTUBE_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(flow.credentials.to_json())

        return HTMLResponse(
            """<!DOCTYPE html><html><body style="background:#0b0f19;color:#10b981;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;">
            <div style="background:#131b2e;padding:2rem;border-radius:12px;border:1px solid #334155;text-align:center;">
              <h2>YouTube Connected!</h2>
              <p style="color:#94a3b8;">You can close this window and return to AutoClipper.</p>
              <script>if(window.opener){window.opener.postMessage({type:'YOUTUBE_AUTH_SUCCESS'},'*');setTimeout(()=>window.close(),1500);}</script>
            </div></body></html>"""
        )
    except Exception as exc:
        return HTMLResponse(
            f"<html><body style='background:#0b0f19;color:#ef4444;font-family:sans-serif;padding:2rem;'><h2>OAuth Exchange Failed</h2><pre>{exc}</pre></body></html>",
            status_code=500,
        )


@router.post("/youtube/exchange_code")
def exchange_youtube_code(req: JsonPayloadRequest) -> Dict[str, Any]:
    raw = req.content.strip()
    if not os.path.isfile(settings.YOUTUBE_CLIENT_SECRETS_FILE):
        return {"success": False, "error": "client_secret.json missing."}
    code = raw
    state = None
    if "code=" in raw:
        qs = urlparse(raw).query if "?" in raw else raw
        parsed = parse_qs(qs)
        code = parsed.get("code", [code])[0]
        state = parsed.get("state", [None])[0]
    try:
        from google_auth_oauthlib.flow import Flow

        code_verifier = _get_pkce_verifier(state)
        kwargs: Dict[str, Any] = {
            "scopes": [
                "https://www.googleapis.com/auth/youtube.upload",
                "https://www.googleapis.com/auth/youtube.readonly",
            ],
            "redirect_uri": f"http://localhost:{settings.PORT}/api/settings/youtube/callback",
        }
        if state:
            kwargs["state"] = state
        if code_verifier:
            kwargs["code_verifier"] = code_verifier
        flow = Flow.from_client_secrets_file(settings.YOUTUBE_CLIENT_SECRETS_FILE, **kwargs)
        if code_verifier:
            flow.code_verifier = code_verifier
        flow.fetch_token(code=code)
        with open(settings.YOUTUBE_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(flow.credentials.to_json())
        return {"success": True, "channel": _get_youtube_channel_info()}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.post("/test_youtube")
def test_youtube_connection() -> Dict[str, Any]:
    try:
        return _get_youtube_channel_info()
    except Exception as exc:
        return {"valid": False, "error": str(exc)}
