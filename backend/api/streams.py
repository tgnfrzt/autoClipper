import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.config import THUMBS_DIR, save_settings_to_env
from backend.core.database import fetch_all_streamers, get_db, upsert_streamer
from backend.modules.streams.discovery import check_twitch_channel_live, check_youtube_target_live
from backend.modules.streams.manager import stream_manager

router = APIRouter(prefix="/api/streams", tags=["streams"])


class AddStreamerRequest(BaseModel):
    target: str
    platform: str = "auto"  # auto | twitch | youtube
    display_name: Optional[str] = None
    priority: int = 90


class ToggleStreamerRequest(BaseModel):
    enabled: Optional[bool] = None
    priority: Optional[int] = None


class EngineToggleRequest(BaseModel):
    enabled: bool


@router.get("/dashboard")
def get_live_dashboard() -> Dict[str, Any]:
    return stream_manager.get_dashboard_snapshot()


@router.get("/watchlist")
def get_watchlist() -> Dict[str, Any]:
    return {"streamers": fetch_all_streamers(enabled_only=False)}


@router.post("/watchlist")
async def add_manual_streamer(req: AddStreamerRequest) -> Dict[str, Any]:
    raw = req.target.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Streamer username or URL is required.")

    plat = req.platform.lower().strip()
    if plat == "auto":
        low = raw.lower()
        if "youtube.com" in low or "youtu.be" in low or raw.startswith("@"):
            plat = "youtube"
        else:
            plat = "twitch"

    if plat == "twitch":
        login = raw.replace("https://www.twitch.tv/", "").replace("https://twitch.tv/", "").strip("/").split("/")[0].lower()
        live_info = await check_twitch_channel_live(login)
        if live_info:
            sid = upsert_streamer(
                platform="twitch",
                channel_id=live_info["channel_id"],
                display_name=req.display_name or live_info["display_name"],
                url=live_info["url"],
                is_manual=True,
                enabled=True,
                priority=req.priority,
                is_live=live_info["is_live"],
                viewer_count=live_info["viewer_count"],
                title=live_info["title"],
                game_name=live_info["game_name"],
            )
        else:
            sid = upsert_streamer(
                platform="twitch",
                channel_id=login,
                display_name=req.display_name or login,
                url=f"https://www.twitch.tv/{login}",
                is_manual=True,
                enabled=True,
                priority=req.priority,
            )
    else:
        yt_info = await check_youtube_target_live(raw)
        if yt_info:
            sid = upsert_streamer(
                platform="youtube",
                channel_id=yt_info["channel_id"],
                display_name=req.display_name or yt_info["display_name"],
                url=yt_info["url"],
                is_manual=True,
                enabled=True,
                priority=req.priority,
                is_live=yt_info["is_live"],
                viewer_count=yt_info["viewer_count"],
                title=yt_info["title"],
                game_name=yt_info["game_name"],
            )
        else:
            url = raw if raw.startswith("http") else f"https://www.youtube.com/{raw}/live"
            cid = raw.strip("/").split("/")[-1]
            sid = upsert_streamer(
                platform="youtube",
                channel_id=cid,
                display_name=req.display_name or cid,
                url=url,
                is_manual=True,
                enabled=True,
                priority=req.priority,
            )

    await stream_manager._sync_active_buffers()
    return {"success": True, "stream_id": sid}


@router.patch("/watchlist/{stream_id}")
async def update_streamer(stream_id: str, req: ToggleStreamerRequest) -> Dict[str, Any]:
    conn = get_db()
    with conn:
        if req.enabled is not None:
            conn.execute("UPDATE streamers SET enabled = ? WHERE id = ?", (1 if req.enabled else 0, stream_id))
        if req.priority is not None:
            conn.execute("UPDATE streamers SET priority = ? WHERE id = ?", (req.priority, stream_id))
    conn.close()
    await stream_manager._sync_active_buffers()
    return {"success": True}


@router.delete("/watchlist/{stream_id}")
async def delete_streamer(stream_id: str) -> Dict[str, Any]:
    conn = get_db()
    with conn:
        conn.execute("DELETE FROM streamers WHERE id = ?", (stream_id,))
    conn.close()
    await stream_manager._sync_active_buffers()
    return {"success": True}


@router.post("/discover")
async def trigger_discovery() -> Dict[str, Any]:
    res = await stream_manager.refresh_discovery_now()
    return {"success": True, **res}


@router.post("/engine")
async def toggle_engine(req: EngineToggleRequest) -> Dict[str, Any]:
    save_settings_to_env({"ENGINE_ENABLED": req.enabled})
    await stream_manager._sync_active_buffers()
    return {"success": True, "engine_enabled": req.enabled}


@router.post("/{stream_id}/clip-now")
async def clip_stream_now(stream_id: str) -> Dict[str, Any]:
    rec = await stream_manager.trigger_manual_clip(stream_id)
    if not rec:
        raise HTTPException(
            status_code=400,
            detail="Stream not active or insufficient buffered video yet (wait ~15s after stream starts).",
        )
    return {"success": True, "clip": rec}


@router.get("/thumb/{filename}")
def get_stream_thumb(filename: str):
    safe = Path(filename).name
    p = THUMBS_DIR / safe
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not ready")
    return FileResponse(str(p), media_type="image/jpeg")
