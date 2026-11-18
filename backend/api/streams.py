import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.config import THUMBS_DIR, save_settings_to_env, settings
from backend.core.database import fetch_all_streamers, get_db, upsert_streamer
from backend.modules.publishers.view_tracker import sync_all_clip_views
from backend.modules.streams.discovery import check_twitch_channel_live, check_youtube_target_live
from backend.modules.streams.manager import stream_manager
from backend.api.ai_models import _get_hardware_telemetry

router = APIRouter(prefix="/api/streams", tags=["streams"])


class AddStreamerRequest(BaseModel):
    target: str
    platform: str = "auto"  # auto | twitch | youtube
    display_name: Optional[str] = None
    priority: int = 92
    in_paid_watchlist: Optional[bool] = True


class ToggleStreamerRequest(BaseModel):
    enabled: Optional[bool] = None
    priority: Optional[int] = None
    in_paid_watchlist: Optional[bool] = None


class EngineToggleRequest(BaseModel):
    enabled: bool


@router.get("/dashboard")
def get_live_dashboard() -> Dict[str, Any]:
    return stream_manager.get_dashboard_snapshot()


@router.get("/telemetry")
def get_system_telemetry() -> Dict[str, Any]:
    snap = stream_manager.get_telemetry_snapshot()
    snap["hardware"] = _get_hardware_telemetry()
    return snap


@router.get("/watchlist")
def get_watchlist() -> Dict[str, Any]:
    rows = fetch_all_streamers(
        enabled_only=False,
        prefer_facecam=settings.PREFER_FACECAM_STREAMS,
        prefer_english=settings.PREFER_ENGLISH_STREAMS,
    )
    active_ids = set(stream_manager._buffers.keys())
    for idx, r in enumerate(rows):
        r["is_buffering"] = r["id"] in active_ids
        r["rank"] = idx + 1
    return {
        "streamers": rows,
        "active_count": len(active_ids),
        "max_concurrent_streams": settings.MAX_CONCURRENT_STREAMS,
        "prefer_english": settings.PREFER_ENGLISH_STREAMS,
    }


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

    paid_flag = True if req.in_paid_watchlist is None else bool(req.in_paid_watchlist)
    if plat == "twitch":
        login = (
            raw.replace("https://www.twitch.tv/", "")
            .replace("https://twitch.tv/", "")
            .replace("http://www.twitch.tv/", "")
            .replace("http://twitch.tv/", "")
            .strip("/")
            .split("/")[0]
            .split("?")[0]
            .lower()
        )
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
                in_paid_watchlist=paid_flag,
                language=live_info.get("language", "en"),
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
                in_paid_watchlist=paid_flag,
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
                in_paid_watchlist=paid_flag,
                language=yt_info.get("language", "en"),
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
                in_paid_watchlist=paid_flag,
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
        if req.in_paid_watchlist is not None:
            conn.execute("UPDATE streamers SET in_paid_watchlist = ? WHERE id = ?", (1 if req.in_paid_watchlist else 0, stream_id))
    conn.close()
    await stream_manager._sync_active_buffers()
    return {"success": True}


def _clip_mentions_streamer(clip: Dict[str, Any], stream_id: str, names: List[str]) -> bool:
    if clip.get("stream_id") == stream_id:
        return True
    sname = (clip.get("streamer_name") or "").strip().lower()
    cap = (clip.get("overlay_caption") or "").lower()
    title = (clip.get("ai_title") or "").lower()
    for n in names:
        nl = (n or "").strip().lower()
        if not nl:
            continue
        if sname == nl:
            return True
        if len(nl) >= 3 and re.search(rf"(?<![a-z0-9_]){re.escape(nl)}(?![a-z0-9_])", f"{cap} {title}"):
            return True
    return False


@router.get("/paid-watchlist")
def get_paid_watchlist() -> Dict[str, Any]:
    conn = get_db()
    try:
        paid_rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT * FROM streamers
                WHERE in_paid_watchlist = 1
                ORDER BY is_live DESC, priority DESC, viewer_count DESC
                """
            ).fetchall()
        ]
        all_clips = [
            dict(r)
            for r in conn.execute("SELECT * FROM clips ORDER BY created_at DESC").fetchall()
        ]
        all_posts = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM post_history WHERE status = 'posted' ORDER BY published_at DESC"
            ).fetchall()
        ]
    finally:
        conn.close()

    posts_by_clip: Dict[str, List[Dict[str, Any]]] = {}
    for p in all_posts:
        posts_by_clip.setdefault(p["clip_id"], []).append(p)

    entries: List[Dict[str, Any]] = list(paid_rows)

    summary_views = 0
    summary_posts = 0
    summary_by_plat = {"youtube_shorts": 0, "instagram": 0, "tiktok": 0}
    seen_global_post_ids: set[str] = set()

    enriched: List[Dict[str, Any]] = []
    for s in entries:
        sid = s["id"]
        match_names = [s.get("display_name") or "", s.get("channel_id") or ""]
        matched_clips = [c for c in all_clips if _clip_mentions_streamer(c, sid, match_names)]

        streamer_posts: List[Dict[str, Any]] = []
        seen_post_ids: set[str] = set()
        total_views = 0
        total_likes = 0
        by_plat = {"youtube_shorts": 0, "instagram": 0, "tiktok": 0}

        clip_cards = []
        for c in matched_clips:
            c_posts = posts_by_clip.get(c["id"], [])
            clip_cards.append(
                {
                    "id": c["id"],
                    "overlay_caption": c.get("overlay_caption") or "",
                    "ai_title": c.get("ai_title") or "",
                    "status": c.get("status") or "pending",
                    "duration_sec": c.get("duration_sec") or 0,
                    "created_at": c.get("created_at") or "",
                    "posts": c_posts,
                }
            )
            for p in c_posts:
                pid = p["id"]
                if pid in seen_post_ids:
                    continue
                seen_post_ids.add(pid)
                v = int(p.get("views") or 0)
                lk = int(p.get("likes") or 0)
                plat_key = "youtube_shorts" if "youtube" in (p.get("platform") or "") else (
                    "instagram" if "instagram" in (p.get("platform") or "") else "tiktok"
                )
                total_views += v
                total_likes += lk
                by_plat[plat_key] = by_plat.get(plat_key, 0) + v
                if pid not in seen_global_post_ids:
                    seen_global_post_ids.add(pid)
                    summary_views += v
                    summary_posts += 1
                    summary_by_plat[plat_key] = summary_by_plat.get(plat_key, 0) + v

                streamer_posts.append(
                    {
                        **p,
                        "clip_caption": c.get("overlay_caption") or c.get("ai_title") or c.get("streamer_name"),
                    }
                )

        streamer_posts.sort(key=lambda x: x.get("published_at") or "", reverse=True)
        enriched.append(
            {
                "id": sid,
                "display_name": s.get("display_name") or s.get("channel_id") or sid,
                "platform": s.get("platform") or "twitch",
                "channel_id": s.get("channel_id") or "",
                "url": s.get("url") or "",
                "is_live": bool(s.get("is_live")),
                "enabled": bool(s.get("enabled", 1)),
                "in_paid_watchlist": bool(s.get("in_paid_watchlist", 1)),
                "clips_count": len(matched_clips),
                "posts_count": len(streamer_posts),
                "total_views": total_views,
                "total_likes": total_likes,
                "views_by_platform": by_plat,
                "posts": streamer_posts,
                "clips": clip_cards,
            }
        )

    enriched.sort(
        key=lambda x: (x["total_views"], x["posts_count"], x["clips_count"], 1 if x["is_live"] else 0),
        reverse=True,
    )

    return {
        "summary": {
            "streamers_count": len(enriched),
            "clips_count": sum(e["clips_count"] for e in enriched),
            "posts_count": summary_posts,
            "total_views": summary_views,
            "views_by_platform": summary_by_plat,
        },
        "streamers": enriched,
    }


@router.post("/paid-watchlist/sync-views")
async def sync_paid_watchlist_views() -> Dict[str, Any]:
    sync_res = await asyncio.to_thread(sync_all_clip_views)
    data = get_paid_watchlist()
    return {"success": True, "sync": sync_res, **data}


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
