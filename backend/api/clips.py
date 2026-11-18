import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.database import get_clip, get_db, list_clips
from backend.modules.discord.bot import discord_bot
from backend.modules.processors.video_processor import render_clip_for_social
from backend.modules.publishers.dispatcher import get_configured_platforms, publish_clip_to_platforms

router = APIRouter(prefix="/api/clips", tags=["clips"])


class ClipUpdateRequest(BaseModel):
    ai_title: Optional[str] = None
    ai_caption: Optional[str] = None


class ClipRenderRequest(BaseModel):
    format_mode: Optional[str] = None
    facecam_position: Optional[str] = None


class ClipPublishRequest(BaseModel):
    platforms: List[str] = ["all"]


@router.get("")
def get_clips(limit: int = 100) -> Dict[str, Any]:
    items = list_clips(limit=limit)
    for c in items:
        raw_kw = c.get("chat_keywords")
        if isinstance(raw_kw, str):
            try:
                c["chat_keywords"] = json.loads(raw_kw)
            except Exception:
                c["chat_keywords"] = []
        c["has_raw"] = bool(c.get("raw_path") and os.path.isfile(c["raw_path"]))
        c["has_rendered"] = bool(c.get("rendered_path") and os.path.isfile(c["rendered_path"]))
        c["has_thumb"] = bool(c.get("thumb_path") and os.path.isfile(c["thumb_path"]))
    return {"clips": items}


@router.get("/{clip_id}/video")
def get_clip_video(clip_id: str, variant: str = "auto"):
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if variant == "rendered":
        path = clip.get("rendered_path")
    elif variant == "raw":
        path = clip.get("raw_path")
    else:
        path = clip.get("rendered_path") if (clip.get("rendered_path") and os.path.isfile(clip["rendered_path"])) else clip.get("raw_path")
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path, media_type="video/mp4")


@router.get("/{clip_id}/thumb")
def get_clip_thumb(clip_id: str):
    clip = get_clip(clip_id)
    if not clip or not clip.get("thumb_path") or not os.path.isfile(clip["thumb_path"]):
        raise HTTPException(status_code=404, detail="Clip thumbnail not found")
    return FileResponse(clip["thumb_path"], media_type="image/jpeg")


@router.patch("/{clip_id}")
def update_clip_metadata(clip_id: str, req: ClipUpdateRequest) -> Dict[str, Any]:
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    conn = get_db()
    with conn:
        if req.ai_title is not None:
            conn.execute("UPDATE clips SET ai_title = ? WHERE id = ?", (req.ai_title, clip_id))
        if req.ai_caption is not None:
            conn.execute("UPDATE clips SET ai_caption = ? WHERE id = ?", (req.ai_caption, clip_id))
    conn.close()
    return {"success": True}


@router.post("/{clip_id}/render")
async def render_clip(clip_id: str, req: ClipRenderRequest) -> Dict[str, Any]:
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    res = await asyncio.to_thread(
        render_clip_for_social,
        clip["raw_path"],
        clip_id,
        req.format_mode,
        req.facecam_position,
    )
    conn = get_db()
    with conn:
        conn.execute("UPDATE clips SET rendered_path = ? WHERE id = ?", (res["output_path"], clip_id))
    conn.close()
    return {"success": True, **res}


@router.post("/{clip_id}/publish")
async def publish_clip(clip_id: str, req: ClipPublishRequest) -> Dict[str, Any]:
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    targets = req.platforms
    if not targets or "all" in [p.lower() for p in targets]:
        targets = get_configured_platforms()
        if not targets:
            raise HTTPException(status_code=400, detail="No social platforms configured yet.")
    res = await asyncio.to_thread(publish_clip_to_platforms, clip_id, targets)
    return {"success": res["any_success"], **res}


@router.post("/{clip_id}/send-discord")
async def send_clip_discord(clip_id: str) -> Dict[str, Any]:
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    msg_id = await discord_bot.send_clip_for_review(clip)
    if not msg_id:
        raise HTTPException(status_code=400, detail=discord_bot.get_status().get("last_error") or "Discord bot not connected or channel invalid.")
    return {"success": True, "discord_message_id": msg_id}


@router.delete("/{clip_id}")
def delete_clip(clip_id: str) -> Dict[str, Any]:
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    for p in (clip.get("raw_path"), clip.get("rendered_path"), clip.get("thumb_path")):
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except Exception:
                pass
    conn = get_db()
    with conn:
        conn.execute("DELETE FROM post_history WHERE clip_id = ?", (clip_id,))
        conn.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
    conn.close()
    return {"success": True}
