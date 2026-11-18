import asyncio
import html
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from backend.core.config import settings
from backend.core.database import get_clip, get_db, list_clips
from backend.modules.discord.bot import discord_bot
from backend.modules.processors.video_processor import render_clip_for_social
from backend.modules.publishers.dispatcher import get_configured_platforms, publish_clip_to_platforms

router = APIRouter(prefix="/api/clips", tags=["clips"])


class ClipUpdateRequest(BaseModel):
    overlay_caption: Optional[str] = None
    ai_title: Optional[str] = None
    ai_caption: Optional[str] = None


class ClipRenderRequest(BaseModel):
    format_mode: Optional[str] = None
    facecam_position: Optional[str] = None
    caption_text: Optional[str] = None


class ClipPublishRequest(BaseModel):
    platforms: List[str] = ["all"]


class ClipStatusRequest(BaseModel):
    status: str = "accepted"


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
        if req.overlay_caption is not None:
            conn.execute("UPDATE clips SET overlay_caption = ? WHERE id = ?", (req.overlay_caption, clip_id))
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
    cap_txt = req.caption_text if req.caption_text is not None else (clip.get("overlay_caption") or clip.get("ai_title") or "")
    res = await asyncio.to_thread(
        render_clip_for_social,
        clip["raw_path"],
        clip_id,
        req.format_mode,
        req.facecam_position,
        None,
        cap_txt,
        clip.get("streamer_name") or "",
    )
    conn = get_db()
    with conn:
        conn.execute("UPDATE clips SET rendered_path = ?, overlay_caption = ? WHERE id = ?", (res["output_path"], cap_txt, clip_id))
    conn.close()
    return {"success": True, **res}


@router.post("/{clip_id}/accept")
async def accept_clip(clip_id: str, req: ClipStatusRequest = ClipStatusRequest()) -> Dict[str, Any]:
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    new_status = req.status.strip().lower()
    if new_status not in ("accepted", "pending", "posted", "rejected"):
        raise HTTPException(status_code=400, detail="Invalid clip status")
    conn = get_db()
    with conn:
        conn.execute("UPDATE clips SET status = ? WHERE id = ?", (new_status, clip_id))
    conn.close()
    if clip.get("discord_message_id") and new_status == "accepted":
        await discord_bot.update_webhook_message_status(
            clip["discord_message_id"],
            f"✅ **Clip `{clip_id}` accepted and moved to Accepted Queue.**",
        )
    return {"success": True, "status": new_status}


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
    if clip.get("discord_message_id"):
        summary_lines = [
            f"✅ **{it.get('platform', '').upper()}**: Posted! {it.get('post_url', '')}"
            if it.get("success")
            else f"❌ **{it.get('platform', '').upper()}**: {it.get('error', 'Failed')}"
            for it in res["results"]
        ]
        await discord_bot.update_webhook_message_status(clip["discord_message_id"], "\n".join(summary_lines))
    return {"success": res["any_success"], **res}


def _action_html(title: str, body_html: str, accent: str = "#10b981") -> HTMLResponse:
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="color-scheme" content="dark" />
<title>{html.escape(title)}</title>
<style>
  body {{ margin:0; background:#090d16; color:#f8fafc; font-family:'Segoe UI',system-ui,sans-serif; display:grid; place-items:center; min-height:100vh; padding:1rem; }}
  .card {{ background:#111827; border:1px solid #26334d; border-top:4px solid {accent}; border-radius:14px; padding:1.75rem 2rem; max-width:540px; width:100%; }}
  h1 {{ margin:0 0 0.75rem; font-size:1.3rem; }}
  p, li {{ color:#cbd5e1; line-height:1.5; }}
  a.btn {{ display:inline-block; margin-top:1rem; background:#6366f1; color:#fff; text-decoration:none; padding:0.55rem 1rem; border-radius:8px; font-weight:600; }}
</style>
</head>
<body>
  <div class="card">
    <h1>{html.escape(title)}</h1>
    {body_html}
    <a class="btn" href="http://127.0.0.1:{settings.PORT}/">Open AutoClipper Dashboard</a>
  </div>
</body>
</html>"""
    )


@router.get("/{clip_id}/quick-action", response_class=HTMLResponse)
async def discord_quick_action(clip_id: str, action: str = "all"):
    clip = get_clip(clip_id)
    if not clip:
        return _action_html("Clip Not Found", f"<p>Clip <code>{html.escape(clip_id)}</code> no longer exists.</p>", "#ef4444")

    act = action.lower().strip()
    if act == "reject":
        conn = get_db()
        with conn:
            conn.execute("UPDATE clips SET status = 'rejected' WHERE id = ?", (clip_id,))
        conn.close()
        if settings.AUTO_DELETE_ON_REJECT:
            for p in (clip.get("raw_path"), clip.get("rendered_path"), clip.get("thumb_path")):
                if p and os.path.isfile(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        if clip.get("discord_message_id"):
            await discord_bot.update_webhook_message_status(
                clip["discord_message_id"],
                f"🗑️ **Clip `{clip_id}` rejected and deleted.**",
            )
        return _action_html("🗑️ Clip Rejected & Deleted", f"<p>Clip <code>{html.escape(clip_id)}</code> was marked rejected and cleaned up.</p>", "#ef4444")

    if act == "all":
        platforms = get_configured_platforms()
        if not platforms:
            return _action_html("⚠️ No Platforms Configured", "<p>Configure YouTube, TikTok, or Instagram in the AutoClipper Configure tab first.</p>", "#f59e0b")
    elif act in ("youtube", "youtube_shorts", "tiktok", "instagram"):
        platforms = ["youtube_shorts" if act == "youtube" else act]
    else:
        return _action_html("Invalid Action", f"<p>Unknown action: {html.escape(action)}</p>", "#ef4444")

    conn = get_db()
    with conn:
        conn.execute("UPDATE clips SET status = 'accepted' WHERE id = ?", (clip_id,))
    conn.close()

    if clip.get("discord_message_id"):
        await discord_bot.update_webhook_message_status(
            clip["discord_message_id"],
            f"⏳ **Publishing `{clip_id}` to {', '.join(platforms)}...**",
        )

    res = await asyncio.to_thread(publish_clip_to_platforms, clip_id, platforms)
    discord_lines: List[str] = []
    html_items: List[str] = []
    for item in res["results"]:
        plat = str(item.get("platform", "")).upper()
        if item.get("success"):
            url = item.get("post_url", "")
            discord_lines.append(f"✅ **{plat}**: Posted! {url}")
            link_html = f' — <a href="{html.escape(url)}" target="_blank" style="color:#38bdf8;">View Post</a>' if url else ""
            html_items.append(f"<li>✅ <strong>{html.escape(plat)}</strong>: Published!{link_html}</li>")
        else:
            err = str(item.get("error", "Failed"))
            discord_lines.append(f"❌ **{plat}**: {err}")
            html_items.append(f"<li>❌ <strong>{html.escape(plat)}</strong>: {html.escape(err)}</li>")

    if clip.get("discord_message_id"):
        await discord_bot.update_webhook_message_status(clip["discord_message_id"], "\n".join(discord_lines))

    return _action_html(
        "🚀 Publishing Complete" if res["any_success"] else "⚠️ Publishing Finished With Errors",
        f"<p>Clip: <strong>{html.escape(clip.get('overlay_caption') or clip['streamer_name'])}</strong></p><ul>{''.join(html_items)}</ul>",
        "#10b981" if res["any_success"] else "#f59e0b",
    )


@router.post("/{clip_id}/send-discord")
async def send_clip_discord(clip_id: str) -> Dict[str, Any]:
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    msg_id = await discord_bot.send_clip_for_review(clip)
    if not msg_id:
        raise HTTPException(status_code=400, detail=discord_bot.get_status().get("last_error") or "Configure a Discord Webhook URL or Bot Token in Settings.")
    return {"success": True, "discord_message_id": msg_id}


@router.delete("/{clip_id}")
async def delete_clip(clip_id: str) -> Dict[str, Any]:
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
    if clip.get("discord_message_id"):
        await discord_bot.update_webhook_message_status(
            clip["discord_message_id"],
            f"🗑️ **Clip `{clip_id}` deleted from dashboard.**",
        )
    return {"success": True}
