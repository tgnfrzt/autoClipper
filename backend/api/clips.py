import asyncio
import html
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from backend.core.config import get_reachable_base_url, settings
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


class PostHistoryUpdateRequest(BaseModel):
    post_url: Optional[str] = None
    views: Optional[int] = None
    likes: Optional[int] = None


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


@router.patch("/posts/{post_id}")
def update_post_history_record(post_id: str, req: PostHistoryUpdateRequest) -> Dict[str, Any]:
    conn = get_db()
    try:
        row = conn.execute("SELECT id FROM post_history WHERE id = ?", (post_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Published post record not found")
        with conn:
            if req.post_url is not None:
                conn.execute("UPDATE post_history SET post_url = ? WHERE id = ?", (req.post_url.strip(), post_id))
            if req.views is not None:
                conn.execute("UPDATE post_history SET views = ? WHERE id = ?", (max(0, int(req.views)), post_id))
            if req.likes is not None:
                conn.execute("UPDATE post_history SET likes = ? WHERE id = ?", (max(0, int(req.likes)), post_id))
    finally:
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
    fc_used = str(res.get("facecam_used") or "")
    actual_fc = "" if fc_used in ("", "none") else fc_used
    conn = get_db()
    with conn:
        conn.execute(
            "UPDATE clips SET rendered_path = ?, overlay_caption = ?, facecam_corner = ? WHERE id = ?",
            (res["output_path"], cap_txt, actual_fc, clip_id),
        )
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
    base = get_reachable_base_url()
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0" />
<meta name="color-scheme" content="dark" />
<title>{html.escape(title)}</title>
<style>
  body {{ margin:0; background:#090d16; color:#f8fafc; font-family:'Segoe UI',system-ui,sans-serif; display:grid; place-items:center; min-height:100vh; padding:1rem; box-sizing:border-box; }}
  .card {{ background:#111827; border:1px solid #26334d; border-top:4px solid {accent}; border-radius:14px; padding:1.75rem 2rem; max-width:540px; width:100%; box-sizing:border-box; }}
  h1 {{ margin:0 0 0.75rem; font-size:1.3rem; }}
  p, li {{ color:#cbd5e1; line-height:1.5; }}
  .btn-row {{ margin-top:1.25rem; display:flex; gap:0.6rem; flex-wrap:wrap; }}
  a.btn {{ display:inline-block; background:#6366f1; color:#fff; text-decoration:none; padding:0.65rem 1.1rem; border-radius:8px; font-weight:600; font-size:0.92rem; text-align:center; }}
  a.btn.sec {{ background:#1e293b; color:#94a3b8; border:1px solid #334155; }}
</style>
</head>
<body>
  <div class="card">
    <h1>{html.escape(title)}</h1>
    {body_html}
    <div class="btn-row">
      <a class="btn" href="{base}/">Open Dashboard</a>
      <a class="btn sec" href="javascript:window.close()">Close Window</a>
    </div>
  </div>
</body>
</html>"""
    )


@router.get("/{clip_id}/review", response_class=HTMLResponse)
async def review_clip_mobile_page(clip_id: str):
    clip = get_clip(clip_id)
    if not clip:
        return _action_html("Clip Not Found", f"<p>Clip <code>{html.escape(clip_id)}</code> was not found.</p>", "#ef4444")

    base = get_reachable_base_url()
    cap_text = clip.get("overlay_caption") or clip.get("ai_title") or clip["streamer_name"]
    streamer = clip.get("streamer_name") or "Streamer"
    platform = (clip.get("platform") or "twitch").upper()
    hype_score = clip.get("combined_score", 0)
    duration = clip.get("duration_sec", 0)
    transcript = clip.get("transcript") or ""
    status = clip.get("status") or "pending"

    conn = get_db()
    existing_posts = conn.execute(
        "SELECT platform, status, post_url FROM post_history WHERE clip_id = ? AND status = 'posted'",
        (clip_id,),
    ).fetchall()
    conn.close()

    already_posted = {p["platform"]: (p["post_url"] or "") for p in existing_posts}

    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
<meta name="theme-color" content="#090d16" />
<title>🎬 Review: {html.escape(cap_text)}</title>
<style>
  :root {{
    --bg: #090d16;
    --card: #111827;
    --border: #26334d;
    --text: #f8fafc;
    --muted: #94a3b8;
    --primary: #6366f1;
    --green: #10b981;
    --red: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    min-height: 100vh;
    padding: 12px;
    display: flex;
    justify-content: center;
  }}
  .app-container {{
    width: 100%;
    max-width: 480px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding-bottom: 24px;
  }}
  .top-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 4px;
  }}
  .brand-badge {{
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--muted);
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}
  .score-badge {{
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .video-wrapper {{
    position: relative;
    width: 100%;
    aspect-ratio: 9/16;
    max-height: 64vh;
    background: #000;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
  }}
  video {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    background: #000;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .clip-caption {{
    font-size: 1.1rem;
    font-weight: 700;
    line-height: 1.35;
    color: #fff;
  }}
  .clip-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.82rem;
    color: var(--muted);
  }}
  .meta-tag {{
    background: #1e293b;
    padding: 2px 8px;
    border-radius: 6px;
  }}
  .transcript-box {{
    background: #0d1322;
    border-left: 3px solid var(--primary);
    padding: 8px 12px;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #cbd5e1;
    font-style: italic;
  }}
  .actions-grid {{
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}
  .btn {{
    width: 100%;
    padding: 14px;
    border: none;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: all 0.15s ease;
    text-decoration: none;
  }}
  .btn:active {{
    transform: scale(0.98);
  }}
  .btn-all {{
    background: linear-gradient(135deg, #10b981, #059669);
    color: #fff;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
  }}
  .platform-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }}
  .btn-plat {{
    background: #1e293b;
    border: 1px solid #334155;
    color: var(--text);
    padding: 10px 4px;
    font-size: 0.85rem;
  }}
  .btn-plat:hover, .btn-plat:active {{
    background: #27354f;
  }}
  .btn-reject {{
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #f87171;
    padding: 11px;
    font-size: 0.9rem;
  }}
  .status-toast {{
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 0.88rem;
    display: none;
    flex-direction: column;
    gap: 6px;
  }}
  .nav-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    padding: 4px;
  }}
  .nav-footer a {{
    color: var(--muted);
    text-decoration: none;
  }}
  .spinner {{
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255,255,255,0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 0.8s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
<div class="app-container">
  <div class="top-header">
    <div class="brand-badge">⚡ GTA VI AutoClipper Mobile</div>
    <div class="score-badge">🔥 {hype_score}/100</div>
  </div>

  <div class="video-wrapper">
    <video id="player" controls playsinline autoplay loop muted poster="/api/clips/{clip_id}/thumb">
      <source src="/api/clips/{clip_id}/video?variant=auto" type="video/mp4" />
      Your phone browser does not support video playback.
    </video>
  </div>

  <div class="card">
    <div class="clip-caption">{html.escape(cap_text)}</div>
    <div class="clip-meta">
      <span class="meta-tag">👤 {html.escape(streamer)} ({platform})</span>
      <span class="meta-tag">⏱️ {duration}s</span>
      <span class="meta-tag">Status: <strong id="lblStatus">{status.upper()}</strong></span>
    </div>
    {f'<div class="transcript-box">"{html.escape(transcript)}"</div>' if transcript else ''}
  </div>

  <div id="statusToast" class="status-toast"></div>

  <div class="actions-grid" id="actionButtons">
    <button class="btn btn-all" id="btnPostAll" onclick="doAction('all')">
      🚀 Approve &amp; Post All Platforms
    </button>
    <div class="platform-row">
      <button class="btn btn-plat" id="btnYt" onclick="doAction('youtube_shorts')">
        ▶️ YouTube
      </button>
      <button class="btn btn-plat" id="btnTt" onclick="doAction('tiktok')">
        🎵 TikTok
      </button>
      <button class="btn btn-plat" id="btnIg" onclick="doAction('instagram')">
        📸 Instagram
      </button>
    </div>
    <button class="btn btn-reject" id="btnReject" onclick="doAction('reject')">
      🗑️ Reject &amp; Delete Clip
    </button>
  </div>

  <div class="nav-footer">
    <a href="{base}/">🌐 Open Full Desktop Dashboard</a>
    <a href="javascript:window.close()">✕ Close</a>
  </div>
</div>

<script>
const clipId = {json.dumps(clip_id)};
const alreadyPosted = {json.dumps(already_posted)};
const toast = document.getElementById('statusToast');
const statusLbl = document.getElementById('lblStatus');

function updatePostedBadges() {{
  for (const [p, url] of Object.entries(alreadyPosted)) {{
    let btn = null;
    if (p === 'youtube_shorts') btn = document.getElementById('btnYt');
    else if (p === 'tiktok') btn = document.getElementById('btnTt');
    else if (p === 'instagram') btn = document.getElementById('btnIg');
    if (btn) {{
      btn.style.borderColor = '#10b981';
      btn.style.color = '#34d399';
      btn.innerHTML = '✅ ' + btn.innerText.replace('✅ ', '');
    }}
  }}
}}
updatePostedBadges();

async function doAction(act) {{
  if (act === 'reject' && !confirm('Are you sure you want to reject and delete this clip?')) return;

  const btnAll = document.getElementById('btnPostAll');
  btnAll.disabled = true;
  toast.style.display = 'flex';

  if (act === 'reject') {{
    toast.innerHTML = '<span class="spinner"></span> Rejecting and cleaning up clip...';
    try {{
      const r = await fetch('/api/clips/' + encodeURIComponent(clipId) + '/quick-action?action=reject');
      toast.innerHTML = '🗑️ <strong>Clip rejected and deleted.</strong>';
      toast.style.borderColor = '#ef4444';
      statusLbl.textContent = 'REJECTED';
      document.getElementById('actionButtons').style.opacity = '0.4';
      document.getElementById('actionButtons').style.pointerEvents = 'none';
      setTimeout(() => {{ window.close(); }}, 2500);
    }} catch(e) {{
      toast.innerHTML = '❌ Error: ' + e.message;
    }}
    return;
  }}

  const targetList = act === 'all' ? ['youtube_shorts', 'tiktok', 'instagram'] : [act];
  toast.innerHTML = '<div style="display:flex;align-items:center;gap:8px;"><span class="spinner"></span> <strong>Publishing in background...</strong></div><ul id="toastList" style="margin-top:6px;padding-left:18px;">' +
    targetList.map(t => '<li>⏳ Uploading to ' + t.toUpperCase() + '...</li>').join('') + '</ul>';

  try {{
    fetch('/api/clips/' + encodeURIComponent(clipId) + '/quick-action?action=' + encodeURIComponent(act));
    statusLbl.textContent = 'PUBLISHING';

    let done = false;
    let pollCount = 0;
    while (!done && pollCount < 30) {{
      await new Promise(res => setTimeout(res, 2000));
      pollCount++;
      const res = await fetch('/api/clips/' + encodeURIComponent(clipId) + '/publish-status');
      if (!res.ok) continue;
      const data = await res.json();
      const history = data.history || [];
      const ul = document.getElementById('toastList');
      if (!ul) break;

      let allFinished = true;
      ul.innerHTML = targetList.map(t => {{
        const match = history.find(h => h.platform === t);
        if (!match) {{
          allFinished = false;
          return '<li>⏳ Uploading to ' + t.toUpperCase() + '...</li>';
        }}
        if (match.status === 'posted') {{
          alreadyPosted[t] = match.post_url || '';
          const lnk = match.post_url ? ' — <a href="' + match.post_url + '" target="_blank" style="color:#38bdf8;font-weight:700;">View</a>' : '';
          return '<li style="color:#34d399;">✅ <strong>' + t.toUpperCase() + '</strong>: Posted!' + lnk + '</li>';
        }}
        return '<li style="color:#f87171;">❌ <strong>' + t.toUpperCase() + '</strong>: ' + (match.error_message || 'Failed') + '</li>';
      }}).join('');

      updatePostedBadges();
      if (allFinished) {{
        statusLbl.textContent = 'POSTED';
        done = true;
      }}
    }}
  }} catch(e) {{
    toast.innerHTML = '❌ Network error: ' + e.message;
  }} finally {{
    btnAll.disabled = false;
  }}
}}
</script>
</body>
</html>"""
    )


@router.get("/{clip_id}/publish-status")
def get_clip_publish_status(clip_id: str) -> Dict[str, Any]:
    clip = get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT platform, status, post_url, error_message, published_at FROM post_history WHERE clip_id = ? ORDER BY published_at DESC LIMIT 12",
            (clip_id,),
        ).fetchall()
        history = [dict(r) for r in rows]
    finally:
        conn.close()
    return {
        "clip_id": clip_id,
        "status": clip["status"],
        "overlay_caption": clip.get("overlay_caption") or clip["streamer_name"],
        "history": history,
    }


async def _background_publish_and_notify(clip_id: str, platforms: List[str], discord_message_id: str) -> None:
    try:
        res = await asyncio.to_thread(publish_clip_to_platforms, clip_id, platforms)
        if discord_message_id:
            discord_lines: List[str] = []
            for item in res.get("results", []):
                plat = str(item.get("platform", "")).upper()
                if item.get("success"):
                    url = item.get("post_url", "")
                    discord_lines.append(f"✅ **{plat}**: Posted! {url}")
                else:
                    err = str(item.get("error", "Failed"))
                    discord_lines.append(f"❌ **{plat}**: {err}")
            if discord_lines:
                await discord_bot.update_webhook_message_status(discord_message_id, "\n".join(discord_lines))
    except Exception as exc:
        if discord_message_id:
            await discord_bot.update_webhook_message_status(discord_message_id, f"❌ **Publish error**: {exc}")


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

    started_iso = datetime.utcnow().isoformat()
    conn = get_db()
    with conn:
        existing_rows = conn.execute(
            "SELECT platform, status, post_url FROM post_history WHERE clip_id = ? AND status = 'posted' ORDER BY published_at DESC",
            (clip_id,),
        ).fetchall()
        already_posted = {}
        for er in existing_rows:
            if er["platform"] not in already_posted:
                already_posted[er["platform"]] = er["post_url"] or ""
        pending_platforms = [p for p in platforms if p not in already_posted]
        if pending_platforms:
            conn.execute("UPDATE clips SET status = 'accepted' WHERE id = ?", (clip_id,))
    conn.close()

    cap_label = html.escape(clip.get("overlay_caption") or clip["streamer_name"])

    if not pending_platforms:
        done_items = []
        for p in platforms:
            url = already_posted.get(p, "")
            lnk = f' — <a href="{html.escape(url)}" target="_blank" style="color:#38bdf8;">View Post</a>' if url else ""
            done_items.append(f"<li>✅ <strong>{html.escape(p.upper())}</strong>: Already Published!{lnk}</li>")
        return _action_html(
            "✅ Already Published",
            f"<p>Clip: <strong>{cap_label}</strong></p><ul>{''.join(done_items)}</ul>",
            "#10b981",
        )

    if clip.get("discord_message_id"):
        asyncio.create_task(
            discord_bot.update_webhook_message_status(
                clip["discord_message_id"],
                f"⏳ **Publishing `{clip_id}` to {', '.join(pending_platforms)}...**",
            )
        )

    asyncio.create_task(_background_publish_and_notify(clip_id, pending_platforms, clip.get("discord_message_id") or ""))

    plat_badges = ", ".join(p.upper() for p in pending_platforms)
    targets_json = json.dumps(platforms)
    initial_rows = "".join(
        (
            f'<li>✅ <strong>{html.escape(p.upper())}</strong>: Published!'
            + (f' — <a href="{html.escape(already_posted[p])}" target="_blank" style="color:#38bdf8;">View Post</a>' if already_posted[p] else "")
            + "</li>"
        )
        if p in already_posted
        else f'<li>⏳ <strong>{html.escape(p.upper())}</strong>: Uploading...</li>'
        for p in platforms
    )
    already_json = json.dumps(already_posted)
    body_html = f"""
    <p>Clip: <strong>{cap_label}</strong></p>
    <p style="font-size:0.92rem; color:#94a3b8;">Clip accepted &amp; publishing started in background for: <strong>{html.escape(plat_badges)}</strong></p>
    <ul id="pubList" style="padding-left:1.2rem;">
      {initial_rows}
    </ul>
    <script>
      (function() {{
        const targets = {targets_json};
        const sinceIso = {json.dumps(started_iso)};
        const prePosted = {already_json};
        const listEl = document.getElementById('pubList');
        async function poll() {{
          try {{
            const r = await fetch('/api/clips/{html.escape(clip_id)}/publish-status');
            if (!r.ok) return;
            const d = await r.json();
            const recent = (d.history || []).filter(h => (h.published_at || '') >= sinceIso);
            const seen = {{}};
            for (const h of recent) {{
              if (!seen[h.platform]) seen[h.platform] = h;
            }}
            let doneCount = 0;
            listEl.innerHTML = targets.map(t => {{
              if (prePosted[t] !== undefined) {{
                doneCount++;
                const lnk = prePosted[t] ? ' — <a href="' + prePosted[t] + '" target="_blank" style="color:#38bdf8;">View Post</a>' : '';
                return '<li>✅ <strong>' + t.toUpperCase() + '</strong>: Published!' + lnk + '</li>';
              }}
              const row = seen[t];
              if (!row) return '<li>⏳ <strong>' + t.toUpperCase() + '</strong>: Uploading...</li>';
              doneCount++;
              if (row.status === 'posted') {{
                const lnk = row.post_url ? ' — <a href="' + row.post_url + '" target="_blank" style="color:#38bdf8;">View Post</a>' : '';
                return '<li>✅ <strong>' + t.toUpperCase() + '</strong>: Published!' + lnk + '</li>';
              }}
              return '<li>❌ <strong>' + t.toUpperCase() + '</strong>: ' + (row.error_message || 'Failed') + '</li>';
            }}).join('');
            if (doneCount < targets.length) {{
              setTimeout(poll, 1800);
            }}
          }} catch (e) {{
            setTimeout(poll, 2500);
          }}
        }}
        setTimeout(poll, 1500);
      }})();
    </script>
    """
    return _action_html("🚀 Accepted & Publishing Live...", body_html, "#10b981")


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
