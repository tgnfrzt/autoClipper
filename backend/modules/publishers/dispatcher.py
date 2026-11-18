import os
import uuid
from datetime import datetime
from typing import Any, Dict, List

from backend.core.config import (
    is_instagram_configured,
    is_tiktok_configured,
    is_youtube_configured,
)
from backend.core.database import get_clip, get_db
from backend.modules.processors.video_processor import render_clip_for_social
from backend.modules.publishers.instagram import publish_instagram_graph_api
from backend.modules.publishers.tiktok import publish_tiktok
from backend.modules.publishers.youtube import publish_youtube_shorts


def get_configured_platforms() -> List[str]:
    platforms: List[str] = []
    if is_youtube_configured():
        platforms.append("youtube_shorts")
    if is_tiktok_configured():
        platforms.append("tiktok")
    if is_instagram_configured():
        platforms.append("instagram")
    return platforms


def publish_clip_to_platforms(clip_id: str, platforms: List[str]) -> Dict[str, Any]:
    """
    Ensures the clip is rendered into the configured format (Smart 9:16 Gaming Split by default)
    and publishes it to each requested platform, recording results in SQLite post_history.
    """
    clip = get_clip(clip_id)
    if not clip:
        raise ValueError(f"Clip not found: {clip_id}")

    # Move clip out of Pending Library immediately once accepted for publishing
    conn = get_db()
    with conn:
        conn.execute("UPDATE clips SET status = 'accepted' WHERE id = ?", (clip_id,))
    conn.close()

    video_to_post = clip.get("rendered_path") or ""
    if not video_to_post or not os.path.isfile(video_to_post):
        render_res = render_clip_for_social(
            clip["raw_path"],
            clip_id,
            caption_text=clip.get("overlay_caption") or clip.get("ai_title") or "",
            streamer_name=clip.get("streamer_name") or "",
        )
        video_to_post = render_res["output_path"]
        conn = get_db()
        with conn:
            conn.execute("UPDATE clips SET rendered_path = ? WHERE id = ?", (video_to_post, clip_id))
        conn.close()

    title = clip.get("ai_title") or f"{clip['streamer_name']} GTA VI Moment"
    caption = clip.get("ai_caption") or title

    results: List[Dict[str, Any]] = []
    any_success = False

    for plat in platforms:
        p_norm = plat.lower().strip()
        try:
            if p_norm in ("youtube", "youtube_shorts", "yt"):
                receipt = publish_youtube_shorts(
                    video_path=video_to_post,
                    title=title,
                    description=caption,
                    is_shorts=True,
                )
            elif p_norm in ("tiktok", "tt"):
                receipt = publish_tiktok(video_path=video_to_post, caption=caption)
            elif p_norm in ("instagram", "instagram_reels", "ig"):
                receipt = publish_instagram_graph_api(video_path=video_to_post, caption=caption)
            else:
                receipt = {"success": False, "platform": p_norm, "error": f"Unknown platform: {plat}"}
        except Exception as exc:
            receipt = {"success": False, "platform": p_norm, "error": str(exc)}

        ok = bool(receipt.get("success"))
        if ok:
            any_success = True

        conn = get_db()
        with conn:
            conn.execute(
                """
                INSERT INTO post_history (id, clip_id, platform, status, post_url, error_message, published_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    clip_id,
                    p_norm,
                    "posted" if ok else "failed",
                    receipt.get("post_url", ""),
                    receipt.get("error", ""),
                    datetime.utcnow().isoformat(),
                ),
            )
        conn.close()
        results.append(receipt)

    if any_success:
        conn = get_db()
        with conn:
            conn.execute("UPDATE clips SET status = 'posted' WHERE id = ?", (clip_id,))
        conn.close()

    return {
        "clip_id": clip_id,
        "video_path": video_to_post,
        "any_success": any_success,
        "results": results,
    }
