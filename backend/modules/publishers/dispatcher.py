import os
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

from backend.core.config import (
    is_instagram_configured,
    is_tiktok_configured,
    is_youtube_configured,
    settings,
)
from backend.core.database import get_clip, get_db
from backend.modules.processors.video_processor import render_clip_for_social
from backend.modules.publishers.instagram import publish_instagram_graph_api
from backend.modules.publishers.tiktok import publish_tiktok
from backend.modules.publishers.youtube import publish_youtube_shorts

_publish_lock = threading.Lock()
_active_publishes: set[str] = set()


def get_configured_platforms() -> List[str]:
    platforms: List[str] = []
    if is_youtube_configured():
        platforms.append("youtube_shorts")
    if is_tiktok_configured():
        platforms.append("tiktok")
    if is_instagram_configured():
        platforms.append("instagram")
    return platforms


def check_autopost_rate_limit() -> Dict[str, Any]:
    """
    Three-layer Anti-Ban / Anti-Flag Safety Guard checking SQLite post_history:
      1. Minimum cooldown minutes between published clips (AUTO_POST_MIN_INTERVAL_MIN)
      2. Rolling 1-hour distinct-clip cap (AUTO_POST_MAX_PER_HOUR)
      3. Rolling 24-hour distinct-clip cap (AUTO_POST_MAX_PER_DAY)
    """
    now_utc = datetime.utcnow()
    one_hour_ago = (now_utc - timedelta(hours=1)).isoformat()
    day_ago = (now_utc - timedelta(hours=24)).isoformat()

    min_gap = max(1, int(settings.AUTO_POST_MIN_INTERVAL_MIN))
    max_hr = max(1, int(settings.AUTO_POST_MAX_PER_HOUR))
    max_day = max(1, int(settings.AUTO_POST_MAX_PER_DAY))

    conn = get_db()
    try:
        last_row = conn.execute(
            "SELECT MAX(published_at) AS ts FROM post_history WHERE status = 'posted'"
        ).fetchone()
        clips_hr = conn.execute(
            "SELECT COUNT(DISTINCT clip_id) FROM post_history WHERE status = 'posted' AND published_at >= ?",
            (one_hour_ago,),
        ).fetchone()[0]
        clips_day = conn.execute(
            "SELECT COUNT(DISTINCT clip_id) FROM post_history WHERE status = 'posted' AND published_at >= ?",
            (day_ago,),
        ).fetchone()[0]
    finally:
        conn.close()

    mins_since_last: float = 9999.0
    cooldown_left = 0.0
    last_ts = last_row["ts"] if last_row else None
    if last_ts:
        try:
            last_dt = datetime.fromisoformat(str(last_ts).replace("Z", ""))
            mins_since_last = max(0.0, (now_utc - last_dt).total_seconds() / 60.0)
            if mins_since_last < min_gap:
                cooldown_left = round(min_gap - mins_since_last, 1)
        except Exception:
            pass

    can_post = True
    reason = f"Ready (≥{settings.AUTO_POST_THRESHOLD} score)"
    if int(clips_day or 0) >= max_day:
        can_post = False
        reason = f"Daily cap reached ({clips_day}/{max_day} in 24h)"
    elif int(clips_hr or 0) >= max_hr:
        can_post = False
        reason = f"Hourly cap reached ({clips_hr}/{max_hr} in 1h)"
    elif cooldown_left > 0.0:
        can_post = False
        reason = f"Cooldown ({cooldown_left:.0f}m left of {min_gap}m gap)"

    return {
        "enabled": settings.AUTO_POST_ENABLED,
        "threshold": settings.AUTO_POST_THRESHOLD,
        "can_post": can_post,
        "reason": reason,
        "clips_last_hour": int(clips_hr or 0),
        "max_per_hour": max_hr,
        "clips_last_day": int(clips_day or 0),
        "max_per_day": max_day,
        "mins_since_last": round(mins_since_last, 1) if last_ts else None,
        "cooldown_remaining_min": cooldown_left,
        "min_interval_min": min_gap,
    }


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
        lock_key = f"{clip_id}:{p_norm}"
        with _publish_lock:
            if lock_key in _active_publishes:
                results.append({"success": False, "platform": p_norm, "error": "Publish already in progress (deduplicated)."})
                continue
            _active_publishes.add(lock_key)

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
        finally:
            with _publish_lock:
                _active_publishes.discard(lock_key)

        ok = bool(receipt.get("success"))
        if ok:
            any_success = True

        # Do not pollute history with redundant lock-skip notices
        if receipt.get("error") == "Publish already in progress (deduplicated).":
            results.append(receipt)
            continue

        ext_id = str(receipt.get("post_id") or receipt.get("video_id") or receipt.get("publish_id") or "")
        conn = get_db()
        with conn:
            conn.execute(
                """
                INSERT INTO post_history (id, clip_id, platform, status, post_url, error_message, external_post_id, published_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    clip_id,
                    p_norm,
                    "posted" if ok else "failed",
                    receipt.get("post_url", ""),
                    receipt.get("error", ""),
                    ext_id,
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
