import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

from backend.core.config import settings
from backend.core.database import get_db


def compute_combined_score(chat_score: float, visual_score: float) -> float:
    """
    Combines 80% Chat Hype + 20% Visual Evaluation (configurable via settings).
    Normalizes weights in case user enters e.g. 80 & 20 or 0.8 & 0.2.
    """
    cw = float(settings.CHAT_WEIGHT)
    vw = float(settings.VISUAL_WEIGHT)
    if (cw + vw) > 0:
        total = cw + vw
        cw /= total
        vw /= total
    else:
        cw, vw = 0.80, 0.20

    score = (cw * float(chat_score)) + (vw * float(visual_score))
    return round(max(0.0, min(100.0, score)), 1)


def save_clip_record(
    stream_id: str,
    streamer_name: str,
    platform: str,
    stream_url: str,
    raw_path: str,
    rendered_path: str,
    thumb_path: str,
    duration_sec: float,
    chat_score: float,
    visual_score: float,
    facecam_score: float,
    combined_score: float,
    chat_mps: float,
    chat_keywords: List[str],
    facecam_corner: str,
    transcript: str,
    overlay_caption: str,
    ai_title: str,
    ai_summary: str,
    ai_caption: str,
) -> Dict[str, Any]:
    clip_id = f"clip_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    now = datetime.utcnow().isoformat()
    kw_json = json.dumps(chat_keywords[:8], ensure_ascii=False)

    conn = get_db()
    with conn:
        conn.execute(
            """
            INSERT INTO clips (
                id, stream_id, streamer_name, platform, stream_url,
                raw_path, rendered_path, thumb_path, duration_sec,
                chat_score, visual_score, facecam_score, combined_score,
                chat_mps, chat_keywords, facecam_corner, transcript, overlay_caption,
                ai_title, ai_summary, ai_caption, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                clip_id,
                stream_id,
                streamer_name,
                platform,
                stream_url,
                raw_path,
                rendered_path,
                thumb_path,
                round(duration_sec, 1),
                round(chat_score, 1),
                round(visual_score, 1),
                round(facecam_score, 1),
                round(combined_score, 1),
                round(chat_mps, 2),
                kw_json,
                facecam_corner,
                transcript,
                overlay_caption,
                ai_title,
                ai_summary,
                ai_caption,
                now,
            ),
        )
    conn.close()

    return {
        "id": clip_id,
        "stream_id": stream_id,
        "streamer_name": streamer_name,
        "platform": platform,
        "stream_url": stream_url,
        "raw_path": raw_path,
        "rendered_path": rendered_path,
        "thumb_path": thumb_path,
        "duration_sec": round(duration_sec, 1),
        "chat_score": round(chat_score, 1),
        "visual_score": round(visual_score, 1),
        "facecam_score": round(facecam_score, 1),
        "combined_score": round(combined_score, 1),
        "chat_mps": round(chat_mps, 2),
        "chat_keywords": chat_keywords[:8],
        "facecam_corner": facecam_corner,
        "transcript": transcript,
        "overlay_caption": overlay_caption,
        "ai_title": ai_title,
        "ai_summary": ai_summary,
        "ai_caption": ai_caption,
        "status": "pending",
        "created_at": now,
    }
