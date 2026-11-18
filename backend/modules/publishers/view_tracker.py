import json
import os
import re
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from backend.core.config import settings
from backend.core.database import get_db
from backend.modules.publishers.tiktok import resolve_tiktok_username
from backend.modules.publishers.youtube import get_youtube_service

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _extract_yt_id(url: str) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"(?:shorts/|watch\?v=|youtu\.be/)([A-Za-z0-9_-]{8,15})", url)
    return m.group(1) if m else None


def _extract_ig_media_id(url: str) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"/reel/(\d{10,25})/?", url)
    return m.group(1) if m else None


def _fetch_yt_stats_ytdlp(video_id: str) -> Optional[Tuple[int, int, int]]:
    try:
        res = subprocess.run(
            [settings.YTDLP_PATH, "-J", "--no-warnings", f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
            creationflags=CREATE_NO_WINDOW,
        )
        if res.returncode == 0 and res.stdout.strip():
            d = json.loads(res.stdout)
            return (
                int(d.get("view_count") or 0),
                int(d.get("like_count") or 0),
                int(d.get("comment_count") or 0),
            )
    except Exception:
        pass
    return None


def _fetch_ig_reel_metrics(media_id: str, token: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        r = requests.get(
            f"{settings.GRAPH_API_URL}/{media_id}",
            params={"fields": "id,permalink,like_count,comments_count", "access_token": token},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("permalink"):
                out["permalink"] = data["permalink"]
            if "like_count" in data:
                out["likes"] = int(data.get("like_count") or 0)
            if "comments_count" in data:
                out["comments"] = int(data.get("comments_count") or 0)
    except Exception:
        pass

    for metric_set in ("views", "plays"):
        try:
            ri = requests.get(
                f"{settings.GRAPH_API_URL}/{media_id}/insights",
                params={"metric": metric_set, "access_token": token},
                timeout=10,
            )
            if ri.status_code == 200:
                items = ri.json().get("data") or []
                for it in items:
                    vals = it.get("values") or []
                    if vals and "value" in vals[0]:
                        out["views"] = int(vals[0]["value"] or 0)
                        return out
        except Exception:
            continue
    return out


def _fetch_tiktok_video_stats(url: str) -> Optional[Tuple[int, int, int]]:
    if "/video/" not in url:
        return None
    try:
        res = subprocess.run(
            [settings.YTDLP_PATH, "-J", "--no-warnings", url],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
            creationflags=CREATE_NO_WINDOW,
        )
        if res.returncode == 0 and res.stdout.strip():
            d = json.loads(res.stdout)
            return (
                int(d.get("view_count") or 0),
                int(d.get("like_count") or 0),
                int(d.get("comment_count") or 0),
            )
    except Exception:
        pass
    return None


def sync_all_clip_views() -> Dict[str, Any]:
    """
    Synchronizes real view/like/comment counts and upgraded permalinks for all
    posted clips in post_history across YouTube Shorts, Instagram Reels, and TikTok.
    """
    now_iso = datetime.utcnow().isoformat()
    conn = get_db()
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT id, clip_id, platform, post_url, views, likes, comments, external_post_id FROM post_history WHERE status = 'posted'"
        ).fetchall()
    ]
    conn.close()

    if not rows:
        return {"synced_posts": 0, "total_views": 0, "errors": []}

    updates: List[Tuple[str, int, int, int, str, str, str]] = []
    errors: List[str] = []

    # 1. Batch YouTube Shorts metrics via YouTube Data API v3 (with yt-dlp fallback)
    yt_rows: List[Tuple[dict, str]] = []
    for r in rows:
        if "youtube" in (r.get("platform") or "").lower():
            vid = _extract_yt_id(r.get("post_url") or "")
            if vid:
                yt_rows.append((r, vid))

    yt_stats: Dict[str, Tuple[int, int, int]] = {}
    if yt_rows:
        unique_ids = list({vid for _, vid in yt_rows})
        if os.path.isfile(settings.YOUTUBE_TOKEN_FILE):
            try:
                yt = get_youtube_service()
                for i in range(0, len(unique_ids), 50):
                    chunk = unique_ids[i : i + 50]
                    resp = yt.videos().list(part="statistics", id=",".join(chunk)).execute()
                    for item in resp.get("items", []):
                        st = item.get("statistics", {})
                        yt_stats[item["id"]] = (
                            int(st.get("viewCount") or 0),
                            int(st.get("likeCount") or 0),
                            int(st.get("commentCount") or 0),
                        )
            except Exception as exc:
                errors.append(f"YouTube stats sync note: {exc}")

        for vid in unique_ids:
            if vid not in yt_stats:
                fb = _fetch_yt_stats_ytdlp(vid)
                if fb:
                    yt_stats[vid] = fb

    for r, vid in yt_rows:
        if vid in yt_stats:
            v, lk, cm = yt_stats[vid]
            updates.append((r["post_url"], v, lk, cm, vid, now_iso, r["id"]))

    # 2. Instagram Reels Graph API Insights & Shortcode Permalink Upgrade
    ig_token = (settings.PAGE_ACCESS_TOKEN or "").strip()
    ig_b_id = (settings.INSTAGRAM_BUSINESS_ID or "").strip()
    permalink_to_id: Dict[str, str] = {}
    if ig_token and ig_b_id:
        try:
            mr = requests.get(
                f"{settings.GRAPH_API_URL}/{ig_b_id}/media",
                params={"fields": "id,permalink", "limit": 50, "access_token": ig_token},
                timeout=12,
            )
            if mr.status_code == 200:
                for it in mr.json().get("data") or []:
                    if it.get("permalink") and it.get("id"):
                        permalink_to_id[it["permalink"].rstrip("/")] = str(it["id"])
        except Exception:
            pass

    ig_id_cache: Dict[str, Dict[str, Any]] = {}
    if ig_token:
        for r in rows:
            if "instagram" not in (r.get("platform") or "").lower():
                continue
            url = (r.get("post_url") or "").strip()
            media_id = (
                (r.get("external_post_id") or "").strip()
                or _extract_ig_media_id(url)
                or permalink_to_id.get(url.rstrip("/"))
            )
            if not media_id:
                continue
            if media_id not in ig_id_cache:
                ig_id_cache[media_id] = _fetch_ig_reel_metrics(media_id, ig_token)
            m = ig_id_cache[media_id]
            if m:
                new_url = m.get("permalink") or url
                new_views = int(m.get("views", r.get("views") or 0))
                new_likes = int(m.get("likes", r.get("likes") or 0))
                new_comments = int(m.get("comments", r.get("comments") or 0))
                updates.append((new_url, new_views, new_likes, new_comments, media_id, now_iso, r["id"]))

    # 3. TikTok Profile Link Upgrade + Direct Video Stats
    tt_rows = [r for r in rows if "tiktok" in (r.get("platform") or "").lower()]
    if tt_rows:
        tt_user = resolve_tiktok_username()
        for r in tt_rows:
            url = (r.get("post_url") or "").strip()
            new_url = url
            if tt_user and (not url or url.rstrip("/") in ("https://www.tiktok.com", "https://tiktok.com")):
                new_url = f"https://www.tiktok.com/@{tt_user}"
            new_views = int(r.get("views") or 0)
            new_likes = int(r.get("likes") or 0)
            new_comments = int(r.get("comments") or 0)
            if "/video/" in new_url:
                st = _fetch_tiktok_video_stats(new_url)
                if st:
                    new_views, new_likes, new_comments = st
            if new_url != url or "/video/" in new_url:
                updates.append((new_url, new_views, new_likes, new_comments, r.get("external_post_id") or "", now_iso, r["id"]))

    if updates:
        conn = get_db()
        with conn:
            conn.executemany(
                """
                UPDATE post_history
                SET post_url = ?, views = ?, likes = ?, comments = ?, external_post_id = ?, last_synced_at = ?
                WHERE id = ?
                """,
                updates,
            )
        conn.close()

    conn = get_db()
    total_views = conn.execute("SELECT COALESCE(SUM(views), 0) FROM post_history WHERE status = 'posted'").fetchone()[0]
    conn.close()

    return {
        "synced_posts": len(updates),
        "total_views": int(total_views or 0),
        "errors": errors,
    }
