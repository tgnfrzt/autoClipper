import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional

import requests

from backend.core.config import settings


def publish_tiktok_via_session(
    video_path: str,
    caption: str = "",
    session_id: Optional[str] = None,
    headless: bool = True,
) -> Dict[str, Any]:
    sess = session_id or settings.TIKTOK_SESSION_ID
    cookies_path = settings.TIKTOK_COOKIES_PATH if os.path.isfile(settings.TIKTOK_COOKIES_PATH) else None

    if not sess and not cookies_path:
        raise ValueError("No TikTok session_id or tiktok_cookies.txt configured.")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    final_caption = (caption or "").strip() or settings.DEFAULT_TIKTOK_CAPTION
    runner_path = os.path.join(os.path.dirname(__file__), "tiktok_runner.py")
    last_err = None

    for browser in ("edge", "chrome", "chromium"):
        try:
            cmd = [
                sys.executable,
                runner_path,
                "--video-path", video_path,
                "--caption", final_caption,
                "--browser", browser,
            ]
            if sess:
                cmd.extend(["--session-id", sess])
            if cookies_path:
                cmd.extend(["--cookies-path", cookies_path])
            if not headless:
                cmd.append("--no-headless")

            sub_env = os.environ.copy()
            sub_env["PYTHONIOENCODING"] = "utf-8"
            sub_env["PYTHONUTF8"] = "1"

            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=sub_env,
                timeout=360,
            )

            lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
            parsed = None
            for ln in reversed(lines):
                if ln.startswith("{") and ln.endswith("}"):
                    try:
                        parsed = json.loads(ln)
                        break
                    except Exception:
                        pass

            if res.returncode == 0 and parsed and parsed.get("success"):
                return {
                    "success": True,
                    "platform": "tiktok",
                    "method": "session_uploader",
                    "browser": browser,
                    "post_url": "https://www.tiktok.com",
                }
            err_msg = (parsed.get("error") if parsed else None) or res.stderr.strip() or f"Exit code {res.returncode}"
            last_err = RuntimeError(err_msg)
        except Exception as exc:
            last_err = exc

    raise RuntimeError(f"TikTok session upload failed: {last_err}")


def publish_tiktok_via_api(
    video_path: str,
    caption: str = "",
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    token = access_token or settings.TIKTOK_ACCESS_TOKEN
    if not token:
        raise ValueError("TikTok API Access Token not configured.")

    init_url = "https://open.tiktokapis.com/v2/post/publish/video/init/"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    final_caption = (caption or "").strip() or settings.DEFAULT_TIKTOK_CAPTION
    file_size = os.path.getsize(video_path)
    payload = {
        "post_info": {
            "title": final_caption[:150],
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "disable_duet": False,
            "disable_stitch": False,
            "disable_comment": False,
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": file_size,
            "chunk_size": file_size,
            "total_chunk_count": 1,
        },
    }

    res = requests.post(init_url, json=payload, headers=headers, timeout=30)
    data = res.json()
    if data.get("error", {}).get("code") != "ok":
        raise RuntimeError(f"TikTok API init failed: {data}")

    upload_url = data["data"]["upload_url"]
    publish_id = data["data"]["publish_id"]

    with open(video_path, "rb") as f:
        put_headers = {
            "Content-Type": "video/mp4",
            "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
        }
        requests.put(upload_url, data=f, headers=put_headers, timeout=180)

    return {
        "success": True,
        "platform": "tiktok",
        "method": "official_api",
        "publish_id": publish_id,
        "post_url": "https://www.tiktok.com",
    }


def publish_tiktok(
    video_path: str,
    caption: str = "",
    access_token: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    sess = session_id or settings.TIKTOK_SESSION_ID
    has_cookies = os.path.isfile(settings.TIKTOK_COOKIES_PATH)
    token = access_token or settings.TIKTOK_ACCESS_TOKEN

    if sess or has_cookies:
        return publish_tiktok_via_session(video_path=video_path, caption=caption, session_id=sess)
    if token:
        return publish_tiktok_via_api(video_path=video_path, caption=caption, access_token=token)
    raise ValueError("No TikTok credentials configured (Session ID, cookies file, or API token).")
