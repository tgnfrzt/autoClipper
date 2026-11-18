import os
import time
from typing import Any, Dict, Optional

import requests

from backend.core.config import settings


def upload_to_temp_hosting(video_path: str, hosting_url: Optional[str] = None) -> Optional[str]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb > 95.0:
        raise RuntimeError(f"Video is {file_size_mb:.1f} MB (exceeds 95 MB Instagram Reels relay limit).")

    errors = []

    # 1. Try user's configured REMOTE_HOSTING_URL relay first (if set)
    if hosting_url and hosting_url.strip().startswith("http"):
        try:
            with open(video_path, "rb") as f:
                files = {"file": (os.path.basename(video_path), f, "video/mp4")}
                res = requests.post(f"{hosting_url.rstrip('/')}/upload_video", files=files, timeout=90)
            txt = res.text.strip()
            if res.status_code == 200 and "<html" not in txt.lower() and "coming soon" not in txt.lower():
                if txt.startswith("{"):
                    payload = res.json()
                    url = payload.get("video_url") or payload.get("url")
                    if url and url.startswith("http"):
                        return url
                elif txt.startswith("http"):
                    return txt
            errors.append(f"Primary relay HTTP {res.status_code}")
        except Exception as exc:
            errors.append(f"Primary relay error: {exc}")

    # 2. Automatic Ephemeral Public Relay Fallbacks (needed when PythonAnywhere is inactive/404)
    fallback_endpoints = [
        ("litterbox", "https://litterbox.catbox.moe/resources/internals/api.php", {"reqtype": "fileupload", "time": "1h"}, "fileToUpload"),
        ("catbox", "https://catbox.moe/user/api.php", {"reqtype": "fileupload"}, "fileToUpload"),
    ]
    for label, endpoint, form_data, file_field in fallback_endpoints:
        try:
            with open(video_path, "rb") as f:
                res = requests.post(
                    endpoint,
                    data=form_data,
                    files={file_field: (os.path.basename(video_path), f, "video/mp4")},
                    timeout=120,
                )
            txt = res.text.strip()
            if res.status_code == 200 and txt.startswith("https://") and "<html" not in txt.lower():
                return txt
            errors.append(f"{label} HTTP {res.status_code}: {txt[:80]}")
        except Exception as exc:
            errors.append(f"{label} error: {exc}")

    raise RuntimeError(f"All Instagram temp hosting relays failed ({'; '.join(errors)})")


def delete_from_temp_hosting(video_url: str, hosting_url: Optional[str]) -> None:
    if not video_url or not hosting_url or not video_url.startswith(hosting_url.rstrip("/")):
        return
    try:
        filename = video_url.split("/")[-1]
        del_url = f"{hosting_url.rstrip('/')}/delete_video/{filename}"
        r = requests.post(del_url, timeout=15)
        if r.status_code not in (200, 404):
            requests.post(f"{hosting_url.rstrip('/')}/delete_video", data={"filename": filename}, timeout=15)
    except Exception:
        pass


def wait_container_ready(
    container_id: str,
    access_token: str,
    max_attempts: int = 25,
    interval_sec: int = 10,
) -> bool:
    status_url = f"{settings.GRAPH_API_URL}/{container_id}"
    for _ in range(max_attempts):
        try:
            r = requests.get(status_url, params={"fields": "status_code", "access_token": access_token}, timeout=20)
            if r.status_code == 200:
                status = r.json().get("status_code")
                if status == "FINISHED":
                    return True
                if status in ("ERROR", "EXPIRED"):
                    return False
        except Exception:
            pass
        time.sleep(interval_sec)
    return False


def publish_instagram_graph_api(
    video_path: str,
    caption: str = "",
    business_id: Optional[str] = None,
    access_token: Optional[str] = None,
    hosting_url: Optional[str] = None,
) -> Dict[str, Any]:
    b_id = business_id or settings.INSTAGRAM_BUSINESS_ID
    token = access_token or settings.PAGE_ACCESS_TOKEN
    host = hosting_url or settings.REMOTE_HOSTING_URL

    if not token or not b_id:
        raise ValueError("Instagram credentials missing: PAGE_ACCESS_TOKEN or INSTAGRAM_BUSINESS_ID not set.")

    public_url = upload_to_temp_hosting(video_path, host)
    if not public_url:
        raise RuntimeError("Failed to obtain public relay URL for Instagram upload.")

    try:
        create_url = f"{settings.GRAPH_API_URL}/{b_id}/media"
        final_caption = (caption or "").strip() or settings.DEFAULT_INSTAGRAM_CAPTION
        params = {
            "access_token": token,
            "caption": final_caption,
            "media_type": "REELS",
            "video_url": public_url,
        }
        res = requests.post(create_url, params=params, timeout=30)
        data = res.json()
        if "id" not in data:
            raise RuntimeError(f"Instagram container creation failed: {data.get('error', {}).get('message', res.text)}")

        container_id = data["id"]
        if not wait_container_ready(container_id, token):
            raise TimeoutError("Instagram media container did not reach FINISHED status in time.")

        pub_url = f"{settings.GRAPH_API_URL}/{b_id}/media_publish"
        pub_res = requests.post(pub_url, params={"creation_id": container_id, "access_token": token}, timeout=30)
        pub_data = pub_res.json()
        if "id" not in pub_data:
            raise RuntimeError(f"Instagram publish failed: {pub_data.get('error', {}).get('message', pub_res.text)}")

        post_id = pub_data["id"]
        return {
            "success": True,
            "platform": "instagram",
            "method": "graph_api",
            "post_id": post_id,
            "post_url": f"https://www.instagram.com/reel/{post_id}/",
        }
    finally:
        delete_from_temp_hosting(public_url, host)
