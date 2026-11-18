import json
import os
from typing import Any, Dict, List, Optional

from backend.core.config import settings


def get_youtube_service():
    """Builds an authenticated YouTube Data API v3 resource using saved credentials."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    token_path = settings.YOUTUBE_TOKEN_FILE
    if not os.path.isfile(token_path):
        raise FileNotFoundError(
            f"YouTube credentials token not found at {token_path}. Authorize YouTube in the Configure tab."
        )

    with open(token_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    creds = Credentials.from_authorized_user_info(info)
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(token_path, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        except Exception as ex:
            raise RuntimeError(
                f"YouTube authorization expired or was revoked ({ex}). Re-authorize in Configure > YouTube."
            ) from ex
    elif creds.expired and not creds.refresh_token:
        raise RuntimeError(
            "YouTube token expired with no refresh token. Re-authorize in Configure > YouTube."
        )

    return build("youtube", "v3", credentials=creds)


def publish_youtube_shorts(
    video_path: str,
    title: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    privacy_status: Optional[str] = None,
    is_shorts: bool = True,
) -> Dict[str, Any]:
    """Uploads a clip to YouTube (Shorts by default) via resumable chunked upload."""
    from googleapiclient.http import MediaFileUpload

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    clean_title = (title or "").strip() or "Insane GTA VI Clip"
    desc = (description or "").strip() or settings.DEFAULT_YOUTUBE_CAPTION
    if is_shorts:
        if "#shorts" not in clean_title.lower():
            clean_title = f"{clean_title[:84]} #Shorts"
        if "#shorts" not in desc.lower():
            desc = f"{desc}\n\n#Shorts #GTA6 #GTAVI #Gaming"

    youtube = get_youtube_service()
    effective_privacy = privacy_status or settings.YOUTUBE_DEFAULT_PRIVACY or "public"
    body = {
        "snippet": {
            "title": clean_title[:100],
            "description": desc,
            "tags": tags or ["GTA6", "GTAVI", "Shorts", "Gaming", "Clips"],
            "categoryId": "20",
        },
        "status": {
            "privacyStatus": effective_privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, chunksize=1024 * 1024 * 5, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        _, response = request.next_chunk()

    video_id = response.get("id")
    post_url = f"https://www.youtube.com/shorts/{video_id}" if is_shorts else f"https://www.youtube.com/watch?v={video_id}"
    return {
        "success": True,
        "platform": "youtube_shorts" if is_shorts else "youtube",
        "video_id": video_id,
        "post_url": post_url,
    }
