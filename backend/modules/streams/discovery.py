import asyncio
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

import httpx

from backend.core.config import settings

TWITCH_GQL_URL = "https://gql.twitch.tv/gql"
TWITCH_PUBLIC_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


async def check_twitch_channel_live(login: str) -> Optional[Dict[str, Any]]:
    """Checks if a single Twitch channel is currently live using Twitch public GQL."""
    clean_login = login.strip().lower().lstrip("@")
    query = """
    query($login: String!) {
      user(login: $login) {
        id
        login
        displayName
        stream {
          id
          title
          viewersCount
          game {
            name
          }
        }
      }
    }
    """
    headers = {
        "Client-ID": TWITCH_PUBLIC_CLIENT_ID,
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                TWITCH_GQL_URL,
                headers=headers,
                json={"query": query, "variables": {"login": clean_login}},
            )
            if r.status_code != 200:
                return None
            user = (r.json().get("data") or {}).get("user")
            if not user:
                return None
            stream = user.get("stream")
            return {
                "platform": "twitch",
                "channel_id": user.get("login", clean_login),
                "display_name": user.get("displayName", clean_login),
                "url": f"https://www.twitch.tv/{user.get('login', clean_login)}",
                "is_live": stream is not None,
                "viewer_count": int(stream.get("viewersCount", 0)) if stream else 0,
                "title": str(stream.get("title", "")) if stream else "",
                "game_name": str(((stream.get("game") or {}).get("name")) or "") if stream else "",
            }
    except Exception:
        return None


async def discover_twitch_game_streams(game_name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetches top live Twitch streams for a game/category (e.g. 'Grand Theft Auto VI')
    ordered by viewer count via Twitch public GQL (no API key needed) or Helix API if configured.
    """
    if settings.TWITCH_CLIENT_ID and settings.TWITCH_CLIENT_SECRET:
        helix_streams = await _discover_twitch_helix(game_name, limit)
        if helix_streams:
            return helix_streams

    query = """
    query($name: String!, $first: Int!) {
      game(name: $name) {
        name
        streams(first: $first, options: {sort: VIEWER_COUNT}) {
          edges {
            node {
              id
              title
              viewersCount
              broadcaster {
                login
                displayName
              }
              game {
                name
              }
            }
          }
        }
      }
    }
    """
    headers = {
        "Client-ID": TWITCH_PUBLIC_CLIENT_ID,
        "Content-Type": "application/json",
    }
    results: List[Dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.post(
                TWITCH_GQL_URL,
                headers=headers,
                json={"query": query, "variables": {"name": game_name, "first": limit}},
            )
            if r.status_code != 200:
                return []
            game = (r.json().get("data") or {}).get("game")
            if not game:
                return []
            edges = (game.get("streams") or {}).get("edges") or []
            for edge in edges:
                node = (edge or {}).get("node") or {}
                b = node.get("broadcaster") or {}
                login = str(b.get("login", "")).strip()
                if not login:
                    continue
                results.append(
                    {
                        "platform": "twitch",
                        "channel_id": login.lower(),
                        "display_name": str(b.get("displayName") or login),
                        "url": f"https://www.twitch.tv/{login}",
                        "is_live": True,
                        "viewer_count": int(node.get("viewersCount") or 0),
                        "title": str(node.get("title") or ""),
                        "game_name": str(((node.get("game") or {}).get("name")) or game_name),
                    }
                )
    except Exception as exc:
        print(f"[Discovery] Twitch GQL error for '{game_name}': {exc}")
    return results


async def _discover_twitch_helix(game_name: str, limit: int = 20) -> List[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            tok_res = await client.post(
                "https://id.twitch.tv/oauth2/token",
                params={
                    "client_id": settings.TWITCH_CLIENT_ID,
                    "client_secret": settings.TWITCH_CLIENT_SECRET,
                    "grant_type": "client_credentials",
                },
            )
            if tok_res.status_code != 200:
                return []
            token = tok_res.json().get("access_token", "")
            headers = {"Client-ID": settings.TWITCH_CLIENT_ID, "Authorization": f"Bearer {token}"}
            g_res = await client.get("https://api.twitch.tv/helix/games", headers=headers, params={"name": game_name})
            games = g_res.json().get("data", [])
            if not games:
                return []
            game_id = games[0]["id"]
            s_res = await client.get(
                "https://api.twitch.tv/helix/streams",
                headers=headers,
                params={"game_id": game_id, "first": min(limit, 100)},
            )
            streams = s_res.json().get("data", [])
            return [
                {
                    "platform": "twitch",
                    "channel_id": s["user_login"].lower(),
                    "display_name": s["user_name"],
                    "url": f"https://www.twitch.tv/{s['user_login']}",
                    "is_live": True,
                    "viewer_count": int(s.get("viewer_count", 0)),
                    "title": s.get("title", ""),
                    "game_name": s.get("game_name", game_name),
                }
                for s in streams
            ]
    except Exception:
        return []


def _ytdlp_search_live_sync(search_query: str, limit: int = 12) -> List[Dict[str, Any]]:
    """Uses yt-dlp with YouTube's Live filter (sp=EgJAAQ%3D%3D) to find live streams with zero API quota."""
    encoded_q = search_query.replace(" ", "+")
    url = f"https://www.youtube.com/results?search_query={encoded_q}&sp=EgJAAQ%253D%253D"
    cmd = [
        settings.YTDLP_PATH,
        "--flat-playlist",
        "--dump-single-json",
        "--no-warnings",
        "--playlist-end", str(limit),
        url,
    ]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=25,
            creationflags=CREATE_NO_WINDOW,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return []
        data = json.loads(res.stdout)
        entries = data.get("entries") or []
        results: List[Dict[str, Any]] = []
        for entry in entries:
            if not entry:
                continue
            vid = str(entry.get("id") or "").strip()
            if not vid or len(vid) < 6:
                continue
            live_status = entry.get("live_status")
            if live_status and live_status not in ("is_live", "is_upcoming"):
                continue
            if live_status == "is_upcoming":
                continue
            channel = str(entry.get("channel") or entry.get("uploader") or "YouTube Live")
            results.append(
                {
                    "platform": "youtube",
                    "channel_id": vid,
                    "display_name": channel,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "is_live": True,
                    "viewer_count": int(entry.get("concurrent_view_count") or entry.get("view_count") or 0),
                    "title": str(entry.get("title") or ""),
                    "game_name": search_query,
                }
            )
        return results
    except Exception as exc:
        print(f"[Discovery] YouTube search error: {exc}")
        return []


async def discover_youtube_live_streams(query: str, limit: int = 12) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(_ytdlp_search_live_sync, query, limit)


async def check_youtube_target_live(url_or_channel: str) -> Optional[Dict[str, Any]]:
    """Checks if a manual YouTube channel/live URL is currently live."""
    target = url_or_channel.strip()
    if target.startswith("@"):
        target = f"https://www.youtube.com/{target}/live"
    elif "youtube.com" not in target and "youtu.be" not in target:
        target = f"https://www.youtube.com/watch?v={target}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            r = await client.get(target, headers=headers)
            if r.status_code != 200:
                return None
            html = r.text
            is_live = '"isLive":true' in html or '"isLiveNow":true' in html
            vid_match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
            title_match = re.search(r'<title>(.*?)</title>', html, re.DOTALL)
            author_match = re.search(r'"ownerChannelName":"([^"]+)"', html) or re.search(r'"author":"([^"]+)"', html)
            vid = vid_match.group(1) if vid_match else ""
            if not vid:
                return None
            title = (title_match.group(1).replace(" - YouTube", "").strip() if title_match else "")
            author = author_match.group(1) if author_match else target
            return {
                "platform": "youtube",
                "channel_id": vid,
                "display_name": author,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "is_live": is_live,
                "viewer_count": 0,
                "title": title,
                "game_name": "",
            }
    except Exception:
        return None
