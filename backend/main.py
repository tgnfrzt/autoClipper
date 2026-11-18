import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.api.ai_models import router as ai_router
from backend.api.clips import router as clips_router
from backend.api.settings import router as settings_router
from backend.api.streams import router as streams_router
from backend.core.config import BASE_DIR, settings
from backend.core.database import get_db, init_db
from backend.modules.ai.vision_evaluator import vision_manager
from backend.modules.discord.bot import discord_bot
from backend.modules.publishers.dispatcher import (
    check_autopost_rate_limit,
    get_configured_platforms,
    publish_clip_to_platforms,
)
from backend.modules.streams.manager import stream_manager


async def _handle_new_clip(clip: dict) -> None:
    score = float(clip.get("combined_score") or 0.0)
    print(f"[AutoClipper] Captured clip {clip['id']} from {clip['streamer_name']} (Score {score})")
    msg_id = await discord_bot.send_clip_for_review(clip)

    if not settings.AUTO_POST_ENABLED or score < float(settings.AUTO_POST_THRESHOLD):
        return

    platforms = get_configured_platforms(respect_yt_limit=True)
    if not platforms:
        stream_manager.log_event("AUTOPOST", f"Score {score}/100 hit Auto-Post threshold, but no available platforms right now.")
        return

    guard = await asyncio.to_thread(check_autopost_rate_limit)
    if guard["can_post"]:
        stream_manager.log_event(
            "AUTOPOST",
            f"Viral clip! {clip['streamer_name']} scored {score}/100 (>= {settings.AUTO_POST_THRESHOLD}) — Auto-Posting to {', '.join(platforms)}.",
        )
        if msg_id:
            await discord_bot.update_webhook_message_status(
                msg_id,
                f"🤖 **Auto-Posting viral clip (Score {score}/100 ≥ {settings.AUTO_POST_THRESHOLD}) to {', '.join(platforms)}...**",
            )
        res = await asyncio.to_thread(publish_clip_to_platforms, clip["id"], platforms)
        if msg_id:
            lines = [
                f"✅ **{it.get('platform', '').upper()}**: Auto-Posted! {it.get('post_url', '')}"
                if it.get("success")
                else f"❌ **{it.get('platform', '').upper()}**: {it.get('error', 'Failed')}"
                for it in res.get("results", [])
            ]
            if lines:
                await discord_bot.update_webhook_message_status(msg_id, "\n".join(lines))
    else:
        conn = get_db()
        try:
            with conn:
                conn.execute("UPDATE clips SET status = 'accepted' WHERE id = ? AND status = 'pending'", (clip["id"],))
        finally:
            conn.close()
        stream_manager.log_event(
            "AUTOPOST",
            f"Score {score}/100 saved to Accepted Queue — Anti-Ban Guard active: {guard['reason']}",
        )
        if msg_id:
            await discord_bot.update_webhook_message_status(
                msg_id,
                f"🛡️ **Viral Score {score}/100 — Saved to Accepted Queue** (Anti-Ban Guard: {guard['reason']})",
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    discord_bot.attach_stream_manager(stream_manager)
    stream_manager.set_on_clip_callback(_handle_new_clip)
    await discord_bot.start_or_restart()
    await stream_manager.start()
    yield
    await stream_manager.stop()
    await discord_bot.stop()
    vision_manager.stop_local_server()


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(settings_router)
app.include_router(streams_router)
app.include_router(clips_router)
app.include_router(ai_router)


@app.get("/api/health")
def health() -> dict:
    d_stat = discord_bot.get_status()
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "engine_enabled": settings.ENGINE_ENABLED,
        "vlm_healthy": vision_manager.is_server_healthy(),
        "discord_ready": d_stat["ready"],
        "discord_mode": d_stat["mode"],
        "discord_connected": d_stat["connected"],
    }


@app.get("/")
def serve_frontend():
    idx = Path(BASE_DIR) / "frontend" / "index.html"
    if not idx.is_file():
        return {"error": "frontend/index.html missing"}
    return FileResponse(str(idx), media_type="text/html")
