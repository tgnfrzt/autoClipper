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
from backend.core.database import init_db
from backend.modules.ai.vision_evaluator import vision_manager
from backend.modules.discord.bot import discord_bot
from backend.modules.streams.manager import stream_manager


async def _handle_new_clip(clip: dict) -> None:
    print(f"[AutoClipper] Captured clip {clip['id']} from {clip['streamer_name']} (Score {clip['combined_score']})")
    await discord_bot.send_clip_for_review(clip)


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
