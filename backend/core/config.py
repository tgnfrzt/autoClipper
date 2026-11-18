import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
BUFFERS_DIR = STORAGE_DIR / "buffers"
CLIPS_RAW_DIR = STORAGE_DIR / "clips" / "raw"
CLIPS_RENDERED_DIR = STORAGE_DIR / "clips" / "rendered"
THUMBS_DIR = STORAGE_DIR / "thumbs"
DATA_DIR = STORAGE_DIR / "data"
MODELS_DIR = STORAGE_DIR / "models"
RESOURCES_DIR = BASE_DIR / "resources"

for _d in (
    BUFFERS_DIR,
    CLIPS_RAW_DIR,
    CLIPS_RENDERED_DIR,
    THUMBS_DIR,
    DATA_DIR,
    MODELS_DIR,
    RESOURCES_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

ENV_FILE = BASE_DIR / ".env"


def _resolve_binary(name: str, fallback_candidates: list[str]) -> str:
    env_override = os.getenv(f"{name.upper()}_PATH", "").strip()
    if env_override and os.path.isfile(env_override):
        return env_override
    found = shutil.which(name)
    if found:
        return found
    for cand in fallback_candidates:
        if cand and os.path.isfile(cand):
            return cand
    return name


def load_env_file() -> None:
    if ENV_FILE.is_file():
        with open(ENV_FILE, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        v = v[1:-1]
                    os.environ[k] = v.replace("\\n", "\n")


load_env_file()


def _env_str(key: str, default: str = "") -> str:
    val = os.getenv(key, "")
    return val if val != "" else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key, "").strip()
    return int(val) if val else default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key, "").strip()
    return float(val) if val else default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, "").strip().lower()
    if val == "":
        return default
    return val in ("true", "1", "yes", "on")


@dataclass
class Settings:
    APP_NAME: str = "GTA VI AI AutoClipper"
    DEBUG: bool = _env_bool("DEBUG", False)
    HOST: str = _env_str("HOST", "0.0.0.0")
    PORT: int = _env_int("PORT", 17430)

    # Storage paths
    BASE_PATH: str = str(BASE_DIR)
    STORAGE_PATH: str = str(STORAGE_DIR)
    BUFFERS_PATH: str = str(BUFFERS_DIR)
    CLIPS_RAW_PATH: str = str(CLIPS_RAW_DIR)
    CLIPS_RENDERED_PATH: str = str(CLIPS_RENDERED_DIR)
    THUMBS_PATH: str = str(THUMBS_DIR)
    DATA_PATH: str = str(DATA_DIR)
    MODELS_PATH: str = str(MODELS_DIR)
    RESOURCES_PATH: str = str(RESOURCES_DIR)
    DB_PATH: str = str(DATA_DIR / "autoclipper.db")
    WATERMARK_PATH: str = str(RESOURCES_DIR / "watermark.png")

    # Tool binaries
    FFMPEG_PATH: str = _resolve_binary(
        "ffmpeg",
        [
            r"C:\Users\mihal\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
        ],
    )
    FFPROBE_PATH: str = _resolve_binary(
        "ffprobe",
        [
            r"C:\Users\mihal\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.exe",
            r"C:\ffmpeg\bin\ffprobe.exe",
        ],
    )
    YTDLP_PATH: str = _resolve_binary("yt-dlp", [])
    STREAMLINK_PATH: str = _resolve_binary("streamlink", [])

    # Stream Discovery & Monitoring
    ENGINE_ENABLED: bool = _env_bool("ENGINE_ENABLED", True)
    GAME_QUERY: str = _env_str("GAME_QUERY", "Grand Theft Auto VI")
    FALLBACK_GAME_QUERY: str = _env_str("FALLBACK_GAME_QUERY", "Grand Theft Auto V")
    AUTO_DISCOVER_TWITCH: bool = _env_bool("AUTO_DISCOVER_TWITCH", True)
    AUTO_DISCOVER_YOUTUBE: bool = _env_bool("AUTO_DISCOVER_YOUTUBE", True)
    PREFER_FACECAM_STREAMS: bool = _env_bool("PREFER_FACECAM_STREAMS", True)
    MAX_CONCURRENT_STREAMS: int = _env_int("MAX_CONCURRENT_STREAMS", 15)
    STREAM_QUALITY: str = _env_str("STREAM_QUALITY", "720p,480p,best")
    DISCOVERY_INTERVAL_SEC: int = _env_int("DISCOVERY_INTERVAL_SEC", 120)

    # 80/20 Hype Detection Engine
    CHAT_WEIGHT: float = _env_float("CHAT_WEIGHT", 0.80)
    VISUAL_WEIGHT: float = _env_float("VISUAL_WEIGHT", 0.20)
    TRIGGER_SCORE_THRESHOLD: float = _env_float("TRIGGER_SCORE_THRESHOLD", 70.0)
    CLIP_DURATION_SEC: int = _env_int("CLIP_DURATION_SEC", 45)
    POST_TRIGGER_DELAY_SEC: int = _env_int("POST_TRIGGER_DELAY_SEC", 4)
    CLIP_COOLDOWN_SEC: int = _env_int("CLIP_COOLDOWN_SEC", 60)
    VISUAL_EVAL_INTERVAL_SEC: int = _env_int("VISUAL_EVAL_INTERVAL_SEC", 4)
    SEGMENT_DURATION_SEC: int = _env_int("SEGMENT_DURATION_SEC", 3)

    # AI Vision Model (Small/MoE local or OpenAI-compatible)
    VLM_PROVIDER: str = _env_str("VLM_PROVIDER", "llamacpp")
    VLM_API_URL: str = _env_str("VLM_API_URL", "http://127.0.0.1:8081/v1")
    VLM_API_KEY: str = _env_str("VLM_API_KEY", "")
    VLM_MODEL_NAME: str = _env_str(
        "VLM_MODEL_NAME",
        "Qwen2.5-VL-3B-Instruct (Q4_K_M)"
        if (MODELS_DIR / "qwen2.5-vl-3b" / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf").is_file()
        else "SmolVLM-500M-Instruct (Q8_0)",
    )
    LLAMA_SERVER_EXE: str = _env_str(
        "LLAMA_SERVER_EXE",
        r"E:\llama-cpp\llama-server.exe" if os.path.isfile(r"E:\llama-cpp\llama-server.exe") else (shutil.which("llama-server") or "llama-server"),
    )
    VLM_MODEL_PATH: str = _env_str(
        "VLM_MODEL_PATH",
        str(MODELS_DIR / "qwen2.5-vl-3b" / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf")
        if (MODELS_DIR / "qwen2.5-vl-3b" / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf").is_file()
        else str(MODELS_DIR / "smolvlm-500m" / "SmolVLM-500M-Instruct-Q8_0.gguf"),
    )
    VLM_MMPROJ_PATH: str = _env_str(
        "VLM_MMPROJ_PATH",
        str(MODELS_DIR / "qwen2.5-vl-3b" / "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf")
        if (MODELS_DIR / "qwen2.5-vl-3b" / "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf").is_file()
        else str(MODELS_DIR / "smolvlm-500m" / "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"),
    )
    VLM_PORT: int = _env_int("VLM_PORT", 8081)
    VLM_AUTO_START: bool = _env_bool("VLM_AUTO_START", True)
    VLM_CTX_SIZE: int = _env_int("VLM_CTX_SIZE", 2048)
    VLM_GPU_LAYERS: int = _env_int("VLM_GPU_LAYERS", 99)

    # Video Output & Smart 9:16 Gaming Split
    VIDEO_FORMAT_MODE: str = _env_str("VIDEO_FORMAT_MODE", "smart_gaming_split")
    FACECAM_POSITION: str = _env_str("FACECAM_POSITION", "auto")
    BURN_IN_CAPTIONS: bool = _env_bool("BURN_IN_CAPTIONS", True)
    ANTI_FINGERPRINT: bool = _env_bool("ANTI_FINGERPRINT", True)
    WATERMARK_ALPHA: float = _env_float("WATERMARK_ALPHA", 0.9)

    # Discord (Webhook URL recommended, or Optional Bot)
    DISCORD_WEBHOOK_URL: str = _env_str("DISCORD_WEBHOOK_URL", "")
    DISCORD_BOT_TOKEN: str = _env_str("DISCORD_BOT_TOKEN", "")
    DISCORD_CHANNEL_ID: str = _env_str("DISCORD_CHANNEL_ID", "")
    DISCORD_USER_ID: str = _env_str("DISCORD_USER_ID", "")
    DISCORD_COMMAND_PREFIX: str = _env_str("DISCORD_COMMAND_PREFIX", "!")
    AUTO_DELETE_ON_REJECT: bool = _env_bool("AUTO_DELETE_ON_REJECT", True)

    # Optional Twitch Helix API
    TWITCH_CLIENT_ID: str = _env_str("TWITCH_CLIENT_ID", "")
    TWITCH_CLIENT_SECRET: str = _env_str("TWITCH_CLIENT_SECRET", "")

    # Social Publishing (All defaults blank — user configures in UI or imports locally)
    INSTAGRAM_BUSINESS_ID: str = _env_str("INSTAGRAM_BUSINESS_ID", "")
    PAGE_ACCESS_TOKEN: str = _env_str("PAGE_ACCESS_TOKEN", "")
    GRAPH_API_VERSION: str = _env_str("GRAPH_API_VERSION", "v23.0")
    GRAPH_API_URL: str = f"https://graph.facebook.com/{_env_str('GRAPH_API_VERSION', 'v23.0')}"
    REMOTE_HOSTING_URL: str = _env_str("REMOTE_HOSTING_URL", "")

    YOUTUBE_CLIENT_SECRETS_FILE: str = str(DATA_DIR / "client_secret.json")
    YOUTUBE_TOKEN_FILE: str = str(DATA_DIR / "youtube_token.json")
    YOUTUBE_DEFAULT_PRIVACY: str = _env_str("YOUTUBE_DEFAULT_PRIVACY", "public")

    TIKTOK_COOKIES_PATH: str = str(DATA_DIR / "tiktok_cookies.txt")
    TIKTOK_ACCESS_TOKEN: str = _env_str("TIKTOK_ACCESS_TOKEN", "")
    TIKTOK_CLIENT_KEY: str = _env_str("TIKTOK_CLIENT_KEY", "")
    TIKTOK_CLIENT_SECRET: str = _env_str("TIKTOK_CLIENT_SECRET", "")
    TIKTOK_SESSION_ID: str = _env_str("TIKTOK_SESSION_ID", "")

    DEFAULT_YOUTUBE_CAPTION: str = _env_str(
        "DEFAULT_YOUTUBE_CAPTION",
        "Insane GTA VI moment! #GTA6 #GTAVI #Shorts #Gaming #Viral",
    )
    DEFAULT_TIKTOK_CAPTION: str = _env_str(
        "DEFAULT_TIKTOK_CAPTION",
        "Wait for the ending 😭🔥 #GTA6 #GTAVI #Gaming #FYP #Viral",
    )
    DEFAULT_INSTAGRAM_CAPTION: str = _env_str(
        "DEFAULT_INSTAGRAM_CAPTION",
        "GTA VI is already wild 🤯🔥 #GTA6 #GTAVI #Reels #Gaming #Viral",
    )


settings = Settings()


def get_watermark_path() -> Optional[str]:
    if os.path.isfile(settings.WATERMARK_PATH):
        return settings.WATERMARK_PATH
    return None


def is_youtube_configured() -> bool:
    return os.path.isfile(settings.YOUTUBE_TOKEN_FILE)


def is_discord_configured() -> bool:
    return bool(settings.DISCORD_WEBHOOK_URL.strip() or settings.DISCORD_BOT_TOKEN.strip())


def is_tiktok_configured() -> bool:
    return bool(
        settings.TIKTOK_SESSION_ID
        or settings.TIKTOK_ACCESS_TOKEN
        or os.path.isfile(settings.TIKTOK_COOKIES_PATH)
    )


def is_instagram_configured() -> bool:
    return bool(
        settings.INSTAGRAM_BUSINESS_ID
        and settings.PAGE_ACCESS_TOKEN
    )


def save_settings_to_env(updates: Dict[str, Any]) -> None:
    """Persist settings updates to local gitignored .env and sync in-memory Settings."""
    current_lines: list[str] = []
    if ENV_FILE.is_file():
        with open(ENV_FILE, "r", encoding="utf-8", errors="replace") as f:
            current_lines = f.readlines()

    updated_keys = set()
    new_lines: list[str] = []

    for line in current_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in updates:
                val = updates[k]
                if isinstance(val, bool):
                    str_val = "true" if val else "false"
                else:
                    str_val = str(val).replace("\n", "\\n")
                new_lines.append(f"{k}={str_val}\n")
                updated_keys.add(k)
                continue
        new_lines.append(line)

    for k, val in updates.items():
        if k not in updated_keys:
            if isinstance(val, bool):
                str_val = "true" if val else "false"
            else:
                str_val = str(val).replace("\n", "\\n")
            new_lines.append(f"{k}={str_val}\n")

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    for k, val in updates.items():
        if isinstance(val, bool):
            os.environ[k] = "true" if val else "false"
        else:
            os.environ[k] = str(val)
        if hasattr(settings, k):
            target_type = type(getattr(settings, k))
            try:
                if target_type is bool:
                    if isinstance(val, bool):
                        setattr(settings, k, val)
                    else:
                        setattr(settings, k, str(val).lower() in ("true", "1", "yes", "on"))
                elif target_type is int:
                    setattr(settings, k, int(float(val)))
                elif target_type is float:
                    setattr(settings, k, float(val))
                else:
                    setattr(settings, k, str(val))
            except Exception as exc:
                print(f"[Config] Warning updating {k}: {exc}")

    if "GRAPH_API_VERSION" in updates:
        settings.GRAPH_API_URL = f"https://graph.facebook.com/{settings.GRAPH_API_VERSION}"
