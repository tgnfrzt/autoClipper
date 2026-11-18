import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from backend.core.config import settings

_nvenc_available: Optional[bool] = None
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def is_nvenc_available() -> bool:
    """Checks if NVIDIA NVENC hardware acceleration works on the host GPU (e.g. GTX 1650)."""
    global _nvenc_available
    if _nvenc_available is not None:
        return _nvenc_available
    try:
        cmd = [
            settings.FFMPEG_PATH,
            "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
            "-c:v", "h264_nvenc",
            "-f", "null", "-",
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            timeout=5,
            creationflags=CREATE_NO_WINDOW,
        )
        _nvenc_available = res.returncode == 0
    except Exception:
        _nvenc_available = False
    return _nvenc_available


def get_fast_encoder_args(quality: int = 21, preset: str = "p4") -> List[str]:
    """Returns NVENC encoding args if supported, falling back cleanly to libx264."""
    if is_nvenc_available():
        return [
            "-c:v", "h264_nvenc",
            "-preset", preset,
            "-cq", str(quality),
            "-b:v", "3500k",
            "-maxrate", "5000k",
            "-bufsize", "8000k",
            "-spatial-aq", "1",
            "-temporal-aq", "1",
        ]
    return [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(quality),
    ]


def probe_video(file_path: str) -> Dict[str, Any]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")

    cmd = [
        settings.FFPROBE_PATH,
        "-v", "error",
        "-show_entries", "format=duration,size,bit_rate:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of", "json",
        file_path,
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
        creationflags=CREATE_NO_WINDOW,
    )
    data = json.loads(res.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    width = int(video_stream.get("width", 0)) if video_stream else 0
    height = int(video_stream.get("height", 0)) if video_stream else 0

    fps = 30.0
    if video_stream and "r_frame_rate" in video_stream:
        try:
            num, den = video_stream["r_frame_rate"].split("/")
            fps = float(num) / float(den)
        except Exception:
            fps = 30.0

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "has_audio": audio_stream is not None,
        "size_bytes": int(data.get("format", {}).get("size", 0) or os.path.getsize(file_path)),
    }


def run_ffmpeg(args: List[str], timeout: int = 240) -> None:
    cmd = [settings.FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error"] + args
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        creationflags=CREATE_NO_WINDOW,
    )
    if res.returncode != 0:
        err_msg = (res.stderr or "").strip()[-700:] or f"FFmpeg exited with code {res.returncode}"
        raise RuntimeError(f"FFmpeg error: {err_msg}")
