import asyncio
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from backend.core.config import BUFFERS_DIR, THUMBS_DIR, settings

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _safe_folder_name(stream_id: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stream_id)


def resolve_hls_url_sync(stream_url: str, quality: str = "720p,480p,best") -> Optional[str]:
    """Resolves a live Twitch or YouTube URL to a direct HLS manifest URL using streamlink or yt-dlp."""
    # 1. Try streamlink first (best for Twitch & live HLS)
    try:
        cmd = [settings.STREAMLINK_PATH, "--stream-url", stream_url, quality]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=18,
            creationflags=CREATE_NO_WINDOW,
        )
        out = (res.stdout or "").strip().splitlines()
        if res.returncode == 0 and out:
            candidate = out[-1].strip()
            if candidate.startswith("http"):
                return candidate
    except Exception:
        pass

    # 2. Fallback to yt-dlp -g
    try:
        cmd = [
            settings.YTDLP_PATH,
            "-f", "best[height<=720]/best",
            "-g",
            "--no-warnings",
            stream_url,
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
            creationflags=CREATE_NO_WINDOW,
        )
        out = (res.stdout or "").strip().splitlines()
        if res.returncode == 0 and out:
            candidate = out[0].strip()
            if candidate.startswith("http"):
                return candidate
    except Exception:
        pass

    return None


class StreamRingBuffer:
    """
    Zero-decode circular HLS segment recorder for one live stream.
    Runs ffmpeg with `-c copy` so CPU/GPU load stays tiny even across 15-25 concurrent streams.
    """

    def __init__(self, stream_id: str, stream_url: str, display_name: str) -> None:
        self.stream_id = stream_id
        self.stream_url = stream_url
        self.display_name = display_name
        self.safe_id = _safe_folder_name(stream_id)
        self.buf_dir = BUFFERS_DIR / self.safe_id
        self.thumb_path = THUMBS_DIR / f"{self.safe_id}.jpg"
        self.proc: Optional[subprocess.Popen] = None
        self.hls_url: Optional[str] = None
        self.started_at: float = 0.0
        self.last_error: str = ""
        self._starting: bool = False

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def get_segments_ordered(self) -> List[Path]:
        if not self.buf_dir.is_dir():
            return []
        files = [Path(p) for p in glob.glob(str(self.buf_dir / "seg_*.ts"))]
        files = [f for f in files if f.is_file() and f.stat().st_size > 4096]
        files.sort(key=lambda p: p.stat().st_mtime)
        return files

    def buffered_seconds(self) -> int:
        segs = self.get_segments_ordered()
        return max(0, (len(segs) - 1) * settings.SEGMENT_DURATION_SEC)

    async def ensure_started(self) -> bool:
        if self.is_running() or self._starting:
            return True
        self._starting = True
        try:
            hls = await asyncio.to_thread(resolve_hls_url_sync, self.stream_url, settings.STREAM_QUALITY)
            if not hls:
                self.last_error = "Could not resolve live HLS stream URL."
                return False
            self.hls_url = hls
            if self.buf_dir.exists():
                shutil.rmtree(self.buf_dir, ignore_errors=True)
            self.buf_dir.mkdir(parents=True, exist_ok=True)

            seg_pattern = str(self.buf_dir / "seg_%03d.ts")
            seg_sec = max(2, int(settings.SEGMENT_DURATION_SEC))
            # Keep ~90 seconds circular buffer (30 segments of 3s)
            wrap_count = max(25, int(90 / seg_sec))
            cmd = [
                settings.FFMPEG_PATH,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5",
                "-i", hls,
                "-c", "copy",
                "-f", "segment",
                "-segment_time", str(seg_sec),
                "-segment_wrap", str(wrap_count),
                "-reset_timestamps", "1",
                seg_pattern,
            ]
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
            )
            self.started_at = time.time()
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False
        finally:
            self._starting = False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    def extract_latest_frame_sync(self) -> Optional[str]:
        """Extracts a single JPEG keyframe from the newest completed .ts segment (~20ms)."""
        segs = self.get_segments_ordered()
        if len(segs) < 2:
            return None
        # Second-newest segment is guaranteed closed/flushed by ffmpeg
        target_seg = segs[-2]
        tmp_thumb = THUMBS_DIR / f"{self.safe_id}.tmp.jpg"
        cmd = [
            settings.FFMPEG_PATH,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-skip_frame", "nokey",
            "-i", str(target_seg),
            "-frames:v", "1",
            "-vf", "scale=512:-2",
            "-q:v", "4",
            str(tmp_thumb),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=6, creationflags=CREATE_NO_WINDOW)
            if res.returncode == 0 and tmp_thumb.is_file() and tmp_thumb.stat().st_size > 1000:
                if self.thumb_path.exists():
                    self.thumb_path.unlink()
                tmp_thumb.rename(self.thumb_path)
                return str(self.thumb_path)
        except Exception:
            pass
        return None

    async def extract_latest_frame(self) -> Optional[str]:
        return await asyncio.to_thread(self.extract_latest_frame_sync)

    def cut_clip_sync(self, duration_sec: int, output_mp4: str) -> Tuple[bool, float, str]:
        """Concatenates the last `duration_sec` of buffered TS segments into a clean MP4 (zero decode)."""
        segs = self.get_segments_ordered()
        if len(segs) < 2:
            return False, 0.0, "Not enough buffered video segments yet."

        # Exclude newest segment in case ffmpeg is actively writing it
        completed = segs[:-1]
        seg_sec = max(2, int(settings.SEGMENT_DURATION_SEC))
        needed_segs = max(2, min(len(completed), int(round(duration_sec / seg_sec))))
        chosen = completed[-needed_segs:]

        os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
        concat_file = self.buf_dir / f"concat_{int(time.time())}.txt"
        try:
            with open(concat_file, "w", encoding="utf-8") as f:
                for p in chosen:
                    safe_p = str(p.resolve()).replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{safe_p}'\n")

            cmd = [
                settings.FFMPEG_PATH,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                "-bsf:a", "aac_adtstoasc",
                "-movflags", "+faststart",
                output_mp4,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=45, creationflags=CREATE_NO_WINDOW)
            if res.returncode != 0 or not os.path.isfile(output_mp4):
                # Retry without aac_adtstoasc if audio codec differs
                cmd_retry = [c for c in cmd if c not in ("-bsf:a", "aac_adtstoasc")]
                res = subprocess.run(cmd_retry, capture_output=True, text=True, errors="replace", timeout=45, creationflags=CREATE_NO_WINDOW)
                if res.returncode != 0 or not os.path.isfile(output_mp4):
                    return False, 0.0, (res.stderr or "FFmpeg concat failed").strip()

            actual_dur = float(len(chosen) * seg_sec)
            return True, actual_dur, ""
        finally:
            try:
                if concat_file.exists():
                    concat_file.unlink()
            except Exception:
                pass
