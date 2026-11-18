import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np

from backend.core.config import MODELS_DIR, save_settings_to_env, settings

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

BLOCKED_CLIP_PHRASES = (
    "commercial break",
    "ad break",
    "advertisement",
    "legal notice",
    "starting soon",
    "be right back",
    "stream ending",
    "stream offline",
    "twitch break",
    "stream break",
    "gaming break",
    "sponsor screen",
    "4-8 word",
    "exciting headline",
    "inaudible yelling",
    "inaudible / yelling",
)

VAGUE_FILLER_PHRASES = (
    "yelling in the game",
    "reacting to chat",
    "reacts to chat",
    "hosting a gaming session",
    "yelling and interacting",
)

MUNDANE_SLOP_PHRASES = (
    "reacting to the chat",
    "reacting to chat",
    "reacts to chat",
    "discussing strategy",
    "discussing his plans",
    "discussing plans",
    "discussing the map",
    "talking to someone in the car",
    "talking to a character",
    "stops at a gas station",
    "stopped at a gas station",
    "currently in a boat",
    "navigating a boat",
    "reacting to bugs",
    "reacting to a phone call",
    "in the middle of a tournament",
    "live streaming on twitch",
    "chat overlay showing",
    "looking at the map",
    "browsing the menu",
    "waiting in the lobby",
)

VLM_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen2.5-vl-3b": {
        "id": "qwen2.5-vl-3b",
        "name": "Qwen2.5-VL-3B-Instruct (Q4_K_M)",
        "size_label": "~2.65 GB (Recommended • Internet/Meme Aware • Fits 4GB VRAM)",
        "recommended_for": "Best accuracy for shocked facecam expressions, GTA action & viral captions",
        "model_filename": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "mmproj_filename": "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
        "model_url": "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "mmproj_url": "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
    },
    "smolvlm-500m": {
        "id": "smolvlm-500m",
        "name": "SmolVLM-500M-Instruct (Q8_0)",
        "size_label": "~520 MB (Ultra-Light • <1GB VRAM)",
        "recommended_for": "Ultra-fast fallback for older GPUs (<2GB VRAM)",
        "model_filename": "SmolVLM-500M-Instruct-Q8_0.gguf",
        "mmproj_filename": "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
        "model_url": "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf",
        "mmproj_url": "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
    },
}


class VisionModelManager:
    """
    Manages the local VLM (Qwen2.5-VL-3B-Instruct or SmolVLM via llama-server.exe / OpenAI API),
    OpenCV facecam reaction detection, side-by-side facecam+gameplay compositing,
    and post-clip Whisper + VLM caption generation.
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self._prev_frames: Dict[str, np.ndarray] = {}
        self._prev_facecams: Dict[str, np.ndarray] = {}
        self._facecam_cache: Dict[str, Dict[str, Any]] = {}
        self._vlm_semaphore: Optional[asyncio.Semaphore] = None
        self._whisper_model: Any = None
        self._whisper_lock = threading.Lock()
        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        cascade_xml = (
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if hasattr(cv2, "data")
            else ""
        )
        self._face_cascade = (
            cascade_cls(cascade_xml)
            if (cascade_cls is not None and cascade_xml and os.path.isfile(cascade_xml))
            else None
        )

        # Telemetry metrics for Statistics & Monitor UI
        self.total_evaluations: int = 0
        self.vlm_evaluations: int = 0
        self.last_latency_ms: int = 0
        self._latencies_ms: Deque[int] = deque(maxlen=50)
        self._last_error: str = ""
        self._download_state: Dict[str, Any] = {
            "active": False,
            "preset_id": "",
            "file": "",
            "progress_pct": 0.0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "error": "",
        }

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._vlm_semaphore is None:
            self._vlm_semaphore = asyncio.Semaphore(1)
        return self._vlm_semaphore

    def get_telemetry(self) -> Dict[str, Any]:
        avg_ms = int(sum(self._latencies_ms) / len(self._latencies_ms)) if self._latencies_ms else 0
        detected_facecams = sum(1 for v in self._facecam_cache.values() if v.get("has_facecam"))
        return {
            "total_evaluations": self.total_evaluations,
            "vlm_evaluations": self.vlm_evaluations,
            "last_latency_ms": self.last_latency_ms,
            "avg_latency_ms": avg_ms,
            "facecams_detected": detected_facecams,
        }

    def list_presets(self) -> List[Dict[str, Any]]:
        items = []
        for pid, p in VLM_PRESETS.items():
            pdir = MODELS_DIR / pid
            mpath = pdir / p["model_filename"]
            mmpath = pdir / p["mmproj_filename"]
            installed = (
                mpath.is_file()
                and mpath.stat().st_size > 10_000_000
                and mmpath.is_file()
                and mmpath.stat().st_size > 5_000_000
            )
            items.append(
                {
                    **p,
                    "installed": installed,
                    "model_path": str(mpath),
                    "mmproj_path": str(mmpath),
                    "is_active": os.path.abspath(settings.VLM_MODEL_PATH) == os.path.abspath(str(mpath)),
                }
            )
        return items

    def is_server_healthy(self) -> bool:
        if settings.VLM_PROVIDER == "opencv_only":
            return True
        base = settings.VLM_API_URL.rstrip("/")
        root = base[:-3] if base.endswith("/v1") else base
        for url in (f"{root}/health", f"{base}/models"):
            try:
                req = urllib.request.Request(url)
                if settings.VLM_API_KEY:
                    req.add_header("Authorization", f"Bearer {settings.VLM_API_KEY}")
                with urllib.request.urlopen(req, timeout=1.2) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                continue
        return False

    def get_status(self) -> Dict[str, Any]:
        running_pid = self._proc.pid if (self._proc and self._proc.poll() is None) else None
        healthy = self.is_server_healthy()
        return {
            "provider": settings.VLM_PROVIDER,
            "api_url": settings.VLM_API_URL,
            "model_name": settings.VLM_MODEL_NAME,
            "model_path": settings.VLM_MODEL_PATH,
            "mmproj_path": settings.VLM_MMPROJ_PATH,
            "model_exists": os.path.isfile(settings.VLM_MODEL_PATH),
            "mmproj_exists": os.path.isfile(settings.VLM_MMPROJ_PATH),
            "llama_server_exe": settings.LLAMA_SERVER_EXE,
            "llama_server_found": os.path.isfile(settings.LLAMA_SERVER_EXE),
            "port": settings.VLM_PORT,
            "pid": running_pid,
            "healthy": healthy,
            "last_error": self._last_error,
            "download": dict(self._download_state),
            "presets": self.list_presets(),
            "telemetry": self.get_telemetry(),
        }

    def start_local_server(self) -> Tuple[bool, str]:
        if settings.VLM_PROVIDER != "llamacpp":
            return True, f"Using external provider '{settings.VLM_PROVIDER}'."
        if self.is_server_healthy():
            return True, "Local VLM server is already online."

        exe = settings.LLAMA_SERVER_EXE
        if not os.path.isfile(exe):
            msg = f"llama-server.exe not found at {exe}. Set LLAMA_SERVER_EXE in Configure."
            self._last_error = msg
            return False, msg

        if not os.path.isfile(settings.VLM_MODEL_PATH):
            msg = f"Model GGUF not found at {settings.VLM_MODEL_PATH}. Download a preset in Configure."
            self._last_error = msg
            return False, msg

        with self._lock:
            if self._proc and self._proc.poll() is None:
                return True, "Local VLM server is starting."

            log_path = Path(settings.STORAGE_PATH) / "vlm_server.log"
            cmd = [
                exe,
                "-m", settings.VLM_MODEL_PATH,
                "--alias", settings.VLM_MODEL_NAME or Path(settings.VLM_MODEL_PATH).stem,
                "-c", str(settings.VLM_CTX_SIZE),
                "-ngl", str(settings.VLM_GPU_LAYERS),
                "--fit", "on",
                "--fit-target", "384",
                "--host", "127.0.0.1",
                "--port", str(settings.VLM_PORT),
                "-np", "1",
                "--no-warmup",
            ]
            if settings.VLM_MMPROJ_PATH and os.path.isfile(settings.VLM_MMPROJ_PATH):
                cmd.extend(["--mmproj", settings.VLM_MMPROJ_PATH])

            try:
                log_f = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    creationflags=CREATE_NO_WINDOW,
                )
                self._last_error = ""
            except Exception as exc:
                self._last_error = str(exc)
                return False, f"Failed to launch llama-server: {exc}"

        for _ in range(45):
            if self._proc and self._proc.poll() is not None:
                self._last_error = "llama-server exited unexpectedly. Check storage/vlm_server.log."
                return False, self._last_error
            if self.is_server_healthy():
                save_settings_to_env({"VLM_API_URL": f"http://127.0.0.1:{settings.VLM_PORT}/v1"})
                return True, f"Local VLM ({settings.VLM_MODEL_NAME}) online on port {settings.VLM_PORT}."
            time.sleep(0.5)

        return True, "Local VLM server started and is finishing model warmup."

    def stop_local_server(self) -> Tuple[bool, str]:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=4)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None
                return True, "Local VLM server stopped."
        return True, "Local VLM server was not running."

    def download_preset(self, preset_id: str, activate: bool = True) -> Tuple[bool, str]:
        if preset_id not in VLM_PRESETS:
            return False, f"Unknown preset '{preset_id}'."
        if self._download_state["active"]:
            return False, "Another model download is already in progress."

        preset = VLM_PRESETS[preset_id]
        target_dir = MODELS_DIR / preset_id
        target_dir.mkdir(parents=True, exist_ok=True)
        model_dst = target_dir / preset["model_filename"]
        mmproj_dst = target_dir / preset["mmproj_filename"]

        def _worker() -> None:
            self._download_state.update(
                {"active": True, "preset_id": preset_id, "file": "", "progress_pct": 0.0, "downloaded_mb": 0.0, "total_mb": 0.0, "error": ""}
            )
            try:
                for label, url, dst in (
                    (preset["model_filename"], preset["model_url"], model_dst),
                    (preset["mmproj_filename"], preset["mmproj_url"], mmproj_dst),
                ):
                    self._download_state["file"] = label
                    tmp_dst = dst.with_suffix(".part")
                    with httpx.stream("GET", url, follow_redirects=True, timeout=180.0) as r:
                        r.raise_for_status()
                        total = int(r.headers.get("content-length") or 0)
                        done = 0
                        with open(tmp_dst, "wb") as f:
                            for chunk in r.iter_bytes(chunk_size=1024 * 256):
                                f.write(chunk)
                                done += len(chunk)
                                self._download_state["downloaded_mb"] = round(done / (1024 * 1024), 1)
                                self._download_state["total_mb"] = round(total / (1024 * 1024), 1) if total else 0.0
                                self._download_state["progress_pct"] = round((done / total) * 100.0, 1) if total else 0.0
                    if dst.exists():
                        dst.unlink()
                    tmp_dst.rename(dst)

                if activate:
                    self.stop_local_server()
                    save_settings_to_env(
                        {
                            "VLM_PROVIDER": "llamacpp",
                            "VLM_MODEL_NAME": preset["name"],
                            "VLM_MODEL_PATH": str(model_dst),
                            "VLM_MMPROJ_PATH": str(mmproj_dst),
                            "VLM_API_URL": f"http://127.0.0.1:{settings.VLM_PORT}/v1",
                        }
                    )
                    self.start_local_server()
            except Exception as exc:
                self._download_state["error"] = str(exc)
            finally:
                self._download_state["active"] = False

        threading.Thread(target=_worker, daemon=True).start()
        return True, f"Started downloading {preset['name']}..."

    # ------------------------------------------------------------------
    # Facecam Corner Detection, Reaction Scoring & OpenCV Action Scoring
    # ------------------------------------------------------------------
    def _crop_corner(self, img: np.ndarray, corner: str) -> np.ndarray:
        h, w = img.shape[:2]
        cw = max(64, int(w * 0.24))
        ch = max(48, int(h * 0.30))
        x0, x1 = (w - cw, w) if "right" in corner else (0, cw)
        if corner.startswith("mid_"):
            y0 = max(0, (h - ch) // 2)
            return img[y0 : y0 + ch, x0:x1]
        if corner.startswith("bottom_"):
            return img[h - ch : h, x0:x1]
        return img[0:ch, x0:x1]

    def inspect_facecam_and_motion(self, stream_id: str, image_path: str) -> Dict[str, Any]:
        """
        Detects stationary edge/corner facecam overlays (including mid-left / mid-right webcams)
        with skin-tone verification & 3-hit hysteresis, computes gameplay motion/flash score,
        and scores sudden movement inside the streamer's facecam box.
        """
        img = cv2.imread(image_path)
        if img is None:
            return {
                "cv_score": 0.0,
                "motion": 0.0,
                "flash": 0.0,
                "has_facecam": False,
                "facecam_corner": "",
                "facecam_score": 0.0,
                "img": None,
            }

        h, w = img.shape[:2]
        small = cv2.resize(img, (256, 144))
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # 1. Gameplay motion + flash + red intensity
        motion_score = 0.0
        flash_score = 0.0
        prev = self._prev_frames.get(stream_id)
        if prev is not None and prev.shape == gray_small.shape:
            diff = cv2.absdiff(gray_small, prev)
            mean_diff = float(np.mean(diff))
            motion_pct = float(np.mean(diff > 28)) * 100.0
            motion_score = min(100.0, mean_diff * 2.2 + motion_pct * 0.8)
            flash_score = min(100.0, abs(float(np.mean(gray_small)) - float(np.mean(prev))) * 2.8)
        self._prev_frames[stream_id] = gray_small

        b, g, r = cv2.split(small.astype(np.float32))
        red_excess = np.maximum(0.0, r - 0.55 * (g + b))
        red_score = min(100.0, float(np.mean(red_excess)) * 2.5)
        cv_score = round(min(100.0, 0.55 * motion_score + 0.30 * flash_score + 0.15 * red_score), 1)

        # 2. Edge/Corner Facecam Detection with Skin Check & Hysteresis (supports top/mid/bottom left/right)
        cache = self._facecam_cache.setdefault(
            stream_id, {"has_facecam": False, "corner": "", "hits": 0, "last_pos": None}
        )
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ycrcb_full = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if self._face_cascade is not None and not self._face_cascade.empty():
            faces = self._face_cascade.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
            detected_corner = ""
            det_pos: Optional[Tuple[float, float]] = None
            for (fx, fy, fw, fh) in faces:
                cx = (fx + fw * 0.5) / w
                cy = (fy + fh * 0.5) / h
                if not (cx <= 0.26 or cx >= 0.74):
                    continue
                if fw / w > 0.22 or fh / h > 0.32:
                    continue
                roi = ycrcb_full[int(fy + fh * 0.15) : int(fy + fh * 0.85), int(fx + fw * 0.15) : int(fx + fw * 0.85)]
                skin_ratio = (
                    float(
                        np.mean(
                            (roi[:, :, 0] >= 35)
                            & (roi[:, :, 1] >= 130)
                            & (roi[:, :, 1] <= 180)
                            & (roi[:, :, 2] >= 75)
                            & (roi[:, :, 2] <= 130)
                        )
                    )
                    if roi.size
                    else 0.0
                )
                if skin_ratio < 0.20:
                    continue
                horiz = "left" if cx < 0.5 else "right"
                vert = "top" if cy < 0.36 else ("mid" if cy <= 0.64 else "bottom")
                detected_corner = f"{vert}_{horiz}"
                det_pos = (cx, cy)
                break

            if detected_corner and det_pos is not None:
                prev_pos = cache.get("last_pos")
                same_spot = (
                    prev_pos is None
                    or (abs(det_pos[0] - prev_pos[0]) < 0.04 and abs(det_pos[1] - prev_pos[1]) < 0.04)
                )
                if same_spot or cache["corner"] == detected_corner:
                    cache["hits"] = min(10, cache["hits"] + 2)
                else:
                    cache["hits"] = 1
                cache["corner"] = detected_corner
                cache["last_pos"] = det_pos
                # Require at least 2 confirming frames before locking a facecam badge
                if cache["hits"] >= 3:
                    cache["has_facecam"] = True
            else:
                cache["hits"] = max(0, cache["hits"] - 1)
                if cache["hits"] == 0:
                    cache["has_facecam"] = False
                    cache["corner"] = ""
                    cache["last_pos"] = None

        # 3. Compute Facecam Reaction Score (sudden webcam motion when streamer reacts/jumps/yells)
        facecam_score = 0.0
        if cache["has_facecam"] and cache["corner"]:
            cam_crop = self._crop_corner(img, cache["corner"])
            cam_gray = cv2.resize(cv2.cvtColor(cam_crop, cv2.COLOR_BGR2GRAY), (96, 72))
            prev_cam = self._prev_facecams.get(stream_id)
            if prev_cam is not None and prev_cam.shape == cam_gray.shape:
                cam_diff = cv2.absdiff(cam_gray, prev_cam)
                cam_motion_pct = float(np.mean(cam_diff > 22)) * 100.0
                cam_mean = float(np.mean(cam_diff))
                facecam_score = round(min(100.0, cam_mean * 3.0 + cam_motion_pct * 1.4), 1)
            self._prev_facecams[stream_id] = cam_gray

        return {
            "cv_score": cv_score,
            "motion": round(motion_score, 1),
            "flash": round(flash_score, 1),
            "has_facecam": bool(cache["has_facecam"]),
            "facecam_corner": cache["corner"],
            "facecam_score": facecam_score,
            "img": img,
        }

    def _build_composite_b64(self, img: np.ndarray, facecam_corner: str) -> str:
        """
        Builds a compact composite image for Qwen2.5-VL-3B:
        If a facecam corner is present, places [Gameplay (256x192) | Zoomed Facecam (160x192)] side-by-side
        so the VLM sees BOTH the game scene and the streamer's facial expression clearly!
        """
        if facecam_corner:
            game_panel = cv2.resize(img, (256, 192))
            cam_raw = self._crop_corner(img, facecam_corner)
            cam_panel = cv2.resize(cam_raw, (160, 192))
            cv2.putText(cam_panel, "CAM", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            composite = np.hstack([game_panel, cam_panel])
        else:
            composite = cv2.resize(img, (320, 180))

        ok, buf = cv2.imencode(".jpg", composite, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ------------------------------------------------------------------
    # Commercial / Ad Slate Detection & Real-Time Visual Evaluation (20%)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_twitch_ad_slate(img: np.ndarray) -> bool:
        """
        Detects Twitch's purple 'Commercial break in progress' slate in <3ms.
        Requires bright Twitch brand purple (H 133..148, S>=135, V>=110) covering >=42%
        of the frame AND low edge complexity so dark blue GTA night skies never false-trigger.
        """
        if img is None:
            return False
        small = cv2.resize(img, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 0] >= 133) & (hsv[:, :, 0] <= 148) & (hsv[:, :, 1] >= 135) & (hsv[:, :, 2] >= 110)
        purple_ratio = float(np.mean(mask))
        if purple_ratio < 0.42:
            return False
        edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 60, 140)
        edge_mean = float(np.mean(edges))
        return edge_mean < 22.0

    @staticmethod
    def _contains_blocked_phrase(*texts: str) -> bool:
        joined = " ".join((t or "") for t in texts).lower()
        return any(p in joined for p in BLOCKED_CLIP_PHRASES)

    async def evaluate_frame(
        self,
        stream_id: str,
        image_path: str,
        streamer_name: str = "",
        chat_context: str = "",
        force_vlm: bool = False,
    ) -> Dict[str, Any]:
        if not os.path.isfile(image_path):
            return {
                "visual_score": 0.0,
                "cv_score": 0.0,
                "vlm_score": None,
                "has_facecam": False,
                "facecam_corner": "",
                "facecam_score": 0.0,
                "is_blocked": False,
                "latency_ms": 0,
                "event": "no_frame",
                "summary": "",
            }

        self.total_evaluations += 1
        cv_res = await asyncio.to_thread(self.inspect_facecam_and_motion, stream_id, image_path)
        img = cv_res["img"]

        if self._is_twitch_ad_slate(img):
            return {
                "visual_score": 0.0,
                "cv_score": 0.0,
                "vlm_score": 0.0,
                "has_facecam": cv_res["has_facecam"],
                "facecam_corner": cv_res["facecam_corner"],
                "facecam_score": 0.0,
                "is_blocked": True,
                "latency_ms": 0,
                "event": "commercial_break",
                "summary": "Twitch commercial break slate detected (auto-filtered)",
            }

        base_fallback_score = (
            round(min(100.0, 0.65 * cv_res["cv_score"] + 0.35 * cv_res["facecam_score"]), 1)
            if cv_res["has_facecam"]
            else cv_res["cv_score"]
        )

        vlm_online = True if settings.VLM_PROVIDER == "opencv_only" else await asyncio.to_thread(self.is_server_healthy)
        if settings.VLM_PROVIDER == "opencv_only" or not vlm_online:
            return {
                "visual_score": base_fallback_score,
                "cv_score": cv_res["cv_score"],
                "vlm_score": None,
                "has_facecam": cv_res["has_facecam"],
                "facecam_corner": cv_res["facecam_corner"],
                "facecam_score": cv_res["facecam_score"],
                "is_blocked": False,
                "latency_ms": 0,
                "event": "opencv_only" if settings.VLM_PROVIDER == "opencv_only" else "vlm_offline",
                "summary": f"Motion {cv_res['motion']}% | Cam {cv_res['facecam_score']}%",
            }

        sem = self._get_semaphore()
        if sem.locked() and not force_vlm:
            return {
                "visual_score": base_fallback_score,
                "cv_score": cv_res["cv_score"],
                "vlm_score": None,
                "has_facecam": cv_res["has_facecam"],
                "facecam_corner": cv_res["facecam_corner"],
                "facecam_score": cv_res["facecam_score"],
                "is_blocked": False,
                "latency_ms": 0,
                "event": "opencv_fastpath",
                "summary": f"Motion {cv_res['motion']}% | Cam {cv_res['facecam_score']}%",
            }

        b64 = self._build_composite_b64(img, cv_res["facecam_corner"]) if img is not None else ""
        if not b64:
            return {
                "visual_score": base_fallback_score,
                "cv_score": cv_res["cv_score"],
                "vlm_score": None,
                "has_facecam": cv_res["has_facecam"],
                "facecam_corner": cv_res["facecam_corner"],
                "facecam_score": cv_res["facecam_score"],
                "is_blocked": False,
                "latency_ms": 0,
                "event": "encode_skip",
                "summary": "",
            }

        cam_hint = (
            "The right panel labeled CAM is a zoomed crop of the streamer's facecam. "
            "Pay special attention to shocked, screaming, laughing, or hyped facial reactions!"
            if cv_res["facecam_corner"]
            else ""
        )
        prompt = (
            f"Analyze this live GTA / gaming stream frame from {streamer_name or 'streamer'}. {cam_hint}\n"
            f"Chat hype right now: {chat_context or 'normal'}.\n"
            "Score 0-100 for how clip-worthy/exciting the gameplay & streamer reaction are "
            "(0=menu/AFK/commercial break, 40=normal gameplay, 75=police chase/crash/shocked facecam/gunfight, 95=explosion/WASTED/insane stunt/screaming reaction).\n"
            "ONLY set is_commercial_break=true if the full screen is literally a Twitch/YouTube 'Commercial break in progress' slate or 'Stream offline / starting soon' slate with no game visible.\n"
            'Respond ONLY with compact JSON: {"visual_score": <0-100>, "is_commercial_break": <true/false>, "event": "<2-4 words>", "summary": "<short 1 sentence>"}'
        )

        t0 = time.time()
        async with sem:
            vlm_res = await self._query_vlm_b64_images([b64], prompt)
        latency_ms = int((time.time() - t0) * 1000)

        if vlm_res is None:
            return {
                "visual_score": base_fallback_score,
                "cv_score": cv_res["cv_score"],
                "vlm_score": None,
                "has_facecam": cv_res["has_facecam"],
                "facecam_corner": cv_res["facecam_corner"],
                "facecam_score": cv_res["facecam_score"],
                "is_blocked": False,
                "latency_ms": latency_ms,
                "event": "vlm_timeout",
                "summary": f"OpenCV fallback (Motion {cv_res['motion']}%)",
            }

        self.vlm_evaluations += 1
        self.last_latency_ms = latency_ms
        self._latencies_ms.append(latency_ms)

        ev_text = str(vlm_res.get("event", "gameplay"))[:60]
        sum_text = str(vlm_res.get("summary", ""))[:220]
        vlm_score = float(max(0.0, min(100.0, vlm_res.get("visual_score", base_fallback_score))))
        flagged_ad = bool(vlm_res.get("is_commercial_break") or vlm_res.get("is_ad_or_break"))
        is_ad = self._contains_blocked_phrase(ev_text, sum_text) or (flagged_ad and vlm_score < 25.0)
        if is_ad:
            return {
                "visual_score": 0.0,
                "cv_score": cv_res["cv_score"],
                "vlm_score": 0.0,
                "has_facecam": cv_res["has_facecam"],
                "facecam_corner": cv_res["facecam_corner"],
                "facecam_score": cv_res["facecam_score"],
                "is_blocked": True,
                "latency_ms": latency_ms,
                "event": "commercial_break",
                "summary": "Commercial break / non-gameplay screen filtered",
            }

        vlm_score = float(max(0.0, min(100.0, vlm_res.get("visual_score", base_fallback_score))))
        if cv_res["has_facecam"]:
            blended = round(min(100.0, 0.65 * vlm_score + 0.20 * cv_res["cv_score"] + 0.15 * cv_res["facecam_score"]), 1)
        else:
            blended = round(min(100.0, 0.75 * vlm_score + 0.25 * cv_res["cv_score"]), 1)

        return {
            "visual_score": blended,
            "cv_score": cv_res["cv_score"],
            "vlm_score": vlm_score,
            "has_facecam": cv_res["has_facecam"],
            "facecam_corner": cv_res["facecam_corner"],
            "facecam_score": cv_res["facecam_score"],
            "is_blocked": False,
            "latency_ms": latency_ms,
            "event": ev_text,
            "summary": sum_text,
        }

    # ------------------------------------------------------------------
    # Post-Clip Audio Transcription + Deep Scene Captioning
    # ------------------------------------------------------------------
    def transcribe_clip_audio_sync(self, video_path: str) -> str:
        """Uses local CPU faster-whisper (tiny.en int8, 0 VRAM) to transcribe what the streamer yelled."""
        if not os.path.isfile(video_path):
            return ""
        try:
            from faster_whisper import WhisperModel

            with self._whisper_lock:
                if self._whisper_model is None:
                    self._whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

            # Pipe last 28 seconds mono 16kHz PCM directly to numpy (avoids PyAV metadata_errors bug)
            cmd = [
                settings.FFMPEG_PATH,
                "-y", "-hide_banner", "-loglevel", "error",
                "-sseof", "-28",
                "-i", video_path,
                "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le",
                "pipe:1",
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=20, creationflags=CREATE_NO_WINDOW)
            if res.returncode != 0 or len(res.stdout) < 3200:
                return ""
            audio_np = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self._whisper_model.transcribe(audio_np, beam_size=1, vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
            return text[:350]
        except Exception as exc:
            print(f"[Whisper] Transcription note: {exc}")
            return ""

    def _extract_clip_keyframes(self, video_path: str, facecam_corner: str) -> Tuple[List[str], bool]:
        """
        Extracts 3 keyframe composites (25%, 55%, 85% of clip) as base64 JPEGs
        and checks if any frame is a Twitch purple commercial break slate.
        """
        b64_frames: List[str] = []
        ad_detected = False
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return b64_frames, False
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            indices = [max(0, int(total * r)) for r in (0.25, 0.55, 0.85)] if total > 15 else [0]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    if self._is_twitch_ad_slate(frame):
                        ad_detected = True
                    b64 = self._build_composite_b64(frame, facecam_corner)
                    if b64:
                        b64_frames.append(b64)
        finally:
            cap.release()
        return b64_frames, ad_detected

    def _clean_and_validate_caption(
        self,
        raw_overlay: str,
        summary: str,
        streamer_name: str,
        fallback_kw: str = "",
    ) -> str:
        # Strip hashtags and quotes
        cap = re.sub(r"#\w+", "", raw_overlay or "").strip().strip('"').strip("'").strip()
        s_clean = streamer_name.strip()
        # Strip accidental other-streamer hallucinated prefix like "LIRIK xQc ..." if model echoed an unrelated handle
        other_handles = ("xqc", "rubius", "kaicenat", "lirik", "squeezie", "jynxzi", "trymacs", "zarbex")
        for h in other_handles:
            if h != s_clean.lower():
                cap = re.sub(rf"^{re.escape(s_clean)}[:\s]+{h}\b\s*", f"{s_clean} ", cap, flags=re.IGNORECASE)
                cap = re.sub(rf"^{h}[:\s]+", "", cap, flags=re.IGNORECASE)

        if not cap or cap.lower() == s_clean.lower() or len(cap) <= len(s_clean) + 3 or cap.lower().startswith("streamer:"):
            cap = f"{s_clean}: {summary[:65]}" if summary else f"{s_clean} pulls off a crazy moment!"
        elif any(v in cap.lower() for v in VAGUE_FILLER_PHRASES):
            if summary and not any(v in summary.lower() for v in VAGUE_FILLER_PHRASES):
                cap = f"{s_clean}: {summary[:65]}"
            elif fallback_kw and fallback_kw != "chat going crazy":
                cap = f"{s_clean} goes wild ({fallback_kw[:45]})"
            else:
                cap = f"{s_clean} triggers an insane gameplay moment!"

        if s_clean.lower() not in cap.lower():
            cap = f"{s_clean} {cap}"
        # Collapse accidental duplicate prefix like "xQc: xQc ..." or "LIRIK: LIRIK ..."
        cap = re.sub(rf"^{re.escape(s_clean)}\s*:?\s+{re.escape(s_clean)}\b", s_clean, cap, flags=re.IGNORECASE)
        cap = " ".join(cap.split())
        if len(cap) > 95:
            cap = cap[:95].rsplit(" ", 1)[0]
        return cap

    async def analyze_clip_and_generate_captions(
        self,
        video_path: str,
        thumb_path: str,
        streamer_name: str,
        facecam_corner: str,
        chat_keywords: List[str],
        chat_sample: List[str],
        chat_mps: float,
    ) -> Dict[str, str]:
        """
        Deep post-clip analysis combining:
        - Multi-frame commercial-break / ad rejection gate
        - Whisper speech transcript from the clip audio (only when real speech detected)
        - 3 keyframes + zoomed streamer facecam
        - Live Twitch/YouTube chat reactions & slang
        """
        transcript = await asyncio.to_thread(self.transcribe_clip_audio_sync, video_path)
        fallback_kw = ", ".join(chat_keywords[:4]) if chat_keywords else "chat going crazy"
        fallback = {
            "overlay_caption": f"{streamer_name} shocked chat during wild gameplay!",
            "title": f"{streamer_name} INSANE Moment! ({fallback_kw})",
            "summary": f"{streamer_name} triggered a {chat_mps:.1f} msg/s chat spike ({fallback_kw}).",
            "caption": f"{streamer_name} went crazy! 🔥 ({fallback_kw}) #GTA6 #GTAVI #Gaming #Viral",
            "transcript": transcript,
        }

        b64_frames, ad_slate_found = await asyncio.to_thread(self._extract_clip_keyframes, video_path, facecam_corner)
        if ad_slate_found:
            return {"reject_reason": "Twitch commercial break slate detected in clip frames"}

        if not b64_frames and thumb_path and os.path.isfile(thumb_path):
            img = cv2.imread(thumb_path)
            if img is not None:
                if self._is_twitch_ad_slate(img):
                    return {"reject_reason": "Twitch commercial break slate detected in thumbnail"}
                b64 = self._build_composite_b64(img, facecam_corner)
                if b64:
                    b64_frames.append(b64)

        if not b64_frames or settings.VLM_PROVIDER == "opencv_only" or not self.is_server_healthy():
            return fallback

        chat_lines = " | ".join(chat_sample[-5:]) if chat_sample else fallback_kw
        speech_line = f'Streamer actually said: "{transcript}"\n' if len(transcript.strip()) >= 5 else ""
        prompt = (
            f"You are a selective viral TikTok/Shorts gaming clip editor evaluating a clip from {streamer_name}.\n"
            f"Streamer: {streamer_name}\n"
            f"{speech_line}"
            f"Chat burst ({chat_mps:.1f} msg/s): {chat_lines}\n"
            "First rate excitement_score (0-100) and check is_mundane_slop:\n"
            "- Set is_mundane_slop=true and excitement_score < 45 if the clip ONLY shows routine walking, casual driving, stopping at a gas station, "
            "talking to chat/teammates, discussing strategy/plans, navigating a menu/map, or low-stakes dialogue with NO crash, explosion, shootout, heist chaos, funny fail, clutch play, or shocked/screaming facecam reaction.\n"
            "- ONLY set is_mundane_slop=false and excitement_score >= 60 if something genuinely entertaining, intense, chaotic, funny, or emotional actually happens.\n"
            f"If worth clipping, write a punchy 4-9 word viral hook starting with '{streamer_name}:' describing the climax (NOT dry wording like '{streamer_name} is in the middle of...').\n"
            "ONLY set is_commercial_break=true if ALL frames are literally a Twitch commercial break slate or offline card with no game visible.\n"
            "Respond ONLY with valid JSON:\n"
            '{"is_commercial_break": <true/false>, "is_mundane_slop": <true/false>, "excitement_score": <0-100>, '
            f'"overlay_caption": "<4-9 word punchy hook starting with {streamer_name}:>", '
            f'"title": "<catchy Shorts title with {streamer_name}>", '
            '"summary": "<1-2 sentence breakdown of the exact hype moment>", '
            '"caption": "<social post caption with #GTA6 #GTAVI #Gaming hashtags>"}'
        )

        sem = self._get_semaphore()
        async with sem:
            res = await self._query_vlm_b64_images(b64_frames[:3], prompt, max_tokens=280)

        if not res:
            return fallback

        raw_overlay = str(res.get("overlay_caption") or "").strip()
        raw_title = str(res.get("title") or "").strip()
        raw_summary = str(res.get("summary") or "").strip()
        excitement = float(max(0.0, min(100.0, res.get("excitement_score", 70.0))))

        flagged_ad = bool(res.get("is_commercial_break") or res.get("is_ad_or_break"))
        if self._contains_blocked_phrase(raw_overlay, raw_title, raw_summary) or (
            flagged_ad and not any(k in raw_summary.lower() for k in ("car", "driv", "shoot", "walk", "run", "fly", "game", "character", "police", "gta"))
        ):
            return {"reject_reason": f"Non-gameplay / ad break detected ({raw_summary[:80]})"}

        joined_low = f"{raw_overlay} {raw_summary}".lower()
        is_slop = (
            bool(res.get("is_mundane_slop"))
            or excitement < 52.0
            or any(p in joined_low for p in MUNDANE_SLOP_PHRASES)
        )
        if is_slop:
            return {"reject_reason": f"Mundane / low-action clip filtered (excite={excitement:.0f}: {raw_summary[:75]})"}

        summary = raw_summary or fallback["summary"]
        overlay = self._clean_and_validate_caption(raw_overlay, summary, streamer_name, fallback_kw=fallback_kw)
        title = raw_title or overlay
        if streamer_name.lower() not in title.lower():
            title = f"{streamer_name} — {title}"

        caption = str(res.get("caption") or "").strip() or f"{overlay} 🔥 #GTA6 #GTAVI #Gaming #Viral"

        return {
            "overlay_caption": overlay,
            "title": title[:110],
            "summary": summary[:280],
            "caption": caption[:300],
            "transcript": transcript,
            "excitement_score": excitement,
        }

    async def _query_vlm_b64_images(
        self,
        b64_images: List[str],
        prompt: str,
        max_tokens: int = 180,
    ) -> Optional[Dict[str, Any]]:
        content_parts: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in b64_images
            if b64
        ]
        if not content_parts:
            return None
        content_parts.append({"type": "text", "text": prompt})

        url = f"{settings.VLM_API_URL.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if settings.VLM_API_KEY:
            headers["Authorization"] = f"Bearer {settings.VLM_API_KEY}"

        payload = {
            "model": settings.VLM_MODEL_NAME or "local-vlm",
            "messages": [{"role": "user", "content": content_parts}],
            "temperature": 0.25,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                r = await client.post(url, json=payload, headers=headers)
                if r.status_code != 200:
                    self._last_error = f"VLM HTTP {r.status_code}: {r.text[:160]}"
                    return None
                data = r.json()
                text = data["choices"][0]["message"]["content"].strip()
                return self._extract_json_object(text)
        except Exception as exc:
            self._last_error = str(exc)
            return None

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        num_match = re.search(r"\b(\d{1,3})\b", cleaned)
        score = min(100, int(num_match.group(1))) if num_match else 45
        return {"visual_score": score, "event": "vlm_text", "summary": cleaned[:180]}


vision_manager = VisionModelManager()
