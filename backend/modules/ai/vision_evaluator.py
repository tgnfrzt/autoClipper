import asyncio
import base64
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np

from backend.core.config import MODELS_DIR, save_settings_to_env, settings

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

VLM_PRESETS: Dict[str, Dict[str, Any]] = {
    "smolvlm-500m": {
        "id": "smolvlm-500m",
        "name": "SmolVLM-500M-Instruct (Q8_0)",
        "size_label": "~520 MB (Ultra-Fast • <1GB VRAM)",
        "recommended_for": "GTX 1650 / 4GB VRAM — Evaluates frames in ~200ms while watching 15+ streams",
        "model_filename": "SmolVLM-500M-Instruct-Q8_0.gguf",
        "mmproj_filename": "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
        "model_url": "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf",
        "mmproj_url": "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
    },
    "qwen2.5-vl-3b": {
        "id": "qwen2.5-vl-3b",
        "name": "Qwen2.5-VL-3B-Instruct (Q4_K_M)",
        "size_label": "~2.6 GB (High Detail • Fits 4GB VRAM)",
        "recommended_for": "Higher visual accuracy for GTA VI HUD/events & viral title generation",
        "model_filename": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "mmproj_filename": "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
        "model_url": "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "mmproj_url": "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
    },
}


class VisionModelManager:
    """
    Manages the lightweight local VLM (via llama-server.exe or any OpenAI-compatible API)
    combined with ultra-fast OpenCV visual action analysis for the 20% visual score.
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self._prev_frames: Dict[str, np.ndarray] = {}
        self._vlm_semaphore: Optional[asyncio.Semaphore] = None
        self._download_state: Dict[str, Any] = {
            "active": False,
            "preset_id": "",
            "file": "",
            "progress_pct": 0.0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "error": "",
        }
        self._last_error: str = ""

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._vlm_semaphore is None:
            self._vlm_semaphore = asyncio.Semaphore(1)
        return self._vlm_semaphore

    def list_presets(self) -> List[Dict[str, Any]]:
        items = []
        for pid, p in VLM_PRESETS.items():
            pdir = MODELS_DIR / pid
            mpath = pdir / p["model_filename"]
            mmpath = pdir / p["mmproj_filename"]
            installed = mpath.is_file() and mpath.stat().st_size > 10_000_000 and mmpath.is_file() and mmpath.stat().st_size > 5_000_000
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
        # Try /health (llama.cpp) or /models (OpenAI-compatible)
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
        model_exists = os.path.isfile(settings.VLM_MODEL_PATH)
        mmproj_exists = os.path.isfile(settings.VLM_MMPROJ_PATH)
        return {
            "provider": settings.VLM_PROVIDER,
            "api_url": settings.VLM_API_URL,
            "model_name": settings.VLM_MODEL_NAME,
            "model_path": settings.VLM_MODEL_PATH,
            "mmproj_path": settings.VLM_MMPROJ_PATH,
            "model_exists": model_exists,
            "mmproj_exists": mmproj_exists,
            "llama_server_exe": settings.LLAMA_SERVER_EXE,
            "llama_server_found": os.path.isfile(settings.LLAMA_SERVER_EXE),
            "port": settings.VLM_PORT,
            "pid": running_pid,
            "healthy": healthy,
            "last_error": self._last_error,
            "download": dict(self._download_state),
            "presets": self.list_presets(),
        }

    def start_local_server(self) -> Tuple[bool, str]:
        if settings.VLM_PROVIDER != "llamacpp":
            return True, f"Using external provider '{settings.VLM_PROVIDER}'."

        if self.is_server_healthy():
            return True, "Local VLM server is already online."

        exe = settings.LLAMA_SERVER_EXE
        if not os.path.isfile(exe):
            msg = f"llama-server.exe not found at {exe}. Set LLAMA_SERVER_EXE in Configure or switch to OpenAI/Ollama API."
            self._last_error = msg
            return False, msg

        if not os.path.isfile(settings.VLM_MODEL_PATH):
            msg = f"Model GGUF not found at {settings.VLM_MODEL_PATH}. Download a preset in Configure first."
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

        # Wait briefly for readiness (small 500M model loads in ~2-4s)
        for _ in range(30):
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
                    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
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
    # Fast OpenCV Visual Scoring + VLM Semantic Evaluation
    # ------------------------------------------------------------------
    def compute_opencv_action_score(self, stream_id: str, image_path: str) -> Dict[str, Any]:
        """
        Ultra-fast (<5ms) visual excitement heuristic:
        - Frame-to-frame motion intensity (explosions, fast driving, camera shake)
        - Sudden luminance flash (gunfire, explosions, scene cuts)
        - High-saturation red/alert screen flashes (GTA Wasted / Wanted / Combat)
        """
        img = cv2.imread(image_path)
        if img is None:
            return {"cv_score": 0.0, "motion": 0.0, "flash": 0.0, "red_alert": 0.0}

        small = cv2.resize(img, (256, 144))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        motion_score = 0.0
        flash_score = 0.0
        prev = self._prev_frames.get(stream_id)
        if prev is not None and prev.shape == gray.shape:
            diff = cv2.absdiff(gray, prev)
            mean_diff = float(np.mean(diff))
            motion_pct = float(np.mean(diff > 28)) * 100.0
            motion_score = min(100.0, mean_diff * 2.2 + motion_pct * 0.8)
            lum_delta = abs(float(np.mean(gray)) - float(np.mean(prev)))
            flash_score = min(100.0, lum_delta * 2.8)
        self._prev_frames[stream_id] = gray

        # Red intensity check (explosions, health critical, GTA Wasted tint)
        b, g, r = cv2.split(small.astype(np.float32))
        red_excess = np.maximum(0.0, r - 0.55 * (g + b))
        red_score = min(100.0, float(np.mean(red_excess)) * 2.5)

        cv_score = min(100.0, 0.55 * motion_score + 0.30 * flash_score + 0.15 * red_score)
        return {
            "cv_score": round(cv_score, 1),
            "motion": round(motion_score, 1),
            "flash": round(flash_score, 1),
            "red_alert": round(red_score, 1),
        }

    async def evaluate_frame(
        self,
        stream_id: str,
        image_path: str,
        streamer_name: str = "",
        chat_context: str = "",
        force_vlm: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates a stream frame for the 20% visual component.
        Combines OpenCV visual action metrics with local VLM semantic inspection.
        """
        if not os.path.isfile(image_path):
            return {"visual_score": 0.0, "cv_score": 0.0, "vlm_score": None, "event": "no_frame", "summary": ""}

        cv_res = self.compute_opencv_action_score(stream_id, image_path)

        if settings.VLM_PROVIDER == "opencv_only":
            return {
                "visual_score": cv_res["cv_score"],
                "cv_score": cv_res["cv_score"],
                "vlm_score": None,
                "event": "visual_action" if cv_res["cv_score"] >= 55 else "gameplay",
                "summary": f"Motion {cv_res['motion']}% | Flash {cv_res['flash']}%",
            }

        # Only call VLM if server is reachable (and either forced or periodic)
        if not self.is_server_healthy():
            return {
                "visual_score": cv_res["cv_score"],
                "cv_score": cv_res["cv_score"],
                "vlm_score": None,
                "event": "vlm_offline_fallback",
                "summary": f"OpenCV fallback (Motion {cv_res['motion']}%)",
            }

        sem = self._get_semaphore()
        if sem.locked() and not force_vlm:
            return {
                "visual_score": cv_res["cv_score"],
                "cv_score": cv_res["cv_score"],
                "vlm_score": None,
                "event": "vlm_busy_fallback",
                "summary": f"OpenCV fast-path (Motion {cv_res['motion']}%)",
            }

        async with sem:
            vlm_res = await self._query_vlm_image(
                image_path=image_path,
                prompt=(
                    f"You are analyzing a live gaming livestream frame from {streamer_name or 'a streamer'} playing GTA VI / GTA.\n"
                    f"Recent chat hype context: {chat_context or 'normal chat'}.\n"
                    "Rate how visually exciting, chaotic, funny, or clip-worthy this frame is on a 0-100 scale "
                    "(0=menu/loading/sitting idle, 40=normal driving/walking, 75=police chase/gunfight/crash/reaction, "
                    "95=huge explosion/WASTED/insane stunt/shocked streamer).\n"
                    'Respond ONLY with compact JSON: {"visual_score": <int 0-100>, "event": "<2-4 words>", "summary": "<short 1-sentence description>"}'
                ),
            )

        if vlm_res is None:
            return {
                "visual_score": cv_res["cv_score"],
                "cv_score": cv_res["cv_score"],
                "vlm_score": None,
                "event": "vlm_timeout",
                "summary": f"OpenCV fallback (Motion {cv_res['motion']}%)",
            }

        vlm_score = float(max(0.0, min(100.0, vlm_res.get("visual_score", cv_res["cv_score"]))))
        blended_visual = round(0.70 * vlm_score + 0.30 * cv_res["cv_score"], 1)
        return {
            "visual_score": blended_visual,
            "cv_score": cv_res["cv_score"],
            "vlm_score": vlm_score,
            "event": str(vlm_res.get("event", "gameplay"))[:60],
            "summary": str(vlm_res.get("summary", ""))[:220],
        }

    async def generate_clip_metadata(
        self,
        image_paths: List[str],
        streamer_name: str,
        chat_keywords: List[str],
        chat_mps: float,
    ) -> Dict[str, str]:
        """Generates a viral clip title, summary, and social caption using the VLM."""
        fallback_kw = ", ".join(chat_keywords[:4]) if chat_keywords else "chat going crazy"
        fallback = {
            "title": f"{streamer_name} INSANE GTA VI Moment! ({fallback_kw})",
            "summary": f"Chat spiked to {chat_mps:.1f} msg/s ({fallback_kw}).",
            "caption": f"{streamer_name} went crazy in GTA VI! 🤯🔥 ({fallback_kw}) #GTA6 #GTAVI #Gaming #Viral",
        }

        valid_imgs = [p for p in image_paths if p and os.path.isfile(p)][:2]
        if not valid_imgs or settings.VLM_PROVIDER == "opencv_only" or not self.is_server_healthy():
            return fallback

        sem = self._get_semaphore()
        async with sem:
            res = await self._query_vlm_multi_images(
                image_paths=valid_imgs,
                prompt=(
                    "Describe what is happening on screen in this livestream frame in one short sentence. "
                    'Return JSON: {"title": "<4-8 word exciting headline>", "summary": "<one sentence describing what is visible>"}'
                ),
            )
        if not res:
            return fallback

        raw_title = str(res.get("title") or "").strip()
        raw_summary = str(res.get("summary") or "").strip()
        if not raw_title or raw_title.lower().startswith("streamer:"):
            title = f"{streamer_name}: {raw_summary[:55]}" if raw_summary else fallback["title"]
        else:
            title = f"{streamer_name} — {raw_title}"

        summary = raw_summary or fallback["summary"]
        caption = f"{title} 🔥 ({fallback_kw}) #GTA6 #GTAVI #Gaming #Viral"
        return {
            "title": title[:100],
            "summary": f"{summary} (Chat: {chat_mps:.1f} msg/s)"[:280],
            "caption": caption[:300],
        }

    async def _query_vlm_image(self, image_path: str, prompt: str) -> Optional[Dict[str, Any]]:
        return await self._query_vlm_multi_images([image_path], prompt)

    async def _query_vlm_multi_images(self, image_paths: List[str], prompt: str) -> Optional[Dict[str, Any]]:
        content_parts: List[Dict[str, Any]] = []
        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                if w > 320:
                    scale = 320.0 / w
                    img = cv2.resize(img, (320, max(64, int(h * scale))))
                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            except Exception:
                continue

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
            "temperature": 0.2,
            "max_tokens": 180,
        }

        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
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
        # Heuristic parse if small VLM returned plain text with a number
        num_match = re.search(r"\b(\d{1,3})\b", cleaned)
        score = min(100, int(num_match.group(1))) if num_match else 45
        return {"visual_score": score, "event": "vlm_text", "summary": cleaned[:160]}


vision_manager = VisionModelManager()
