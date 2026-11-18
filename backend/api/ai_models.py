import asyncio
import os
import subprocess
import tempfile
from typing import Any, Dict

import cv2
import numpy as np
import psutil
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from backend.core.config import MODELS_DIR, save_settings_to_env, settings
from backend.modules.ai.vision_evaluator import VLM_PRESETS, vision_manager

router = APIRouter(prefix="/api/ai", tags=["ai"])
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class DownloadModelRequest(BaseModel):
    preset_id: str
    activate: bool = True


class ActivateModelRequest(BaseModel):
    preset_id: str


def _get_hardware_telemetry() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    hw: Dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": round((vm.total - vm.available) / (1024 ** 3), 2),
        "ram_total_gb": round(vm.total / (1024 ** 3), 2),
        "gpu_name": "N/A",
        "vram_used_mb": 0,
        "vram_total_mb": 0,
        "gpu_util": 0,
    }
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1.5,
            creationflags=CREATE_NO_WINDOW,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 4:
                hw["gpu_name"] = parts[0]
                hw["vram_used_mb"] = int(float(parts[1]))
                hw["vram_total_mb"] = int(float(parts[2]))
                hw["gpu_util"] = int(float(parts[3]))
    except Exception:
        pass
    return hw


@router.get("/status")
def get_ai_status() -> Dict[str, Any]:
    return {
        "vlm": vision_manager.get_status(),
        "hardware": _get_hardware_telemetry(),
    }


@router.post("/start")
async def start_vlm_server() -> Dict[str, Any]:
    ok, msg = await asyncio.to_thread(vision_manager.start_local_server)
    return {"ok": ok, "message": msg}


@router.post("/stop")
async def stop_vlm_server() -> Dict[str, Any]:
    ok, msg = await asyncio.to_thread(vision_manager.stop_local_server)
    return {"ok": ok, "message": msg}


@router.post("/download")
def download_vlm_preset(req: DownloadModelRequest) -> Dict[str, Any]:
    ok, msg = vision_manager.download_preset(req.preset_id, activate=req.activate)
    return {"ok": ok, "message": msg}


@router.post("/activate")
async def activate_vlm_preset(req: ActivateModelRequest) -> Dict[str, Any]:
    if req.preset_id not in VLM_PRESETS:
        return {"ok": False, "message": f"Unknown preset: {req.preset_id}"}
    preset = VLM_PRESETS[req.preset_id]
    pdir = MODELS_DIR / req.preset_id
    mpath = pdir / preset["model_filename"]
    mmpath = pdir / preset["mmproj_filename"]
    if not mpath.is_file():
        return {"ok": False, "message": "Preset not downloaded yet."}
    await asyncio.to_thread(vision_manager.stop_local_server)
    save_settings_to_env(
        {
            "VLM_PROVIDER": "llamacpp",
            "VLM_MODEL_NAME": preset["name"],
            "VLM_MODEL_PATH": str(mpath),
            "VLM_MMPROJ_PATH": str(mmpath),
            "VLM_API_URL": f"http://127.0.0.1:{settings.VLM_PORT}/v1",
        }
    )
    ok, msg = await asyncio.to_thread(vision_manager.start_local_server)
    return {"ok": ok, "message": msg}


@router.post("/test-vision")
async def test_vision_evaluation(file: UploadFile | None = File(default=None)) -> Dict[str, Any]:
    """Tests the 20% visual evaluation pipeline on an uploaded image or generated synthetic frame."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tmp_path = tf.name
        if file is not None:
            content = await file.read()
            with open(tmp_path, "wb") as f:
                f.write(content)
        else:
            # Generate synthetic bright action frame
            img = np.zeros((288, 512, 3), dtype=np.uint8)
            cv2.rectangle(img, (0, 0), (512, 288), (20, 40, 120), -1)
            cv2.circle(img, (256, 144), 60, (0, 140, 255), -1)
            cv2.putText(img, "GTA VI ACTION TEST", (95, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(tmp_path, img)

        res = await vision_manager.evaluate_frame(
            stream_id="_test_frame",
            image_path=tmp_path,
            streamer_name="TestStreamer",
            chat_context="OMG CLIP IT WASTED",
            force_vlm=True,
        )
        return {"ok": True, "evaluation": res, "vlm_healthy": vision_manager.is_server_healthy()}
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
