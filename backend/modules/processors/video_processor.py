import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

from backend.core.config import CLIPS_RENDERED_DIR, get_watermark_path, settings
from backend.utils.ffmpeg_runner import get_fast_encoder_args, probe_video, run_ffmpeg


def _detect_facecam_corner(video_path: str, width: int, height: int) -> Optional[str]:
    """
    Samples 3 frames across the clip and checks the 4 corner regions with OpenCV Haar cascade
    to detect if the streamer has a corner facecam overlay.
    """
    if width <= 0 or height <= 0:
        return None
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_indices = (
            [max(0, int(total_frames * r)) for r in (0.2, 0.5, 0.8)]
            if total_frames > 10
            else [0, 15, 30]
        )

        corner_votes: Dict[str, int] = {
            "top_left": 0,
            "top_right": 0,
            "bottom_left": 0,
            "bottom_right": 0,
        }

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(36, 36))
            for (fx, fy, fw, fh) in faces:
                cx = (fx + fw * 0.5) / width
                cy = (fy + fh * 0.5) / height
                # Check if face lies in outer corner webcam zones
                is_left = cx <= 0.34
                is_right = cx >= 0.66
                is_top = cy <= 0.44
                is_bottom = cy >= 0.56
                if is_left and is_top:
                    corner_votes["top_left"] += 1
                elif is_right and is_top:
                    corner_votes["top_right"] += 1
                elif is_left and is_bottom:
                    corner_votes["bottom_left"] += 1
                elif is_right and is_bottom:
                    corner_votes["bottom_right"] += 1

        cap.release()
        best_corner, votes = max(corner_votes.items(), key=lambda kv: kv[1])
        return best_corner if votes >= 1 else None
    except Exception:
        return None


def _corner_crop_expr(corner: str, width: int, height: int) -> Tuple[int, int, int, int]:
    """Returns (cw, ch, cx, cy) for a corner facecam box."""
    cw = max(160, int(width * 0.28) // 2 * 2)
    ch = max(120, int(height * 0.32) // 2 * 2)
    if corner == "top_left":
        return cw, ch, 0, 0
    if corner == "top_right":
        return cw, ch, max(0, (width - cw) // 2 * 2), 0
    if corner == "bottom_left":
        return cw, ch, 0, max(0, (height - ch) // 2 * 2)
    if corner == "bottom_right":
        return cw, ch, max(0, (width - cw) // 2 * 2), max(0, (height - ch) // 2 * 2)
    return cw, ch, 0, 0


def render_clip_for_social(
    raw_path: str,
    clip_id: str,
    format_mode: Optional[str] = None,
    facecam_position: Optional[str] = None,
    anti_fingerprint: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Formats a raw clip according to the configured video mode:
    - smart_gaming_split (default): Top 1080x620 facecam + Bottom 1080x1300 gameplay,
      or Action-Zoomed 9:16 gameplay over blurred canvas if no corner facecam detected.
    - blurred_9_16: Full 16:9 centered on 1080x1920 blurred background (AutoPoster style).
    - original_16_9: Keeps original horizontal 16:9 layout.
    """
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Raw clip not found: {raw_path}")

    mode = (format_mode or settings.VIDEO_FORMAT_MODE or "smart_gaming_split").strip().lower()
    cam_pref = (facecam_position or settings.FACECAM_POSITION or "auto").strip().lower()
    use_af = settings.ANTI_FINGERPRINT if anti_fingerprint is None else anti_fingerprint

    out_path = str(CLIPS_RENDERED_DIR / f"{clip_id}_{mode}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    info = probe_video(raw_path)
    src_w = info["width"] or 1280
    src_h = info["height"] or 720

    if mode == "original_16_9":
        shutil.copy2(raw_path, out_path)
        rendered_info = probe_video(out_path)
        return {
            "output_path": out_path,
            "mode": mode,
            "facecam_used": "none",
            "duration": rendered_info["duration"],
            "width": rendered_info["width"],
            "height": rendered_info["height"],
            "size_bytes": rendered_info["size_bytes"],
        }

    # Determine facecam corner if smart_gaming_split
    detected_corner: Optional[str] = None
    if mode == "smart_gaming_split":
        if cam_pref in ("top_left", "top_right", "bottom_left", "bottom_right"):
            detected_corner = cam_pref
        elif cam_pref == "auto":
            detected_corner = _detect_facecam_corner(raw_path, src_w, src_h)

    target_w, target_h = 1080, 1920
    wm_path = get_watermark_path()

    if mode == "smart_gaming_split" and detected_corner:
        cw, ch, cx, cy = _corner_crop_expr(detected_corner, src_w, src_h)
        gw = max(320, int(src_w * 0.62) // 2 * 2)
        gx = max(0, ((src_w - gw) // 2) // 2 * 2)

        af_filter = (
            ",eq=contrast=1.02:brightness=0.005:saturation=1.03,noise=alls=1:allf=t+u"
            if use_af
            else ""
        )
        filter_complex = (
            f"[0:v]split=2[vcam_in][vgame_in];"
            f"[vcam_in]crop={cw}:{ch}:{cx}:{cy},"
            f"scale=1080:620:force_original_aspect_ratio=increase,crop=1080:620,setsar=1[cam];"
            f"[vgame_in]crop={gw}:{src_h}:{gx}:0,"
            f"scale=1080:1300:force_original_aspect_ratio=increase,crop=1080:1300,setsar=1[game];"
            f"[cam][game]vstack=inputs=2,"
            f"drawbox=x=0:y=618:w=1080:h=4:color=#6366f1@0.95:t=fill{af_filter}[base]"
        )
    elif mode == "smart_gaming_split":
        # Pure gameplay without facecam: action-zoomed center crop over fast blurred 9:16 canvas
        gw = max(320, int(src_w * 0.72) // 2 * 2)
        gx = max(0, ((src_w - gw) // 2) // 2 * 2)
        af_fg = (
            ",eq=contrast=1.02:brightness=0.006:saturation=1.04,noise=alls=1:allf=t+u"
            if use_af
            else ""
        )
        filter_complex = (
            f"[0:v]scale=270:480:force_original_aspect_ratio=increase,"
            f"crop=270:480,boxblur=6:2,scale={target_w}:{target_h},setsar=1[bg];"
            f"[0:v]crop={gw}:{src_h}:{gx}:0,"
            f"scale=1080:1350:force_original_aspect_ratio=decrease,setsar=1{af_fg}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
        )
    else:
        # blurred_9_16 (AutoPoster classic)
        af_fg = (
            ",scale=iw*1.02:ih*1.02,crop=iw/1.02:ih/1.02,"
            "eq=contrast=1.03:brightness=0.008:saturation=1.04,noise=alls=2:allf=t+u"
            if use_af
            else ""
        )
        filter_complex = (
            f"[0:v]scale=270:480:force_original_aspect_ratio=increase,"
            f"crop=270:480,boxblur=6:2,scale={target_w}:{target_h},setsar=1[bg];"
            f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,setsar=1{af_fg}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
        )

    last_v = "[base]"
    inputs = ["-i", raw_path]
    next_idx = 1

    if wm_path and os.path.isfile(wm_path):
        inputs.extend(["-loop", "1", "-i", wm_path])
        filter_complex += (
            f";[{next_idx}:v]format=rgba,colorchannelmixer=aa={settings.WATERMARK_ALPHA},scale=240:-1[wm];"
            f"{last_v}[wm]overlay=48:H-h-360:shortest=1[v_wm]"
        )
        last_v = "[v_wm]"
        next_idx += 1

    args = list(inputs)
    if not info["has_audio"]:
        args.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])
        audio_map = f"{next_idx}:a"
    else:
        if use_af:
            filter_complex += ";[0:a]atempo=1.005[outa]"
            audio_map = "[outa]"
        else:
            audio_map = "0:a"

    encoder_args = get_fast_encoder_args(quality=21)
    args.extend(
        ["-filter_complex", filter_complex, "-map", last_v, "-map", audio_map]
        + encoder_args
        + [
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-shortest",
            out_path,
        ]
    )

    run_ffmpeg(args, timeout=300)
    rendered_info = probe_video(out_path)
    return {
        "output_path": out_path,
        "mode": mode,
        "facecam_used": detected_corner or "none",
        "duration": rendered_info["duration"],
        "width": rendered_info["width"],
        "height": rendered_info["height"],
        "size_bytes": rendered_info["size_bytes"],
    }


def ensure_discord_attachable_mp4(video_path: str, max_mb: float = 7.8) -> str:
    """
    Discord non-boosted channels allow 8-10MB uploads. If the clip exceeds max_mb,
    creates a compressed 720p preview copy for Discord so the video always embeds cleanly.
    """
    if not os.path.isfile(video_path):
        return video_path
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb <= max_mb:
        return video_path

    p = Path(video_path)
    preview_path = str(p.with_name(f"{p.stem}_discord.mp4"))
    info = probe_video(video_path)
    dur = max(10.0, info["duration"] or 45.0)
    # Calculate target video bitrate in kbps for ~7.2 MB total
    target_total_kbps = int((7.2 * 8192) / dur)
    target_v_kbps = max(450, min(2200, target_total_kbps - 96))

    args = [
        "-i", video_path,
        "-vf", "scale=-2:720",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-b:v", f"{target_v_kbps}k",
        "-maxrate", f"{int(target_v_kbps * 1.25)}k",
        "-bufsize", f"{target_v_kbps * 2}k",
        "-c:a", "aac",
        "-b:a", "96k",
        "-movflags", "+faststart",
        preview_path,
    ]
    try:
        run_ffmpeg(args, timeout=120)
        if os.path.isfile(preview_path):
            return preview_path
    except Exception:
        pass
    return video_path
