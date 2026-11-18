import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PIL import Image, ImageDraw, ImageFont

from backend.core.config import CLIPS_RENDERED_DIR, get_watermark_path, settings
from backend.utils.ffmpeg_runner import get_fast_encoder_args, probe_video, run_ffmpeg


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\impact.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _create_caption_overlay_png(
    caption_text: str,
    streamer_name: str = "",
    width: int = 1080,
    height: int = 1920,
    y_center: int = 620,
) -> Optional[str]:
    """
    Generates a crisp 1080x1920 transparent PNG caption card with the streamer's name
    highlighted in gold and dark pill backing so it reads cleanly over gameplay.
    """
    clean = " ".join((caption_text or "").split()).strip()
    if not clean:
        return None

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_bold_font(42)

    max_text_w = int(width * 0.84)
    words = clean.split(" ")
    lines: List[str] = []
    cur_line = ""
    for w in words:
        test = f"{cur_line} {w}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) <= max_text_w or not cur_line:
            cur_line = test
        else:
            lines.append(cur_line)
            cur_line = w
    if cur_line:
        lines.append(cur_line)
    lines = lines[:3]

    line_heights: List[int] = []
    line_widths: List[int] = []
    for ln in lines:
        bb = draw.textbbox((0, 0), ln, font=font)
        line_widths.append(bb[2] - bb[0])
        line_heights.append(max(46, (bb[3] - bb[1]) + 10))

    line_gap = 8
    total_text_h = sum(line_heights) + line_gap * max(0, len(lines) - 1)
    box_w = min(width - 64, max(line_widths) + 64)
    box_h = total_text_h + 44
    box_x0 = (width - box_w) // 2
    box_y0 = max(32, min(height - box_h - 32, y_center - box_h // 2))
    box_x1 = box_x0 + box_w
    box_y1 = box_y0 + box_h

    # Subtle shadow + pill container + indigo accent outline
    draw.rounded_rectangle(
        (box_x0 + 4, box_y0 + 6, box_x1 + 4, box_y1 + 6),
        radius=26,
        fill=(0, 0, 0, 150),
    )
    draw.rounded_rectangle(
        (box_x0, box_y0, box_x1, box_y1),
        radius=26,
        fill=(11, 15, 25, 228),
        outline=(99, 102, 241, 255),
        width=4,
    )

    s_lower = (streamer_name or "").strip().lower()
    cur_y = box_y0 + 22
    for idx, ln in enumerate(lines):
        lw = line_widths[idx]
        cur_x = (width - lw) // 2
        tokens = ln.split(" ")
        for t_idx, tok in enumerate(tokens):
            tok_str = tok + (" " if t_idx < len(tokens) - 1 else "")
            is_streamer = bool(s_lower and s_lower in tok.lower())
            color = (251, 191, 36, 255) if is_streamer else (248, 250, 252, 255)
            draw.text((cur_x + 2, cur_y + 2), tok_str, font=font, fill=(0, 0, 0, 180))
            draw.text((cur_x, cur_y), tok_str, font=font, fill=color)
            tb = draw.textbbox((0, 0), tok_str, font=font)
            cur_x += (tb[2] - tb[0])
        cur_y += line_heights[idx] + line_gap

    fd, png_path = tempfile.mkstemp(prefix="cap_", suffix=".png")
    os.close(fd)
    img.save(png_path, "PNG")
    return png_path


def _detect_facecam_corner(video_path: str, width: int, height: int) -> Optional[str]:
    if width <= 0 or height <= 0:
        return None
    try:
        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if cascade_cls is None or not hasattr(cv2, "data"):
            return None
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.isfile(cascade_path):
            return None
        face_cascade = cascade_cls(cascade_path)
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
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
            for (fx, fy, fw, fh) in faces:
                cx = (fx + fw * 0.5) / width
                cy = (fy + fh * 0.5) / height
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
    caption_text: Optional[str] = None,
    streamer_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Formats a raw clip into Smart 9:16 Gaming Split (or blurred 9:16 / original 16:9)
    and optionally burns in a crisp on-screen caption banner naming the streamer.
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

    if mode == "original_16_9" and not (settings.BURN_IN_CAPTIONS and caption_text):
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

    detected_corner: Optional[str] = None
    if mode == "smart_gaming_split":
        if cam_pref in ("top_left", "top_right", "bottom_left", "bottom_right"):
            detected_corner = cam_pref
        elif cam_pref == "auto":
            detected_corner = _detect_facecam_corner(raw_path, src_w, src_h)

    target_w, target_h = (src_w, src_h) if mode == "original_16_9" else (1080, 1920)
    wm_path = get_watermark_path()

    caption_y = 620
    if mode == "original_16_9":
        filter_complex = f"[0:v]setsar=1[base]"
        caption_y = max(90, int(target_h * 0.14))
    elif mode == "smart_gaming_split" and detected_corner:
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
        caption_y = 620
    elif mode == "smart_gaming_split":
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
        caption_y = 250
    else:
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
        caption_y = 320

    last_v = "[base]"
    inputs = ["-i", raw_path]
    next_idx = 1

    caption_png: Optional[str] = None
    if settings.BURN_IN_CAPTIONS and caption_text:
        caption_png = _create_caption_overlay_png(
            caption_text=caption_text,
            streamer_name=streamer_name or "",
            width=target_w,
            height=target_h,
            y_center=caption_y,
        )

    try:
        if caption_png and os.path.isfile(caption_png):
            inputs.extend(["-loop", "1", "-i", caption_png])
            filter_complex += f";{last_v}[{next_idx}:v]overlay=0:0:shortest=1[v_cap]"
            last_v = "[v_cap]"
            next_idx += 1

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
        tail_args = [
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-shortest",
            out_path,
        ]
        base_cmd = ["-filter_complex", filter_complex, "-map", last_v, "-map", audio_map]
        try:
            run_ffmpeg(args + base_cmd + encoder_args + tail_args, timeout=300)
        except RuntimeError:
            if "h264_nvenc" in encoder_args:
                cpu_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
                run_ffmpeg(args + base_cmd + cpu_args + tail_args, timeout=300)
            else:
                raise
        rendered_info = probe_video(out_path)
        return {
            "output_path": out_path,
            "mode": mode,
            "facecam_used": detected_corner or "none",
            "caption_burned": bool(caption_png),
            "duration": rendered_info["duration"],
            "width": rendered_info["width"],
            "height": rendered_info["height"],
            "size_bytes": rendered_info["size_bytes"],
        }
    finally:
        if caption_png and os.path.isfile(caption_png):
            try:
                os.remove(caption_png)
            except Exception:
                pass


def ensure_discord_attachable_mp4(video_path: str, max_mb: float = 7.8) -> str:
    if not os.path.isfile(video_path):
        return video_path
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb <= max_mb:
        return video_path

    p = Path(video_path)
    preview_path = str(p.with_name(f"{p.stem}_discord.mp4"))
    info = probe_video(video_path)
    dur = max(10.0, info["duration"] or 45.0)
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
