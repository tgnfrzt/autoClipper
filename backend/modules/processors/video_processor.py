import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
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


def _classify_cam_zone(cx_norm: float, cy_norm: float) -> str:
    horiz = "left" if cx_norm < 0.5 else "right"
    if cy_norm < 0.36:
        vert = "top"
    elif cy_norm <= 0.64:
        vert = "mid"
    else:
        vert = "bottom"
    return f"{vert}_{horiz}"


def _detect_facecam_box(
    video_path: str,
    width: int,
    height: int,
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
    """
    Accurately detects a stationary streamer webcam overlay across multiple frames using:
    1. Multi-frame frontal face clustering along edge zones (including mid-left/mid-right webcams)
    2. Tight spatial stability (eliminates moving gameplay NPCs / car wheels / HUD circles)
    3. Human skin-tone presence check in YCrCb space
    4. Temporal mean-frame rectangular border snapping so the crop hugs the webcam cleanly.

    Returns (corner_label, (cw, ch, cx, cy)) or (None, None) if no real facecam is found.
    """
    if width <= 0 or height <= 0:
        return None, None
    try:
        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if cascade_cls is None or not hasattr(cv2, "data"):
            return None, None
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.isfile(cascade_path):
            return None, None
        face_cascade = cascade_cls(cascade_path)
        if face_cascade.empty():
            return None, None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_indices = (
            [max(0, int(total_frames * r)) for r in (0.15, 0.32, 0.50, 0.68, 0.85)]
            if total_frames > 15
            else [0, 8, 16, 24, 32]
        )

        sampled_frames: List[np.ndarray] = []
        candidates: List[Tuple[int, float, float, float, float, float]] = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            fidx = len(sampled_frames)
            sampled_frames.append(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(34, 34))
            for (fx, fy, fw, fh) in faces:
                cx_norm = (fx + fw * 0.5) / width
                cy_norm = (fy + fh * 0.5) / height
                w_norm = fw / width
                h_norm = fh / height
                # Streamer webcams live on the left/right edges (or corners), not dead-center gameplay
                if not (cx_norm <= 0.26 or cx_norm >= 0.74):
                    continue
                if w_norm > 0.22 or h_norm > 0.32:
                    continue
                ix0, iy0 = int(fx + fw * 0.15), int(fy + fh * 0.15)
                ix1, iy1 = int(fx + fw * 0.85), int(fy + fh * 0.85)
                roi = ycrcb[iy0:iy1, ix0:ix1]
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
                candidates.append((fidx, cx_norm, cy_norm, w_norm, h_norm, skin_ratio))

        cap.release()
        if len(sampled_frames) < 3 or not candidates:
            return None, None

        # Find the densest spatially consistent cluster across unique frames
        best_cluster: List[Tuple[int, float, float, float, float, float]] = []
        for c in candidates:
            nearby = [x for x in candidates if abs(x[1] - c[1]) < 0.05 and abs(x[2] - c[2]) < 0.05]
            by_frame: Dict[int, Tuple[int, float, float, float, float, float]] = {}
            for x in nearby:
                if x[0] not in by_frame or x[5] > by_frame[x[0]][5]:
                    by_frame[x[0]] = x
            if len(by_frame) > len(best_cluster):
                best_cluster = list(by_frame.values())

        if len(best_cluster) < 3:
            return None, None

        mcx = float(np.median([x[1] for x in best_cluster]))
        mcy = float(np.median([x[2] for x in best_cluster]))
        mfh = float(np.median([x[4] for x in best_cluster]))
        stdx = float(np.std([x[1] for x in best_cluster]))
        stdy = float(np.std([x[2] for x in best_cluster]))
        max_skin = max(x[5] for x in best_cluster)

        # Webcam overlay does not walk across screen between keyframes
        if stdx > 0.022 or stdy > 0.022 or max_skin < 0.20:
            return None, None

        # Snap to stationary picture-in-picture webcam edges using temporal mean frame
        mean_gray = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in sampled_frames], axis=0)
        is_left = mcx < 0.5
        sx0, sx1 = (0, int(width * 0.24)) if is_left else (int(width * 0.76), width)
        row_diff = np.mean(np.abs(np.diff(mean_gray[:, sx0:sx1], axis=0)), axis=1)
        yb0, yb1 = max(0, int((mcy - 0.16) * height)), min(height, int((mcy + 0.16) * height))
        col_diff = np.mean(np.abs(np.diff(mean_gray[yb0:yb1, :], axis=1)), axis=0)

        t0 = max(2, int((mcy - 0.24) * height))
        t1 = max(t0 + 2, int((mcy - max(0.03, mfh * 0.45)) * height))
        b0 = min(height - 4, int((mcy + max(0.03, mfh * 0.45)) * height))
        b1 = max(b0 + 2, min(height - 2, int((mcy + 0.26) * height)))

        top_seg = row_diff[t0:t1]
        bot_seg = row_diff[b0:b1]
        top_peak = float(np.max(top_seg)) if top_seg.size else 0.0
        bot_peak = float(np.max(bot_seg)) if bot_seg.size else 0.0
        top_y = (t0 + int(np.argmax(top_seg))) if top_peak >= 12.0 else max(0, int((mcy - 0.13) * height))
        bot_y = (b0 + int(np.argmax(bot_seg))) if bot_peak >= 12.0 else min(height, int((mcy + 0.13) * height))

        if is_left:
            r0, r1 = int(width * 0.14), int(width * 0.30)
            side_seg = col_diff[r0:r1]
            side_peak = float(np.max(side_seg)) if side_seg.size else 0.0
            right_x = (r0 + int(np.argmax(side_seg))) if side_peak >= 12.0 else int(width * 0.22)
            left_x = 0
            l1 = max(2, int(mcx * width - width * 0.03))
            if l1 > 6 and float(np.max(col_diff[0:l1])) >= 18.0:
                left_x = int(np.argmax(col_diff[0:l1]))
        else:
            l0, l1 = int(width * 0.70), int(width * 0.86)
            side_seg = col_diff[l0:l1]
            side_peak = float(np.max(side_seg)) if side_seg.size else 0.0
            left_x = (l0 + int(np.argmax(side_seg))) if side_peak >= 12.0 else int(width * 0.78)
            right_x = width

        # Require at least one genuine static PIP border unless position stability is ultra-tight
        if max(top_peak, bot_peak, side_peak) < 12.0 and (stdx > 0.012 or stdy > 0.012):
            return None, None

        cw = max(120, right_x - left_x)
        # Prevent weaker chat-box horizontal lines below the webcam from stretching height too far
        if bot_peak < 25.0 and (bot_y - top_y) > int(cw * 0.68):
            bot_y = min(height, top_y + int(cw * 0.66))

        ch = max(90, bot_y - top_y)
        # Ensure even integer bounds inside frame for ffmpeg crop
        cx = max(0, min(width - cw, left_x) // 2 * 2)
        cy = max(0, min(height - ch, top_y) // 2 * 2)
        cw = min(width - cx, cw // 2 * 2)
        ch = min(height - cy, ch // 2 * 2)
        if cw < 100 or ch < 80:
            return None, None

        return _classify_cam_zone(mcx, mcy), (cw, ch, cx, cy)
    except Exception:
        return None, None


def _corner_crop_expr(corner: str, width: int, height: int) -> Tuple[int, int, int, int]:
    cw = max(160, int(width * 0.24) // 2 * 2)
    ch = max(120, int(height * 0.30) // 2 * 2)
    is_right = "right" in corner
    cx = max(0, (width - cw) // 2 * 2) if is_right else 0
    if corner.startswith("mid_"):
        cy = max(0, ((height - ch) // 2) // 2 * 2)
    elif corner.startswith("bottom_"):
        cy = max(0, (height - ch) // 2 * 2)
    else:
        cy = 0
    return cw, ch, cx, cy


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
    with a zoomed-out 16:9 gameplay crop, subtle blurred mirror top/bottom margins,
    accurate multi-frame facecam cropping, and a burned-in streamer caption banner.
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
    cam_box: Optional[Tuple[int, int, int, int]] = None
    if mode == "smart_gaming_split":
        if cam_pref in ("top_left", "top_right", "mid_left", "mid_right", "bottom_left", "bottom_right"):
            detected_corner = cam_pref
            cam_box = _corner_crop_expr(cam_pref, src_w, src_h)
        elif cam_pref == "auto":
            detected_corner, cam_box = _detect_facecam_box(raw_path, src_w, src_h)

    target_w, target_h = (src_w, src_h) if mode == "original_16_9" else (1080, 1920)
    wm_path = get_watermark_path()

    caption_y = 560
    if mode == "original_16_9":
        filter_complex = f"[0:v]setsar=1[base]"
        caption_y = max(90, int(target_h * 0.14))
    elif mode == "smart_gaming_split" and detected_corner and cam_box:
        cw, ch, cx, cy = cam_box
        # Keep 76% of the 16:9 horizontal width (zoomed out vs 46.7% before) and fill margins with blurred mirror
        gw = max(320, int(src_w * 0.76) // 2 * 2)
        gx = max(0, ((src_w - gw) // 2) // 2 * 2)
        fg_h = max(360, min(1160, (int(round(1080 * src_h / gw)) // 2) * 2))

        af_fg = (
            ",eq=contrast=1.02:brightness=0.005:saturation=1.03,noise=alls=1:allf=t+u"
            if use_af
            else ""
        )
        filter_complex = (
            f"[0:v]split=3[vcam_in][vgame_bg][vgame_fg];"
            f"[vcam_in]crop={cw}:{ch}:{cx}:{cy},"
            f"scale=1080:560:force_original_aspect_ratio=increase,crop=1080:560,setsar=1[cam];"
            f"[vgame_bg]scale=270:340:force_original_aspect_ratio=increase,"
            f"crop=270:340,boxblur=14:3,eq=brightness=-0.06:saturation=1.15,scale=1080:1360,setsar=1[gbg];"
            f"[vgame_fg]crop={gw}:{src_h}:{gx}:0,"
            f"scale=1080:{fg_h},setsar=1{af_fg}[gfg];"
            f"[gbg][gfg]overlay=(W-w)/2:(H-h)/2[game];"
            f"[cam][game]vstack=inputs=2,"
            f"drawbox=x=0:y=558:w=1080:h=4:color=#6366f1@0.95:t=fill[base]"
        )
        caption_y = 560
    elif mode == "smart_gaming_split":
        # Pure 16:9 gameplay without facecam: show 76% horizontal view + clean blurred mirror margins
        gw = max(320, int(src_w * 0.76) // 2 * 2)
        gx = max(0, ((src_w - gw) // 2) // 2 * 2)
        fg_h = max(360, min(1320, (int(round(1080 * src_h / gw)) // 2) * 2))
        af_fg = (
            ",eq=contrast=1.02:brightness=0.006:saturation=1.04,noise=alls=1:allf=t+u"
            if use_af
            else ""
        )
        filter_complex = (
            f"[0:v]split=2[vbg][vfg];"
            f"[vbg]scale=270:480:force_original_aspect_ratio=increase,"
            f"crop=270:480,boxblur=14:3,eq=brightness=-0.06:saturation=1.15,scale={target_w}:{target_h},setsar=1[bg];"
            f"[vfg]crop={gw}:{src_h}:{gx}:0,"
            f"scale=1080:{fg_h},setsar=1{af_fg}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
        )
        caption_y = max(240, (target_h - fg_h) // 2 - 90)
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


def _extract_audio_rms_per_sec(video_path: str, n_sec: int) -> List[float]:
    if n_sec <= 0:
        return []
    create_flags = 0x08000000 if os.name == "nt" else 0
    try:
        res = subprocess.run(
            [
                settings.FFMPEG_PATH,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "8000",
                "-f",
                "s16le",
                "pipe:1",
            ],
            capture_output=True,
            timeout=15,
            creationflags=create_flags,
        )
        if res.returncode != 0 or len(res.stdout) < 1600:
            return [0.0] * n_sec
        samples = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32)
        sr = 8000
        out: List[float] = []
        for s in range(n_sec):
            seg = samples[s * sr : (s + 1) * sr]
            out.append(float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0)
        return out
    except Exception:
        return [0.0] * n_sec


def _normalize_series(vals: List[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, 5))
    hi = float(np.percentile(arr, 95))
    if hi - lo < 1e-5:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo) * 100.0, 0.0, 100.0)


def auto_trim_to_action_start(
    video_path: str,
    min_duration_sec: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyzes a raw captured clip (typically 45-60s) backwards from its hype climax using
    1fps motion + scene-cut detection and per-second audio RMS energy, then trims dead/unrelated
    intro before the action started (preserving ~3.5s pre-roll context and >= min_duration_sec).
    """
    if not os.path.isfile(video_path):
        return {"trimmed": False, "trim_start": 0.0, "duration": 0.0, "reason": "missing_file"}

    info = probe_video(video_path)
    dur = float(info.get("duration") or 0.0)
    min_dur = max(18, int(min_duration_sec or settings.MIN_CLIP_DURATION_SEC))
    if not settings.AUTO_TRIM_ENABLED or dur <= float(min_dur) + 5.0:
        return {"trimmed": False, "trim_start": 0.0, "duration": dur, "reason": "kept_full"}

    n_sec = int(dur)
    motion_vals = [0.0] * n_sec
    hist_corrs = [1.0] * n_sec

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        prev_gray: Optional[np.ndarray] = None
        prev_hist: Optional[np.ndarray] = None
        for t in range(n_sec):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_vals[t] = float(np.mean(diff)) * 2.2 + float(np.mean(diff > 25)) * 80.0
                if prev_hist is not None:
                    hist_corrs[t] = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
            prev_gray = gray
            prev_hist = hist
        cap.release()

    audio_vals = _extract_audio_rms_per_sec(video_path, n_sec)
    motion_norm = _normalize_series(motion_vals)
    audio_norm = _normalize_series(audio_vals)
    has_audio = float(np.max(audio_norm)) > 5.0
    activity = (0.52 * audio_norm + 0.48 * motion_norm) if has_audio else motion_norm
    if activity.size < min_dur or float(np.max(activity)) < 8.0:
        return {"trimmed": False, "trim_start": 0.0, "duration": dur, "reason": "flat_signal"}

    # Locate action climax using 3-second rolling window in latter 70% of clip
    peak_lo = max(4, int(n_sec * 0.30))
    peak_hi = max(peak_lo + 1, n_sec - 2)
    best_t = peak_hi - 1
    best_val = -1.0
    for t in range(peak_lo, peak_hi):
        w_val = float(np.mean(activity[max(0, t - 1) : min(n_sec, t + 2)]))
        if w_val > best_val:
            best_val = w_val
            best_t = t

    max_trim = max(0.0, dur - float(min_dur))
    trim_start = 0.0
    reason = "action_starts_early"

    # Priority A: hard scene cut / pause-menu exit between t=4 and (best_t - 4) with low prior activity
    cut_limit = min(int(max_trim) + 1, max(4, best_t - 4))
    chosen_cut: Optional[int] = None
    for t in range(4, cut_limit):
        is_hard_cut = hist_corrs[t] < 0.62 and motion_vals[t] > 32.0
        was_static_menu = (t >= 2 and float(np.mean(motion_vals[t - 2 : t])) < 3.5 and motion_vals[t] > 20.0)
        if (is_hard_cut or was_static_menu) and float(np.mean(activity[:t])) < 0.60 * best_val:
            chosen_cut = t

    if chosen_cut is not None:
        trim_start = max(0.0, float(chosen_cut) - 1.5)
        reason = "scene_cut"
    else:
        # Priority B: walk backward from climax to find earliest onset above quiet lead-in baseline
        baseline = float(np.percentile(activity[:best_t], 25)) if best_t >= 4 else 0.0
        onset_thresh = baseline + 0.25 * max(0.0, best_val - baseline)
        onset_t = best_t
        quiet_streak = 0
        for t in range(best_t, -1, -1):
            local_avg = float(np.mean(activity[max(0, t - 1) : min(n_sec, t + 1)]))
            if local_avg >= onset_thresh:
                onset_t = t
                quiet_streak = 0
            else:
                quiet_streak += 1
                if quiet_streak >= 3:
                    break
        if onset_t >= 5:
            trim_start = max(0.0, float(onset_t) - 3.5)
            reason = "action_onset"

    trim_start = round(min(trim_start, max_trim), 2)
    if trim_start < 4.0:
        return {"trimmed": False, "trim_start": 0.0, "duration": dur, "reason": "action_starts_early"}

    p = Path(video_path)
    tmp_out = str(p.with_name(f"{p.stem}_autotrim.mp4"))
    try:
        # Try fast keyframe stream copy first
        run_ffmpeg(
            ["-ss", f"{trim_start:.2f}", "-i", video_path, "-c", "copy", "-movflags", "+faststart", tmp_out],
            timeout=60,
        )
        new_dur = float(probe_video(tmp_out).get("duration") or 0.0)
        expected_dur = dur - trim_start
        # If keyframe alignment made the clip way too short or missed trimming by >4.5s, perform accurate cut
        if new_dur < float(min_dur) - 2.0 or (new_dur - expected_dur) > 4.5:
            if os.path.isfile(tmp_out):
                os.remove(tmp_out)
            run_ffmpeg(
                [
                    "-ss",
                    f"{trim_start:.2f}",
                    "-i",
                    video_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    tmp_out,
                ],
                timeout=120,
            )
            new_dur = float(probe_video(tmp_out).get("duration") or 0.0)

        if os.path.isfile(tmp_out) and new_dur >= float(min_dur) - 2.0:
            os.replace(tmp_out, video_path)
            return {
                "trimmed": True,
                "trim_start": trim_start,
                "original_duration": dur,
                "duration": new_dur,
                "reason": reason,
            }
    except Exception:
        pass
    finally:
        if os.path.isfile(tmp_out):
            try:
                os.remove(tmp_out)
            except Exception:
                pass

    return {"trimmed": False, "trim_start": 0.0, "duration": dur, "reason": "fallback_full"}
