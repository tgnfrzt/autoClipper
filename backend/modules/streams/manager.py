import asyncio
import os
import shutil
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional

from backend.core.config import CLIPS_RAW_DIR, THUMBS_DIR, settings
from backend.core.database import fetch_all_streamers, get_db, update_streamer_facecam, upsert_streamer
from backend.modules.ai.clip_detector import compute_combined_score, save_clip_record
from backend.modules.ai.vision_evaluator import vision_manager
from backend.modules.processors.video_processor import render_clip_for_social
from backend.modules.streams.chat_monitor import chat_monitor
from backend.modules.streams.discovery import (
    check_twitch_channel_live,
    check_youtube_target_live,
    discover_twitch_game_streams,
    discover_youtube_live_streams,
)
from backend.modules.streams.stream_buffer import StreamRingBuffer


class StreamManagerService:
    """
    Central runtime orchestrator:
    - Discovers top GTA VI (and fallback GTA V) streams + manual streamers, preferring facecam streams
    - Syncs live Twitch/YouTube chat monitors (80% trigger signal)
    - Maintains circular zero-decode HLS buffers for active streams
    - Evaluates keyframes + zoomed facecam reactions via Qwen2.5-VL-3B + OpenCV (20% trigger signal)
    - Runs post-clip Whisper + VLM analysis, burns in streamer caption on Smart 9:16 Split, and notifies Discord
    - Publishes live progress, statistics, and diagnostics for the Monitoring Dashboard
    """

    def __init__(self) -> None:
        self._running = False
        self._started_at: float = time.time()
        self._buffers: Dict[str, StreamRingBuffer] = {}
        self._live_state: Dict[str, Dict[str, Any]] = {}
        self._last_clip_at: Dict[str, float] = {}
        self._clipping_locks: set[str] = set()
        self._on_clip_created: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = None
        self._tasks: List[asyncio.Task] = []
        self._last_discovery_at: float = 0.0

        # Live Progress, Activity Console & Stats
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        self._activity_log: Deque[Dict[str, str]] = deque(maxlen=100)
        self._peak_chat_mps: float = 0.0
        self._clips_captured_session: int = 0

    def log_event(self, category: str, message: str) -> None:
        self._activity_log.appendleft(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "category": category.upper(),
                "message": message,
            }
        )

    def set_on_clip_callback(self, cb: Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]) -> None:
        self._on_clip_created = cb

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._started_at = time.time()
        self.log_event("SYSTEM", "AutoClipper engine started.")
        await chat_monitor.start()
        if settings.VLM_AUTO_START and settings.VLM_PROVIDER == "llamacpp":
            asyncio.create_task(asyncio.to_thread(self._boot_vlm))

        self._tasks = [
            asyncio.create_task(self._discovery_loop()),
            asyncio.create_task(self._monitor_and_evaluate_loop()),
        ]

    def _boot_vlm(self) -> None:
        ok, msg = vision_manager.start_local_server()
        self.log_event("VLM", msg if ok else f"VLM start warning: {msg}")

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()
        await chat_monitor.stop()
        for buf in list(self._buffers.values()):
            buf.stop()
        self._buffers.clear()
        self.log_event("SYSTEM", "AutoClipper engine stopped.")

    async def refresh_discovery_now(self) -> Dict[str, Any]:
        """Discovers live streams for the target game and checks enabled manual/popular streamers."""
        discovered: List[Dict[str, Any]] = []
        manual_streamers = fetch_all_streamers(enabled_only=True)

        twitch_checks = [
            check_twitch_channel_live(s["channel_id"])
            for s in manual_streamers
            if s["platform"] == "twitch"
        ]
        yt_checks = [
            check_youtube_target_live(s["url"])
            for s in manual_streamers
            if s["platform"] == "youtube"
        ]

        checked_results = await asyncio.gather(*twitch_checks, *yt_checks, return_exceptions=True)
        for res in checked_results:
            if isinstance(res, dict):
                upsert_streamer(
                    platform=res["platform"],
                    channel_id=res["channel_id"],
                    display_name=res["display_name"],
                    url=res["url"],
                    is_manual=True,
                    enabled=True,
                    is_live=res["is_live"],
                    viewer_count=res["viewer_count"],
                    title=res["title"],
                    game_name=res["game_name"],
                )
                if res["is_live"]:
                    discovered.append(res)

        if settings.AUTO_DISCOVER_TWITCH:
            tw_streams = await discover_twitch_game_streams(settings.GAME_QUERY, limit=25)
            if not tw_streams and settings.FALLBACK_GAME_QUERY:
                tw_streams = await discover_twitch_game_streams(settings.FALLBACK_GAME_QUERY, limit=20)
            for s in tw_streams:
                upsert_streamer(
                    platform="twitch",
                    channel_id=s["channel_id"],
                    display_name=s["display_name"],
                    url=s["url"],
                    is_manual=False,
                    enabled=True,
                    priority=65,
                    is_live=True,
                    viewer_count=s["viewer_count"],
                    title=s["title"],
                    game_name=s["game_name"],
                )
                discovered.append(s)

        if settings.AUTO_DISCOVER_YOUTUBE:
            yt_streams = await discover_youtube_live_streams(f"{settings.GAME_QUERY} live", limit=10)
            if not yt_streams and settings.FALLBACK_GAME_QUERY:
                yt_streams = await discover_youtube_live_streams(f"{settings.FALLBACK_GAME_QUERY} live", limit=8)
            for s in yt_streams:
                upsert_streamer(
                    platform="youtube",
                    channel_id=s["channel_id"],
                    display_name=s["display_name"],
                    url=s["url"],
                    is_manual=False,
                    enabled=True,
                    priority=60,
                    is_live=True,
                    viewer_count=s["viewer_count"],
                    title=s["title"],
                    game_name=s["game_name"],
                )
                discovered.append(s)

        self._last_discovery_at = time.time()
        await self._sync_active_buffers()
        self.log_event(
            "DISCOVERY",
            f"Discovered {len(discovered)} live streams; {len(self._buffers)} active buffers running.",
        )
        return {"live_discovered": len(discovered), "active_buffers": len(self._buffers)}

    async def _sync_active_buffers(self) -> None:
        all_rows = fetch_all_streamers(enabled_only=True, prefer_facecam=settings.PREFER_FACECAM_STREAMS)
        live_rows = [r for r in all_rows if r["is_live"] == 1]
        target_rows = live_rows[: max(1, int(settings.MAX_CONCURRENT_STREAMS))]
        target_ids = {r["id"] for r in target_rows}

        for sid in list(self._buffers.keys()):
            if sid not in target_ids or not settings.ENGINE_ENABLED:
                self._buffers.pop(sid).stop()
                self._live_state.pop(sid, None)

        if not settings.ENGINE_ENABLED:
            await chat_monitor.sync_streams([])
            return

        for row in target_rows:
            sid = row["id"]
            if sid not in self._buffers:
                buf = StreamRingBuffer(sid, row["url"], row["display_name"])
                self._buffers[sid] = buf
                self._live_state[sid] = {
                    "id": sid,
                    "platform": row["platform"],
                    "channel_id": row["channel_id"],
                    "display_name": row["display_name"],
                    "url": row["url"],
                    "viewer_count": row["viewer_count"],
                    "title": row["title"],
                    "game_name": row["game_name"],
                    "has_facecam": bool(row.get("has_facecam")),
                    "facecam_corner": row.get("facecam_corner") or "",
                    "facecam_score": 0.0,
                    "chat_score": 0.0,
                    "visual_score": 0.0,
                    "combined_score": 0.0,
                    "chat_mps": 0.0,
                    "baseline_mps": 0.0,
                    "spike_ratio": 1.0,
                    "total_messages": 0,
                    "recent_chat": [],
                    "top_keywords": [],
                    "visual_event": "buffering",
                    "visual_summary": "",
                    "visual_blocked": False,
                    "last_vlm_ms": 0,
                    "eval_count": 0,
                    "thumb_url": "",
                    "buffered_sec": 0,
                    "segments_count": 0,
                    "buffer_size_mb": 0.0,
                    "ffmpeg_pid": None,
                    "buffering": False,
                    "last_error": "",
                    "last_eval_at": 0.0,
                }
                asyncio.create_task(buf.ensure_started())

        await chat_monitor.sync_streams(target_rows)

    async def _discovery_loop(self) -> None:
        await asyncio.sleep(2)
        while self._running:
            try:
                if settings.ENGINE_ENABLED:
                    await self.refresh_discovery_now()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.log_event("ERROR", f"Discovery error: {exc}")
            await asyncio.sleep(max(30, int(settings.DISCOVERY_INTERVAL_SEC)))

    async def _monitor_and_evaluate_loop(self) -> None:
        while self._running:
            try:
                if not settings.ENGINE_ENABLED:
                    await asyncio.sleep(2)
                    continue

                now = time.time()
                eval_interval = max(2, int(settings.VISUAL_EVAL_INTERVAL_SEC))

                for sid, buf in list(self._buffers.items()):
                    if not buf.is_running() and (now - buf.started_at) > 15:
                        asyncio.create_task(buf.ensure_started())

                    tracker = chat_monitor.get_tracker(sid)
                    chat_metrics = tracker.get_metrics()
                    state = self._live_state.get(sid)
                    if not state:
                        continue

                    state["chat_score"] = chat_metrics["chat_score"]
                    state["chat_mps"] = chat_metrics["chat_mps"]
                    state["baseline_mps"] = chat_metrics["baseline_mps"]
                    state["spike_ratio"] = chat_metrics["spike_ratio"]
                    state["total_messages"] = chat_metrics["total_messages"]
                    state["recent_chat"] = chat_metrics["recent_sample"]
                    state["top_keywords"] = chat_metrics["top_keywords"]
                    state["buffered_sec"] = buf.buffered_seconds()
                    state["segments_count"] = buf.segments_count()
                    state["buffer_size_mb"] = buf.buffer_size_mb()
                    state["ffmpeg_pid"] = buf.pid()
                    state["buffering"] = buf.is_running()
                    state["last_error"] = buf.last_error

                    due_for_visual = (now - state["last_eval_at"]) >= eval_interval
                    chat_spiking = chat_metrics["chat_score"] >= 50.0 and (now - state["last_eval_at"]) >= 1.8

                    if (due_for_visual or chat_spiking) and buf.buffered_seconds() >= 6:
                        state["last_eval_at"] = now
                        frame_path = await buf.extract_latest_frame()
                        if frame_path:
                            state["thumb_url"] = f"/api/streams/thumb/{buf.safe_id}.jpg?t={int(now)}"
                            vis = await vision_manager.evaluate_frame(
                                stream_id=sid,
                                image_path=frame_path,
                                streamer_name=state["display_name"],
                                chat_context=", ".join(chat_metrics["top_keywords"][:4]),
                                force_vlm=chat_spiking,
                            )
                            prev_fc = state["has_facecam"]
                            state["has_facecam"] = vis["has_facecam"]
                            state["facecam_corner"] = vis["facecam_corner"]
                            state["facecam_score"] = vis["facecam_score"]
                            state["visual_score"] = vis["visual_score"]
                            state["visual_event"] = vis["event"]
                            state["visual_summary"] = vis["summary"]
                            state["visual_blocked"] = bool(vis.get("is_blocked"))
                            state["last_vlm_ms"] = vis["latency_ms"]
                            state["eval_count"] += 1

                            if vis.get("is_blocked"):
                                self.log_event("FILTERED", f"Skipped {state['display_name']}: {vis['summary']}")

                            if vis["has_facecam"] and not prev_fc:
                                update_streamer_facecam(sid, True, vis["facecam_corner"])
                                self.log_event(
                                    "FACECAM",
                                    f"Locked onto {state['display_name']}'s facecam ({vis['facecam_corner']}).",
                                )

                    if (
                        buf.last_error == "Could not resolve live HLS stream URL."
                        and not buf.is_running()
                        and (now - buf.started_at) > 45
                    ):
                        conn = get_db()
                        try:
                            with conn:
                                conn.execute("UPDATE streamers SET is_live = 0 WHERE id = ?", (sid,))
                        finally:
                            conn.close()
                        self.log_event("STREAM", f"Marked {state['display_name']} offline (stream URL ended).")
                        buf.stop()
                        self._buffers.pop(sid, None)
                        self._live_state.pop(sid, None)
                        continue

                    if state.get("visual_blocked") or state["visual_score"] < 12.0:
                        combined = 0.0
                    else:
                        combined = compute_combined_score(state["chat_score"], state["visual_score"])
                    state["combined_score"] = combined

                    in_cooldown = (now - self._last_clip_at.get(sid, 0.0)) < settings.CLIP_COOLDOWN_SEC
                    if (
                        combined >= settings.TRIGGER_SCORE_THRESHOLD
                        and buf.buffered_seconds() >= 15
                        and not in_cooldown
                        and sid not in self._clipping_locks
                    ):
                        self._last_clip_at[sid] = now
                        self.log_event(
                            "SPIKE",
                            f"Hype spike on {state['display_name']}! Score {combined}/100 (Chat {state['chat_score']}, Vision {state['visual_score']}, Cam {state['facecam_score']})",
                        )
                        asyncio.create_task(
                            self._execute_clip_capture(
                                stream_id=sid,
                                chat_score=state["chat_score"],
                                visual_score=state["visual_score"],
                                facecam_score=state["facecam_score"],
                                facecam_corner=state["facecam_corner"],
                                combined_score=combined,
                                chat_mps=state["chat_mps"],
                                chat_keywords=list(state["top_keywords"]),
                                chat_sample=list(state["recent_chat"]),
                                delay_sec=settings.POST_TRIGGER_DELAY_SEC,
                            )
                        )

                total_mps = round(sum(s["chat_mps"] for s in self._live_state.values()), 2)
                if total_mps > self._peak_chat_mps:
                    self._peak_chat_mps = total_mps
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.log_event("ERROR", f"Monitor loop error: {exc}")

            await asyncio.sleep(1.5)

    def _set_job_progress(self, job_id: str, streamer: str, pct: int, stage: str) -> None:
        self._active_jobs[job_id] = {
            "job_id": job_id,
            "streamer": streamer,
            "progress_pct": pct,
            "stage": stage,
            "updated_at": time.time(),
        }

    async def _execute_clip_capture(
        self,
        stream_id: str,
        chat_score: float,
        visual_score: float,
        facecam_score: float,
        facecam_corner: str,
        combined_score: float,
        chat_mps: float,
        chat_keywords: List[str],
        chat_sample: List[str],
        delay_sec: int = 0,
    ) -> Optional[Dict[str, Any]]:
        if stream_id in self._clipping_locks:
            return None
        self._clipping_locks.add(stream_id)
        job_id = f"job_{uuid.uuid4().hex[:6]}"
        state = self._live_state.get(stream_id)
        streamer_name = state["display_name"] if state else stream_id
        try:
            buf = self._buffers.get(stream_id)
            if not buf or not state:
                return None

            if delay_sec > 0:
                self._set_job_progress(job_id, streamer_name, 10, f"Waiting {delay_sec}s to capture climax...")
                await asyncio.sleep(delay_sec)

            self._set_job_progress(job_id, streamer_name, 22, "Cutting raw 30-60s HLS clip (0-decode)...")
            clip_uid = f"raw_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            raw_mp4 = str(CLIPS_RAW_DIR / f"{clip_uid}.mp4")
            clip_dur = max(30, min(60, int(settings.CLIP_DURATION_SEC)))
            ok, actual_dur, err = await asyncio.to_thread(buf.cut_clip_sync, clip_dur, raw_mp4)
            if not ok:
                self.log_event("ERROR", f"Clip cut failed for {streamer_name}: {err}")
                return None

            frame_path = await buf.extract_latest_frame()
            clip_thumb = str(THUMBS_DIR / f"{clip_uid}.jpg")
            if frame_path and os.path.isfile(frame_path):
                shutil.copy2(frame_path, clip_thumb)
            else:
                clip_thumb = ""

            self._set_job_progress(job_id, streamer_name, 45, "Transcribing audio & analyzing clip with Qwen2.5-VL...")
            meta = await vision_manager.analyze_clip_and_generate_captions(
                video_path=raw_mp4,
                thumb_path=clip_thumb,
                streamer_name=streamer_name,
                facecam_corner=facecam_corner,
                chat_keywords=chat_keywords,
                chat_sample=chat_sample,
                chat_mps=chat_mps,
            )

            if meta.get("reject_reason"):
                for p in (raw_mp4, clip_thumb):
                    if p and os.path.isfile(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                self.log_event("FILTERED", f"Auto-filtered clip from {streamer_name}: {meta['reject_reason']}")
                self._set_job_progress(job_id, streamer_name, 100, "Filtered (Commercial / Non-gameplay)")
                return None

            self._set_job_progress(job_id, streamer_name, 72, "Rendering Smart 9:16 Gaming Split + Burned-In Caption...")
            rendered_path = ""
            actual_facecam_corner = facecam_corner
            try:
                render_res = await asyncio.to_thread(
                    render_clip_for_social,
                    raw_mp4,
                    clip_uid,
                    settings.VIDEO_FORMAT_MODE,
                    settings.FACECAM_POSITION,
                    settings.ANTI_FINGERPRINT,
                    meta["overlay_caption"],
                    streamer_name,
                )
                rendered_path = render_res["output_path"]
                fc_used = str(render_res.get("facecam_used") or "")
                actual_facecam_corner = "" if fc_used in ("", "none") else fc_used
            except Exception as exc:
                self.log_event("WARN", f"Auto-render note for {streamer_name}: {exc}")

            self._set_job_progress(job_id, streamer_name, 92, "Saving clip & sending to Discord bot...")
            record = save_clip_record(
                stream_id=stream_id,
                streamer_name=streamer_name,
                platform=state["platform"],
                stream_url=state["url"],
                raw_path=raw_mp4,
                rendered_path=rendered_path,
                thumb_path=clip_thumb,
                duration_sec=actual_dur,
                chat_score=chat_score,
                visual_score=visual_score,
                facecam_score=facecam_score,
                combined_score=combined_score,
                chat_mps=chat_mps,
                chat_keywords=chat_keywords,
                facecam_corner=actual_facecam_corner,
                transcript=meta["transcript"],
                overlay_caption=meta["overlay_caption"],
                ai_title=meta["title"],
                ai_summary=meta["summary"],
                ai_caption=meta["caption"],
            )

            self._clips_captured_session += 1
            self.log_event("CLIP", f"Captured & captioned '{meta['overlay_caption']}' ({actual_dur:.0f}s)")

            if self._on_clip_created:
                asyncio.create_task(self._on_clip_created(record))

            self._set_job_progress(job_id, streamer_name, 100, "Completed!")
            return record
        finally:
            self._clipping_locks.discard(stream_id)
            asyncio.create_task(self._expire_job(job_id))

    async def _expire_job(self, job_id: str) -> None:
        await asyncio.sleep(8)
        self._active_jobs.pop(job_id, None)

    async def trigger_manual_clip(self, stream_id: str) -> Optional[Dict[str, Any]]:
        state = self._live_state.get(stream_id)
        if not state:
            return None
        self.log_event("MANUAL", f"Manual clip requested for {state['display_name']}.")
        return await self._execute_clip_capture(
            stream_id=stream_id,
            chat_score=state["chat_score"],
            visual_score=state["visual_score"],
            facecam_score=state["facecam_score"],
            facecam_corner=state["facecam_corner"],
            combined_score=max(state["combined_score"], 100.0),
            chat_mps=state["chat_mps"],
            chat_keywords=list(state["top_keywords"]) or ["MANUAL CLIP"],
            chat_sample=list(state["recent_chat"]),
            delay_sec=0,
        )

    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        active = list(self._live_state.values())
        active.sort(
            key=lambda x: (x["has_facecam"] if settings.PREFER_FACECAM_STREAMS else False, x["combined_score"], x["viewer_count"]),
            reverse=True,
        )
        total_mps = round(sum(s["chat_mps"] for s in active), 2)
        return {
            "engine_enabled": settings.ENGINE_ENABLED,
            "prefer_facecam": settings.PREFER_FACECAM_STREAMS,
            "active_count": len(active),
            "facecam_count": sum(1 for s in active if s["has_facecam"]),
            "total_chat_mps": total_mps,
            "last_discovery_at": self._last_discovery_at,
            "active_jobs": list(self._active_jobs.values()),
            "recent_activity": list(self._activity_log)[:15],
            "streams": active,
        }

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        active = list(self._live_state.values())
        total_chat_msgs = sum(t.total_messages for t in chat_monitor.trackers.values())
        total_buf_mb = round(sum(s["buffer_size_mb"] for s in active), 2)
        conn = get_db()
        try:
            clip_counts = {
                row["status"]: row["cnt"]
                for row in conn.execute("SELECT status, COUNT(*) AS cnt FROM clips GROUP BY status").fetchall()
            }
            total_clips_db = sum(clip_counts.values())
            total_posts = conn.execute("SELECT COUNT(*) FROM post_history WHERE status='posted'").fetchone()[0]
        finally:
            conn.close()

        return {
            "uptime_sec": int(time.time() - self._started_at),
            "engine_enabled": settings.ENGINE_ENABLED,
            "prefer_facecam": settings.PREFER_FACECAM_STREAMS,
            "active_streams": len(active),
            "facecam_streams": sum(1 for s in active if s["has_facecam"]),
            "total_chat_messages": total_chat_msgs,
            "current_chat_mps": round(sum(s["chat_mps"] for s in active), 2),
            "peak_chat_mps": self._peak_chat_mps,
            "total_buffer_mb": total_buf_mb,
            "clips_session": self._clips_captured_session,
            "clips_total": total_clips_db,
            "clips_posted": total_posts,
            "clip_status_counts": clip_counts,
            "vlm": vision_manager.get_status(),
            "active_jobs": list(self._active_jobs.values()),
            "activity_log": list(self._activity_log)[:60],
            "streams_diagnostics": active,
        }


stream_manager = StreamManagerService()
