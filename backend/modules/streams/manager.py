import asyncio
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from backend.core.config import CLIPS_RAW_DIR, THUMBS_DIR, settings
from backend.core.database import fetch_all_streamers, get_db, upsert_streamer
from backend.modules.ai.clip_detector import compute_combined_score, save_clip_record
from backend.modules.ai.vision_evaluator import vision_manager
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
    - Discovers top GTA VI (and fallback GTA V) streams + manually added streamers
    - Syncs live Twitch/YouTube chat monitors (80% trigger signal)
    - Maintains circular zero-decode HLS buffers for active streams
    - Evaluates keyframes via lightweight VLM + OpenCV (20% trigger signal)
    - Automatically clips the last 30-60s on hype spikes and dispatches to Discord
    """

    def __init__(self) -> None:
        self._running = False
        self._buffers: Dict[str, StreamRingBuffer] = {}
        self._live_state: Dict[str, Dict[str, Any]] = {}
        self._last_clip_at: Dict[str, float] = {}
        self._clipping_locks: set[str] = set()
        self._on_clip_created: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = None
        self._tasks: List[asyncio.Task] = []
        self._last_discovery_at: float = 0.0

    def set_on_clip_callback(self, cb: Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]) -> None:
        self._on_clip_created = cb

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await chat_monitor.start()
        if settings.VLM_AUTO_START and settings.VLM_PROVIDER == "llamacpp":
            asyncio.create_task(asyncio.to_thread(vision_manager.start_local_server))

        self._tasks = [
            asyncio.create_task(self._discovery_loop()),
            asyncio.create_task(self._monitor_and_evaluate_loop()),
        ]

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()
        await chat_monitor.stop()
        for buf in list(self._buffers.values()):
            buf.stop()
        self._buffers.clear()

    async def refresh_discovery_now(self) -> Dict[str, Any]:
        """Discovers live streams for the target game and checks enabled manual/popular streamers."""
        discovered: List[Dict[str, Any]] = []
        manual_streamers = fetch_all_streamers(enabled_only=True)

        # 1. Check enabled manual & preset streamers (Twitch + YouTube)
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

        # 2. Auto-discover top Twitch streams for GTA VI (or fallback game)
        if settings.AUTO_DISCOVER_TWITCH:
            tw_streams = await discover_twitch_game_streams(settings.GAME_QUERY, limit=20)
            if not tw_streams and settings.FALLBACK_GAME_QUERY:
                tw_streams = await discover_twitch_game_streams(settings.FALLBACK_GAME_QUERY, limit=15)
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

        # 3. Auto-discover live YouTube streams
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
        return {"live_discovered": len(discovered), "active_buffers": len(self._buffers)}

    async def _sync_active_buffers(self) -> None:
        all_rows = fetch_all_streamers(enabled_only=True)
        live_rows = [r for r in all_rows if r["is_live"] == 1]
        # Limit to MAX_CONCURRENT_STREAMS ordered by manual priority & viewer count
        target_rows = live_rows[: max(1, int(settings.MAX_CONCURRENT_STREAMS))]
        target_ids = {r["id"] for r in target_rows}

        # Stop buffers no longer active
        for sid in list(self._buffers.keys()):
            if sid not in target_ids or not settings.ENGINE_ENABLED:
                self._buffers.pop(sid).stop()
                self._live_state.pop(sid, None)

        if not settings.ENGINE_ENABLED:
            await chat_monitor.sync_streams([])
            return

        # Start new buffers
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
                    "chat_score": 0.0,
                    "visual_score": 0.0,
                    "combined_score": 0.0,
                    "chat_mps": 0.0,
                    "top_keywords": [],
                    "visual_event": "buffering",
                    "visual_summary": "",
                    "thumb_url": "",
                    "buffered_sec": 0,
                    "buffering": False,
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
                print(f"[StreamManager] Discovery error: {exc}")
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
                    state["top_keywords"] = chat_metrics["top_keywords"]
                    state["buffered_sec"] = buf.buffered_seconds()
                    state["buffering"] = buf.is_running()

                    # Periodic visual inspection (every VISUAL_EVAL_INTERVAL_SEC, or immediately if chat spikes)
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
                            state["visual_score"] = vis["visual_score"]
                            state["visual_event"] = vis["event"]
                            state["visual_summary"] = vis["summary"]

                    combined = compute_combined_score(state["chat_score"], state["visual_score"])
                    state["combined_score"] = combined

                    # Trigger clip if threshold met and outside cooldown
                    in_cooldown = (now - self._last_clip_at.get(sid, 0.0)) < settings.CLIP_COOLDOWN_SEC
                    if (
                        combined >= settings.TRIGGER_SCORE_THRESHOLD
                        and buf.buffered_seconds() >= 15
                        and not in_cooldown
                        and sid not in self._clipping_locks
                    ):
                        self._last_clip_at[sid] = now
                        asyncio.create_task(
                            self._execute_clip_capture(
                                stream_id=sid,
                                chat_score=state["chat_score"],
                                visual_score=state["visual_score"],
                                combined_score=combined,
                                chat_mps=state["chat_mps"],
                                chat_keywords=list(state["top_keywords"]),
                                delay_sec=settings.POST_TRIGGER_DELAY_SEC,
                            )
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[StreamManager] Monitor loop error: {exc}")

            await asyncio.sleep(1.5)

    async def _execute_clip_capture(
        self,
        stream_id: str,
        chat_score: float,
        visual_score: float,
        combined_score: float,
        chat_mps: float,
        chat_keywords: List[str],
        delay_sec: int = 0,
    ) -> Optional[Dict[str, Any]]:
        if stream_id in self._clipping_locks:
            return None
        self._clipping_locks.add(stream_id)
        try:
            buf = self._buffers.get(stream_id)
            state = self._live_state.get(stream_id)
            if not buf or not state:
                return None

            if delay_sec > 0:
                await asyncio.sleep(delay_sec)

            clip_uid = f"raw_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            raw_mp4 = str(CLIPS_RAW_DIR / f"{clip_uid}.mp4")
            clip_dur = max(30, min(60, int(settings.CLIP_DURATION_SEC)))
            ok, actual_dur, err = await asyncio.to_thread(buf.cut_clip_sync, clip_dur, raw_mp4)
            if not ok:
                print(f"[ClipCapture] Failed for {stream_id}: {err}")
                return None

            # Grab thumbnail snapshot
            frame_path = await buf.extract_latest_frame()
            clip_thumb = str(THUMBS_DIR / f"{clip_uid}.jpg")
            if frame_path and os.path.isfile(frame_path):
                shutil.copy2(frame_path, clip_thumb)
            else:
                clip_thumb = ""

            meta = await vision_manager.generate_clip_metadata(
                image_paths=[clip_thumb] if clip_thumb else [],
                streamer_name=state["display_name"],
                chat_keywords=chat_keywords,
                chat_mps=chat_mps,
            )

            record = save_clip_record(
                stream_id=stream_id,
                streamer_name=state["display_name"],
                platform=state["platform"],
                stream_url=state["url"],
                raw_path=raw_mp4,
                thumb_path=clip_thumb,
                duration_sec=actual_dur,
                chat_score=chat_score,
                visual_score=visual_score,
                combined_score=combined_score,
                chat_mps=chat_mps,
                chat_keywords=chat_keywords,
                ai_title=meta["title"],
                ai_summary=meta["summary"],
                ai_caption=meta["caption"],
            )

            if self._on_clip_created:
                asyncio.create_task(self._on_clip_created(record))
            return record
        finally:
            self._clipping_locks.discard(stream_id)

    async def trigger_manual_clip(self, stream_id: str) -> Optional[Dict[str, Any]]:
        state = self._live_state.get(stream_id)
        if not state:
            return None
        return await self._execute_clip_capture(
            stream_id=stream_id,
            chat_score=state["chat_score"],
            visual_score=state["visual_score"],
            combined_score=max(state["combined_score"], 100.0),
            chat_mps=state["chat_mps"],
            chat_keywords=list(state["top_keywords"]) or ["MANUAL CLIP"],
            delay_sec=0,
        )

    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        active = list(self._live_state.values())
        active.sort(key=lambda x: (x["combined_score"], x["viewer_count"]), reverse=True)
        total_mps = round(sum(s["chat_mps"] for s in active), 2)
        return {
            "engine_enabled": settings.ENGINE_ENABLED,
            "active_count": len(active),
            "total_chat_mps": total_mps,
            "last_discovery_at": self._last_discovery_at,
            "streams": active,
        }


stream_manager = StreamManagerService()
