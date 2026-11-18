import asyncio
import math
import random
import re
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import httpx
import websockets

HYPE_TERMS = {
    "omg", "wtf", "clip", "clipped", "no way", "noway", "insane", "crazy", "lmao",
    "lmfao", "ayo", "pog", "poggers", "kekw", "omegalul", "w", "wasted", "busted",
    "6 stars", "gta", "gta6", "gtavi", "holy", "what", "bro", "nah", "rip", "f",
    "sheesh", "fire", "goat", "clutch", "explosion", "dead", "cooked", "real",
}


class StreamChatTracker:
    """
    Tracks 1-second message velocity buckets and hype keyword density for a single stream.
    Computes an adaptive 0..100 chat_score representing the 80% trigger signal.
    """

    def __init__(self, stream_id: str) -> None:
        self.stream_id = stream_id
        self.started_at: float = time.time()
        self.buckets: Dict[int, int] = {}
        self.recent_messages: Deque[Tuple[float, str, str]] = deque(maxlen=180)
        self.total_messages: int = 0
        self.last_message_at: float = 0.0

    def add_message(self, user: str, text: str) -> None:
        now = time.time()
        sec = int(now)
        self.buckets[sec] = self.buckets.get(sec, 0) + 1
        self.recent_messages.append((now, user, text))
        self.total_messages += 1
        self.last_message_at = now

        # Prune buckets older than 120s
        cutoff = sec - 120
        stale = [k for k in self.buckets if k < cutoff]
        for k in stale:
            del self.buckets[k]

    def get_metrics(self) -> Dict[str, Any]:
        now = time.time()
        sec = int(now)

        # Current 5-second burst rate (messages per second)
        burst_counts = [self.buckets.get(sec - i, 0) for i in range(5)]
        current_mps = sum(burst_counts) / 5.0

        # Adaptive trailing baseline strictly over seconds observed since tracker start
        elapsed = max(0, int(now - self.started_at))
        if elapsed <= 8:
            mean_mps = current_mps
            std_mps = max(1.0, current_mps * 0.5)
        else:
            max_lookback = min(95, elapsed)
            baseline_counts = [self.buckets.get(sec - i, 0) for i in range(5, max_lookback)]
            mean_mps = sum(baseline_counts) / max(1, len(baseline_counts))
            variance = sum((c - mean_mps) ** 2 for c in baseline_counts) / max(1, len(baseline_counts))
            std_mps = math.sqrt(variance)

        # Spike ratio & Z-score
        spike_ratio = current_mps / max(0.5, mean_mps)
        z_score = (current_mps - mean_mps) / max(0.6, std_mps)

        # Hype keyword & caps density over last 6 seconds
        recent_texts = [t for ts, _, t in self.recent_messages if (now - ts) <= 6.0]
        hype_hits = 0
        kw_counter: Counter[str] = Counter()
        for msg in recent_texts:
            low = msg.lower()
            words = re.findall(r"[a-z0-9']+", low)
            matched_msg = False
            for w in words:
                if w in HYPE_TERMS:
                    kw_counter[w.upper()] += 1
                    matched_msg = True
            if "no way" in low:
                kw_counter["NO WAY"] += 1
                matched_msg = True
            if "???" in msg or "!!!" in msg or (len(msg) >= 4 and msg.isupper()):
                matched_msg = True
            if matched_msg:
                hype_hits += 1

        hype_density = (hype_hits / len(recent_texts)) if recent_texts else 0.0

        # Combine velocity burst + hype token density into 0..100 score
        # - Wait until >= 18s baseline warmup has elapsed before full scoring
        # - Needs real message volume (current_mps >= 0.8) so 1 random message in an empty chat doesn't false-trigger
        warmup_gate = min(1.0, max(0.0, (elapsed - 10) / 15.0))
        volume_gate = min(1.0, current_mps / 1.8)
        velocity_component = min(75.0, max(0.0, (z_score * 13.0) + max(0.0, spike_ratio - 1.35) * 16.0))
        absolute_boost = min(12.0, current_mps * 0.4)
        hype_component = min(25.0, hype_density * 35.0)

        chat_score = round(min(100.0, warmup_gate * volume_gate * (velocity_component + absolute_boost + hype_component)), 1)
        top_keywords = [k for k, _ in kw_counter.most_common(6)]
        if not top_keywords and recent_texts:
            # Grab most repeated non-trivial word
            all_words = [
                w.upper()
                for m in recent_texts
                for w in re.findall(r"[a-zA-Z0-9]{2,}", m)
            ]
            top_keywords = [k for k, _ in Counter(all_words).most_common(4)]

        return {
            "chat_score": chat_score,
            "chat_mps": round(current_mps, 2),
            "baseline_mps": round(mean_mps, 2),
            "spike_ratio": round(spike_ratio, 2),
            "hype_density": round(hype_density, 2),
            "top_keywords": top_keywords,
            "total_messages": self.total_messages,
            "recent_sample": [t for _, _, t in list(self.recent_messages)[-5:]],
        }


class ChatMonitorService:
    """
    Manages live chat ingestion across all active Twitch (shared anonymous WebSocket IRC)
    and YouTube (async Innertube live_chat poller) streams.
    """

    def __init__(self) -> None:
        self.trackers: Dict[str, StreamChatTracker] = {}
        self._twitch_channels: Set[str] = set()
        self._joined_channels: Set[str] = set()
        self._youtube_tasks: Dict[str, asyncio.Task] = {}
        self._ws: Any = None
        self._running = False
        self._twitch_task: Optional[asyncio.Task] = None

    def get_tracker(self, stream_id: str) -> StreamChatTracker:
        if stream_id not in self.trackers:
            self.trackers[stream_id] = StreamChatTracker(stream_id)
        return self.trackers[stream_id]

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._twitch_task = asyncio.create_task(self._twitch_irc_loop())

    async def stop(self) -> None:
        self._running = False
        if self._twitch_task:
            self._twitch_task.cancel()
        for t in list(self._youtube_tasks.values()):
            t.cancel()
        self._youtube_tasks.clear()

    async def sync_streams(self, active_streams: List[Dict[str, Any]]) -> None:
        """Aligns joined Twitch channels & YouTube chat pollers with currently active streams."""
        desired_twitch: Set[str] = set()
        desired_youtube: Dict[str, str] = {}

        for s in active_streams:
            sid = s["id"]
            self.get_tracker(sid)
            if s["platform"] == "twitch":
                desired_twitch.add(s["channel_id"].lower())
            elif s["platform"] == "youtube":
                desired_youtube[sid] = s["url"]

        # Sync Twitch channels on live socket
        to_part = self._twitch_channels - desired_twitch
        to_join = desired_twitch - self._twitch_channels
        self._twitch_channels = desired_twitch

        if self._ws is not None:
            for ch in to_part:
                try:
                    await self._ws.send(f"PART #{ch}")
                    self._joined_channels.discard(ch)
                except Exception:
                    pass
            for ch in to_join:
                try:
                    await self._ws.send(f"JOIN #{ch}")
                    self._joined_channels.add(ch)
                except Exception:
                    pass

        # Sync YouTube chat pollers
        for sid in list(self._youtube_tasks.keys()):
            if sid not in desired_youtube:
                self._youtube_tasks.pop(sid).cancel()

        for sid, url in desired_youtube.items():
            if sid not in self._youtube_tasks or self._youtube_tasks[sid].done():
                self._youtube_tasks[sid] = asyncio.create_task(self._youtube_chat_loop(sid, url))

    async def _twitch_irc_loop(self) -> None:
        while self._running:
            try:
                async with websockets.connect(
                    "wss://irc-ws.chat.twitch.tv:443",
                    ping_interval=30,
                    ping_timeout=15,
                ) as ws:
                    self._ws = ws
                    self._joined_channels.clear()
                    nick = f"justinfan{random.randint(10000, 99999)}"
                    await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
                    await ws.send(f"NICK {nick}")

                    for ch in list(self._twitch_channels):
                        await ws.send(f"JOIN #{ch}")
                        self._joined_channels.add(ch)

                    async for raw in ws:
                        if not self._running:
                            break
                        for line in raw.splitlines():
                            if line.startswith("PING"):
                                await ws.send(line.replace("PING", "PONG", 1))
                                continue
                            if " PRIVMSG #" in line:
                                self._handle_twitch_privmsg(line)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._ws = None
                self._joined_channels.clear()
                print(f"[ChatMonitor] Twitch IRC reconnect in 4s: {exc}")
                await asyncio.sleep(4)

    def _handle_twitch_privmsg(self, line: str) -> None:
        try:
            # Parse channel and message text
            m = re.search(r"PRIVMSG #([a-zA-Z0-9_]+)\s+:(.*)$", line)
            if not m:
                return
            ch = m.group(1).lower()
            text = m.group(2).strip()
            user_match = re.search(r":([^!]+)!", line)
            user = user_match.group(1) if user_match else "viewer"
            sid = f"twitch:{ch}"
            self.get_tracker(sid).add_message(user, text)
        except Exception:
            pass

    async def _youtube_chat_loop(self, stream_id: str, video_url: str) -> None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        while self._running:
            try:
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    page = await client.get(video_url, headers=headers)
                    if page.status_code != 200:
                        await asyncio.sleep(15)
                        continue
                    html = page.text
                    key_m = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', html)
                    cont_m = re.search(r'"liveChatRenderer".*?"continuation":"([^"]+)"', html, re.DOTALL)
                    if not cont_m:
                        cont_m = re.search(r'"invalidationContinuationData".*?"continuation":"([^"]+)"', html, re.DOTALL)
                    if not key_m or not cont_m:
                        await asyncio.sleep(20)
                        continue

                    api_key = key_m.group(1)
                    continuation = cont_m.group(1)
                    tracker = self.get_tracker(stream_id)
                    seen_ids: Deque[str] = deque(maxlen=400)

                    while self._running and continuation:
                        payload = {
                            "context": {
                                "client": {
                                    "clientName": "WEB",
                                    "clientVersion": "2.20260101.00.00",
                                }
                            },
                            "continuation": continuation,
                        }
                        r = await client.post(
                            f"https://www.youtube.com/youtubei/v1/live_chat/get_live_chat?key={api_key}",
                            json=payload,
                            headers=headers,
                        )
                        if r.status_code != 200:
                            break
                        data = r.json()
                        live_cont = (
                            data.get("continuationContents", {})
                            .get("liveChatContinuation", {})
                        )
                        actions = live_cont.get("actions") or []
                        for act in actions:
                            item = (
                                act.get("addChatItemAction", {})
                                .get("item", {})
                                .get("liveChatTextMessageRenderer")
                            )
                            if not item:
                                continue
                            mid = item.get("id", "")
                            if mid and mid in seen_ids:
                                continue
                            if mid:
                                seen_ids.append(mid)
                            runs = (item.get("message") or {}).get("runs") or []
                            text = "".join(part.get("text", "") for part in runs).strip()
                            user = (item.get("authorName") or {}).get("simpleText") or "viewer"
                            if text:
                                tracker.add_message(user, text)

                        # Next continuation
                        next_cont = None
                        for c in live_cont.get("continuations") or []:
                            for key_name in ("invalidationContinuationData", "timedContinuationData", "reloadContinuationData"):
                                if key_name in c and c[key_name].get("continuation"):
                                    next_cont = c[key_name]["continuation"]
                                    break
                            if next_cont:
                                break
                        continuation = next_cont
                        await asyncio.sleep(2.5)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(10)


chat_monitor = ChatMonitorService()
