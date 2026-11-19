import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import discord
import httpx

from backend.core.config import get_reachable_base_url, settings
from backend.core.database import get_clip_by_discord_message, get_db, upsert_streamer
from backend.modules.processors.video_processor import ensure_discord_attachable_mp4
from backend.modules.publishers.dispatcher import get_configured_platforms, publish_clip_to_platforms

REACTION_APPROVE_ALL = "✅"
REACTION_YOUTUBE = "▶️"
REACTION_TIKTOK = "🎵"
REACTION_INSTAGRAM = "📸"
REACTION_REJECT = "❌"

ALL_REVIEW_REACTIONS = (
    REACTION_APPROVE_ALL,
    REACTION_YOUTUBE,
    REACTION_TIKTOK,
    REACTION_INSTAGRAM,
    REACTION_REJECT,
)


class DiscordClipBotService:
    """
    Hybrid Discord integration supporting:
    1. Discord Webhook URL (Zero setup — sends clip + 1-click Publish/Reject links & updates message live)
    2. Optional Discord Bot Token (Adds emoji reaction controls & !status / !clip / !add commands)
    """

    def __init__(self) -> None:
        self._client: Optional[discord.Client] = None
        self._task: Optional[asyncio.Task] = None
        self._connected: bool = False
        self._bot_user: str = ""
        self._last_error: str = ""
        self._message_intent_enabled: bool = True
        self._processing_msgs: set[str] = set()
        self._stream_manager: Any = None

    def attach_stream_manager(self, manager: Any) -> None:
        self._stream_manager = manager

    def has_webhook(self) -> bool:
        url = (settings.DISCORD_WEBHOOK_URL or "").strip()
        return url.startswith("https://") and "/api/webhooks/" in url

    def _webhook_base_url(self) -> str:
        return (settings.DISCORD_WEBHOOK_URL or "").strip().split("?")[0].rstrip("/")

    def get_status(self) -> Dict[str, Any]:
        wh = self.has_webhook()
        mode = "webhook" if wh else ("bot" if self._connected else "none")
        if wh and self._connected:
            mode = "webhook+bot"
        return {
            "webhook_configured": wh,
            "has_token": bool(settings.DISCORD_BOT_TOKEN),
            "channel_id": settings.DISCORD_CHANNEL_ID,
            "user_id": settings.DISCORD_USER_ID,
            "connected": self._connected,
            "bot_user": self._bot_user,
            "mode": mode,
            "ready": wh or self._connected,
            "message_intent_enabled": self._message_intent_enabled,
            "last_error": self._last_error,
        }

    async def start_or_restart(self) -> None:
        self._last_error = ""
        await self.stop()
        token = (settings.DISCORD_BOT_TOKEN or "").strip()
        if not token:
            return
        self._task = asyncio.create_task(self._run_bot(token))

    async def stop(self) -> None:
        self._connected = False
        self._bot_user = ""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
            self._task = None

    async def _run_bot(self, token: str) -> None:
        for try_message_content in (True, False):
            intents = discord.Intents.default()
            intents.reactions = True
            intents.message_content = try_message_content
            client = discord.Client(intents=intents)
            self._client = client
            self._message_intent_enabled = try_message_content

            @client.event
            async def on_ready():
                self._connected = True
                self._bot_user = str(client.user)
                self._last_error = ""
                print(f"[DiscordBot] Logged in as {client.user}")

            @client.event
            async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
                if client.user and payload.user_id == client.user.id:
                    return
                await self._handle_reaction(payload)

            @client.event
            async def on_message(message: discord.Message):
                if message.author.bot:
                    return
                prefix = settings.DISCORD_COMMAND_PREFIX or "!"
                content = (message.content or "").strip()
                if not content.startswith(prefix):
                    return
                await self._handle_command(message, content[len(prefix):].strip())

            try:
                await client.start(token)
                break
            except discord.errors.PrivilegedIntentsRequired:
                self._last_error = "Message Content Intent disabled in Discord Dev Portal; running reaction-only mode."
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._connected = False
                self._last_error = str(exc)
                break

    async def _resolve_destinations(self) -> List[Any]:
        if not self._client or not self._connected:
            return []
        targets: List[Any] = []
        if settings.DISCORD_CHANNEL_ID:
            try:
                cid = int(settings.DISCORD_CHANNEL_ID.strip())
                ch = self._client.get_channel(cid) or await self._client.fetch_channel(cid)
                if ch:
                    targets.append(ch)
            except Exception as exc:
                self._last_error = f"Channel fetch error: {exc}"
        if settings.DISCORD_USER_ID:
            try:
                uid = int(settings.DISCORD_USER_ID.strip())
                user = self._client.get_user(uid) or await self._client.fetch_user(uid)
                if user:
                    dm = await user.create_dm()
                    targets.append(dm)
            except Exception as exc:
                self._last_error = f"DM fetch error: {exc}"
        return targets

    async def send_test_message(self) -> Dict[str, Any]:
        sent_count = 0
        errors: List[str] = []

        if self.has_webhook():
            try:
                base = get_reachable_base_url()
                payload = {
                    "username": "GTA VI AI AutoClipper",
                    "embeds": [
                        {
                            "title": "⚡ GTA VI AI AutoClipper Webhook Connected!",
                            "description": (
                                "New clips will arrive here automatically with the rendered 9:16 video "
                                "and interactive **Post All / YouTube / TikTok / Instagram / Reject** buttons below."
                            ),
                            "color": 0x6366F1,
                            "fields": [
                                {
                                    "name": "📱 Mobile Reachable Base URL",
                                    "value": (
                                        f"`{base}`\n\n"
                                        f"🚀 **[  ✅ TEST POST ALL  ]({base}/)**\n\n"
                                        f"📱 **[  🎬 TEST MOBILE REVIEW PLAYER  ]({base}/)**"
                                    ),
                                    "inline": False,
                                }
                            ],
                        }
                    ],
                    "components": [
                        {
                            "type": 1,
                            "components": [
                                {"type": 2, "style": 5, "label": "🌐 Test Phone Connection", "emoji": {"name": "📱"}, "url": f"{base}/"},
                                {"type": 2, "style": 5, "label": "AutoClipper Dashboard", "emoji": {"name": "⚡"}, "url": f"{base}/"},
                            ],
                        }
                    ],
                }
                async with httpx.AsyncClient(timeout=15.0) as client:
                    r = await client.post(f"{self._webhook_base_url()}?wait=true", json=payload)
                    if r.status_code in (200, 204):
                        sent_count += 1
                    else:
                        errors.append(f"Webhook HTTP {r.status_code}: {r.text[:120]}")
            except Exception as exc:
                errors.append(f"Webhook error: {exc}")

        if self._connected and self._client:
            targets = await self._resolve_destinations()
            for t in targets:
                try:
                    await t.send(
                        embed=discord.Embed(
                            title="⚡ GTA VI AI AutoClipper Bot Connected!",
                            description=(
                                "Clips will appear here automatically.\n"
                                f"{REACTION_APPROVE_ALL} All • {REACTION_YOUTUBE} YouTube • "
                                f"{REACTION_TIKTOK} TikTok • {REACTION_INSTAGRAM} Instagram • {REACTION_REJECT} Delete"
                            ),
                            color=0x6366F1,
                        )
                    )
                    sent_count += 1
                except Exception as exc:
                    errors.append(f"Bot send error: {exc}")

        if sent_count > 0:
            self._last_error = ""
            return {"ok": True, "sent_to": sent_count, "mode": self.get_status()["mode"]}
        return {
            "ok": False,
            "error": "; ".join(errors) or self._last_error or "Configure a Discord Webhook URL or Bot Token first.",
        }

    def _build_clip_components(self, clip: Dict[str, Any]) -> List[Dict[str, Any]]:
        base = get_reachable_base_url()
        clip_id = clip["id"]
        return [
            {
                "type": 1,  # ActionRow 1: Quick Publishing & Reject Buttons
                "components": [
                    {
                        "type": 2,  # Button
                        "style": 5,  # Link button
                        "label": "Post All",
                        "emoji": {"name": "✅"},
                        "url": f"{base}/api/clips/{clip_id}/quick-action?action=all",
                    },
                    {
                        "type": 2,
                        "style": 5,
                        "label": "YouTube",
                        "emoji": {"name": "▶️"},
                        "url": f"{base}/api/clips/{clip_id}/quick-action?action=youtube_shorts",
                    },
                    {
                        "type": 2,
                        "style": 5,
                        "label": "TikTok",
                        "emoji": {"name": "🎵"},
                        "url": f"{base}/api/clips/{clip_id}/quick-action?action=tiktok",
                    },
                    {
                        "type": 2,
                        "style": 5,
                        "label": "Instagram",
                        "emoji": {"name": "📸"},
                        "url": f"{base}/api/clips/{clip_id}/quick-action?action=instagram",
                    },
                    {
                        "type": 2,
                        "style": 5,
                        "label": "Reject",
                        "emoji": {"name": "❌"},
                        "url": f"{base}/api/clips/{clip_id}/quick-action?action=reject",
                    },
                ],
            },
            {
                "type": 1,  # ActionRow 2: Mobile Review Player & Dashboard
                "components": [
                    {
                        "type": 2,
                        "style": 5,
                        "label": "📱 Mobile Review Player",
                        "url": f"{base}/api/clips/{clip_id}/review",
                    },
                    {
                        "type": 2,
                        "style": 5,
                        "label": "🌐 Dashboard",
                        "url": f"{base}/",
                    },
                ],
            },
        ]

    def _create_bot_review_view(self, clip_id: str) -> discord.ui.View:
        base = get_reachable_base_url()
        view = discord.ui.View(timeout=None)
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.link,
                label="Post All",
                emoji="✅",
                url=f"{base}/api/clips/{clip_id}/quick-action?action=all",
                row=0,
            )
        )
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.link,
                label="YouTube",
                emoji="▶️",
                url=f"{base}/api/clips/{clip_id}/quick-action?action=youtube_shorts",
                row=0,
            )
        )
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.link,
                label="TikTok",
                emoji="🎵",
                url=f"{base}/api/clips/{clip_id}/quick-action?action=tiktok",
                row=0,
            )
        )
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.link,
                label="Instagram",
                emoji="📸",
                url=f"{base}/api/clips/{clip_id}/quick-action?action=instagram",
                row=0,
            )
        )
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.link,
                label="Reject",
                emoji="❌",
                url=f"{base}/api/clips/{clip_id}/quick-action?action=reject",
                row=0,
            )
        )
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.link,
                label="📱 Mobile Review Player",
                url=f"{base}/api/clips/{clip_id}/review",
                row=1,
            )
        )
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.link,
                label="🌐 Dashboard",
                url=f"{base}/",
                row=1,
            )
        )
        return view

    def _build_clip_embed_dict(self, clip: Dict[str, Any]) -> Dict[str, Any]:
        kw_raw = clip.get("chat_keywords")
        if isinstance(kw_raw, str):
            try:
                kw_list = json.loads(kw_raw)
            except Exception:
                kw_list = []
        else:
            kw_list = kw_raw or []

        fields: List[Dict[str, Any]] = []
        if clip.get("overlay_caption"):
            fields.append({"name": "On-Screen Caption", "value": f"**{clip['overlay_caption']}**", "inline": False})
        if clip.get("transcript"):
            fields.append({"name": "Streamer Said", "value": f"\"{clip['transcript'][:200]}\"", "inline": False})
        fields.append({"name": "Streamer", "value": f"{clip['streamer_name']} ({clip['platform'].upper()})", "inline": True})
        fields.append(
            {
                "name": "Hype Score (80/20)",
                "value": f"**{clip['combined_score']}/100** (Chat {clip['chat_score']} • Vision {clip['visual_score']} • Cam {clip.get('facecam_score', 0)})",
                "inline": True,
            }
        )
        fields.append(
            {
                "name": "Chat Burst",
                "value": f"{clip['chat_mps']} msg/s • {', '.join(kw_list[:5]) or 'HYPE'}",
                "inline": False,
            }
        )
        base = get_reachable_base_url()
        clip_id = clip["id"]
        action_buttons_md = (
            f"🚀 **[  ✅ POST ALL PLATFORMS  ]({base}/api/clips/{clip_id}/quick-action?action=all)**\n\n"
            f"▶️ **[  YouTube Shorts  ]({base}/api/clips/{clip_id}/quick-action?action=youtube_shorts)**  •  "
            f"🎵 **[  TikTok  ]({base}/api/clips/{clip_id}/quick-action?action=tiktok)**  •  "
            f"📸 **[  Instagram  ]({base}/api/clips/{clip_id}/quick-action?action=instagram)**\n\n"
            f"🗑️ **[  ❌ Reject & Delete  ]({base}/api/clips/{clip_id}/quick-action?action=reject)**\n\n"
            f"📱 **[  🎬 OPEN MOBILE REVIEW PLAYER  ]({base}/api/clips/{clip_id}/review)**"
        )
        fields.append({
            "name": "⚡ 1-Click Phone Actions",
            "value": action_buttons_md,
            "inline": False,
        })

        return {
            "title": clip.get("ai_title") or f"🔥 {clip['streamer_name']} Clip",
            "description": clip.get("ai_summary") or "Hype spike detected!",
            "url": clip.get("stream_url") or None,
            "color": 0x10B981,
            "fields": fields,
            "footer": {"text": f"Clip ID: {clip['id']} • Duration: {clip['duration_sec']}s"},
        }

    async def send_clip_for_review(self, clip: Dict[str, Any]) -> Optional[str]:
        raw_path = clip.get("rendered_path") or clip.get("raw_path") or ""
        attach_path = await asyncio.to_thread(ensure_discord_attachable_mp4, raw_path)
        embed_dict = self._build_clip_embed_dict(clip)
        components = self._build_clip_components(clip)
        base = get_reachable_base_url()
        clip_id = clip["id"]

        first_msg_id: Optional[str] = None
        first_ch_id: Optional[str] = None

        # 1. Discord Webhook (Primary / Zero-Bot Setup)
        if self.has_webhook():
            try:
                content_text = (
                    f"🎬 **New Clip Ready: {clip.get('overlay_caption') or clip['streamer_name']}**\n"
                    f"> 🚀 **[  ✅ POST ALL  ]({base}/api/clips/{clip_id}/quick-action?action=all)**  •  "
                    f"📱 **[  🎬 MOBILE REVIEW  ]({base}/api/clips/{clip_id}/review)**  •  "
                    f"❌ **[  REJECT  ]({base}/api/clips/{clip_id}/quick-action?action=reject)**"
                )
                payload_json = json.dumps(
                    {
                        "username": "GTA VI AI AutoClipper",
                        "content": content_text,
                        "embeds": [embed_dict],
                        "components": components,
                    }
                )
                async with httpx.AsyncClient(timeout=60.0) as client:
                    if attach_path and os.path.isfile(attach_path):
                        with open(attach_path, "rb") as f:
                            file_bytes = f.read()
                        resp = await client.post(
                            f"{self._webhook_base_url()}?wait=true",
                            data={"payload_json": payload_json},
                            files={"file": (os.path.basename(attach_path), file_bytes, "video/mp4")},
                        )
                    else:
                        resp = await client.post(
                            f"{self._webhook_base_url()}?wait=true",
                            json=json.loads(payload_json),
                        )

                    if resp.status_code in (200, 201):
                        data = resp.json()
                        first_msg_id = str(data.get("id") or "")
                        first_ch_id = "webhook"
                        self._last_error = ""
                    else:
                        self._last_error = f"Webhook send HTTP {resp.status_code}: {resp.text[:160]}"
            except Exception as exc:
                self._last_error = f"Webhook send error: {exc}"

        # 2. Optional Discord Bot (if also configured)
        if self._connected and self._client:
            targets = await self._resolve_destinations()
            embed = discord.Embed.from_dict(embed_dict)
            view = self._create_bot_review_view(clip["id"])
            for dest in targets:
                try:
                    if attach_path and os.path.isfile(attach_path):
                        msg = await dest.send(embed=embed, file=discord.File(attach_path), view=view)
                    else:
                        msg = await dest.send(embed=embed, view=view)
                    for emoji in ALL_REVIEW_REACTIONS:
                        try:
                            await msg.add_reaction(emoji)
                        except Exception:
                            pass
                    if not first_msg_id:
                        first_msg_id = str(msg.id)
                        first_ch_id = str(msg.channel.id)
                except Exception as exc:
                    self._last_error = f"Bot send error: {exc}"

        if first_msg_id:
            conn = get_db()
            with conn:
                conn.execute(
                    "UPDATE clips SET discord_message_id = ?, discord_channel_id = ? WHERE id = ?",
                    (first_msg_id, first_ch_id or "", clip["id"]),
                )
            conn.close()
        return first_msg_id

    async def update_webhook_message_status(self, discord_message_id: str, status_text: str) -> None:
        """Edits the webhook message content in-place to show live publish/reject status."""
        if not self.has_webhook() or not discord_message_id:
            return
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                await client.patch(
                    f"{self._webhook_base_url()}/messages/{discord_message_id}",
                    json={"content": status_text[:1900]},
                )
        except Exception:
            pass

    async def _handle_reaction(self, payload: discord.RawReactionActionEvent) -> None:
        emoji = str(payload.emoji)
        if emoji not in ALL_REVIEW_REACTIONS:
            return
        msg_id = str(payload.message_id)
        if msg_id in self._processing_msgs:
            return
        clip = get_clip_by_discord_message(msg_id)
        if not clip:
            return

        self._processing_msgs.add(msg_id)
        try:
            channel = self._client.get_channel(payload.channel_id) or await self._client.fetch_channel(payload.channel_id)
            message = await channel.fetch_message(payload.message_id)

            if emoji == REACTION_REJECT:
                conn = get_db()
                with conn:
                    conn.execute("UPDATE clips SET status = 'rejected' WHERE id = ?", (clip["id"],))
                conn.close()
                if settings.AUTO_DELETE_ON_REJECT:
                    for p in (clip.get("raw_path"), clip.get("rendered_path")):
                        if p and os.path.isfile(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                await message.reply(f"🗑️ Clip `{clip['id']}` rejected and cleaned up.")
                return

            if emoji == REACTION_APPROVE_ALL:
                platforms = get_configured_platforms()
                if not platforms:
                    await message.reply("⚠️ No social platforms are configured yet. Open Web UI > Configure.")
                    return
            elif emoji == REACTION_YOUTUBE:
                platforms = ["youtube_shorts"]
            elif emoji == REACTION_TIKTOK:
                platforms = ["tiktok"]
            elif emoji == REACTION_INSTAGRAM:
                platforms = ["instagram"]
            else:
                return

            status_msg = await message.reply(
                f"⏳ Rendering **{settings.VIDEO_FORMAT_MODE}** & publishing `{clip['id']}` to: `{', '.join(platforms)}`..."
            )
            res = await asyncio.to_thread(publish_clip_to_platforms, clip["id"], platforms)
            lines = [
                f"✅ **{item.get('platform', '').upper()}**: Posted! {item.get('post_url', '')}"
                if item.get("success")
                else f"❌ **{item.get('platform', '').upper()}**: {item.get('error', 'Failed')}"
                for item in res["results"]
            ]
            await status_msg.edit(content="\n".join(lines) or "Finished publishing.")
        except Exception as exc:
            print(f"[DiscordBot] Reaction handler error: {exc}")
        finally:
            self._processing_msgs.discard(msg_id)

    async def _handle_command(self, message: discord.Message, raw_cmd: str) -> None:
        parts = raw_cmd.split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "status" and self._stream_manager:
            snap = self._stream_manager.get_dashboard_snapshot()
            top = "\n".join(
                f"• **{s['display_name']}** ({s['platform']}): {s['chat_mps']} msg/s | Score `{s['combined_score']}`"
                for s in snap["streams"][:8]
            )
            await message.reply(
                f"📡 Watching **{snap['active_count']}** live streams | Total Chat: **{snap['total_chat_mps']} msg/s**\n{top or 'No live streams buffered yet.'}"
            )
        elif cmd == "clip" and arg and self._stream_manager:
            snap = self._stream_manager.get_dashboard_snapshot()
            match = next(
                (s for s in snap["streams"] if arg.lower() in s["display_name"].lower() or arg.lower() in s["channel_id"].lower()),
                None,
            )
            if not match:
                await message.reply(f"Could not find an active buffered stream matching `{arg}`.")
                return
            await message.reply(f"🎬 Clipping last {settings.CLIP_DURATION_SEC}s of **{match['display_name']}**...")
            rec = await self._stream_manager.trigger_manual_clip(match["id"])
            if rec:
                await self.send_clip_for_review(rec)
            else:
                await message.reply("⚠️ Not enough video buffered for that stream yet.")
        elif cmd == "add" and arg:
            low = arg.lower()
            if "youtube.com" in low or "youtu.be" in low or arg.startswith("@"):
                platform = "youtube"
                channel_id = arg.split("/")[-1]
                url = arg if arg.startswith("http") else f"https://www.youtube.com/{arg}/live"
            else:
                platform = "twitch"
                channel_id = arg.replace("https://www.twitch.tv/", "").strip("/").lower()
                url = f"https://www.twitch.tv/{channel_id}"
            upsert_streamer(platform, channel_id, channel_id, url, is_manual=True, enabled=True, priority=95)
            await message.reply(f"➕ Added `{channel_id}` ({platform}) to watch list! Refreshing discovery...")
            if self._stream_manager:
                asyncio.create_task(self._stream_manager.refresh_discovery_now())


discord_bot = DiscordClipBotService()
