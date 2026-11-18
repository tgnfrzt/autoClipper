import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import discord

from backend.core.config import settings
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
    Discord Bot that sends every captured clip to the configured channel/DM
    and publishes or deletes clips based on emoji reactions.
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

    def get_status(self) -> Dict[str, Any]:
        return {
            "has_token": bool(settings.DISCORD_BOT_TOKEN),
            "channel_id": settings.DISCORD_CHANNEL_ID,
            "user_id": settings.DISCORD_USER_ID,
            "connected": self._connected,
            "bot_user": self._bot_user,
            "message_intent_enabled": self._message_intent_enabled,
            "last_error": self._last_error,
        }

    async def start_or_restart(self) -> None:
        await self.stop()
        token = (settings.DISCORD_BOT_TOKEN or "").strip()
        if not token:
            self._last_error = ""
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
                print(f"[DiscordBot] {self._last_error}")
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._connected = False
                self._last_error = str(exc)
                print(f"[DiscordBot] Error: {exc}")
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
        if not self._connected or not self._client:
            return {"ok": False, "error": self._last_error or "Discord bot is not connected yet."}
        targets = await self._resolve_destinations()
        if not targets:
            return {"ok": False, "error": "No valid DISCORD_CHANNEL_ID or DISCORD_USER_ID reachable."}
        for t in targets:
            await t.send(
                embed=discord.Embed(
                    title="⚡ GTA VI AI AutoClipper Connected!",
                    description=(
                        "Clips will appear here automatically when chat + AI vision detect a viral moment.\n\n"
                        "**Reactions:**\n"
                        f"{REACTION_APPROVE_ALL} Post to ALL configured platforms\n"
                        f"{REACTION_YOUTUBE} YouTube Shorts only\n"
                        f"{REACTION_TIKTOK} TikTok only\n"
                        f"{REACTION_INSTAGRAM} Instagram Reels only\n"
                        f"{REACTION_REJECT} Reject & delete clip"
                    ),
                    color=0x6366F1,
                )
            )
        return {"ok": True, "sent_to": len(targets)}

    async def send_clip_for_review(self, clip: Dict[str, Any]) -> Optional[str]:
        if not self._connected or not self._client:
            return None
        targets = await self._resolve_destinations()
        if not targets:
            return None

        raw_path = clip.get("rendered_path") or clip.get("raw_path") or ""
        attach_path = await asyncio.to_thread(ensure_discord_attachable_mp4, raw_path)

        kw_raw = clip.get("chat_keywords")
        if isinstance(kw_raw, str):
            try:
                kw_list = json.loads(kw_raw)
            except Exception:
                kw_list = []
        else:
            kw_list = kw_raw or []

        embed = discord.Embed(
            title=clip.get("ai_title") or f"🔥 {clip['streamer_name']} Clip",
            description=clip.get("ai_summary") or "Hype spike detected!",
            url=clip.get("stream_url") or None,
            color=0x10B981,
        )
        embed.add_field(name="Streamer", value=f"{clip['streamer_name']} ({clip['platform'].upper()})", inline=True)
        embed.add_field(
            name="Hype Score (80/20)",
            value=f"**{clip['combined_score']}/100** (Chat {clip['chat_score']} • Visual {clip['visual_score']})",
            inline=True,
        )
        embed.add_field(
            name="Chat Burst",
            value=f"{clip['chat_mps']} msg/s • {', '.join(kw_list[:5]) or 'HYPE'}",
            inline=False,
        )
        embed.add_field(
            name="React to Publish",
            value=(
                f"{REACTION_APPROVE_ALL} All Enabled • "
                f"{REACTION_YOUTUBE} YouTube • "
                f"{REACTION_TIKTOK} TikTok • "
                f"{REACTION_INSTAGRAM} Instagram • "
                f"{REACTION_REJECT} Reject"
            ),
            inline=False,
        )
        embed.set_footer(text=f"Clip ID: {clip['id']} • Duration: {clip['duration_sec']}s")

        first_msg_id: Optional[str] = None
        first_ch_id: Optional[str] = None

        for dest in targets:
            try:
                if attach_path and os.path.isfile(attach_path):
                    msg = await dest.send(embed=embed, file=discord.File(attach_path))
                else:
                    msg = await dest.send(embed=embed)

                for emoji in ALL_REVIEW_REACTIONS:
                    try:
                        await msg.add_reaction(emoji)
                    except Exception:
                        pass

                if first_msg_id is None:
                    first_msg_id = str(msg.id)
                    first_ch_id = str(msg.channel.id)
            except Exception as exc:
                self._last_error = f"Send clip error: {exc}"

        if first_msg_id:
            conn = get_db()
            with conn:
                conn.execute(
                    "UPDATE clips SET discord_message_id = ?, discord_channel_id = ? WHERE id = ?",
                    (first_msg_id, first_ch_id or "", clip["id"]),
                )
            conn.close()
        return first_msg_id

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

            lines = []
            for item in res["results"]:
                plat = item.get("platform", "unknown").upper()
                if item.get("success"):
                    url = item.get("post_url", "")
                    lines.append(f"✅ **{plat}**: Posted! {url}")
                else:
                    lines.append(f"❌ **{plat}**: {item.get('error', 'Failed')}")

            await status_msg.edit(content="\n".join(lines) or "Finished publishing.")
        except Exception as exc:
            print(f"[DiscordBot] Reaction handler error: {exc}")
        finally:
            self._processing_msgs.discard(msg_id)

    async def _handle_command(self, message: discord.Message, raw_cmd: str) -> None:
        parts = raw_cmd.split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "status":
            if self._stream_manager:
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
