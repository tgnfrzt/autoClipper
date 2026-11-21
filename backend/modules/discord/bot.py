import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import discord
import httpx

from backend.core.config import get_reachable_base_url, settings
from backend.core.database import get_clip, get_clip_by_discord_message, get_db, upsert_streamer
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
        self._webhook_channel_id: str = ""

    def attach_stream_manager(self, manager: Any) -> None:
        self._stream_manager = manager

    def has_webhook(self) -> bool:
        url = (settings.DISCORD_WEBHOOK_URL or "").strip()
        return url.startswith("https://") and "/api/webhooks/" in url

    def _webhook_base_url(self) -> str:
        return (settings.DISCORD_WEBHOOK_URL or "").strip().split("?")[0].rstrip("/")

    async def _resolve_webhook_channel_id(self) -> str:
        if self._webhook_channel_id:
            return self._webhook_channel_id
        if not self.has_webhook():
            return ""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(self._webhook_base_url())
                if r.status_code == 200:
                    cid = str(r.json().get("channel_id") or "")
                    if cid:
                        self._webhook_channel_id = cid
                        return cid
        except Exception:
            pass
        return ""

    def get_status(self) -> Dict[str, Any]:
        wh = self.has_webhook()
        mode = "bot" if self._connected else ("webhook" if wh else "none")
        if wh and self._connected:
            mode = "webhook+bot"
        return {
            "webhook_configured": wh,
            "has_token": bool(settings.DISCORD_BOT_TOKEN),
            "channel_id": settings.DISCORD_CHANNEL_ID or self._webhook_channel_id,
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
        self._webhook_channel_id = ""
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
            async def on_interaction(interaction: discord.Interaction):
                if interaction.type == discord.InteractionType.component:
                    data = interaction.data or {}
                    cid = str(data.get("custom_id") or "")
                    if cid.startswith("ac:"):
                        await self._handle_button_interaction(interaction, cid)

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
        raw_cid = (settings.DISCORD_CHANNEL_ID or "").strip()
        if not raw_cid and self.has_webhook():
            raw_cid = await self._resolve_webhook_channel_id()
        if raw_cid:
            try:
                cid = int(raw_cid)
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

        if self._connected and self._client:
            targets = await self._resolve_destinations()
            if not targets:
                errors.append(
                    self._last_error
                    or "Bot connected, but no destination channel found. Set Channel ID or Webhook URL (auto-detects channel)."
                )
            for t in targets:
                try:
                    preview = discord.ui.View(timeout=None)
                    preview.add_item(discord.ui.Button(style=discord.ButtonStyle.success, label="Post All", emoji="✅", disabled=True, row=0))
                    preview.add_item(discord.ui.Button(style=discord.ButtonStyle.secondary, label="YouTube", emoji="▶️", disabled=True, row=0))
                    preview.add_item(discord.ui.Button(style=discord.ButtonStyle.secondary, label="TikTok", emoji="🎵", disabled=True, row=0))
                    preview.add_item(discord.ui.Button(style=discord.ButtonStyle.secondary, label="Instagram", emoji="📸", disabled=True, row=0))
                    preview.add_item(discord.ui.Button(style=discord.ButtonStyle.danger, label="Reject", emoji="❌", disabled=True, row=0))
                    await t.send(
                        embed=discord.Embed(
                            title="⚡ GTA VI AI AutoClipper Bot Connected!",
                            description=(
                                "Native interactive Discord buttons are active and work anywhere on mobile data (4G/5G)!\n"
                                "Real clips will arrive with active **Post All / YouTube / TikTok / Instagram / Reject** buttons below."
                            ),
                            color=0x10B981,
                        ),
                        view=preview,
                    )
                    sent_count += 1
                except Exception as exc:
                    errors.append(f"Bot send error: {exc}")

        if sent_count == 0 and self.has_webhook():
            try:
                base = get_reachable_base_url()
                payload = {
                    "username": "GTA VI AI AutoClipper",
                    "embeds": [
                        {
                            "title": "⚡ GTA VI AI AutoClipper Webhook Connected!",
                            "description": (
                                "New clips will arrive here with touch action links (same Wi-Fi / Tailscale). "
                                "Note: Discord strips native component buttons from incoming webhooks — add a **Discord Bot Token** in Configure for native buttons on 4G/5G."
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
                }
                async with httpx.AsyncClient(timeout=15.0) as client:
                    r = await client.post(f"{self._webhook_base_url()}?wait=true", json=payload)
                    if r.status_code in (200, 204):
                        sent_count += 1
                    else:
                        errors.append(f"Webhook HTTP {r.status_code}: {r.text[:120]}")
            except Exception as exc:
                errors.append(f"Webhook error: {exc}")

        if sent_count > 0:
            self._last_error = ""
            return {"ok": True, "sent_to": sent_count, "mode": self.get_status()["mode"]}
        return {
            "ok": False,
            "error": "; ".join(errors) or self._last_error or "Configure a Discord Bot Token or Webhook URL first.",
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

    def _create_bot_review_view(
        self,
        clip_id: str,
        posted_platforms: Optional[set[str]] = None,
        rejected: bool = False,
    ) -> discord.ui.View:
        posted = posted_platforms or set()
        configured = set(get_configured_platforms())
        all_done = rejected or (bool(configured) and configured.issubset(posted))

        view = discord.ui.View(timeout=None)
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.success,
                label="Post All",
                emoji="✅",
                custom_id=f"ac:all:{clip_id}",
                disabled=all_done,
                row=0,
            )
        )
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.success if "youtube_shorts" in posted else discord.ButtonStyle.secondary,
                label="YouTube",
                emoji="▶️",
                custom_id=f"ac:youtube_shorts:{clip_id}",
                disabled=rejected or ("youtube_shorts" in posted),
                row=0,
            )
        )
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.success if "tiktok" in posted else discord.ButtonStyle.secondary,
                label="TikTok",
                emoji="🎵",
                custom_id=f"ac:tiktok:{clip_id}",
                disabled=rejected or ("tiktok" in posted),
                row=0,
            )
        )
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.success if "instagram" in posted else discord.ButtonStyle.secondary,
                label="Instagram",
                emoji="📸",
                custom_id=f"ac:instagram:{clip_id}",
                disabled=rejected or ("instagram" in posted),
                row=0,
            )
        )
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.danger,
                label="Reject",
                emoji="❌",
                custom_id=f"ac:reject:{clip_id}",
                disabled=rejected,
                row=0,
            )
        )
        return view

    def _build_clip_embed_dict(self, clip: Dict[str, Any], include_webhook_links: bool = True) -> Dict[str, Any]:
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
        if include_webhook_links:
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
        else:
            fields.append({
                "name": "⚡ Interactive Controls",
                "value": "Tap the native interactive buttons below to **Post All**, pick a platform, or **Reject** — works anywhere on mobile data!",
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
        clip_id = clip["id"]

        first_msg_id: Optional[str] = None
        first_ch_id: Optional[str] = None

        # 1. Preferred: Discord Bot with True Native Interactive Buttons (works anywhere on 4G/5G)
        if self._connected and self._client:
            targets = await self._resolve_destinations()
            embed = discord.Embed.from_dict(self._build_clip_embed_dict(clip, include_webhook_links=False))
            view = self._create_bot_review_view(clip_id)
            content_bot = f"🎬 **New Clip Ready:** {clip.get('overlay_caption') or clip['streamer_name']}"
            for dest in targets:
                try:
                    if attach_path and os.path.isfile(attach_path):
                        msg = await dest.send(content=content_bot, embed=embed, file=discord.File(attach_path), view=view)
                    else:
                        msg = await dest.send(content=content_bot, embed=embed, view=view)
                    if not first_msg_id:
                        first_msg_id = str(msg.id)
                        first_ch_id = str(msg.channel.id)
                        self._last_error = ""
                except Exception as exc:
                    self._last_error = f"Bot send error: {exc}"

        # 2. Fallback: Discord Webhook (Zero-Bot Setup using Wi-Fi / Tailscale action links)
        if not first_msg_id and self.has_webhook():
            try:
                base = get_reachable_base_url()
                embed_dict = self._build_clip_embed_dict(clip, include_webhook_links=True)
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

        if first_msg_id:
            conn = get_db()
            with conn:
                conn.execute(
                    "UPDATE clips SET discord_message_id = ?, discord_channel_id = ? WHERE id = ?",
                    (first_msg_id, first_ch_id or "", clip_id),
                )
            conn.close()
        return first_msg_id

    async def update_webhook_message_status(self, discord_message_id: str, status_text: str) -> None:
        """Edits the Discord message content in-place (Bot channel message or Webhook) to show live publish/reject status."""
        if not discord_message_id:
            return
        try:
            clip = get_clip_by_discord_message(discord_message_id)
            ch_id_raw = str((clip or {}).get("discord_channel_id") or "")
            if ch_id_raw.isdigit() and self._connected and self._client:
                try:
                    cid = int(ch_id_raw)
                    channel = self._client.get_channel(cid) or await self._client.fetch_channel(cid)
                    if channel:
                        msg = await channel.fetch_message(int(discord_message_id))
                        await msg.edit(content=status_text[:1900])
                        return
                except Exception:
                    pass
            if self.has_webhook():
                async with httpx.AsyncClient(timeout=12.0) as client:
                    await client.patch(
                        f"{self._webhook_base_url()}/messages/{discord_message_id}",
                        json={"content": status_text[:1900]},
                    )
        except Exception:
            pass

    async def _handle_button_interaction(self, interaction: discord.Interaction, custom_id: str) -> None:
        parts = custom_id.split(":", 2)
        if len(parts) != 3:
            return
        _, action, clip_id = parts
        clip = get_clip(clip_id)
        if not clip:
            await interaction.response.send_message(f"⚠️ Clip `{clip_id}` no longer exists.", ephemeral=True)
            return

        lock_key = f"btn:{clip_id}:{action}"
        if lock_key in self._processing_msgs:
            await interaction.response.send_message("⏳ Action already in progress for this clip.", ephemeral=True)
            return

        self._processing_msgs.add(lock_key)
        try:
            await interaction.response.defer(ephemeral=True)

            if action == "reject":
                conn = get_db()
                with conn:
                    conn.execute("UPDATE clips SET status = 'rejected' WHERE id = ?", (clip_id,))
                conn.close()
                if settings.AUTO_DELETE_ON_REJECT:
                    for p in (clip.get("raw_path"), clip.get("rendered_path"), clip.get("thumb_path")):
                        if p and os.path.isfile(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                if interaction.message:
                    try:
                        await interaction.message.edit(
                            content=f"🗑️ **Clip `{clip_id}` rejected and deleted.**",
                            view=self._create_bot_review_view(clip_id, rejected=True),
                        )
                    except Exception:
                        pass
                await interaction.followup.send(f"🗑️ Clip `{clip_id}` rejected and cleaned up.", ephemeral=True)
                return

            if action == "all":
                platforms = get_configured_platforms()
                if not platforms:
                    await interaction.followup.send("⚠️ No social platforms are configured yet. Open Web UI > Configure.", ephemeral=True)
                    return
            elif action in ("youtube_shorts", "tiktok", "instagram"):
                platforms = [action]
            else:
                await interaction.followup.send(f"⚠️ Unknown action `{action}`.", ephemeral=True)
                return

            conn = get_db()
            existing_rows = conn.execute(
                "SELECT platform, post_url FROM post_history WHERE clip_id = ? AND status = 'posted' ORDER BY published_at DESC",
                (clip_id,),
            ).fetchall()
            already_posted: Dict[str, str] = {}
            for er in existing_rows:
                if er["platform"] not in already_posted:
                    already_posted[er["platform"]] = er["post_url"] or ""
            pending = [p for p in platforms if p not in already_posted]

            if not pending:
                conn.close()
                if interaction.message:
                    try:
                        await interaction.message.edit(
                            view=self._create_bot_review_view(clip_id, posted_platforms=set(already_posted.keys()))
                        )
                    except Exception:
                        pass
                done_lines = [
                    f"✅ **{p.upper()}**: Already Posted! {already_posted[p]}".strip()
                    for p in platforms
                ]
                await interaction.followup.send("\n".join(done_lines), ephemeral=True)
                return

            with conn:
                conn.execute("UPDATE clips SET status = 'accepted' WHERE id = ?", (clip_id,))
            conn.close()

            if interaction.message:
                try:
                    await interaction.message.edit(
                        content=f"⏳ **Publishing `{clip_id}` to {', '.join(pending)}...**"
                    )
                except Exception:
                    pass

            res = await asyncio.to_thread(publish_clip_to_platforms, clip_id, pending)

            conn = get_db()
            posted_rows = conn.execute(
                "SELECT DISTINCT platform FROM post_history WHERE clip_id = ? AND status = 'posted'",
                (clip_id,),
            ).fetchall()
            conn.close()
            posted_set = {r["platform"] for r in posted_rows}

            lines = [
                f"✅ **{item.get('platform', '').upper()}**: Posted! {item.get('post_url', '')}".strip()
                if item.get("success")
                else f"❌ **{item.get('platform', '').upper()}**: {item.get('error', 'Failed')}"
                for item in res.get("results", [])
            ]
            summary = "\n".join(lines) or "Finished publishing."

            if interaction.message:
                try:
                    await interaction.message.edit(
                        content=summary[:1900],
                        view=self._create_bot_review_view(clip_id, posted_platforms=posted_set),
                    )
                except Exception:
                    pass
            await interaction.followup.send(summary[:1900], ephemeral=True)
        except Exception as exc:
            print(f"[DiscordBot] Button interaction error: {exc}")
            try:
                await interaction.followup.send(f"❌ Error processing button: {exc}", ephemeral=True)
            except Exception:
                pass
        finally:
            self._processing_msgs.discard(lock_key)

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
