# ⚡ GTA VI AI AutoClipper

Local-first multi-stream AI AutoClipper built to monitor **tens of Twitch & YouTube live streams simultaneously** on modest hardware (tested on **GTX 1650 4GB VRAM + 32GB RAM**), detect viral moments using **80% live chat velocity/hype + 20% AI visual inspection**, cut the last 30–60 seconds, format into **Smart 9:16 Gaming Split** vertical clips, send them to your **Discord bot** for emoji-reaction approval, and auto-publish to **YouTube Shorts, TikTok, and Instagram Reels**.

---

## ✨ Key Features

- **Multi-Stream Zero-Decode Ring Buffer**: Uses `streamlink` / `yt-dlp` + `ffmpeg -c copy` circular MPEG-TS segments (~90s rolling window per stream). Watching 15–25 live streams uses **0% GPU** and minimal CPU.
- **80% Chat Hype Velocity + 20% AI Visual & Facecam Reaction Evaluation**:
  - **Twitch Anonymous WebSocket IRC** + **YouTube Live Chat Innertube Poller** (no Twitch/YouTube API keys required) compute adaptive baseline z-scores and hype-keyword density (`OMG`, `WTF`, `CLIP IT`, `WASTED`, `6 STARS`, etc.).
  - **Local Vision AI (`Qwen2.5-VL-3B-Instruct` recommended or `SmolVLM-500M-Instruct` via `llama-server`)** inspects side-by-side `[Gameplay | 2.5x Zoomed Facecam]` frames so it recognizes both game action and shocked streamer reactions while staying inside 4GB VRAM.
  - **Facecam-First Preference**: Auto-detects streamer facecams via OpenCV, prioritizes facecam streams in active buffer slots, and scores sudden webcam motion spikes.
- **Post-Clip Whisper + Burned-In Streamer Captions & Smart 9:16 Split**:
  - Transcribes what the streamer shouted via CPU `faster-whisper`, analyzes 3 keyframes + chat slang, generates a short punchy caption naming the streamer, and burns it crisply onto the **1080×1920 Smart 9:16 Gaming Split** video using NVIDIA **NVENC (`h264_nvenc`)** (with automatic `libx264` CPU fallback).
- **Live Statistics & System Monitor Dashboard**:
  - Real-time clip pipeline progress bars, per-stream FFmpeg PID / buffer MB / facecam / chat baseline diagnostics table, and live engine activity console.
- **Discord Review Bot**:
  - Sends every captured clip directly to your Discord channel/DM with AI title, hype breakdown, and inline video preview.
  - React with:
    - `✅` Publish to **all** configured platforms
    - `▶️` Publish to **YouTube Shorts**
    - `🎵` Publish to **TikTok**
    - `📸` Publish to **Instagram Reels**
    - `❌` Reject & delete clip
  - Supports `!status`, `!clip <streamer>`, and `!add <streamer_or_url>` commands.
- **Built-in Configure Web UI (No Manual `.env` Editing Required)**:
  - Manage streams, 80/20 weights, clip duration (30–60s), VLM model downloads, Discord bot token, and YouTube OAuth / TikTok session / Instagram Graph API directly in the browser.

---

## 🚀 Quick Start (For You & Friends)

### 1. Prerequisites
- **Python 3.10+**
- **FFmpeg** (must be on `PATH` or configured in `FFMPEG_PATH`)
- Optional for local Vision AI: `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) (or Ollama / OpenAI / Gemini API)

### 2. Install & Launch
```bash
git clone https://github.com/tgnfrzt/autoClipper.git
cd autoClipper

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
python run.py
```
Then open **http://127.0.0.1:17430** in your browser and go to the **Configure** tab!

*(On Windows, you can also double-click `run_local.bat`.)*

---

## ⚙️ Setting Up Discord & Social Publishing (Configure Tab)

1. **Vision AI**: In **Configure → AI Vision**, click **Download** on `SmolVLM-500M-Instruct (~520 MB)` (recommended for 4GB VRAM GPUs like GTX 1650) and click **Start VLM Server**, or choose any OpenAI-compatible API endpoint.
2. **Discord Bot**:
   - Create an application in the [Discord Developer Portal](https://discord.com/developers/applications) → **Bot** → copy the Token.
   - Enable **Message Content Intent** (for `!status` / `!clip` commands) and invite the bot with **Send Messages**, **Attach Files**, **Embed Links**, and **Add Reactions** permissions.
   - Paste the **Bot Token** and target **Channel ID** (or your **User ID** for DMs) in the **Configure** tab and click **Save Settings & Connect**.
3. **YouTube Shorts**: Upload your Google OAuth `client_secret.json` (`http://localhost:17430/api/settings/youtube/callback` as authorized redirect URI) and click **1-Click Authorize YouTube**.
4. **TikTok**: Paste your `sessionid` cookie from `tiktok.com` (or upload `tiktok_cookies.txt`, or use Official Content Posting API token).
5. **Instagram Reels**: Enter your Instagram Business ID, Meta Page Access Token, and Temporary Public Hosting Relay URL.

---

## 🔒 Credential Safety
`.env`, `storage/`, `*.db`, `*.gguf`, `client_secret.json`, `youtube_token.json`, and `*cookies*.txt` are all gitignored so personal credentials are never committed when sharing this repository.
