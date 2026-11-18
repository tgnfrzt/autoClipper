import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.core.config import settings

DEFAULT_POPULAR_STREAMERS = [
    ("twitch", "xqc", "xQc", "https://www.twitch.tv/xqc", 0, 1, 90),
    ("twitch", "kaicenat", "KaiCenat", "https://www.twitch.tv/kaicenat", 0, 1, 95),
    ("twitch", "lirik", "LIRIK", "https://www.twitch.tv/lirik", 0, 1, 85),
    ("twitch", "summit1g", "summit1g", "https://www.twitch.tv/summit1g", 0, 1, 80),
    ("twitch", "shroud", "shroud", "https://www.twitch.tv/shroud", 0, 1, 78),
    ("twitch", "caseoh_", "caseoh_", "https://www.twitch.tv/caseoh_", 0, 1, 82),
    ("twitch", "jynxzi", "Jynxzi", "https://www.twitch.tv/jynxzi", 0, 1, 76),
    ("twitch", "hasanabi", "HasanAbi", "https://www.twitch.tv/hasanabi", 0, 0, 60),
    ("twitch", "tarik", "tarik", "https://www.twitch.tv/tarik", 0, 0, 60),
    ("youtube", "@IShowSpeed", "IShowSpeed", "https://www.youtube.com/@IShowSpeed/live", 0, 1, 92),
]


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    conn = get_db()
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS streamers (
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                url TEXT NOT NULL,
                is_manual INTEGER NOT NULL DEFAULT 1,
                enabled INTEGER NOT NULL DEFAULT 1,
                priority INTEGER NOT NULL DEFAULT 50,
                is_live INTEGER NOT NULL DEFAULT 0,
                viewer_count INTEGER NOT NULL DEFAULT 0,
                title TEXT DEFAULT '',
                game_name TEXT DEFAULT '',
                has_facecam INTEGER NOT NULL DEFAULT 0,
                facecam_corner TEXT DEFAULT '',
                last_checked_at TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS clips (
                id TEXT PRIMARY KEY,
                stream_id TEXT NOT NULL,
                streamer_name TEXT NOT NULL,
                platform TEXT NOT NULL,
                stream_url TEXT NOT NULL,
                raw_path TEXT NOT NULL,
                rendered_path TEXT DEFAULT '',
                thumb_path TEXT DEFAULT '',
                duration_sec REAL DEFAULT 0.0,
                chat_score REAL DEFAULT 0.0,
                visual_score REAL DEFAULT 0.0,
                facecam_score REAL DEFAULT 0.0,
                combined_score REAL DEFAULT 0.0,
                chat_mps REAL DEFAULT 0.0,
                chat_keywords TEXT DEFAULT '[]',
                facecam_corner TEXT DEFAULT '',
                transcript TEXT DEFAULT '',
                overlay_caption TEXT DEFAULT '',
                ai_title TEXT DEFAULT '',
                ai_summary TEXT DEFAULT '',
                ai_caption TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                discord_message_id TEXT DEFAULT '',
                discord_channel_id TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS post_history (
                id TEXT PRIMARY KEY,
                clip_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                status TEXT NOT NULL,
                post_url TEXT DEFAULT '',
                error_message TEXT DEFAULT '',
                published_at TEXT NOT NULL
            );
            """
        )

        # Idempotent schema migrations for existing SQLite databases
        migrations = [
            "ALTER TABLE streamers ADD COLUMN has_facecam INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE streamers ADD COLUMN facecam_corner TEXT DEFAULT ''",
            "ALTER TABLE streamers ADD COLUMN in_paid_watchlist INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE streamers ADD COLUMN rpm_rate REAL NOT NULL DEFAULT 0.0",
            "ALTER TABLE streamers ADD COLUMN language TEXT DEFAULT 'en'",
            "ALTER TABLE clips ADD COLUMN facecam_score REAL DEFAULT 0.0",
            "ALTER TABLE clips ADD COLUMN facecam_corner TEXT DEFAULT ''",
            "ALTER TABLE clips ADD COLUMN transcript TEXT DEFAULT ''",
            "ALTER TABLE clips ADD COLUMN overlay_caption TEXT DEFAULT ''",
            "ALTER TABLE post_history ADD COLUMN views INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE post_history ADD COLUMN likes INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE post_history ADD COLUMN comments INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE post_history ADD COLUMN external_post_id TEXT DEFAULT ''",
            "ALTER TABLE post_history ADD COLUMN last_synced_at TEXT DEFAULT ''",
        ]
        for sql in migrations:
            try:
                conn.execute(sql)
            except Exception:
                pass

        # Seed paid watchlist initially from manual streamers and already-clipped streamers
        try:
            paid_cnt = conn.execute("SELECT COUNT(*) FROM streamers WHERE in_paid_watchlist = 1").fetchone()[0]
            if paid_cnt == 0:
                conn.execute(
                    """
                    UPDATE streamers SET in_paid_watchlist = 1
                    WHERE is_manual = 1
                       OR id IN (SELECT DISTINCT stream_id FROM clips)
                       OR LOWER(display_name) IN (SELECT LOWER(streamer_name) FROM clips)
                    """
                )
        except Exception:
            pass

        cur = conn.execute("SELECT COUNT(*) FROM streamers")
        if cur.fetchone()[0] == 0:
            now = datetime.utcnow().isoformat()
            for platform, channel_id, name, url, is_manual, enabled, priority in DEFAULT_POPULAR_STREAMERS:
                sid = f"{platform}:{channel_id.lower()}"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO streamers
                    (id, platform, channel_id, display_name, url, is_manual, enabled, priority, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (sid, platform, channel_id, name, url, is_manual, enabled, priority, now),
                )
    conn.close()


def dict_from_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


CORE_POPULAR_HANDLES = {
    "kaicenat", "xqc", "ishowspeed", "@ishowspeed", "caseoh_", "jynxzi",
    "lirik", "stableronaldo", "lacy", "clix", "yourrage", "yourragegaming",
    "shroud", "summit1g", "zackrawrr", "theburntpeanut", "rubius",
}


def compute_stream_rank_score(
    row: Dict[str, Any],
    prefer_facecam: Optional[bool] = None,
    prefer_english: Optional[bool] = None,
) -> float:
    use_fc = settings.PREFER_FACECAM_STREAMS if prefer_facecam is None else prefer_facecam
    prefer_en = settings.PREFER_ENGLISH_STREAMS if prefer_english is None else prefer_english
    lang = (row.get("language") or "en").strip().lower()
    is_en = lang == "en" or lang.startswith("en-")
    cid_low = (row.get("channel_id") or "").strip().lower()
    name_low = (row.get("display_name") or "").strip().lower()
    is_pinned = bool(
        row.get("in_paid_watchlist")
        or row.get("is_manual")
        or cid_low in CORE_POPULAR_HANDLES
        or name_low in CORE_POPULAR_HANDLES
    )
    base_v = max(int(row.get("viewer_count") or 0), 8000 if is_pinned else 0)
    lang_mult = 2.6 if (prefer_en and is_en) else 1.0
    fc_mult = 1.25 if (use_fc and row.get("has_facecam")) else 1.0
    prio_mult = 1.0 + max(0, int(row.get("priority") or 50)) / 150.0
    return base_v * lang_mult * fc_mult * prio_mult


def fetch_all_streamers(
    enabled_only: bool = False,
    prefer_facecam: Optional[bool] = None,
    prefer_english: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    conn = get_db()
    try:
        q = "SELECT * FROM streamers"
        if enabled_only:
            q += " WHERE enabled = 1"
        rows = [dict(r) for r in conn.execute(q).fetchall()]
        rows.sort(
            key=lambda r: (
                1 if r.get("is_live") == 1 else 0,
                1 if r.get("in_paid_watchlist") == 1 else 0,
                compute_stream_rank_score(r, prefer_facecam=prefer_facecam, prefer_english=prefer_english),
                int(r.get("viewer_count") or 0),
            ),
            reverse=True,
        )
        return rows
    finally:
        conn.close()


def update_streamer_facecam(stream_id: str, has_facecam: bool, corner: str = "") -> None:
    conn = get_db()
    with conn:
        conn.execute(
            "UPDATE streamers SET has_facecam = ?, facecam_corner = ? WHERE id = ?",
            (1 if has_facecam else 0, corner, stream_id),
        )
    conn.close()


def upsert_streamer(
    platform: str,
    channel_id: str,
    display_name: str,
    url: str,
    is_manual: bool = True,
    enabled: bool = True,
    priority: int = 50,
    is_live: bool = False,
    viewer_count: int = 0,
    title: str = "",
    game_name: str = "",
    in_paid_watchlist: Optional[bool] = None,
    language: str = "en",
) -> str:
    sid = f"{platform.lower()}:{channel_id.lower()}"
    now = datetime.utcnow().isoformat()
    lang_clean = (language or "en").strip().lower()
    conn = get_db()
    with conn:
        existing = conn.execute(
            "SELECT id, is_manual, enabled, priority, in_paid_watchlist FROM streamers WHERE id = ?",
            (sid,),
        ).fetchone()
        if existing:
            next_paid = existing["in_paid_watchlist"] if in_paid_watchlist is None else (1 if in_paid_watchlist else 0)
            next_manual = 1 if is_manual else existing["is_manual"]
            next_enabled = 1 if in_paid_watchlist else existing["enabled"]
            next_prio = max(int(existing["priority"] or 50), int(priority)) if is_manual else int(existing["priority"] or priority)
            conn.execute(
                """
                UPDATE streamers
                SET display_name = ?, url = ?, is_live = ?, viewer_count = ?,
                    title = ?, game_name = ?, language = ?, priority = ?, last_checked_at = ?,
                    is_manual = ?, enabled = ?, in_paid_watchlist = ?
                WHERE id = ?
                """,
                (
                    display_name,
                    url,
                    1 if is_live else 0,
                    viewer_count,
                    title,
                    game_name,
                    lang_clean,
                    next_prio,
                    now,
                    next_manual,
                    next_enabled,
                    next_paid,
                    sid,
                ),
            )
        else:
            paid_flag = 1 if (in_paid_watchlist is True or (in_paid_watchlist is None and is_manual)) else 0
            conn.execute(
                """
                INSERT INTO streamers
                (id, platform, channel_id, display_name, url, is_manual, enabled, priority,
                 is_live, viewer_count, title, game_name, language, in_paid_watchlist, last_checked_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    platform.lower(),
                    channel_id,
                    display_name,
                    url,
                    1 if is_manual else 0,
                    1 if enabled else 0,
                    priority,
                    1 if is_live else 0,
                    viewer_count,
                    title,
                    game_name,
                    lang_clean,
                    paid_flag,
                    now,
                    now,
                ),
            )
    conn.close()
    return sid


def list_clips(limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM clips ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        clips = [dict(r) for r in rows]
        for c in clips:
            posts = conn.execute(
                "SELECT * FROM post_history WHERE clip_id = ? ORDER BY published_at DESC",
                (c["id"],),
            ).fetchall()
            c["posts"] = [dict(p) for p in posts]
        return clips
    finally:
        conn.close()


def get_clip(clip_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
        if not row:
            return None
        c = dict(row)
        posts = conn.execute(
            "SELECT * FROM post_history WHERE clip_id = ? ORDER BY published_at DESC",
            (clip_id,),
        ).fetchall()
        c["posts"] = [dict(p) for p in posts]
        return c
    finally:
        conn.close()


def get_clip_by_discord_message(message_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM clips WHERE discord_message_id = ?",
            (str(message_id),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
