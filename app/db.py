import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from . import config

_lock = threading.Lock()


def _connect():
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


_conn = None


@contextmanager
def cursor():
    global _conn
    with _lock:
        if _conn is None:
            _conn = _connect()
        cur = _conn.cursor()
        try:
            yield cur
            _conn.commit()
        except Exception:
            _conn.rollback()
            raise


def init():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.SNAP_DIR.mkdir(parents=True, exist_ok=True)
    with cursor() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS whitelist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL UNIQUE,
                owner TEXT DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                channel_id TEXT NOT NULL DEFAULT '0',
                name TEXT DEFAULT '',
                rtsp_url TEXT DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                UNIQUE(device_id, channel_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                device_id TEXT, channel_id TEXT, msg_type TEXT,
                plate TEXT, confidence REAL,
                allowed INTEGER, action TEXT, details TEXT,
                image TEXT
            );
            CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
            CREATE TABLE IF NOT EXISTS raw_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL, body TEXT
            );
            """
        )
        # миграции
        cols = {r["name"] for r in c.execute("PRAGMA table_info(cameras)").fetchall()}
        if "role" not in cols:
            c.execute("ALTER TABLE cameras ADD COLUMN role TEXT NOT NULL DEFAULT 'entry'")
        cols = {r["name"] for r in c.execute("PRAGMA table_info(events)").fetchall()}
        if "images" not in cols:
            c.execute("ALTER TABLE events ADD COLUMN images TEXT")
        if "label_type" not in cols:
            c.execute("ALTER TABLE events ADD COLUMN label_type TEXT")
        old = c.execute("SELECT value FROM settings WHERE key='trigger_msg_types'").fetchone()
        if old and old["value"].startswith("aiVehDetect,aiVehLine,aiVehArea,detection.vehicle.detected"):
            # старый фильтр по msgType заменён фильтром labelType=carAlarm
            c.execute("UPDATE settings SET value='' WHERE key='trigger_msg_types'")
        ver = c.execute("SELECT value FROM settings WHERE key='_schema'").fetchone()
        if not ver or int(ver["value"]) < 2:
            # v2: по умолчанию кадр берётся на момент события (фото тревоги / запись SD), а не «сейчас»
            c.execute("UPDATE settings SET value='smart' WHERE key='image_source' AND value IN ('snapshot','event')")
            c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES ('_schema', '2')")
        for k, v in config.DEFAULT_SETTINGS.items():
            c.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (k, v))


# ---------- settings ----------
def get_settings() -> dict:
    with cursor() as c:
        rows = c.execute("SELECT key, value FROM settings").fetchall()
    s = dict(config.DEFAULT_SETTINGS)
    s.update({r["key"]: r["value"] for r in rows})
    return s


def set_settings(values: dict):
    with cursor() as c:
        for k, v in values.items():
            if k in config.DEFAULT_SETTINGS:
                c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)", (k, str(v)))


# ---------- whitelist ----------
def whitelist(active_only=False):
    q = "SELECT * FROM whitelist" + (" WHERE active=1" if active_only else "") + " ORDER BY plate"
    with cursor() as c:
        return [dict(r) for r in c.execute(q).fetchall()]


def whitelist_add(plate, owner=""):
    with cursor() as c:
        c.execute(
            "INSERT INTO whitelist(plate, owner, active, created_at) VALUES (?,?,1,?) "
            "ON CONFLICT(plate) DO UPDATE SET owner=excluded.owner, active=1",
            (plate, owner, time.time()),
        )


def whitelist_delete(wid):
    with cursor() as c:
        c.execute("DELETE FROM whitelist WHERE id=?", (wid,))


def whitelist_toggle(wid):
    with cursor() as c:
        c.execute("UPDATE whitelist SET active = 1 - active WHERE id=?", (wid,))


# ---------- cameras ----------
def cameras(active_only=False):
    q = "SELECT * FROM cameras" + (" WHERE active=1" if active_only else "") + " ORDER BY id"
    with cursor() as c:
        return [dict(r) for r in c.execute(q).fetchall()]


def camera_get(cid):
    with cursor() as c:
        r = c.execute("SELECT * FROM cameras WHERE id=?", (cid,)).fetchone()
    return dict(r) if r else None


def camera_find(device_id, channel_id):
    with cursor() as c:
        r = c.execute(
            "SELECT * FROM cameras WHERE device_id=? AND (channel_id=? OR ?='')",
            (device_id, str(channel_id or ""), str(channel_id or "")),
        ).fetchone()
        if not r:
            r = c.execute("SELECT * FROM cameras WHERE device_id=?", (device_id,)).fetchone()
    return dict(r) if r else None


def camera_upsert(device_id, channel_id, name="", rtsp_url="", role=None):
    with cursor() as c:
        c.execute(
            "INSERT INTO cameras(device_id, channel_id, name, rtsp_url, role) VALUES (?,?,?,?,?) "
            "ON CONFLICT(device_id, channel_id) DO UPDATE SET name=excluded.name, rtsp_url=excluded.rtsp_url"
            + (", role=excluded.role" if role else ""),
            (device_id, str(channel_id), name, rtsp_url, role or "entry"),
        )


def camera_set_role(cid, role):
    with cursor() as c:
        c.execute("UPDATE cameras SET role=? WHERE id=?", (role, cid))


def camera_delete(cid):
    with cursor() as c:
        c.execute("DELETE FROM cameras WHERE id=?", (cid,))


def camera_toggle(cid):
    with cursor() as c:
        c.execute("UPDATE cameras SET active = 1 - active WHERE id=?", (cid,))


# ---------- events ----------
def event_add(**kw):
    kw.setdefault("ts", time.time())
    cols = ",".join(kw)
    with cursor() as c:
        c.execute(f"INSERT INTO events({cols}) VALUES ({','.join('?' * len(kw))})", tuple(kw.values()))
        return c.lastrowid


def _ev(r):
    d = dict(r)
    try:
        d["frames"] = json.loads(d.get("images") or "[]")
    except Exception:
        d["frames"] = []
    if not d["frames"] and d.get("image"):
        d["frames"] = [{"file": d["image"], "note": ""}]
    return d


EVENT_FILTERS = {
    "allowed": "allowed=1 AND plate IS NOT NULL",
    "denied": "allowed=0 AND plate IS NOT NULL",
    "unknown": "plate IS NULL AND details NOT LIKE '%\"role\": \"exit\"%' AND msg_type!='manual'",
    "exit": "details LIKE '%\"role\": \"exit\"%'",
    "manual": "msg_type='manual'",
}


def events(limit=100, flt="", q=""):
    where, args = [], []
    if flt in EVENT_FILTERS:
        where.append(EVENT_FILTERS[flt])
    if q:
        where.append("plate LIKE ?")
        args.append(f"%{q}%")
    sql = "SELECT * FROM events" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT ?"
    with cursor() as c:
        return [_ev(r) for r in c.execute(sql, (*args, limit)).fetchall()]


def stats_today(since_ts):
    with cursor() as c:
        r = c.execute(
            "SELECT COUNT(*) AS total, SUM(action IN ('open','open-manual')) AS opened, "
            "SUM(allowed=0 AND plate IS NOT NULL) AS denied FROM events WHERE ts>=?", (since_ts,)).fetchone()
    return {k: r[k] or 0 for k in ("total", "opened", "denied")}


def event_get(eid):
    with cursor() as c:
        r = c.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
    return _ev(r) if r else None


def last_action_ts(device_id, channel_id, action):
    with cursor() as c:
        r = c.execute("SELECT MAX(ts) AS t FROM events WHERE device_id=? AND channel_id=? AND action=?",
                      (device_id, str(channel_id), action)).fetchone()
    return r["t"] if r and r["t"] else 0


def raw_add(body: str):
    with cursor() as c:
        c.execute("INSERT INTO raw_messages(ts, body) VALUES (?, ?)", (time.time(), body[:20000]))
        c.execute("DELETE FROM raw_messages WHERE id < (SELECT MAX(id) - 200 FROM raw_messages)")


def raw_list(limit=30):
    with cursor() as c:
        return [dict(r) for r in c.execute("SELECT * FROM raw_messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


def last_allowed_ts(plate):
    with cursor() as c:
        r = c.execute(
            "SELECT MAX(ts) AS t FROM events WHERE plate=? AND allowed=1 AND action='open'", (plate,)
        ).fetchone()
    return r["t"] if r and r["t"] else 0


def cleanup(keep_days):
    cutoff = time.time() - keep_days * 86400
    with cursor() as c:
        rows = c.execute("SELECT image, images FROM events WHERE ts < ? AND (image IS NOT NULL OR images IS NOT NULL)",
                         (cutoff,)).fetchall()
        c.execute("UPDATE events SET image=NULL, images=NULL WHERE ts < ?", (cutoff,))
        c.execute("DELETE FROM events WHERE ts < ?", (cutoff - 90 * 86400,))
    for r in rows:
        files = [r["image"]] if r["image"] else []
        try:
            files += [f["file"] for f in json.loads(r["images"] or "[]")]
        except Exception:
            pass
        for f in set(files):
            try:
                (config.SNAP_DIR / f).unlink(missing_ok=True)
            except Exception:
                pass
