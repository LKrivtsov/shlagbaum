"""Обработка события с камеры.

Камера «въезд»: кадры -> распознавание -> белый список -> SIP + Telegram.
Камера «выезд»: событие -> сразу SIP (без распознавания) + Telegram.
"""
import asyncio
import html
import json
import logging
import threading
import time
import uuid
from datetime import datetime

import cv2

from . import anpr, config, db, sip, telegram
from .imou import client as imou

log = logging.getLogger("processor")

_busy: set[str] = set()          # камеры, которые сейчас обрабатываются
_seen_ids: dict[str, float] = {}  # дедупликация сообщений Imou по id

SOURCE_NAMES = {"snapshot": "снимок", "hls": "видеопоток HD", "event": "фото события", "rtsp": "RTSP"}


# ---------------------------------------------------------------- разбор push
def _find_key(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k.lower() == key.lower() and isinstance(v, (str, int)):
                return str(v)
        for v in o.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def parse_message(msg: dict) -> dict:
    """Приводим форматы push-сообщений Imou к одному виду."""
    msg_type = msg.get("msgType") or msg.get("type") or ""
    if msg_type == "iotEvent":
        msg_type = f"iotEvent:{(msg.get('content') or {}).get('event', '')}"
    device = msg.get("did") or msg.get("deviceId") or ""
    channel = msg.get("cid", msg.get("channelId", "0"))
    if channel in (None, "", -1, "-1"):
        channel = "0"
    pics = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and o.startswith("http") and any(
            x in o.lower() for x in (".jpg", ".jpeg", ".png", "pic", "image", "snap")
        ):
            pics.append(o)

    walk(msg)
    # миниатюры — в конец списка
    pics.sort(key=lambda u: ("thumb" in u.lower() or "small" in u.lower()))
    return {"id": str(msg.get("id") or msg.get("alarmId") or ""), "msg_type": str(msg_type),
            "label_type": _find_key(msg, "labelType") or "",
            "device_id": str(device), "channel_id": str(channel), "pictures": pics}


def _csv(v):
    return [t.strip() for t in (v or "").split(",") if t.strip()]


def should_trigger(ev: dict, settings: dict) -> bool:
    labels = _csv(settings.get("trigger_label_types"))
    if labels and ev["label_type"] not in labels:
        return False
    types = _csv(settings.get("trigger_msg_types"))
    if types and ev["msg_type"] not in types:
        return False
    return bool(ev["device_id"])


async def handle_push(msg: dict):
    ev = parse_message(msg)
    now = time.time()
    for k in [k for k, t in _seen_ids.items() if now - t > 600]:
        _seen_ids.pop(k, None)
    if ev["id"]:
        if ev["id"] in _seen_ids:
            return
        _seen_ids[ev["id"]] = now
    settings = db.get_settings()
    if not should_trigger(ev, settings):
        log.info("пропуск события msgType=%s labelType=%s", ev["msg_type"], ev["label_type"])
        return
    cam = db.camera_find(ev["device_id"], ev["channel_id"])
    if not cam or not cam["active"]:
        log.info("событие с неизвестной/выключенной камеры %s/%s", ev["device_id"], ev["channel_id"])
        return
    await process_camera(cam, ev["msg_type"], ev["pictures"], label_type=ev["label_type"])


# ---------------------------------------------------------------- источники кадров
class StreamReader:
    """Фоновое чтение RTSP/HLS: всегда держим самый свежий кадр."""

    def __init__(self, url):
        self.url = url
        self.frame = None
        self.seq = 0
        self.stop = False
        self.error = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                self.error = "не удалось открыть поток"
                return
            fails = 0
            while not self.stop:
                ok, fr = cap.read()
                if not ok:
                    fails += 1
                    if fails > 50:
                        self.error = "поток оборвался"
                        return
                    time.sleep(0.05)
                    continue
                fails = 0
                self.frame, self.seq = fr, self.seq + 1
        finally:
            cap.release()

    async def next(self, after_seq=0, timeout=15):
        end = time.time() + timeout
        while time.time() < end:
            # первые кадры потока часто битые — ждём хотя бы 5
            if self.frame is not None and self.seq > max(after_seq, 5):
                return self.frame, self.seq
            if self.error:
                raise RuntimeError(self.error)
            await asyncio.sleep(0.1)
        raise RuntimeError("нет кадров из потока")

    def close(self):
        self.stop = True


class FrameSource:
    def __init__(self, cam, pictures, settings):
        self.cam = cam
        self.pictures = list(pictures or [])
        self.mode = "rtsp" if cam.get("rtsp_url") else (settings.get("image_source") or "snapshot")
        self.stream = None
        self.seq = 0

    async def get(self):
        """Возвращает (img, источник). При ошибке основного источника — снимок через облако."""
        try:
            return await self._get(self.mode)
        except Exception as e:
            if self.mode == "snapshot":
                if self.pictures:
                    log.warning("снимок не получен (%s) — берём фото события", e)
                    return await self._get("event")
                raise
            log.warning("источник %s: %s — берём снимок через облако", self.mode, e)
            return await self._get("snapshot")

    async def _get(self, mode):
        if mode == "event":
            if not self.pictures:
                raise RuntimeError("в событии нет фото")
            url = self.pictures.pop(0)
            return anpr.decode_image(await imou.download(url)), mode
        if mode in ("rtsp", "hls"):
            if self.stream is None:
                url = self.cam["rtsp_url"] if mode == "rtsp" else \
                    await imou.hls_url(self.cam["device_id"], self.cam["channel_id"])
                self.stream = StreamReader(url)
            frame, self.seq = await self.stream.next(self.seq)
            return frame, mode
        data = await imou.snapshot(self.cam["device_id"], self.cam["channel_id"])
        return anpr.decode_image(data), "snapshot"

    def close(self):
        if self.stream:
            self.stream.close()


# ---------------------------------------------------------------- утилиты
def match_whitelist(plate: str, settings: dict):
    tol = int(settings.get("fuzzy_match") or 0)
    best = None
    for w in db.whitelist(active_only=True):
        wp = anpr.normalize(w["plate"])
        d = anpr.levenshtein(plate, wp)
        if d <= tol and (best is None or d < best[1]):
            best = (w, d)
    return best


async def recognize_bytes(data: bytes):
    img = anpr.decode_image(data)
    if img is None:
        return None, []
    results = await asyncio.to_thread(anpr.get().recognize, img)
    return img, results


def _save_frame(img, results, cam, note) -> tuple[str, bytes]:
    ann = anpr.annotate(img, results)
    h, w = ann.shape[:2]
    # сохраняем крупно, чтобы номер было видно; превью делает браузер
    if w > 1920:
        ann = cv2.resize(ann, (1920, int(h * 1920 / w)))
    cv2.putText(ann, note, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(ann, note, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    fname = f"{datetime.now(config.TZ):%Y%m%d_%H%M%S}_{cam['id']}_{uuid.uuid4().hex[:6]}.jpg"
    (config.SNAP_DIR / fname).write_bytes(jpg)
    return fname, jpg


async def _open_gate(s):
    if s.get("gate_enabled") != "1":
        return "gate-disabled", ""
    try:
        res = await asyncio.to_thread(sip.call, sip.SipConfig.from_settings(s))
        return ("open" if res["ok"] else "open-failed"), res["status"]
    except Exception as e:
        return "open-failed", str(e)


# ---------------------------------------------------------------- основной процесс
async def process_camera(cam: dict, msg_type: str = "manual", pictures: list | None = None,
                         force_open=True, label_type=""):
    key = f"{cam['device_id']}/{cam['channel_id']}"
    if key in _busy:
        return {"skipped": "уже обрабатывается"}
    _busy.add(key)
    try:
        if cam.get("role") == "exit":
            return await _process_exit(cam, msg_type, pictures or [], force_open, label_type)
        return await _process_entry(cam, msg_type, pictures or [], force_open, label_type)
    finally:
        _busy.discard(key)


async def _process_exit(cam, msg_type, pictures, do_actions, label_type):
    """Выезд: открываем сразу по событию, параллельно делаем один кадр для журнала."""
    s = db.get_settings()
    t0 = time.time()
    cooldown = float(s.get("cooldown_seconds") or 30)
    action, details = "none", ""
    gate_task = None
    if do_actions:
        if time.time() - db.last_action_ts(cam["device_id"], cam["channel_id"], "open") < cooldown:
            action = "skip-cooldown"
        else:
            gate_task = asyncio.create_task(_open_gate(s))

    frames, jpg = [], None
    src = FrameSource(cam, pictures, s)
    try:
        img, mode = await src.get()
        if img is not None:
            h, w = img.shape[:2]
            note = f"{SOURCE_NAMES.get(mode, mode)} {w}x{h}"
            fname, jpg = await asyncio.to_thread(_save_frame, img, [], cam, note)
            frames.append({"file": fname, "note": note})
    except Exception as e:
        details = f"кадр не получен: {e}"
    finally:
        src.close()

    if gate_task:
        action, info = await gate_task
        details = (info + ("; " + details if details else "")).strip("; ")

    ev_id = db.event_add(
        device_id=cam["device_id"], channel_id=cam["channel_id"], msg_type=msg_type, label_type=label_type,
        plate=None, confidence=None, allowed=1, action=action,
        details=json.dumps({"info": details, "owner": "", "role": "exit",
                            "elapsed": round(time.time() - t0, 2)}, ensure_ascii=False),
        image=frames[0]["file"] if frames else None, images=json.dumps(frames, ensure_ascii=False),
    )
    if do_actions and s.get("tg_notify_exit") == "1" and action != "skip-cooldown":
        status = {"open": "✅ шлагбаум открыт", "open-failed": f"⚠️ не удалось открыть: {html.escape(details)}",
                  "gate-disabled": "⏸ открытие выключено в настройках"}.get(action, action)
        await telegram.send(s, f"🚙⬅️ <b>Выезд</b>\n{status}\n📷 {html.escape(cam.get('name') or cam['device_id'])}", jpg)
    return {"event_id": ev_id, "plate": None, "raw": None, "confidence": None, "allowed": True,
            "action": action, "details": details}


async def _process_entry(cam, msg_type, pictures, do_actions, label_type):
    s = db.get_settings()
    attempts = max(1, int(s.get("attempts") or 3))
    interval = float(s.get("attempt_interval") or 1)
    min_conf = float(s.get("min_confidence") or 0.8)
    kz_only = s.get("kz_format_only") == "1"
    t0 = time.time()

    best, best_jpg, errors, frames = None, None, [], []
    src = FrameSource(cam, pictures, s)
    try:
        for i in range(attempts):
            try:
                img, mode = await src.get()
                if img is None:
                    raise RuntimeError("пустой кадр")
                results = await asyncio.to_thread(anpr.get().recognize, img)
                h, w = img.shape[:2]
                found = ", ".join(f"{r.plate} {r.confidence:.0%}" for r in results) or "номер не найден"
                note = f"{i + 1}/{attempts} {SOURCE_NAMES.get(mode, mode)} {w}x{h}: {found}"
                fname, jpg = await asyncio.to_thread(_save_frame, img, results, cam, f"{i + 1}/{attempts} {w}x{h}")
                frames.append({"file": fname, "note": note})
                for r in results:
                    if kz_only and not r.kz_valid:
                        continue
                    if best is None or r.score > best.score:
                        best, best_jpg = r, jpg
                if best and best.confidence >= min_conf and (best.kz_valid or not kz_only):
                    if match_whitelist(best.plate, s) or best.confidence > 0.95:
                        break
            except Exception as e:
                errors.append(f"кадр {i + 1}: {e}")
                frames.append({"file": None, "note": f"{i + 1}/{attempts}: ошибка — {e}"})
                log.warning("попытка %s: %s", i + 1, e)
            if i < attempts - 1:
                await asyncio.sleep(interval)
    finally:
        src.close()

    plate = best.plate if best and best.confidence >= min_conf else None
    matched = match_whitelist(plate, s) if plate else None
    allowed = bool(matched)
    action, details = "none", ""

    if plate and allowed and do_actions:
        cooldown = float(s.get("cooldown_seconds") or 30)
        if time.time() - db.last_allowed_ts(plate) < cooldown:
            action = "skip-cooldown"
        else:
            action, details = await _open_gate(s)
    elif not plate:
        details = "; ".join(errors)[:500] if errors else "номер не найден"

    saved = [f for f in frames if f["file"]]
    if best_jpg is None and saved:
        best_jpg = (config.SNAP_DIR / saved[-1]["file"]).read_bytes()
    ev_id = db.event_add(
        device_id=cam["device_id"], channel_id=cam["channel_id"], msg_type=msg_type, label_type=label_type,
        plate=plate or (best.plate if best else None), confidence=best.confidence if best else None,
        allowed=int(allowed), action=action,
        details=json.dumps({"info": details, "owner": matched[0]["owner"] if matched else "", "role": "entry",
                            "elapsed": round(time.time() - t0, 2)}, ensure_ascii=False),
        image=saved[0]["file"] if saved else None, images=json.dumps(frames, ensure_ascii=False),
    )

    if do_actions:
        await _notify(s, cam, plate, best, allowed, matched, action, details, best_jpg)
    return {"event_id": ev_id, "plate": plate, "raw": best.plate if best else None,
            "confidence": best.confidence if best else None, "allowed": allowed, "action": action,
            "details": details}


async def _notify(s, cam, plate, best, allowed, matched, action, details, jpg):
    cam_name = html.escape(cam.get("name") or cam["device_id"])
    if plate and allowed:
        if s.get("tg_notify_allowed") != "1" or action == "skip-cooldown":
            return
        owner = f" ({html.escape(matched[0]['owner'])})" if matched[0].get("owner") else ""
        status = {"open": "✅ шлагбаум открыт", "open-failed": f"⚠️ не удалось открыть: {html.escape(details)}",
                  "gate-disabled": "⏸ открытие выключено в настройках"}.get(action, action)
        text = f"🚗➡️ <b>{plate}</b>{owner}\n{status}\n📷 {cam_name}"
    elif plate:
        if s.get("tg_notify_denied") != "1":
            return
        text = f"🚫 <b>{plate}</b> — нет в белом списке\n📷 {cam_name}"
    else:
        if s.get("tg_notify_unrecognized") != "1":
            return
        raw = f" (похоже на {best.plate}, {best.confidence:.0%})" if best else ""
        text = f"❓ Номер не распознан{html.escape(raw)}\n📷 {cam_name}"
    await telegram.send(s, text, jpg)
