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

SOURCE_NAMES = {"snapshot": "снимок", "hls": "видеопоток HD", "event": "фото события", "rtsp": "RTSP",
                "smart": "снимок"}


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


def event_time(msg: dict) -> float:
    """Момент срабатывания (unix-время). utcTime бывает в секундах или миллисекундах;
    time — unix или строка yyyyMMddHHmmss в местном времени камеры."""
    for key in ("utcTime", "time"):
        v = msg.get(key)
        if v in (None, ""):
            continue
        try:
            sv = str(v)
            if sv.isdigit() and len(sv) == 14:
                return datetime.strptime(sv, "%Y%m%d%H%M%S").replace(tzinfo=config.TZ).timestamp()
            f = float(sv)
            if f > 1e12:
                f /= 1000
            if 1.5e9 < f < 4e9:
                return f
        except Exception:
            continue
    return time.time()


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
            "ts": event_time(msg), "alarm_id": str(msg.get("alarmId") or msg.get("id") or ""),
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
    await process_camera(cam, ev["msg_type"], ev["pictures"], label_type=ev["label_type"], event_ts=ev["ts"],
                         alarm_id=ev["alarm_id"])


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


# ---------------------------------------------------------------- кадры на момент события
def _local(ts: float) -> str:
    return datetime.fromtimestamp(ts, config.TZ).strftime("%Y-%m-%d %H:%M:%S")


async def alarm_pictures(cam, ts, alarm_id="", wait=8.0):
    """Фото, которое камера сняла в момент тревоги (getAlarmMessage → picurlArray).
    Фото появляется в облаке с небольшой задержкой — опрашиваем до wait секунд."""
    end_wait = time.time() + wait
    while True:
        try:
            alarms = await imou.alarm_messages(cam["device_id"], cam["channel_id"], _local(ts - 120), _local(ts + 60))
        except Exception as e:
            log.warning("getAlarmMessage: %s", e)
            alarms = []

        def dist(a):
            if alarm_id and str(a.get("alarmId")) == alarm_id:
                return -1
            try:
                return abs(float(a.get("time")) - ts)
            except Exception:
                return 1e9

        alarms = sorted((a for a in alarms if a.get("picurlArray")), key=dist)
        if alarms and dist(alarms[0]) <= 15:
            urls = [u for u in alarms[0]["picurlArray"] if u and ".dav" not in u.lower()]
            out = []
            for u in urls[:3]:
                try:
                    img = anpr.decode_image(await imou.download(u))
                    if img is not None:
                        out.append(img)
                except Exception as e:
                    log.warning("фото тревоги не скачано: %s", e)
            if out:
                return out
        if time.time() >= end_wait:
            return []
        await asyncio.sleep(1.5)


def _read_record(url, duration, fps, max_frames=40, timeout=25):
    """Читает HLS записи и берёт fps кадров в секунду записи. Возвращает [(секунда, кадр)]."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    out, t_end = [], time.time() + timeout
    try:
        if not cap.isOpened():
            raise RuntimeError("не удалось открыть запись")
        step, next_t, n = 1.0 / max(fps, 0.1), 0.0, 0
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 15
        while time.time() < t_end and len(out) < max_frames:
            if not cap.grab():
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 or n / src_fps
            n += 1
            if pos > duration + 1:
                break
            if pos + 1e-3 >= next_t:
                ok, fr = cap.retrieve()
                if ok:
                    out.append((pos, fr))
                next_t += step
    finally:
        cap.release()
    return out


async def record_frames(cam, ts, before, after, fps):
    """Кадры из записи SD-карты за отрезок [ts-before, ts+after]."""
    ready = ts + after + 2 - time.time()   # запись должна успеть дописаться
    if ready > 0:
        await asyncio.sleep(ready)
    last = None
    for attempt in range(3):
        try:
            url = await imou.record_hls(cam["device_id"], cam["channel_id"], _local(ts - before), _local(ts + after))
            frames = await asyncio.to_thread(_read_record, url, before + after, fps)
            if frames:
                return frames
            last = "в записи нет кадров"
        except Exception as e:
            last = str(e)
        await asyncio.sleep(2 + attempt * 2)
    raise RuntimeError(f"запись SD: {last}")


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
    fname = f"{datetime.now(config.TZ):%Y%m%d_%H%M%S}_{cam['id']}_{uuid.uuid4().hex[:6]}.jpg"
    # оригинал без разметки — для разбора ошибок распознавания и пополнения тестов
    cv2.imwrite(str(config.SNAP_DIR / fname.replace(".jpg", "_raw.jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    ann = anpr.annotate(img, results)
    h, w = ann.shape[:2]
    # сохраняем крупно, чтобы номер было видно; превью делает браузер
    if w > 1920:
        ann = cv2.resize(ann, (1920, int(h * 1920 / w)))
    cv2.putText(ann, note, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(ann, note, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    (config.SNAP_DIR / fname).write_bytes(jpg)
    return fname, jpg


async def open_manual(user: str):
    """Ручное открытие из админки."""
    s = db.get_settings()
    s = dict(s, gate_enabled="1")  # ручное открытие работает даже при выключенном автооткрытии
    action, info = await _open_gate(s)
    action = "open-manual" if action == "open" else action
    db.event_add(device_id="", channel_id="", msg_type="manual", label_type="", plate=None, confidence=None,
                 allowed=1, action=action, details=json.dumps({"info": info, "role": "manual", "owner": user},
                                                              ensure_ascii=False), image=None, images="[]")
    return action, info


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
                         force_open=True, label_type="", event_ts=None, alarm_id=""):
    key = f"{cam['device_id']}/{cam['channel_id']}"
    if key in _busy:
        return {"skipped": "уже обрабатывается"}
    _busy.add(key)
    try:
        if cam.get("role") == "exit":
            return await _process_exit(cam, msg_type, pictures or [], force_open, label_type, event_ts, alarm_id)
        return await _process_entry(cam, msg_type, pictures or [], force_open, label_type, event_ts, alarm_id)
    finally:
        _busy.discard(key)


async def _process_exit(cam, msg_type, pictures, do_actions, label_type, event_ts=None, alarm_id=""):
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
    src = FrameSource(cam, pictures, dict(s, image_source="snapshot" if s.get("image_source") == "smart"
                                         else s.get("image_source")))
    try:
        img, mode = None, "snapshot"
        if event_ts and s.get("image_source") == "smart":
            pics = await alarm_pictures(cam, event_ts, alarm_id, float(s.get("alarm_wait") or 8))
            if pics:
                img, mode = pics[0], "фото тревоги"
        if img is None:
            img, mode = await src.get()
        if img is not None:
            h, w = img.shape[:2]
            note = f"{SOURCE_NAMES.get(mode, mode)} {w}x{h}"
            fname, jpg = await asyncio.to_thread(_save_frame, img, [], cam, note)
            frames.append({"file": fname, "note": note, "cover": True})
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


async def _process_entry(cam, msg_type, pictures, do_actions, label_type, event_ts=None, alarm_id=""):
    s = db.get_settings()
    attempts = max(1, int(s.get("attempts") or 3))
    interval = float(s.get("attempt_interval") or 1)
    min_conf = float(s.get("min_confidence") or 0.8)
    kz_only = s.get("kz_format_only") == "1"
    t0 = time.time()

    best, errors, frames = None, [], []
    best_idx = None              # кадр с лучшим номером
    det_idx, det_score = None, 0.0  # кадр, где детектор видел номерную рамку (значит, есть машина)
    ts = event_ts or time.time()
    mode = s.get("image_source") or "smart"
    done = False

    def good_enough():
        return bool(best and best.confidence >= min_conf and (best.kz_valid or not kz_only)
                    and (match_whitelist(best.plate, s) or best.confidence > 0.95))

    async def handle(img, label, keep_always=True):
        nonlocal best, best_idx, det_idx, det_score
        results, n_dets, max_det = await asyncio.to_thread(anpr.get().recognize, img, True)
        h, w = img.shape[:2]
        if not keep_always and not n_dets:
            return  # кадры записи без номера не храним, чтобы не засорять событие
        found = ", ".join(f"{r.plate} {r.confidence:.0%}" for r in results) or \
            ("номер виден, но не прочитан" if n_dets else "номер не найден")
        fname, _ = await asyncio.to_thread(_save_frame, img, results, cam, f"{label} {w}x{h}")
        frames.append({"file": fname, "note": f"{label} · {w}×{h} · {found}",
                       "plates": [r.plate for r in results], "dets": n_dets})
        idx = len(frames) - 1
        if max_det > det_score:
            det_idx, det_score = idx, max_det
        for r in results:
            if kz_only and not r.kz_valid:
                continue
            if best is None or r.score > best.score:
                best, best_idx = r, idx

    # Параллельно сразу просим текущий снимок: если машина стоит у шлагбаума, он самый быстрый
    snap_task = None
    if mode == "smart" and event_ts and not cam.get("rtsp_url"):
        snap_task = asyncio.create_task(imou.snapshot(cam["device_id"], cam["channel_id"]))

    # 1. Фото тревоги — снято камерой в момент срабатывания
    if mode == "smart" and event_ts:
        try:
            pics = await alarm_pictures(cam, ts, alarm_id, float(s.get("alarm_wait") or 8))
            for k, img in enumerate(pics):
                await handle(img, f"фото тревоги {k + 1}")
            if not pics:
                errors.append("фото тревоги не найдено")
        except Exception as e:
            errors.append(f"фото тревоги: {e}")
        done = good_enough()

    # 1б. Текущий снимок, запрошенный параллельно
    if snap_task and done:
        snap_task.cancel()   # номер уже прочитан с фото тревоги — не ждём
    elif snap_task:
        try:
            data = await snap_task
            if True:
                img = anpr.decode_image(data)
                if img is not None:
                    await handle(img, "снимок сейчас")
                    done = good_enough()
        except Exception as e:
            errors.append(f"снимок: {e}")

    # 2. Кадры из записи SD-карты на время события
    if mode == "smart" and event_ts and not done:
        before, after = float(s.get("record_before") or 2), float(s.get("record_after") or 6)
        try:
            rec = await record_frames(cam, ts, before, after, float(s.get("record_fps") or 2))
            kept_before = len(frames)
            for pos, img in rec:
                await handle(img, f"запись {pos - before:+.1f} с", keep_always=False)
                if good_enough():
                    break
            if len(frames) == kept_before and rec:   # ни на одном кадре нет номера — сохраним первый для истории
                await handle(rec[0][1], f"запись {rec[0][0] - before:+.1f} с")
        except Exception as e:
            errors.append(str(e))
            log.warning("%s", e)
        done = good_enough()

    # 3. Текущие снимки (старый способ) — для ручной проверки и режимов без smart
    if not done and not snap_task:
        src = FrameSource(cam, pictures, dict(s, image_source="snapshot" if mode == "smart" else mode))
        try:
            n_live = attempts
            for i in range(n_live):
                try:
                    img, m = await src.get()
                    if img is None:
                        raise RuntimeError("пустой кадр")
                    await handle(img, f"{SOURCE_NAMES.get(m, m)} {i + 1}/{n_live}")
                    if good_enough():
                        break
                except Exception as e:
                    errors.append(f"снимок {i + 1}: {e}")
                    frames.append({"file": None, "note": f"снимок {i + 1}/{n_live} · ошибка: {e}"})
                if i < n_live - 1:
                    await asyncio.sleep(interval)
        finally:
            src.close()

    # обложка события: кадр с номером -> кадр, где видна номерная рамка -> первый кадр
    # (событие приходит, когда машина уже в кадре, поэтому первый кадр надёжнее последнего)
    saved_idx = [k for k, f in enumerate(frames) if f["file"]]
    cover_idx = best_idx if best_idx is not None else det_idx if det_idx is not None else \
        (saved_idx[0] if saved_idx else None)
    if cover_idx is not None:
        frames[cover_idx]["cover"] = True
    cover = frames[cover_idx]["file"] if cover_idx is not None else None
    best_jpg = (config.SNAP_DIR / cover).read_bytes() if cover else None

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

    ev_id = db.event_add(
        device_id=cam["device_id"], channel_id=cam["channel_id"], msg_type=msg_type, label_type=label_type,
        plate=plate or (best.plate if best else None), confidence=best.confidence if best else None,
        allowed=int(allowed), action=action,
        details=json.dumps({"info": details, "owner": matched[0]["owner"] if matched else "", "role": "entry",
                            "elapsed": round(time.time() - t0, 2)}, ensure_ascii=False),
        image=cover, images=json.dumps(frames, ensure_ascii=False),
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
