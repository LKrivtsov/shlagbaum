"""Обработка события с камеры: снимки -> распознавание -> белый список -> SIP + Telegram."""
import asyncio
import html
import json
import logging
import time
from datetime import datetime

import cv2

from . import anpr, config, db, sip, telegram
from .imou import client as imou

log = logging.getLogger("processor")

_busy: set[str] = set()          # камеры, которые сейчас обрабатываются
_seen_ids: dict[str, float] = {}  # дедупликация сообщений Imou по id


def parse_message(msg: dict) -> dict:
    """Приводим три формата push-сообщений Imou к одному виду."""
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
    return {"id": str(msg.get("id") or msg.get("alarmId") or ""), "msg_type": str(msg_type),
            "device_id": str(device), "channel_id": str(channel), "pictures": pics}


def should_trigger(ev: dict, settings: dict) -> bool:
    types = [t.strip() for t in settings.get("trigger_msg_types", "").split(",") if t.strip()]
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
        return
    cam = db.camera_find(ev["device_id"], ev["channel_id"])
    if not cam or not cam["active"]:
        log.info("событие с неизвестной/выключенной камеры %s/%s", ev["device_id"], ev["channel_id"])
        return
    await process_camera(cam, ev["msg_type"], ev["pictures"])


async def _grab(cam: dict, pictures: list, attempt: int) -> bytes | None:
    # 1) картинка, приложенная к событию; 2) RTSP, если камера доступна напрямую; 3) снимок через облако Imou
    if attempt == 0 and pictures:
        for url in pictures:
            try:
                return await imou.download(url)
            except Exception as e:
                log.warning("не скачали картинку события: %s", e)
    if cam.get("rtsp_url"):
        data = await asyncio.to_thread(_rtsp_frame, cam["rtsp_url"])
        if data:
            return data
    return await imou.snapshot(cam["device_id"], cam["channel_id"])


def _rtsp_frame(url: str) -> bytes | None:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        ok, frame = False, None
        for _ in range(5):  # пропускаем первые (часто битые) кадры
            ok, frame = cap.read()
        if not ok:
            return None
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    finally:
        cap.release()


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


async def process_camera(cam: dict, msg_type: str = "manual", pictures: list | None = None, force_open=True):
    key = f"{cam['device_id']}/{cam['channel_id']}"
    if key in _busy:
        return {"skipped": "уже обрабатывается"}
    _busy.add(key)
    try:
        return await _process(cam, msg_type, pictures or [], force_open)
    finally:
        _busy.discard(key)


async def _process(cam, msg_type, pictures, do_actions):
    s = db.get_settings()
    attempts = max(1, int(s.get("attempts") or 3))
    interval = float(s.get("attempt_interval") or 1)
    min_conf = float(s.get("min_confidence") or 0.8)
    kz_only = s.get("kz_format_only") == "1"
    t0 = time.time()

    best, best_img, last_img, errors = None, None, None, []
    for i in range(attempts):
        try:
            data = await _grab(cam, pictures, i)
            img, results = await recognize_bytes(data)
            last_img = img if img is not None else last_img
            for r in results:
                if kz_only and not r.kz_valid:
                    continue
                if best is None or r.score > best.score:
                    best, best_img = r, (img, results)
            if best and best.confidence >= min_conf and (best.kz_valid or not kz_only):
                if match_whitelist(best.plate, s) or best.confidence > 0.95:
                    break
        except Exception as e:
            errors.append(str(e))
            log.warning("попытка %s: %s", i + 1, e)
        if i < attempts - 1:
            await asyncio.sleep(interval)

    plate = best.plate if best and best.confidence >= min_conf else None
    matched = match_whitelist(plate, s) if plate else None
    allowed = bool(matched)
    action, details = "none", ""

    if plate and allowed and do_actions:
        last = db.last_allowed_ts(plate)
        cooldown = float(s.get("cooldown_seconds") or 30)
        if time.time() - last < cooldown:
            action = "skip-cooldown"
        elif s.get("gate_enabled") != "1":
            action = "gate-disabled"
        else:
            try:
                res = await asyncio.to_thread(sip.call, sip.SipConfig.from_settings(s))
                action = "open" if res["ok"] else "open-failed"
                details = res["status"]
            except Exception as e:
                action, details = "open-failed", str(e)
    elif not plate:
        details = "; ".join(errors)[:500] if errors else "номер не найден"

    # сохраняем снимок с разметкой
    fname, jpg = None, None
    img_for_save = best_img[0] if best_img else last_img
    if img_for_save is not None:
        ann = anpr.annotate(img_for_save, best_img[1] if best_img else [])
        h, w = ann.shape[:2]
        if w > 1280:
            ann = cv2.resize(ann, (1280, int(h * 1280 / w)))
        jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        fname = f"{datetime.now(config.TZ):%Y%m%d_%H%M%S}_{cam['id']}.jpg"
        (config.SNAP_DIR / fname).write_bytes(jpg)

    ev_id = db.event_add(
        device_id=cam["device_id"], channel_id=cam["channel_id"], msg_type=msg_type,
        plate=plate or (best.plate if best else None), confidence=best.confidence if best else None,
        allowed=int(allowed), action=action,
        details=json.dumps({"info": details, "owner": matched[0]["owner"] if matched else "",
                            "elapsed": round(time.time() - t0, 2)}, ensure_ascii=False),
        image=fname,
    )

    if do_actions:
        await _notify(s, cam, plate, best, allowed, matched, action, details, jpg)
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
        text = f"🚗 <b>{plate}</b>{owner}\n{status}\n📷 {cam_name}"
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
