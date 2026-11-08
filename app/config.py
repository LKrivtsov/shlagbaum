import os
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
except Exception:
    TZ = ZoneInfo("UTC")

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
SNAP_DIR = DATA_DIR / "snapshots"
DB_PATH = DATA_DIR / "plate_gate.sqlite3"

IMOU_APP_ID = os.getenv("IMOU_APP_ID", "")
IMOU_APP_SECRET = os.getenv("IMOU_APP_SECRET", "")
IMOU_API_HOST = os.getenv("IMOU_API_HOST", "https://openapi-sg.easy4ip.com").rstrip("/")

DOMAIN = os.getenv("DOMAIN", "").strip()
PUBLIC_IP = os.getenv("PUBLIC_IP", "").strip()


def resolve_domain() -> str:
    """Без своего домена используем <ip-через-дефисы>.sslip.io — он резолвится в IP сервера."""
    global DOMAIN, PUBLIC_IP
    import re
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", DOMAIN):  # в DOMAIN указали IP
        PUBLIC_IP, DOMAIN = DOMAIN, ""
    if DOMAIN:
        return DOMAIN
    if not PUBLIC_IP:
        import urllib.request
        for url in ("https://api.ipify.org", "https://ifconfig.me/ip", "https://icanhazip.com"):
            try:
                PUBLIC_IP = urllib.request.urlopen(url, timeout=5).read().decode().strip()
                if PUBLIC_IP:
                    break
            except Exception:
                continue
    if PUBLIC_IP:
        DOMAIN = PUBLIC_IP.replace(".", "-") + ".sslip.io"
    return DOMAIN
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

ANPR_THREADS = int(os.getenv("ANPR_THREADS", "2"))
DETECTOR_MODEL = os.getenv("DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end")
OCR_MODEL = os.getenv("OCR_MODEL", "cct-xs-v2-global-model")
KEEP_DAYS = int(os.getenv("KEEP_DAYS", "14"))

# Настройки, редактируемые в админке (значения по умолчанию)
DEFAULT_SETTINGS = {
    # Imou
    "imou_callback_flags": "alarm,faceAnalysis,capabilityEvent",
    # Фильтр событий по полю labelType (через запятую). Пусто = не фильтровать.
    "trigger_label_types": "carAlarm",
    # Дополнительный фильтр по msgType (через запятую). Пусто = любой.
    "trigger_msg_types": "",
    # Откуда брать кадр: snapshot (снимок через облако), hls (кадр из HD-видеопотока), event (картинка из события)
    "image_source": "smart",
    # Режим smart: фото тревоги -> кадры из записи SD-карты на время события -> текущий снимок
    "alarm_wait": "8",            # сколько секунд ждать появления фото тревоги
    "record_before": "2",         # секунд записи до события
    "record_after": "6",          # секунд записи после события
    "record_fps": "2",            # сколько кадров в секунду записи распознавать
    # Распознавание
    "attempts": "4",              # сколько снимков пробовать
    "attempt_interval": "1.0",    # пауза между снимками, сек
    "min_confidence": "0.80",     # минимальная средняя уверенность OCR
    "kz_format_only": "1",        # принимать только номера в формате РК
    "fuzzy_match": "0",           # допуск ошибок при сверке с белым списком (0 = строго)
    "cooldown_seconds": "30",     # не открывать повторно для того же номера
    "gate_enabled": "1",          # глобальный выключатель открытия
    # Telegram
    "tg_token": "",
    "tg_chat_ids": "",
    "tg_notify_allowed": "1",
    "tg_notify_denied": "1",
    "tg_notify_unrecognized": "0",
    "tg_notify_exit": "1",
    "tg_send_photo": "1",
    # SIP
    "sip_server": "",
    "sip_port": "5060",
    "sip_user": "",
    "sip_auth_user": "",
    "sip_password": "",
    "sip_domain": "",
    "sip_target": "",             # номер/extension шлагбаума
    "sip_local_ip": "",           # публичный IP VPS (если пусто — определяется сам)
    "sip_ring_timeout": "15",     # сколько ждать ответа, сек
    "sip_hold_seconds": "3",      # сколько держать разговор после ответа
    "sip_dtmf": "",               # DTMF после ответа (например "1" или "#"), пусто = не слать
}
