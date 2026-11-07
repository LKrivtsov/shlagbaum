import logging

import httpx

log = logging.getLogger("telegram")


async def send(settings: dict, text: str, photo: bytes | None = None) -> list[str]:
    token = settings.get("tg_token", "").strip()
    chats = [c.strip() for c in settings.get("tg_chat_ids", "").replace(";", ",").split(",") if c.strip()]
    if not token or not chats:
        return ["telegram не настроен"]
    errors = []
    base = f"https://api.telegram.org/bot{token}"
    async with httpx.AsyncClient(timeout=20) as cl:
        for chat in chats:
            try:
                if photo and settings.get("tg_send_photo") == "1":
                    r = await cl.post(f"{base}/sendPhoto", data={"chat_id": chat, "caption": text, "parse_mode": "HTML"},
                                      files={"photo": ("plate.jpg", photo, "image/jpeg")})
                else:
                    r = await cl.post(f"{base}/sendMessage", data={"chat_id": chat, "text": text, "parse_mode": "HTML"})
                if r.status_code != 200:
                    errors.append(f"{chat}: {r.text[:200]}")
            except Exception as e:
                errors.append(f"{chat}: {e}")
    if errors:
        log.warning("telegram errors: %s", errors)
    return errors
