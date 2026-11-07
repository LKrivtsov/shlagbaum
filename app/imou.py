"""Клиент Imou Open Platform (open.imoulife.com).

Подпись: sign = Base64(HMAC-SHA256("time:{t},nonce:{n},appSecret:{s}", key=hex(sha256(appSecret))))
"""
import base64
import hashlib
import hmac
import logging
import time
import uuid

import httpx

from . import config

log = logging.getLogger("imou")


class ImouError(Exception):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code
        self.msg = msg


def make_sign(ts: int, nonce: str, secret: str) -> str:
    password = hashlib.sha256(secret.encode()).hexdigest().lower()
    src = f"time:{ts},nonce:{nonce},appSecret:{secret}"
    return base64.b64encode(hmac.new(password.encode(), src.encode(), hashlib.sha256).digest()).decode()


class Imou:
    def __init__(self, app_id=None, secret=None, host=None):
        self.app_id = app_id or config.IMOU_APP_ID
        self.secret = secret or config.IMOU_APP_SECRET
        self.host = (host or config.IMOU_API_HOST).rstrip("/")
        self._token = None
        self._token_exp = 0
        self.client = httpx.AsyncClient(timeout=20)

    async def call(self, method: str, params: dict | None = None, with_token=True, _retry=True):
        if not self.app_id or not self.secret:
            raise ImouError("CONFIG", "IMOU_APP_ID / IMOU_APP_SECRET не заданы в .env")
        params = dict(params or {})
        if with_token:
            params["token"] = await self.token()
        ts = int(time.time())
        nonce = uuid.uuid4().hex
        body = {
            "system": {"ver": "1.0", "appId": self.app_id, "sign": make_sign(ts, nonce, self.secret),
                       "time": ts, "nonce": nonce},
            "id": uuid.uuid4().hex,
            "params": params,
        }
        r = await self.client.post(f"{self.host}/openapi/{method}", json=body)
        r.raise_for_status()
        res = r.json().get("result", {})
        code = str(res.get("code"))
        if code == "TK1002" and with_token and _retry:  # токен истёк
            self._token = None
            return await self.call(method, {k: v for k, v in params.items() if k != "token"}, True, False)
        if code != "0":
            raise ImouError(code, res.get("msg"))
        return res.get("data") or {}

    async def token(self) -> str:
        if self._token and time.time() < self._token_exp:
            return self._token
        data = await self.call("accessToken", {}, with_token=False)
        self._token = data["accessToken"]
        self._token_exp = time.time() + int(data.get("expireTime", 3600)) - 600
        return self._token

    # ---- push ----
    async def set_callback(self, url: str, flags: str, on=True):
        params = {"status": "on" if on else "off", "basePush": "2"}
        if on:
            params.update(callbackUrl=url, callbackFlag=flags)
        return await self.call("setMessageCallback", params)

    async def get_callback(self):
        return await self.call("getMessageCallback", {})

    # ---- devices ----
    async def list_devices(self):
        """Список устройств аккаунта. Пробуем несколько методов — у разных версий API они отличаются."""
        errors = []
        for method, params in [
            ("listDeviceDetailsByPage", {"page": 1, "pageSize": 50}),
            ("deviceBaseList", {"bindId": -1, "limit": 50, "type": "bindAndShare", "needApInfo": False}),
        ]:
            try:
                data = await self.call(method, params)
            except Exception as e:
                errors.append(f"{method}: {e}")
                continue
            out = []
            for d in data.get("deviceList") or data.get("devices") or []:
                did = d.get("deviceId")
                chans = d.get("channels") or d.get("channelList") or [{"channelId": "0"}]
                for ch in chans:
                    out.append({
                        "device_id": did,
                        "channel_id": str(ch.get("channelId", "0")),
                        "name": ch.get("channelName") or d.get("deviceName") or d.get("name") or did,
                    })
            return out
        raise ImouError("LIST", "; ".join(errors))

    async def snapshot_url(self, device_id: str, channel_id: str) -> str:
        try:
            data = await self.call("setDeviceSnapEnhanced", {"deviceId": device_id, "channelId": str(channel_id)})
        except ImouError as e:
            log.warning("setDeviceSnapEnhanced failed (%s), fallback to setDeviceSnap", e)
            data = await self.call("setDeviceSnap", {"deviceId": device_id, "channelId": str(channel_id)})
        return data["url"]

    async def download(self, url: str) -> bytes:
        r = await self.client.get(url, follow_redirects=True)
        r.raise_for_status()
        return r.content

    async def snapshot(self, device_id: str, channel_id: str) -> bytes:
        url = await self.snapshot_url(device_id, channel_id)
        # Снимок на облаке появляется не мгновенно — несколько попыток скачивания
        last = None
        for _ in range(6):
            try:
                data = await self.download(url)
                if len(data) > 1000:
                    return data
            except Exception as e:
                last = e
            import asyncio
            await asyncio.sleep(0.7)
        raise ImouError("SNAP", f"не удалось скачать снимок: {last}")


client = Imou()
