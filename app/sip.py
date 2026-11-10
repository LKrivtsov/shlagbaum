"""Минимальный SIP-клиент (UDP) для «звонка на шлагбаум».

Большинство GSM/SIP-контроллеров шлагбаума открываются по входящему звонку
с разрешённого номера: им достаточно, чтобы вызов дошёл (они сбрасывают его
или отвечают). Поэтому клиент:
  1. шлёт INVITE (с Digest-авторизацией при 401/407),
  2. ждёт ответа до ring_timeout,
  3. если ответили — опционально шлёт DTMF (SIP INFO) и через hold_seconds кладёт трубку (BYE),
  4. если не ответили — CANCEL.
Без аудио (RTP не передаётся) — для открытия шлагбаума это не нужно.
"""
import hashlib
import logging
import random
import re
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass

log = logging.getLogger("sip")


@dataclass
class SipConfig:
    server: str
    port: int = 5060
    user: str = ""
    auth_user: str = ""
    password: str = ""
    domain: str = ""
    target: str = ""
    local_ip: str = ""
    ring_timeout: float = 15
    hold_seconds: float = 3
    dtmf: str = ""

    @classmethod
    def from_settings(cls, s: dict):
        return cls(
            server=s.get("sip_server", "").strip(),
            port=int(s.get("sip_port") or 5060),
            user=s.get("sip_user", "").strip(),
            auth_user=(s.get("sip_auth_user") or s.get("sip_user") or "").strip(),
            password=s.get("sip_password", ""),
            domain=(s.get("sip_domain") or s.get("sip_server") or "").strip(),
            target=s.get("sip_target", "").strip(),
            local_ip=s.get("sip_local_ip", "").strip(),
            ring_timeout=float(s.get("sip_ring_timeout") or 15),
            hold_seconds=float(s.get("sip_hold_seconds") or 3),
            dtmf=s.get("sip_dtmf", "").strip(),
        )


def _md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def _parse(data: bytes):
    text = data.decode(errors="replace")
    head, _, body = text.partition("\r\n\r\n")
    lines = head.split("\r\n")
    first = lines[0]
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers.setdefault(k.strip().lower(), []).append(v.strip())
    m = re.match(r"SIP/2.0 (\d{3}) ?(.*)", first)
    code = int(m.group(1)) if m else 0
    reason = m.group(2) if m else first
    return code, reason, headers, body


def _auth_params(h: str) -> dict:
    return {k.lower(): v.strip('"') for k, v in re.findall(r'(\w+)=("[^"]*"|[^,\s]+)', h)}


class SipCall:
    def __init__(self, cfg: SipConfig):
        if not cfg.server or not cfg.target:
            raise ValueError("SIP: не заданы сервер и номер шлагбаума")
        self.cfg = cfg
        self.addr = (socket.gethostbyname(cfg.server), cfg.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))
        self.lport = self.sock.getsockname()[1]
        self.lip = cfg.local_ip or self._detect_ip()
        # RTP-сокет: провайдеры (Zadarma и др.) сбрасывают вызов через несколько секунд, если нет звука
        self.rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtp.bind(("0.0.0.0", 0))
        self.rtp_port = self.rtp.getsockname()[1]
        self.last_body = ""
        self.call_id = f"{uuid.uuid4().hex}@{self.lip}"
        self.from_tag = uuid.uuid4().hex[:10]
        self.to_tag = ""
        self.cseq = random.randint(1, 1000)
        self.branch = ""
        self.auth_header = ""
        self.domain = cfg.domain or cfg.server
        self.target_uri = cfg.target if cfg.target.startswith("sip:") else f"sip:{cfg.target}@{self.domain}"
        self.from_uri = f"sip:{cfg.user or 'gate'}@{self.domain}"
        self.contact = f"<sip:{cfg.user or 'gate'}@{self.lip}:{self.lport}>"
        self.route_set: list[str] = []   # Route для запросов внутри диалога (из Record-Route ответа 200)
        self.log: list[str] = []
        self.t_start = time.time()

    def _detect_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(self.addr)
            return s.getsockname()[0]
        finally:
            s.close()

    def _send(self, msg: str, remember=True):
        if remember:
            self.last_msg = msg
        try:
            self.sock.sendto(msg.encode(), self.addr)
        except OSError as e:
            self.log.append(f"! {e}")

    def _recv(self, timeout):
        self.sock.settimeout(max(0.05, timeout))
        try:
            data, _ = self.sock.recvfrom(65535)
            return data
        except (socket.timeout, ConnectionRefusedError, OSError):
            return None

    def _sdp(self):
        sid = random.randint(10**6, 10**7)
        return (
            "v=0\r\n"
            f"o=- {sid} {sid} IN IP4 {self.lip}\r\n"
            "s=plate-gate\r\n"
            f"c=IN IP4 {self.lip}\r\n"
            "t=0 0\r\n"
            f"m=audio {self.rtp_port} RTP/AVP 8 0 101\r\n"
            "a=rtpmap:0 PCMU/8000\r\na=rtpmap:8 PCMA/8000\r\n"
            "a=rtpmap:101 telephone-event/8000\r\na=fmtp:101 0-16\r\na=sendrecv\r\n"
        )

    def _request(self, method, uri, cseq=None, body="", ctype="", extra="", new_branch=True,
                 in_dialog=False, auth=None):
        if new_branch:
            self.branch = "z9hG4bK" + uuid.uuid4().hex[:16]
        to = f"<{self.target_uri}>" + (f";tag={self.to_tag}" if self.to_tag else "")
        msg = (
            f"{method} {uri} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.lip}:{self.lport};branch={self.branch};rport\r\n"
            "Max-Forwards: 70\r\n"
            f"From: <{self.from_uri}>;tag={self.from_tag}\r\n"
            f"To: {to}\r\n"
            f"Call-ID: {self.call_id}\r\n"
            f"CSeq: {cseq or self.cseq} {method}\r\n"
            f"Contact: {self.contact}\r\n"
            "User-Agent: plate-gate/1.0\r\n"
            + "".join(f"Route: {r}\r\n" for r in (self.route_set if in_dialog else []))
            + (auth if auth is not None else (self.auth_header if method == "INVITE" else ""))
            + extra
            + (f"Content-Type: {ctype}\r\n" if ctype else "")
            + f"Content-Length: {len(body.encode())}\r\n\r\n{body}"
        )
        self._send(msg)
        self.log.append(f"{self._ts()} > {method}" + (f" {uri}" if method == "INVITE" else ""))

    def _ack(self, uri=None, same_branch=True, in_dialog=False):
        self._request("ACK", uri or self.target_uri, cseq=self.cseq, new_branch=not same_branch, in_dialog=in_dialog)

    def _in_dialog(self, method, uri, body="", ctype="", timeout=3):
        """Запрос внутри диалога (BYE/INFO): с Route из Record-Route и своей Digest-авторизацией,
        если прокси её запросит (подпись INVITE для другого метода не подходит)."""
        auth = ""
        res = None
        for _ in range(2):
            self.cseq += 1
            self._request(method, uri, body=body, ctype=ctype, in_dialog=True, auth=auth)
            res = self._wait_final(method, time.time() + timeout)
            if res and res[0] in (401, 407):
                auth = self._make_auth(res[2], method, uri)
                continue
            break
        return res

    def _make_auth(self, headers, method, uri):
        is_proxy = "proxy-authenticate" in headers
        h = (headers.get("proxy-authenticate") or headers.get("www-authenticate"))[0]
        p = _auth_params(h)
        realm, nonce = p.get("realm", ""), p.get("nonce", "")
        user = self.cfg.auth_user or self.cfg.user
        ha1 = _md5(f"{user}:{realm}:{self.cfg.password}")
        ha2 = _md5(f"{method}:{uri}")
        qop = p.get("qop", "")
        parts = [f'username="{user}"', f'realm="{realm}"', f'nonce="{nonce}"', f'uri="{uri}"', "algorithm=MD5"]
        if "auth" in qop.split(","):
            cnonce, nc = uuid.uuid4().hex[:8], "00000001"
            resp = _md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")
            parts += [f"qop=auth", f"nc={nc}", f'cnonce="{cnonce}"']
        else:
            resp = _md5(f"{ha1}:{nonce}:{ha2}")
        parts.append(f'response="{resp}"')
        if "opaque" in p:
            parts.append(f'opaque="{p["opaque"]}"')
        name = "Proxy-Authorization" if is_proxy else "Authorization"
        return f"{name}: Digest {', '.join(parts)}\r\n"

    def _wait_final(self, method, deadline, on_provisional=None):
        """Ждём финальный ответ на транзакцию. Возвращает (code, reason, headers) или None по таймауту."""
        # UDP: пока нет ни одного ответа, повторяем запрос (таймер T1 = 0.5 с, удваивается)
        req, got_any, t1 = self.last_msg, False, 0.5
        next_rt = time.time() + t1
        while time.time() < deadline:
            data = self._recv(min(deadline, next_rt) - time.time())
            if not data:
                if not got_any and time.time() >= next_rt and t1 <= 4:
                    self._send(req)
                    t1 *= 2
                    next_rt = time.time() + t1
                continue
            code, reason, headers, body = _parse(data)
            if code == 0:
                # входящий запрос (например BYE от шлагбаума) — отвечаем 200
                self._reply_ok(data)
                continue
            cseq = (headers.get("cseq") or [""])[0]
            if not cseq.endswith(method):
                continue
            got_any = True
            extra = ""
            if method == "INVITE" and code >= 180:
                # кто ответил: по Server/User-Agent и Contact видно, абонент это или автоответчик провайдера
                who = (headers.get("server") or headers.get("user-agent") or [""])[0]
                ct = (headers.get("contact") or [""])[0]
                sdp = "sdp" if "application/sdp" in " ".join(headers.get("content-type") or []) else "без sdp"
                extra = f" [{', '.join(x for x in (who, ct, sdp) if x)}]"
            self.log.append(f"{self._ts()} < {code} {reason}{extra}")
            if code < 200:
                if on_provisional:
                    on_provisional(code)
                continue
            self.last_body = body
            return code, reason, headers
        return None

    def _ts(self):
        return f"{time.time() - self.t_start:.1f}s"

    def _reply_ok(self, data: bytes):
        text = data.decode(errors="replace")
        head = text.split("\r\n\r\n")[0].split("\r\n")
        method = head[0].split(" ")[0]
        keep = [ln for ln in head[1:] if ln.lower().split(":")[0] in ("via", "from", "to", "call-id", "cseq")]
        self._send("SIP/2.0 200 OK\r\n" + "\r\n".join(keep) + "\r\nContent-Length: 0\r\n\r\n", remember=False)
        self.log.append(f"< {method} (ответили 200)")
        if method == "BYE":
            self.remote_bye = True

    def run(self) -> dict:
        cfg = self.cfg
        self.remote_bye = False
        t0 = time.time()
        try:
            for attempt in range(3):
                self.cseq += 1
                t_inv = time.time()
                self._request("INVITE", self.target_uri, body=self._sdp(), ctype="application/sdp")
                ringing = {"v": False}
                res = self._wait_final(
                    "INVITE", time.time() + cfg.ring_timeout,
                    on_provisional=lambda c: ringing.__setitem__("v", c in (180, 183)),
                )
                if res is None:
                    # не ответили — отменяем вызов
                    self._request("CANCEL", self.target_uri, cseq=self.cseq, new_branch=False)
                    self._wait_final("CANCEL", time.time() + 3)
                    self._wait_final("INVITE", time.time() + 2)  # 487 Request Terminated
                    self._ack()
                    return self._result(ringing["v"], "no-answer" if ringing["v"] else "timeout", t0)
                code, reason, headers = res
                if code >= 300:
                    # ACK на отказ должен нести to-tag из ответа
                    mt = re.search(r";tag=([^;>\s]+)", (headers.get("to") or [""])[0])
                    self.to_tag = mt.group(1) if mt else ""
                if code in (401, 407) and attempt < 2:
                    self._ack()
                    self.to_tag = ""
                    self.auth_header = self._make_auth(headers, "INVITE", self.target_uri)
                    continue
                if code >= 300:
                    self._ack()
                    # 486 Busy / 603 Decline — типичная реакция GSM-модуля шлагбаума: принял вызов и сбросил
                    ok = code in (486, 480, 603, 600)
                    return self._result(ok, f"{code} {reason}", t0)
                # 2xx — ответили
                to = (headers.get("to") or [""])[0]
                m = re.search(r";tag=([^;>\s]+)", to)
                self.to_tag = m.group(1) if m else ""
                contact = (headers.get("contact") or [""])[0]
                mc = re.search(r"<([^>]+)>", contact)
                remote = mc.group(1) if mc else (contact.split(";")[0].strip() or self.target_uri)
                # Record-Route -> Route (для вызывающей стороны в обратном порядке), иначе прокси
                # не узнаёт диалог и отвечает на BYE «404 Not here»
                rr = []
                for v in headers.get("record-route") or []:
                    rr += [x.strip() for x in re.findall(r"<[^>]+>[^,]*", v)] or [v.strip()]
                self.route_set = list(reversed(rr))
                if self.route_set:
                    self.log.append(f"  route: {', '.join(self.route_set)}")
                self._ack(remote, same_branch=False, in_dialog=True)
                rtp = _RtpSilence(self.rtp, self.last_body)
                if rtp.dest:
                    rtp.start()
                    self.log.append(f"{self._ts()} RTP -> {rtp.dest[0]}:{rtp.dest[1]} ({rtp.codec})")
                # Zadarma и многие провайдеры отвечают 200 OK сразу, ещё до того, как зазвонил
                # мобильный (гудки идут звуком). Значит, шлагбаум ещё даже не получил вызов —
                # держим линию до его сброса (BYE), но не дольше ring_timeout.
                early = not ringing["v"] and time.time() - t_inv < 2.5
                hold = max(cfg.hold_seconds, cfg.ring_timeout) if early else cfg.hold_seconds
                if early:
                    self.log.append(f"  ранний ответ провайдера — ждём до {hold:.0f} с")
                if cfg.dtmf:
                    for ch in cfg.dtmf:
                        self._in_dialog("INFO", remote, body=f"Signal={ch}\r\nDuration=250\r\n",
                                        ctype="application/dtmf-relay", timeout=2)
                        time.sleep(0.3)
                end = time.time() + hold
                while time.time() < end and not self.remote_bye:
                    data = self._recv(end - time.time())
                    if data and _parse(data)[0] == 0:
                        self._reply_ok(data)
                rtp.stop()
                if rtp.rx:
                    try:
                        from . import config
                        path = config.DATA_DIR / "sip_last.wav"
                        if rtp.save_wav(path):
                            self.log.append(f"  звук от провайдера: {len(rtp.rx) * 0.02:.1f} с -> {path}")
                    except Exception as e:
                        self.log.append(f"  звук не сохранён: {e}")
                else:
                    self.log.append("  звука от провайдера не было")
                note = ""
                if not self.remote_bye:
                    bye = self._in_dialog("BYE", remote)
                    if bye and bye[0] in (404, 481) and not self.route_set and remote != self.target_uri:
                        # прокси без Record-Route не знает Contact абонента — пробуем через исходный адрес
                        bye = self._in_dialog("BYE", self.target_uri)
                    if bye is None or bye[0] >= 300:
                        # шлагбаум уже ответил — открытие состоялось, но трубку корректно положить не удалось
                        note = f", BYE: {bye[0]} {bye[1]}" if bye else ", BYE: нет ответа"
                return self._result(True, f"answered {code}{note}", t0)
            return self._result(False, "auth failed", t0)
        finally:
            self.sock.close()
            self.rtp.close()

    def _result(self, ok, status, t0):
        r = {"ok": ok, "status": status, "duration": round(time.time() - t0, 1), "trace": self.log}
        log.info("SIP call %s -> %s", self.target_uri, r)
        return r


class _RtpSilence(threading.Thread):
    """Шлёт тишину (G.711, 20 мс) на адрес из SDP ответа и читает входящий звук (симметричный RTP,
    нужен и для прохождения NAT Docker/VPS)."""

    def __init__(self, sock, sdp: str):
        super().__init__(daemon=True)
        self.sock, self.dest, self.codec, self.pt = sock, None, "", 8
        self.rx = []          # входящий звук (pt, payload) — чтобы услышать, что говорит провайдер
        self._halt = threading.Event()
        ip = re.search(r"^c=IN IP4 (\S+)", sdp or "", re.M)
        m = re.search(r"^m=audio (\d+) RTP/\S+ ([\d ]+)", sdp or "", re.M)
        if ip and m and int(m.group(1)):
            pts = [int(x) for x in m.group(2).split()]
            self.pt = next((p for p in pts if p in (0, 8)), 8)
            self.codec = "PCMU" if self.pt == 0 else "PCMA"
            self.dest = (ip.group(1), int(m.group(1)))

    def run(self):
        seq, ts, ssrc = random.randint(0, 65535), random.randint(0, 2**31), random.randint(1, 2**31)
        payload = (b"\xff" if self.pt == 0 else b"\xd5") * 160
        self.sock.setblocking(False)
        nxt = time.time()
        while not self._halt.is_set():
            hdr = struct.pack("!BBHII", 0x80, self.pt, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)
            try:
                self.sock.sendto(hdr + payload, self.dest)
                while True:
                    pkt, _ = self.sock.recvfrom(2048)
                    if len(pkt) > 12 and (pkt[1] & 0x7F) in (0, 8) and len(self.rx) < 3000:
                        cc = pkt[0] & 0x0F
                        self.rx.append((pkt[1] & 0x7F, pkt[12 + 4 * cc:]))
            except (BlockingIOError, OSError):
                pass
            seq, ts, nxt = seq + 1, ts + 160, nxt + 0.02
            self._halt.wait(max(0, nxt - time.time()))

    def stop(self):
        self._halt.set()
        if self.is_alive():
            self.join(1)

    def save_wav(self, path) -> bool:
        """Сохраняет входящий звук в WAV (16 бит, 8 кГц)."""
        if not self.rx:
            return False
        import wave
        pcm = bytearray()
        for pt, data in self.rx:
            for b in data:
                pcm += struct.pack("<h", _ulaw(b) if pt == 0 else _alaw(b))
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(bytes(pcm))
        return True


def _alaw(a):
    a ^= 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    t = t + 8 if seg == 0 else ((t + 0x108) << (seg - 1))
    return t if a & 0x80 else -t


def _ulaw(u):
    u = ~u & 0xFF
    t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4)
    return (0x84 - t) if u & 0x80 else (t - 0x84)


def call(cfg: SipConfig) -> dict:
    return SipCall(cfg).run()
