import asyncio
import base64
import json
import logging
import secrets
import time
from datetime import datetime
from urllib.parse import quote

import cv2
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from . import anpr, config, db, processor, sip, telegram
from .imou import client as imou

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")

app = FastAPI(title="Plate Gate", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(__import__("pathlib").Path(__file__).parent / "templates"))
templates.env.filters["dt"] = lambda ts: datetime.fromtimestamp(ts, config.TZ).strftime("%d.%m %H:%M:%S") if ts else ""
templates.env.filters["fromjson"] = lambda s: json.loads(s) if s else {}
security = HTTPBasic()


def auth(cred: HTTPBasicCredentials = Depends(security)):
    if not config.ADMIN_PASSWORD:
        raise HTTPException(500, "ADMIN_PASSWORD не задан в .env")
    ok = secrets.compare_digest(cred.username.encode(), config.ADMIN_USER.encode()) and \
        secrets.compare_digest(cred.password.encode(), config.ADMIN_PASSWORD.encode())
    if not ok:
        raise HTTPException(401, headers={"WWW-Authenticate": "Basic"})
    return cred.username


def callback_url():
    return f"https://{config.DOMAIN}/webhook/imou/{config.WEBHOOK_SECRET}" if config.DOMAIN else ""


def back(path, msg="", err=False):
    q = f"?{'err' if err else 'msg'}={quote(msg)}" if msg else ""
    return RedirectResponse(path + q, status_code=303)


def render(request, name, **ctx):
    ctx.update(request=request, msg=request.query_params.get("msg"), err=request.query_params.get("err"),
               page=name.split(".")[0])
    return templates.TemplateResponse(request, name, ctx)


@app.on_event("startup")
async def startup():
    db.init()
    dom = await asyncio.to_thread(config.resolve_domain)
    log.info("Админка: https://%s/  Вебхук: https://%s/webhook/imou/<WEBHOOK_SECRET>", dom, dom)
    # модели грузим заранее, чтобы первое событие не ждало
    await asyncio.to_thread(anpr.get)

    async def janitor():
        while True:
            try:
                db.cleanup(config.KEEP_DAYS)
            except Exception as e:
                log.warning("cleanup: %s", e)
            await asyncio.sleep(3600)

    asyncio.create_task(janitor())


@app.get("/health")
def health():
    return {"ok": True}


# ================= Webhook от Imou =================
@app.api_route("/webhook/imou/{secret}", methods=["GET", "HEAD", "POST"])
async def imou_webhook(secret: str, request: Request, bg: BackgroundTasks):
    if not config.WEBHOOK_SECRET or not secrets.compare_digest(secret, config.WEBHOOK_SECRET):
        raise HTTPException(404)
    if request.method != "POST":
        return {"code": 0}
    raw = await request.body()
    db.raw_add(raw.decode(errors="replace"))
    try:
        body = json.loads(raw or b"{}")
    except Exception:
        form = await request.form()
        body = dict(form)
    msgs = body if isinstance(body, list) else [body]
    for m in msgs:
        if isinstance(m, dict):
            bg.add_task(_safe_handle, m)
    # Imou считает доставку успешной при HTTP 2xx — отвечаем сразу, обрабатываем в фоне
    return JSONResponse({"code": 0, "msg": "ok"})


async def _safe_handle(m):
    try:
        await processor.handle_push(m)
    except Exception:
        log.exception("ошибка обработки события")


# ================= Админка =================
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, user=Depends(auth)):
    return render(request, "dashboard.html", events=db.events(100), cameras=db.cameras())


@app.get("/snap/{name}")
def snap(name: str, user=Depends(auth)):
    p = (config.SNAP_DIR / name).resolve()
    if p.parent != config.SNAP_DIR.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/test/recognize", response_class=HTMLResponse)
async def test_recognize(request: Request, file: UploadFile = File(...), user=Depends(auth)):
    data = await file.read()
    t = time.time()
    img, results = await processor.recognize_bytes(data)
    elapsed = time.time() - t
    if img is None:
        return back("/", "Не удалось прочитать изображение", True)
    ann = anpr.annotate(img, results)
    h, w = ann.shape[:2]
    if w > 1000:
        ann = cv2.resize(ann, (1000, int(h * 1000 / w)))
    b64 = base64.b64encode(cv2.imencode(".jpg", ann)[1].tobytes()).decode()
    s = db.get_settings()
    rows = [{"plate": r.plate, "conf": r.confidence, "kz": r.kz_valid,
             "wl": bool(processor.match_whitelist(r.plate, s))} for r in results]
    return render(request, "recognize.html", image=b64, results=rows, elapsed=elapsed)


# ---- белый список ----
@app.get("/whitelist", response_class=HTMLResponse)
def wl_page(request: Request, user=Depends(auth)):
    return render(request, "whitelist.html", items=db.whitelist())


def _split_plate_line(line: str):
    """'901 CX 01 Иванов' -> ('901CX01', 'Иванов'). Номер может быть с пробелами."""
    for sep in ("\t", ";", " - "):
        if sep in line:
            a, b = line.split(sep, 1)
            return anpr.normalize(a), b.strip()
    tokens = line.split()
    for k in range(len(tokens), 0, -1):
        cand = anpr.normalize("".join(tokens[:k]))
        if anpr.is_kz(cand):
            return cand, " ".join(tokens[k:])
    return (anpr.normalize(tokens[0]), " ".join(tokens[1:])) if tokens else ("", "")


@app.post("/whitelist/add")
def wl_add(plates: str = Form(...), owner: str = Form(""), user=Depends(auth)):
    added = []
    for line in plates.replace(",", "\n").splitlines():
        p, who = _split_plate_line(line)
        if len(p) < 4:
            continue
        db.whitelist_add(p, who or owner.strip())
        added.append(p)
    return back("/whitelist", f"Добавлено: {', '.join(added) or 'ничего'}")


@app.post("/whitelist/{wid}/delete")
def wl_del(wid: int, user=Depends(auth)):
    db.whitelist_delete(wid)
    return back("/whitelist", "Удалено")


@app.post("/whitelist/{wid}/toggle")
def wl_toggle(wid: int, user=Depends(auth)):
    db.whitelist_toggle(wid)
    return back("/whitelist")


# ---- камеры ----
@app.get("/cameras", response_class=HTMLResponse)
def cam_page(request: Request, user=Depends(auth)):
    return render(request, "cameras.html", cameras=db.cameras())


@app.post("/cameras/add")
def cam_add(device_id: str = Form(...), channel_id: str = Form("0"), name: str = Form(""),
            rtsp_url: str = Form(""), role: str = Form("entry"), user=Depends(auth)):
    db.camera_upsert(device_id.strip(), channel_id.strip() or "0", name.strip(), rtsp_url.strip(),
                     role if role in ("entry", "exit") else "entry")
    return back("/cameras", "Камера сохранена")


@app.post("/cameras/import")
async def cam_import(user=Depends(auth)):
    try:
        devs = await imou.list_devices()
    except Exception as e:
        return back("/cameras", f"Imou: {e}", True)
    for d in devs:
        db.camera_upsert(d["device_id"], d["channel_id"], d["name"])
    return back("/cameras", f"Импортировано камер: {len(devs)}")


@app.post("/cameras/{cid}/delete")
def cam_del(cid: int, user=Depends(auth)):
    db.camera_delete(cid)
    return back("/cameras", "Удалено")


@app.post("/cameras/{cid}/role")
def cam_role(cid: int, role: str = Form(...), user=Depends(auth)):
    if role in ("entry", "exit"):
        db.camera_set_role(cid, role)
    return back("/cameras", "Роль камеры сохранена")


@app.get("/event/{eid}", response_class=HTMLResponse)
def event_page(eid: int, request: Request, user=Depends(auth)):
    e = db.event_get(eid)
    if not e:
        raise HTTPException(404)
    return render(request, "event.html", e=e)


@app.post("/cameras/{cid}/toggle")
def cam_toggle(cid: int, user=Depends(auth)):
    db.camera_toggle(cid)
    return back("/cameras")


@app.post("/cameras/{cid}/test")
async def cam_test(cid: int, open_gate: str = Form("0"), user=Depends(auth)):
    cam = db.camera_get(cid)
    if not cam:
        return back("/cameras", "Нет камеры", True)
    res = await processor.process_camera(cam, "manual-test", force_open=open_gate == "1")
    if "skipped" in res:
        return back("/cameras", res["skipped"], True)
    if cam.get("role") == "exit":
        txt = f"Выезд: действие {res['action']} {res['details']}"
    else:
        txt = f"Номер: {res['plate'] or '—'} (сырой: {res['raw'] or '—'}), в списке: {'да' if res['allowed'] else 'нет'}, действие: {res['action']} {res['details']}"
    return back("/", txt)


# ---- Imou ----
@app.get("/imou", response_class=HTMLResponse)
async def imou_page(request: Request, user=Depends(auth)):
    return render(request, "imou.html", callback_url=callback_url(), settings=db.get_settings(),
                  raw=db.raw_list(), app_id=config.IMOU_APP_ID, host=config.IMOU_API_HOST)


@app.post("/imou/callback")
async def imou_set_cb(action: str = Form(...), flags: str = Form(""), user=Depends(auth)):
    try:
        if action == "on":
            if not callback_url():
                return back("/imou", "Не задан DOMAIN в .env", True)
            db.set_settings({"imou_callback_flags": flags})
            await imou.set_callback(callback_url(), flags, True)
            return back("/imou", "Вебхук включён")
        if action == "off":
            await imou.set_callback("", "", False)
            return back("/imou", "Вебхук выключен")
        data = await imou.get_callback()
        return back("/imou", "Текущие настройки: " + json.dumps(data, ensure_ascii=False))
    except Exception as e:
        return back("/imou", f"Ошибка Imou: {e}", True)


@app.post("/imou/token")
async def imou_token(user=Depends(auth)):
    try:
        imou._token = None
        t = await imou.token()
        return back("/imou", f"Подключение работает, токен {t[:8]}…")
    except Exception as e:
        return back("/imou", f"Ошибка: {e}", True)


@app.post("/imou/replay/{rid}")
async def imou_replay(rid: int, user=Depends(auth)):
    for r in db.raw_list(200):
        if r["id"] == rid:
            m = json.loads(r["body"])
            processor._seen_ids.pop(str(m.get("id", "")), None)
            await processor.handle_push(m)
            return back("/", "Событие обработано повторно")
    return back("/imou", "Не найдено", True)


# ---- настройки ----
@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, user=Depends(auth)):
    return render(request, "settings.html", s=db.get_settings())


@app.post("/settings")
async def settings_save(request: Request, user=Depends(auth)):
    form = await request.form()
    vals = {k: v for k, v in form.items() if k in config.DEFAULT_SETTINGS}
    # чекбоксы: отсутствуют в форме, если сняты
    for k in ("kz_format_only", "gate_enabled", "tg_notify_allowed", "tg_notify_denied",
              "tg_notify_unrecognized", "tg_send_photo", "tg_notify_exit"):
        if form.get("_section") == _section_of(k):
            vals[k] = "1" if form.get(k) else "0"
    if "sip_password" in vals and vals["sip_password"] == "********":
        vals.pop("sip_password")
    db.set_settings(vals)
    return back("/settings", "Сохранено")


def _section_of(k):
    return "tg" if k.startswith("tg_") else "rec"


@app.post("/settings/test-telegram")
async def test_tg(user=Depends(auth)):
    errs = await telegram.send(db.get_settings(), "✅ Тест: уведомления Plate Gate работают")
    return back("/settings", "Ошибки: " + "; ".join(errs) if errs else "Сообщение отправлено", bool(errs))


@app.post("/settings/test-sip")
async def test_sip(user=Depends(auth)):
    try:
        res = await asyncio.to_thread(sip.call, sip.SipConfig.from_settings(db.get_settings()))
        txt = f"{'OK' if res['ok'] else 'Не удалось'}: {res['status']} за {res['duration']} с. Трасса: {' | '.join(res['trace'])}"
        return back("/settings", txt, not res["ok"])
    except Exception as e:
        return back("/settings", f"SIP ошибка: {e}", True)
