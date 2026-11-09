"""Лёгкое распознавание номеров на CPU.

Детектор: YOLOv9-tiny (ONNX, ~8 МБ) из open-image-models.
OCR: fast-plate-ocr cct-xs-v2-global (ONNX, ~3 МБ).
На одном слабом ядре: ~50-150 мс детекция + ~10 мс OCR на номер.
"""
import logging
import re
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

from . import config
from .rectify import Rectified as _Rect, rectify as _rectify

log = logging.getLogger("anpr")

# Казахстанские форматы
KZ_PATTERNS = [
    re.compile(r"^\d{3}[A-Z]{2,3}\d{2}$"),   # 901CX01, 724AJW02 (с 2012 г.)
    re.compile(r"^[A-Z]\d{3}[A-Z]{2,3}$"),   # старый образец A123BCD
]

# Кириллица -> латиница (для ввода номеров в белый список с русской раскладкой)
_CYR = str.maketrans("АВЕКМНОРСТУХІ", "ABEKMHOPCTYXI")


# Коды регионов РК (01 Астана … 20 Улытау)
KZ_REGIONS = [f"{i:02d}" for i in range(1, 21)]


def region_ok(plate: str) -> bool:
    """Для новых номеров (901CX01) проверяем, что регион существует."""
    if re.match(r"^\d{3}[A-Z]{2,3}\d{2}$", plate):
        return plate[-2:] in KZ_REGIONS
    return True


def normalize(plate: str) -> str:
    p = (plate or "").upper().translate(_CYR)
    return re.sub(r"[^A-Z0-9]", "", p)


def is_kz(plate: str) -> bool:
    return any(p.match(plate) for p in KZ_PATTERNS) and region_ok(plate)


@dataclass
class PlateResult:
    plate: str
    confidence: float       # средняя уверенность OCR по символам
    det_confidence: float
    box: tuple              # x1, y1, x2, y2
    kz_valid: bool = False
    char_probs: list = field(default_factory=list)
    square: bool = False    # двухстрочный (квадратный) номер
    quad: object = None     # 4 угла номера в кадре (после выравнивания)
    norm: object = None     # выпрямленное изображение номера

    @property
    def score(self):
        return self.confidence * (1.0 if self.kz_valid else 0.7)


class ANPR:
    def __init__(self):
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import create_detector

        so = ort.SessionOptions()
        so.intra_op_num_threads = config.ANPR_THREADS
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        prov = ["CPUExecutionProvider"]
        self.detector = create_detector(config.DETECTOR_MODEL, conf_thresh=0.3, providers=prov, sess_options=so)
        so2 = ort.SessionOptions()
        so2.intra_op_num_threads = config.ANPR_THREADS
        so2.inter_op_num_threads = 1
        self.ocr = LicensePlateRecognizer(config.OCR_MODEL, device="cpu", providers=prov, sess_options=so2)
        cfg = self.ocr.config
        self.alphabet, self.pad, self.slots = cfg.alphabet, cfg.pad_char, cfg.max_plate_slots
        self._dig = [self.alphabet.index(c) for c in "0123456789"]
        self._let = [self.alphabet.index(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        self._pad = self.alphabet.index(self.pad)
        self._lock = threading.Lock()  # одна инференс-сессия за раз — экономим CPU и RAM
        log.info("ANPR ready (выравнивание номеров v2): %s + %s, threads=%s", config.DETECTOR_MODEL, config.OCR_MODEL, config.ANPR_THREADS)

    # ---- OCR с доступом к вероятностям по каждому символу ----
    def _probs(self, imgs):
        from fast_plate_ocr.inference.plate_recognizer import _load_image_from_source, preprocess_image
        x = preprocess_image(_load_image_from_source(imgs, self.ocr.config))
        out = self.ocr.model.run([self.ocr.plate_output_name], {"input": x})[0]
        return out.reshape(len(imgs), self.slots, len(self.alphabet))

    def _decode_free(self, P):
        idx = P.argmax(axis=1)
        chars = [self.alphabet[i] for i in idx]
        n = len("".join(chars).rstrip(self.pad))
        txt = "".join(chars[:n]).replace(self.pad, "")
        probs = [float(P[i, idx[i]]) for i in range(n)]
        return txt, probs

    def _decode_tmpl(self, P, templates):
        """Лучшее прочтение под один из шаблонов (цифра/буква на каждой позиции).
        Исправляет типичные путаницы 8↔B, 0↔O, 1↔I, 5↔S, 2↔Z."""
        best = None
        for t in templates:
            txt, probs, logp = "", [], 0.0
            rpos = t.find("RR")
            reg = None
            if rpos >= 0:   # регион выбираем целиком из списка существующих
                def rp(code):
                    a, b2 = self.alphabet.index(code[0]), self.alphabet.index(code[1])
                    return P[rpos, a] * P[rpos + 1, b2]
                reg = max(KZ_REGIONS, key=rp)
            for i in range(self.slots):
                if reg is not None and i in (rpos, rpos + 1):
                    j = self.alphabet.index(reg[i - rpos])
                    txt += reg[i - rpos]
                    probs.append(float(P[i, j] / (P[i, self._dig].sum() + 1e-9)))
                    logp += float(np.log(P[i, j] + 1e-9))
                elif i < len(t):
                    cand = self._dig if t[i] == "D" else self._let
                    j = max(cand, key=lambda k: P[i, k])
                    txt += self.alphabet[j]
                    # уверенность внутри класса (среди цифр или среди букв)
                    probs.append(float(P[i, j] / (P[i, cand].sum() + 1e-9)))
                    logp += float(np.log(P[i, j] + 1e-9))
                else:
                    logp += float(np.log(P[i, self._pad] + 1e-9))
            if best is None or logp > best[2]:
                best = (txt, probs, logp)
        return best[0], best[1]

    def _read(self, img, b):
        """Читает номер в рамке детектора b. Возвращает (номер, уверенность, вероятности, квадратный?, рамка).

        Детектор часто даёт рамку впритык или обрезает край (особенно на номерах под углом —
        отсекается регион «02» справа). Поэтому читаем несколько вырезок с разным запасом
        и выбираем результат голосованием, отдавая предпочтение формату РК."""
        H, W = img.shape[:2]
        bw, bh = b.x2 - b.x1, b.y2 - b.y1
        if bw < 15 or bh < 8:
            return None
        crops, boxes = [], []
        for px, py in CROP_PADS:
            x1, y1 = max(0, b.x1 - int(bw * px)), max(0, b.y1 - int(bh * py))
            x2, y2 = min(W, b.x2 + int(bw * px)), min(H, b.y2 + int(bh * py))
            crops.append(img[y1:y2, x1:x2])
            boxes.append((x1, y1, x2, y2))
        # Выравнивание: поворот + скос перспективы -> прямоугольник/квадрат
        try:
            rc = _rectify(img, (b.x1, b.y1, b.x2, b.y2))
        except Exception:   # выравнивание — улучшение, а не обязательный шаг
            log.exception("rectify failed")
            rc = None
        if rc is None:
            # символы не нашлись (мелкий/грязный номер) — показываем хотя бы ровную вырезку рамки
            x1, y1, x2, y2 = boxes[1]
            q = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            crop = img[y1:y2, x1:x2]
            oh = 200 if bw / max(bh, 1) < SQUARE_MAX_ASPECT else 110
            norm = cv2.resize(crop, (max(1, int(crop.shape[1] * oh / max(1, crop.shape[0]))), oh),
                              interpolation=cv2.INTER_CUBIC)
            log.info("rectify: символы не найдены, рамка %dx%d — без выравнивания", bw, bh)
            rc_fallback = _Rect(norm, q, oh == 200, [], 0.0, 0.0)
        else:
            rc_fallback = None
        rect_imgs = []
        if rc is not None:
            rect_imgs.append(rc.image)
            if rc.square and len(rc.rows) == 2:
                (a1, a2), (b1, b2) = rc.rows
                top, bot = rc.image[a1:a2], rc.image[b1:b2]
                if top.size and bot.size:
                    tw = top.shape[1]
                    top = top[:, int(tw * 0.3):]          # без флага и «KZ»
                    hh = max(top.shape[0], bot.shape[0])
                    rs = lambda im: cv2.resize(im, (max(1, int(im.shape[1] * hh / im.shape[0])), hh))
                    rect_imgs += [np.hstack([rs(top), rs(bot)]), top, bot]
        square_like = bw / max(bh, 1) < SQUARE_MAX_ASPECT
        tops_bots = []
        if square_like:
            # Квадратный номер РК: сверху «флаг/KZ + 3 цифры», снизу «регион + 2–3 буквы».
            for c in crops[:2]:
                ch, cw = c.shape[:2]
                tops_bots += [c[: int(ch * 0.52), int(cw * 0.28):], c[int(ch * 0.48):]]
        P = self._probs(crops + tops_bots + rect_imgs)

        cands = []   # (номер, уверенность, вероятности, квадратный, рамка)
        for i, box in enumerate(boxes):
            free, fp = self._decode_free(P[i])
            free = normalize(free)
            fconf = float(np.mean(fp)) if fp else 0.0
            m = SQUARE_AS_ROW.match(free)
            if m:   # квадратный номер, прочитанный в одну строку: 172 02BUR -> 172BUR02
                cands.append((m.group(1) + m.group(3) + m.group(2), fconf, fp, True, box))
            elif is_kz(free):
                cands.append((free, fconf, fp, False, box))
            else:
                kz, kp = self._decode_tmpl(P[i], KZ_TEMPLATES)
                cands.append((kz, float(np.mean(kp)) * 0.95, kp, False, box))
                if len(free) >= 4:
                    cands.append((free, fconf * 0.9, fp, False, box))
        # Ансамбль: усредняем вероятности по всем вырезкам и читаем по шаблонам РК.
        # Шумные символы (мелкий регион «02») так читаются надёжнее, чем в любой одной вырезке.
        Pavg = P[: len(crops)].mean(axis=0)
        kz, kp = self._decode_tmpl(Pavg, KZ_TEMPLATES)
        cands.append((kz, float(np.mean(kp)), kp, False, boxes[1]))
        cands.append((kz, float(np.mean(kp)), kp, False, boxes[1]))   # двойной вес в голосовании
        for j in range(0, len(tops_bots), 2):
            k = len(crops) + j
            t_txt, t_p = self._decode_tmpl(P[k], SQUARE_TOP)
            b_txt, b_p = self._decode_tmpl(P[k + 1], SQUARE_BOTTOM)
            probs = t_p + b_p[2:] + b_p[:2]
            cands.append((t_txt + b_txt[2:] + b_txt[:2], float(np.mean(probs)), probs, True, boxes[j // 2]))
        # Кандидаты с выровненного номера — самые надёжные, вес x1.5
        k0 = len(crops) + len(tops_bots)
        rbox = boxes[1]
        if rc is not None:
            q = rc.quad
            rbox = (max(0, int(q[:, 0].min())), max(0, int(q[:, 1].min())),
                    min(W, int(q[:, 0].max())), min(H, int(q[:, 1].max())))
            idx = [k0] + ([k0 + 1] if len(rect_imgs) > 1 else [])
            for k in idx:
                free, fp = self._decode_free(P[k])
                free = normalize(free)
                fconf = float(np.mean(fp)) if fp else 0.0
                m = SQUARE_AS_ROW.match(free)
                if m:
                    cands.append((m.group(1) + m.group(3) + m.group(2), fconf, fp, True, rbox, RECT_WEIGHT))
                elif is_kz(free):
                    cands.append((free, fconf, fp, rc.square, rbox, RECT_WEIGHT))
                if not rc.square:
                    kz, kp = self._decode_tmpl(P[k], KZ_TEMPLATES)
                    cands.append((kz, float(np.mean(kp)), kp, False, rbox, RECT_WEIGHT))
            if len(rect_imgs) == 4:
                t_txt, t_p = self._decode_tmpl(P[k0 + 2], SQUARE_TOP)
                b_txt, b_p = self._decode_tmpl(P[k0 + 3], SQUARE_BOTTOM)
                probs = t_p + b_p[2:] + b_p[:2]
                cands.append((t_txt + b_txt[2:] + b_txt[:2], float(np.mean(probs)), probs, True, rbox, RECT_WEIGHT))
        if not cands:
            return None
        # голосование: суммарная уверенность одинаковых прочтений, формат РК в приоритете
        votes = {}
        for c in cands:
            votes[c[0]] = votes.get(c[0], 0.0) + c[1] * (c[5] if len(c) > 5 else 1.0) * (1.0 if is_kz(c[0]) else 0.6)
        win = max(votes, key=votes.get)
        best = max((c for c in cands if c[0] == win), key=lambda c: c[1])
        return (best[0], best[1], best[2], best[3], best[4], rc or rc_fallback)

    def recognize(self, img: np.ndarray, with_stats=False):
        """Список номеров. with_stats=True — ещё и (кол-во найденных рамок номеров, макс. уверенность детектора):
        по ним видно, есть ли в кадре машина, даже если номер не прочитался."""
        if img is None or img.size == 0:
            return ([], 0, 0.0) if with_stats else []
        h, w = img.shape[:2]
        results = []
        with self._lock:
            dets = self.detector.predict(img)
            n_dets = len(dets)
            max_det = max((float(d.confidence) for d in dets), default=0.0)
            for d in dets:
                r = self._read(img, d.bounding_box)
                if r is None or len(r[0]) < 4:
                    continue
                plate, conf, probs, square, box, rc = r
                results.append(PlateResult(plate, conf, float(d.confidence), box, is_kz(plate), probs, square,
                                           rc.quad if rc is not None else None,
                                           rc.image if rc is not None else None))
        results.sort(key=lambda r: r.score, reverse=True)
        return (results, n_dets, max_det) if with_stats else results


# Шаблоны номеров РК: D — цифра, L — буква
# D — цифра, L — буква, RR — код региона из списка
KZ_TEMPLATES = ["DDDLLRR", "DDDLLLRR", "LDDDLL", "LDDDLLL"]
SQUARE_TOP = ["DDD"]
SQUARE_BOTTOM = ["RRLL", "RRLLL"]
# запас вокруг рамки детектора (доля ширины, доля высоты) — читаем все варианты
CROP_PADS = [(0.04, 0.08), (0.12, 0.15), (0.2, 0.2), (0.3, 0.25)]
# квадратный номер, прочитанный в одну строку: «172» + «02BUR»
SQUARE_AS_ROW = re.compile(r"^(\d{3})(\d{2})([A-Z]{2,3})$")
RECT_WEIGHT = 1.5        # вес голосов с выровненного номера
SQUARE_MAX_ASPECT = 1.6   # ширина/высота рамки: однострочный номер РК обычно 2.3–4.5, квадратный ~0.9–1.2


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def annotate(img: np.ndarray, results: list[PlateResult]) -> np.ndarray:
    """Рисует контур номера и рядом — выровненный номер с результатом распознавания.
    Вставка не перекрывает ни один найденный номер на кадре."""
    out = img.copy()
    H, W = img.shape[:2]
    t = max(2, W // 640)
    busy = []   # занятые области: сами номера и уже вставленные карточки
    for r in results:
        if r.quad is not None:
            q = np.asarray(r.quad)
            busy.append((int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max())))
        busy.append(tuple(int(v) for v in r.box))
    for r in results:
        color = (89, 199, 52) if r.kz_valid else (0, 149, 255)   # системные зелёный/оранжевый
        if r.quad is not None:
            pts = np.asarray(r.quad).round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, color, t, cv2.LINE_AA)
            q = np.asarray(r.quad)
            px1, py1, px2, py2 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
        else:
            px1, py1, px2, py2 = r.box
            cv2.rectangle(out, (int(px1), int(py1)), (int(px2), int(py2)), color, t)

        if not r.kz_valid and r.confidence < 0.5:
            continue   # мусорное прочтение — только контур, без карточки
        # карточка: выровненный номер + подпись
        label = f"{r.plate}  {r.confidence:.2f}"
        fs = max(0.7, W / 1100)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        pad = max(6, W // 240)
        if r.norm is not None and r.norm.size:
            nh, nw = r.norm.shape[:2]
            target_w = int(min(W * 0.32, max(W * 0.18, (px2 - px1) * 1.6)))
            if r.square:
                target_w = int(target_w * 0.55)
            sc = target_w / nw
            plate_img = cv2.resize(r.norm, (target_w, max(1, int(nh * sc))), interpolation=cv2.INTER_CUBIC)
        else:
            plate_img = None
        cw = max(tw, plate_img.shape[1] if plate_img is not None else 0) + pad * 2
        chh = (plate_img.shape[0] + pad if plate_img is not None else 0) + th + base + pad * 2
        if cw > W or chh > H:
            continue
        cx = (px1 + px2) / 2
        gap = pad * 2
        options = [
            (cx - cw / 2, py1 - gap - chh),            # над номером
            (cx - cw / 2, py2 + gap),                  # под номером
            (px2 + gap, (py1 + py2) / 2 - chh / 2),    # справа
            (px1 - gap - cw, (py1 + py2) / 2 - chh / 2),  # слева
        ]
        pos = None
        for ox_, oy_ in options:
            ox_ = int(min(max(0, ox_), W - cw))
            oy_ = int(min(max(0, oy_), H - chh))
            rect = (ox_, oy_, ox_ + cw, oy_ + chh)
            if not any(_overlap(rect, b) for b in busy):
                pos = rect
                break
        if pos is None:   # всё занято — в угол кадра, дальний от номера
            ox_ = 0 if cx > W / 2 else W - cw
            oy_ = 0 if (py1 + py2) / 2 > H / 2 else H - chh
            pos = (ox_, oy_, ox_ + cw, oy_ + chh)
        x0, y0, x1, y1 = pos
        busy.append(pos)
        # линия от карточки к номеру
        a = (int(min(max(cx, x0), x1)), y1 if y1 <= py1 else (y0 if y0 >= py2 else (y0 + y1) // 2))
        bpt = (int(cx), int(py1) if a[1] <= py1 else int(py2))
        cv2.line(out, a, bpt, color, max(1, t - 1), cv2.LINE_AA)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), -1)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, t)
        yy = y0 + pad
        if plate_img is not None:
            xx = x0 + (cw - plate_img.shape[1]) // 2
            out[yy:yy + plate_img.shape[0], xx:xx + plate_img.shape[1]] = plate_img
            yy += plate_img.shape[0] + pad
        cv2.putText(out, label, (x0 + (cw - tw) // 2, yy + th), cv2.FONT_HERSHEY_SIMPLEX, fs, (30, 30, 30), t, cv2.LINE_AA)
    return out


def decode_image(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_instance = None
_inst_lock = threading.Lock()


def get() -> ANPR:
    global _instance
    with _inst_lock:
        if _instance is None:
            _instance = ANPR()
    return _instance
