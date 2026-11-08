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
        log.info("ANPR ready: %s + %s, threads=%s", config.DETECTOR_MODEL, config.OCR_MODEL, config.ANPR_THREADS)

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
        square_like = bw / max(bh, 1) < SQUARE_MAX_ASPECT
        tops_bots = []
        if square_like:
            # Квадратный номер РК: сверху «флаг/KZ + 3 цифры», снизу «регион + 2–3 буквы».
            for c in crops[:2]:
                ch, cw = c.shape[:2]
                tops_bots += [c[: int(ch * 0.52), int(cw * 0.28):], c[int(ch * 0.48):]]
        P = self._probs(crops + tops_bots)

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
        if not cands:
            return None
        # голосование: суммарная уверенность одинаковых прочтений, формат РК в приоритете
        votes = {}
        for c in cands:
            votes[c[0]] = votes.get(c[0], 0.0) + c[1] * (1.0 if is_kz(c[0]) else 0.6)
        win = max(votes, key=votes.get)
        best = max((c for c in cands if c[0] == win), key=lambda c: c[1])
        return best

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
                plate, conf, probs, square, box = r
                results.append(PlateResult(plate, conf, float(d.confidence), box, is_kz(plate), probs, square))
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
SQUARE_MAX_ASPECT = 1.6   # ширина/высота рамки: однострочный номер РК обычно 2.3–4.5, квадратный ~0.9–1.2


def annotate(img: np.ndarray, results: list[PlateResult]) -> np.ndarray:
    out = img.copy()
    for r in results:
        x1, y1, x2, y2 = r.box
        color = (89, 199, 52) if r.kz_valid else (0, 149, 255)  # системные зелёный/оранжевый
        t = max(2, img.shape[1] // 400)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
        label = f"{r.plate} {r.confidence:.2f}"
        fs = max(0.6, img.shape[1] / 1200)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t, cv2.LINE_AA)
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
