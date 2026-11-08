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
    re.compile(r"^\d{2}[A-Z]{2}\d{3}$"),     # мото/прицепы и др.
    re.compile(r"^\d{3}[A-Z]{2}\d{3}$"),
]

# Кириллица -> латиница (для ввода номеров в белый список с русской раскладкой)
_CYR = str.maketrans("АВЕКМНОРСТУХІ", "ABEKMHOPCTYXI")


def normalize(plate: str) -> str:
    p = (plate or "").upper().translate(_CYR)
    return re.sub(r"[^A-Z0-9]", "", p)


def is_kz(plate: str) -> bool:
    return any(p.match(plate) for p in KZ_PATTERNS)


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
            for i in range(self.slots):
                if i < len(t):
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

    def _read(self, crop, w, h):
        """Читает номер на вырезанной рамке. Возвращает (номер, уверенность, вероятности, квадратный?)."""
        if w / max(h, 1) < SQUARE_MAX_ASPECT:
            # Квадратный номер РК: сверху «KZ/флаг + 3 цифры», снизу «регион + 2–3 буквы».
            # Режем на две строки и читаем каждую отдельно.
            H, W = crop.shape[:2]
            top = crop[: int(H * 0.52), int(W * 0.28):]   # без флага и «KZ»
            bot = crop[int(H * 0.48):]
            P = self._probs([top, bot])
            t_txt, t_p = self._decode_tmpl(P[0], SQUARE_TOP)
            b_txt, b_p = self._decode_tmpl(P[1], SQUARE_BOTTOM)
            plate = t_txt + b_txt[2:] + b_txt[:2]   # 172 + BUR + 02
            probs = t_p + b_p[2:] + b_p[:2]
            return plate, float(np.mean(probs)), probs, True
        P = self._probs([crop])[0]
        free, fp = self._decode_free(P)
        free = normalize(free)
        if is_kz(free) or not free:
            if len(free) < 4:
                return None
            return free, float(np.mean(fp)) if fp else 0.0, fp, False
        # не похоже на формат РК — пробуем прочитать под шаблоны РК
        kz, kp = self._decode_tmpl(P, KZ_TEMPLATES)
        kconf = float(np.mean(kp))
        fconf = float(np.mean(fp)) if fp else 0.0
        if kconf >= fconf - 0.15:
            return kz, kconf, kp, False
        if len(free) < 4:
            return None
        return free, fconf, fp, False

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
                b = d.bounding_box
                # небольшой запас вокруг рамки помогает OCR
                px, py = int((b.x2 - b.x1) * 0.04), int((b.y2 - b.y1) * 0.08)
                x1, y1 = max(0, b.x1 - px), max(0, b.y1 - py)
                x2, y2 = min(w, b.x2 + px), min(h, b.y2 + py)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0 or (x2 - x1) < 20:
                    continue
                r = self._read(crop, x2 - x1, y2 - y1)
                if r is None:
                    continue
                plate, conf, probs, square = r
                results.append(PlateResult(plate, conf, float(d.confidence), (x1, y1, x2, y2), is_kz(plate), probs,
                                           square))
        results.sort(key=lambda r: r.score, reverse=True)
        return (results, n_dets, max_det) if with_stats else results


# Шаблоны номеров РК: D — цифра, L — буква
KZ_TEMPLATES = ["DDDLLDD", "DDDLLLDD", "LDDDLL", "LDDDLLL"]
SQUARE_TOP = ["DDD"]
SQUARE_BOTTOM = ["DDLL", "DDLLL"]
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
