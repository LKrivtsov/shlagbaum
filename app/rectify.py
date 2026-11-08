"""Выравнивание номера перед распознаванием.

Номер на кадре обычно повёрнут и скошен перспективой (камера сбоку/сверху).
1. В области вокруг рамки детектора ищем тёмные компоненты похожие на символы.
2. Угол поворота: грубо — по максимуму дисперсии горизонтальной проекции, точно —
   прямой через центры символов основной строки.
3. Наклон (скос перспективы): по максимуму дисперсии вертикальной проекции символов,
   т.е. когда штрихи символов становятся вертикальными.
4. Границы номера: по символам + светлый фон номера (регион «02» часто сливается
   с разделителем и в маску символов не попадает).
5. Перспективное преобразование 4 углов в прямоугольник (однострочный номер)
   или почти квадрат (двухстрочный).
"""
from dataclasses import dataclass

import cv2
import numpy as np

WORK_H = 160          # высота рамки детектора после масштабирования, px
OUT_H_ROW = 110       # высота выпрямленного однострочного номера
OUT_H_SQUARE = 200    # высота выпрямленного квадратного номера


@dataclass
class Rectified:
    image: np.ndarray        # выпрямленный номер
    quad: np.ndarray         # 4 угла номера в кадре: tl, tr, br, bl (float32)
    square: bool             # двухстрочный номер
    rows: list               # для квадратного: [(y1, y2) верхней строки, (y1, y2) нижней] в координатах image
    angle: float
    shear: float


def _rows(boxes):
    """Группирует компоненты (x1,y1,x2,y2) в строки символов похожей высоты (самая большая — первая)."""
    left = list(range(len(boxes)))
    rows = []
    while len(left) >= 2:
        best, best_h = [], 0
        for i in left:
            hi = boxes[i][3] - boxes[i][1]
            ci = (boxes[i][1] + boxes[i][3]) / 2
            grp = [j for j in left
                   if 0.65 < (boxes[j][3] - boxes[j][1]) / hi < 1.5
                   and abs((boxes[j][1] + boxes[j][3]) / 2 - ci) < hi * 0.3]
            if len(grp) > len(best) or (len(grp) == len(best) and hi > best_h):
                best, best_h = grp, hi
        if len(best) < 2:
            break
        rows.append(best)
        left = [i for i in left if i not in best]
    return rows


def _best_rot(m, angles, w, h):
    best, ba = -1.0, 0.0
    for a in angles:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0)
        v = np.var(cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST).sum(1, dtype=np.float64))
        if v > best:
            best, ba = v, float(a)
    return ba


def _best_shear(m, angles, w, h):
    best, ba = -1.0, 0.0
    for a in angles:
        t = np.tan(np.radians(a))
        M = np.float32([[1, t, -t * h / 2], [0, 1, 0]])
        v = np.var(cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST).sum(0, dtype=np.float64))
        if v > best:
            best, ba = v, float(a)
    return ba


def rectify(img, box):
    """box = (x1, y1, x2, y2) рамки детектора. Возвращает Rectified или None."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 15 or bh < 8:
        return None
    ox, oy = max(0, x1 - int(bw * .5)), max(0, y1 - int(bh * .4))
    ex, ey = min(W, x2 + int(bw * .5)), min(H, y2 + int(bh * .4))
    sc = WORK_H / max(bh, 1)
    c = cv2.resize(img[oy:ey, ox:ex], None, fx=sc, fy=sc,
                   interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
    g = cv2.createCLAHE(2.0, (4, 4)).apply(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))
    h, w = g.shape
    bi = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, 8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bi, connectivity=8)
    cand = []
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if hh < WORK_H * 0.15 or hh > WORK_H or ww > hh * 1.5 or ww < 3 or a < hh * ww * 0.12:
            continue
        if x <= 1 or y <= 1 or x + ww >= w - 1 or y + hh >= h - 1:
            continue
        cand.append(i)
    if len(cand) < 3:
        return None

    def tbox(A, i):
        x, y, ww, hh, _ = st[i]
        p = np.array([[x, y, 1], [x + ww, y, 1], [x + ww, y + hh, 1], [x, y + hh, 1]], float) @ A[:2].T
        return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())

    def rot(a):
        return np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), [0, 0, 1]])

    # 1) угол: грубо по проекции, затем прямая через центры символов основной строки
    m = np.isin(lab, cand).astype(np.uint8)
    ang = _best_rot(m, np.arange(-40, 41, 2), w, h)
    R = rot(ang)
    bx = [tbox(R, i) for i in cand]
    rows = _rows(bx)
    if rows and len(rows[0]) >= 3:
        cx = np.array([st[cand[k]][0] + st[cand[k]][2] / 2 for k in rows[0]])
        cy = np.array([st[cand[k]][1] + st[cand[k]][3] / 2 for k in rows[0]])
        if np.ptp(cx) > 0:
            ang = float(np.degrees(np.arctan(np.polyfit(cx, cy, 1)[0])))
            R = rot(ang)
            bx = [tbox(R, i) for i in cand]
            rows = _rows(bx)
    if not rows or len(rows[0]) < 3:
        return None
    main = rows[0]
    hm = np.median([bx[k][3] - bx[k][1] for k in main])

    # 2) вторая строка — квадратный номер РК (сверху «172», снизу «02BUR»)
    sel, second = list(main), []
    c1 = np.mean([(bx[k][1] + bx[k][3]) / 2 for k in main])
    r1 = (min(bx[k][0] for k in main), max(bx[k][2] for k in main))
    if len(main) <= 5:
        for r in rows[1:]:
            cy2 = np.mean([(bx[k][1] + bx[k][3]) / 2 for k in r])
            rx = (min(bx[k][0] for k in r), max(bx[k][2] for k in r))
            hr = np.median([bx[k][3] - bx[k][1] for k in r])
            ov = min(r1[1], rx[1]) - max(r1[0], rx[0])
            if 0.8 * hm < abs(cy2 - c1) < 2.0 * hm and ov > -0.3 * hm and 0.6 < hr / hm < 1.6:
                second = r
                sel += r
                break
    square = bool(second)

    # 3) скос: штрихи символов должны стать вертикальными
    ms = cv2.warpAffine(np.isin(lab, [cand[k] for k in sel]).astype(np.uint8), R[:2], (w, h), flags=cv2.INTER_NEAREST)
    sh = _best_shear(ms, np.arange(-40, 41, 2.5), w, h)
    sh = _best_shear(ms, np.arange(sh - 2, sh + 2.1, 0.5), w, h)
    t = np.tan(np.radians(sh))
    A = np.array([[1, t, -t * h / 2], [0, 1, 0], [0, 0, 1]]) @ R

    def pbox(k):   # точная рамка символа после преобразования (по пикселям)
        x, y, ww, hh, _ = st[cand[k]]
        ys, xs = np.where(lab[y:y + hh, x:x + ww] == cand[k])
        p = np.stack([xs + x, ys + y, np.ones_like(xs)], 1).astype(float) @ A[:2].T
        return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())

    bs = {k: pbox(k) for k in sel}
    arr = np.array(list(bs.values()))
    tx1, ty1, tx2, ty2 = arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()
    ch = float(np.median(arr[:, 3] - arr[:, 1]))

    # 4) границы номера по светлому фону вокруг строки символов
    gw = cv2.warpAffine(g, A[:2], (w, h), flags=cv2.INTER_LINEAR)
    band = gw[int(max(0, ty1)):int(min(h, ty2 + 1))]
    left_x, right_x = tx1, tx2
    if band.size:
        col = np.percentile(band, 80, axis=0)
        bg = float(np.median(col[int(max(0, tx1)):int(min(w, tx2 + 1))]))

        def walk(x, step, limit):
            x, x0, miss = int(x), int(x), 0
            while 0 <= x + step < w and abs(x + step - x0) < limit:
                x += step
                miss = miss + 1 if col[x] < bg * 0.72 else 0
                if miss > ch * 0.12:
                    return x - step * miss
            return x

        if not square:
            right_x = max(tx2, walk(tx2, 1, ch * 2.2))   # регион «02» за разделителем
        left_x = min(tx1, walk(tx1, -1, ch * 1.0))       # флаг и «KZ»
    if square:
        px1, px2 = min(left_x, tx1 - 0.25 * ch), tx2 + 0.25 * ch
        py1, py2 = ty1 - 0.2 * ch, ty2 + 0.2 * ch
    else:
        px1, px2 = min(left_x, tx1 - 0.3 * ch) - 0.1 * ch, right_x + 0.15 * ch
        py1, py2 = ty1 - 0.22 * ch, ty2 + 0.22 * ch
    pw, ph = px2 - px1, py2 - py1
    if ph <= 0 or not (0.6 < pw / ph < 7.5):
        return None

    # 5) 4 угла в координатах кадра и перспективное выпрямление
    rect = np.array([[px1, py1, 1], [px2, py1, 1], [px2, py2, 1], [px1, py2, 1]], float)
    quad = ((rect @ np.linalg.inv(A).T)[:, :2] / sc + [ox, oy]).astype(np.float32)
    oh = OUT_H_SQUARE if square else OUT_H_ROW
    ow = int(round(oh * pw / ph))
    M = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]]))
    out = cv2.warpPerspective(img, M, (ow, oh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    rows_out = []
    if square:
        k = oh / ph
        for grp in sorted([main, second], key=lambda r: np.mean([bs[i][1] for i in r])):
            ya = min(bs[i][1] for i in grp) - py1
            yb = max(bs[i][3] for i in grp) - py1
            rows_out.append((max(0, int((ya - 0.15 * ch) * k)), min(oh, int((yb + 0.15 * ch) * k))))
    return Rectified(out, quad, square, rows_out, ang, sh)
