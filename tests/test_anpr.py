"""Проверка распознавания на примерах: python -m tests.test_anpr
(в контейнере: docker compose exec app python -m tests.test_anpr)"""
import time
from pathlib import Path

from app import anpr

SAMPLES = Path(__file__).parent / "samples"


def main():
    eng = anpr.get()
    ok = 0
    files = sorted(SAMPLES.glob("*.jpg"))
    for f in files:
        img = anpr.decode_image(f.read_bytes())
        t = time.time()
        res = eng.recognize(img)
        ms = (time.time() - t) * 1000
        got = res[0].plate if res else "-"
        good = got == f.stem
        ok += good
        print(f"{'OK ' if good else 'ERR'} ожидали {f.stem:10} получили {got:10} "
              f"conf={res[0].confidence if res else 0:.3f} {ms:.0f} мс")
    print(f"Итого: {ok}/{len(files)}")
    return ok == len(files)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
