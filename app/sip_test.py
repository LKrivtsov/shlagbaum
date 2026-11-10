"""Пробный звонок на шлагбаум из консоли сервера, по очереди в нескольких форматах номера.

    docker compose exec app python -m app.sip_test                 # 3 формата номера из настроек
    docker compose exec app python -m app.sip_test 77083232351     # конкретные номера через пробел

Сервер, логин и пароль берутся из настроек админки. Между звонками — пауза 20 с,
чтобы было понятно, какой из них дошёл до шлагбаума.
"""
import re
import sys
import time

from . import db, sip


def variants(num: str):
    d = re.sub(r"\D", "", num)
    if len(d) == 11 and d[0] in "78":
        return ["7" + d[1:], "8" + d[1:], "+7" + d[1:]]
    return [num]


def main():
    db.init()
    cfg = sip.SipConfig.from_settings(db.get_settings())
    nums = sys.argv[1:] or variants(cfg.target)
    for i, n in enumerate(nums):
        if i:
            print("\nпауза 20 с…", flush=True)
            time.sleep(20)
        cfg.target = n
        print(f"\n[{i + 1}/{len(nums)}] {time.strftime('%H:%M:%S')} звоню на {n}", flush=True)
        r = sip.call(cfg)
        print(("OK " if r["ok"] else "ERR ") + r["status"], f"{r['duration']} с")
        print("  " + "\n  ".join(r["trace"]))


if __name__ == "__main__":
    main()
