#!/usr/bin/env python3
"""
join_energy.py — Bashko requests CSV (benchmark) me power CSV (host) → J/token i sistemit.

Për çdo kërkesë, integron fuqinë e sistemit mbi dritaren [t_start_unix, t_end_unix] me rregull
trapezoidal, zbret idle baseline → energji inkrementale; pjesëton me eval_count → J/token sistemi.
Agregon median + IQR për (model, num_ctx).

Kërkon që host-i (power) dhe VM-ja (requests) të jenë NTP-sync (janë).

Përdorim:
  python3 join_energy.py --requests results/requests_vm105-energy_*.csv \
                         --power power_vm105-energy.csv --out results/energy_vm105.csv
Idle: jepe me --idle-w, ose lihet auto = perc.10 e watt-eve (përafrim i idle-it).
"""
import argparse, csv, statistics
from pathlib import Path


def load_power(path):
    pts = []
    for r in csv.DictReader(open(path)):
        try:
            pts.append((float(r["t_unix"]), float(r["watts"])))
        except (ValueError, KeyError):
            continue
    pts.sort()
    return pts


def trapz(pts, idle):
    if len(pts) < 2:
        return float("nan")
    e = 0.0
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        e += (max(p0 - idle, 0.0) + max(p1 - idle, 0.0)) / 2.0 * (t1 - t0)
    return e


def median_iqr(xs):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return (float("nan"),) * 3
    med = statistics.median(xs)
    if len(xs) >= 4:
        q1 = statistics.median(xs[:len(xs) // 2])
        q3 = statistics.median(xs[(len(xs) + 1) // 2:])
    else:
        q1 = q3 = med
    return med, q1, q3


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--idle-w", type=float, default=None, help="watt idle; auto=perc.10 nëse mungon")
    ap.add_argument("--out", default="energy_summary.csv")
    args = ap.parse_args()

    power = load_power(args.power)
    if len(power) < 2:
        raise SystemExit("Power CSV bosh ose i pamjaftueshëm.")
    idle = args.idle_w if args.idle_w is not None else percentile([w for _, w in power], 10)
    print(f"# mostra fuqie: {len(power)} | idle baseline = {idle:.1f} W "
          f"({'e dhënë' if args.idle_w is not None else 'auto perc.10'})")

    rows = list(csv.DictReader(open(args.requests)))
    groups = {}
    per_req = []
    for r in rows:
        if r.get("ok", "True") not in ("True", "true", "1"):
            continue
        try:
            t0 = float(r["t_start_unix"]); t1 = float(r["t_end_unix"])
            ec = int(float(r["eval_count"]))
        except (ValueError, KeyError):
            continue
        if t0 != t0 or t1 != t1 or ec <= 0:
            continue
        win = [(t, w) for (t, w) in power if t0 <= t <= t1]
        e = trapz(win, idle)
        avgw = statistics.mean([w for _, w in win]) if win else float("nan")
        jtok = e / ec if e == e else float("nan")
        per_req.append({**r, "sys_energy_j": e, "sys_avg_w": avgw, "sys_j_per_token": jtok})
        groups.setdefault((r["model"], int(r["target_ctx"])), []).append((jtok, avgw))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "num_ctx", "n", "sys_avg_w_median",
                    "sys_j_per_token_median", "sys_j_per_token_q1", "sys_j_per_token_q3",
                    "idle_baseline_w"])
        for (model, ctx), vals in sorted(groups.items()):
            jt = [v[0] for v in vals]; aw = [v[1] for v in vals]
            m, q1, q3 = median_iqr(jt)
            w.writerow([model, ctx, len(vals), f"{median_iqr(aw)[0]:.1f}",
                        f"{m:.3f}", f"{q1:.3f}", f"{q3:.3f}", f"{idle:.1f}"])
    print(f"# u shkrua: {args.out}  ({len(groups)} konfigurime, {len(per_req)} kërkesa)")


if __name__ == "__main__":
    main()
