#!/usr/bin/env python3
"""
power_logger.py — Logger i fuqisë së sistemit (AC) via ipmitool DCMI, për host-in Dell.

Xhiron NË HOST-in fizik (pve2/pve1, jo brenda VM-së) sepse ipmitool flet me BMC-në
(iDRAC) lokale. Mostron `ipmitool dcmi power reading` (instantaneous, ~1s) çdo --interval
sekonda dhe shkruan (t_unix, iso, watts) në CSV, që bashkohet me requests CSV të benchmark-ut
sipas timestamp-it (shih join_energy.py).

Përdorim tipik (në pve2):
  # 1) idle baseline 30s pa ngarkesë, pastaj lëre të xhirojë gjatë benchmark-ut
  python3 power_logger.py --out power_vm105-energy.csv
  # (Ctrl+C për ta ndalur pas benchmark-ut)

Kërkon: ipmitool + akses in-band te BMC (root). Pa rrjet/pass — përdor /dev/ipmi0.
"""
import argparse, csv, re, subprocess, sys, time
from datetime import datetime, timezone

_RE = re.compile(r"Instantaneous power reading:\s*([\d.]+)\s*Watts", re.I)


def read_power() -> float:
    """Kthen watt instantan nga DCMI, ose nan nëse dështon."""
    try:
        out = subprocess.run(["ipmitool", "dcmi", "power", "reading"],
                             capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return float("nan")
    m = _RE.search(out)
    return float(m.group(1)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Logger fuqie sistemi via ipmitool DCMI.")
    ap.add_argument("--interval", type=float, default=1.0, help="sekonda mes mostrave (default 1)")
    ap.add_argument("--out", default="power_log.csv")
    ap.add_argument("--duration", type=float, default=0, help="sekonda; 0 = derisa Ctrl+C")
    args = ap.parse_args()

    # Sanity: a punon ipmitool?
    w0 = read_power()
    if w0 != w0:
        print("[err] ipmitool dcmi power reading nuk ktheu watt — kontrollo ipmitool/BMC.",
              file=sys.stderr)
        sys.exit(1)
    print(f"[ok] fuqia fillestare = {w0:.0f} W — po shkruaj te {args.out} çdo {args.interval}s "
          f"(Ctrl+C për ndalim)", file=sys.stderr)

    n = 0
    t_end = time.time() + args.duration if args.duration > 0 else float("inf")
    try:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_unix", "iso", "watts"])
            while time.time() < t_end:
                t = time.time()
                watts = read_power()
                w.writerow([f"{t:.3f}", datetime.fromtimestamp(t, timezone.utc).isoformat(),
                            f"{watts:.1f}"])
                f.flush()
                n += 1
                if n % 30 == 0:
                    print(f"  {n} mostra, e fundit {watts:.0f} W", file=sys.stderr)
                # ruaj kadencën
                dt = args.interval - (time.time() - t)
                if dt > 0:
                    time.sleep(dt)
    except KeyboardInterrupt:
        pass
    print(f"[done] {n} mostra → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
