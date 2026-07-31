#!/usr/bin/env python3
"""
power_logger.py — System (AC) power logger via ipmitool DCMI, for Dell hosts.

Runs ON THE PHYSICAL HOST (not inside the VM), because ipmitool talks to the local BMC
(iDRAC). It samples `ipmitool dcmi power reading` (instantaneous, ~1 s) every --interval
seconds and writes (t_unix, iso, watts) to a CSV, which is then joined with the benchmark
requests CSV by timestamp (see join_energy.py).

Typical use (on the hypervisor host):
  # 1) 30 s idle baseline with no load, then leave it running for the whole benchmark
  python3 power_logger.py --out power_energy.csv
  # (Ctrl+C to stop it after the benchmark)

Requires: ipmitool + in-band BMC access (root). No network or password — uses /dev/ipmi0.
"""
import argparse, csv, re, subprocess, sys, time
from datetime import datetime, timezone

_RE = re.compile(r"Instantaneous power reading:\s*([\d.]+)\s*Watts", re.I)


def read_power() -> float:
    """Return the instantaneous wattage from DCMI, or nan on failure."""
    try:
        out = subprocess.run(["ipmitool", "dcmi", "power", "reading"],
                             capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return float("nan")
    m = _RE.search(out)
    return float(m.group(1)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser(description="System power logger via ipmitool DCMI.")
    ap.add_argument("--interval", type=float, default=1.0, help="sekonda mes mostrave (default 1)")
    ap.add_argument("--out", default="power_log.csv")
    ap.add_argument("--duration", type=float, default=0, help="sekonda; 0 = derisa Ctrl+C")
    args = ap.parse_args()

    # Sanity: a punon ipmitool?
    w0 = read_power()
    if w0 != w0:
        print("[err] ipmitool dcmi power reading returned no wattage — check ipmitool/BMC.",
              file=sys.stderr)
        sys.exit(1)
    print(f"[ok] initial power = {w0:.0f} W — writing to {args.out} every {args.interval}s "
          f"(Ctrl+C to stop)", file=sys.stderr)

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
