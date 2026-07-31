#!/usr/bin/env python3
"""
bench_llm.py — Benchmark of local LLM inference (Ollama) on a modest GPU (NVIDIA T400 4 GB)

Measured for every (model × num_ctx × repetition):
  - Generation throughput (tokens/s)         from eval_count / eval_duration
  - Prompt eval rate (tokens/s)              from prompt_eval_count / prompt_eval_duration
  - Time-to-first-token TTFT (ms)            from request dispatch to the first chunk
  - Total latency (ms)                       from request to completion
  - Peak VRAM (MB), GPU power avg/max (W)    from the nvidia-smi sampler
  - GPU energy (J) & J/token                 trapezoidal integral of power − idle baseline
  - GPU/CPU offload split (%)                from `ollama ps`
Concurrency test (1/2/4/8 parallel requests) → per-user degradation & aggregate throughput.

RQ2 requires the context window (num_ctx = the KV-cache allocation) to be a CONTROLLED
variable, so `num_ctx` is passed explicitly to Ollama. Otherwise Ollama allocates 4096 by
default and the context effect appears flat. The prompt (input) length is held fixed
(--prompt-tokens) while num_ctx is varied (--ctx).

Design: the script runs INSIDE the GPU-equipped VM (local Ollama + local nvidia-smi), so
that GPU samples and requests share the same clock. It can also run against a remote host
(--host), in which case pass --no-gpu since nvidia-smi is not local.

Author: AI-LAB / FIE Measurement Lab
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError

# ----------------------------------------------------------------------------
# Default configuration (overridden by CLI arguments)
# ----------------------------------------------------------------------------

DEFAULT_MODELS = ["qwen2.5:1.5b", "phi3.5", "qwen2.5-coder:7b", "phi4"]
DEFAULT_CTX_LENS = [512, 2048, 8192]  # num_ctx = allocated KV window (the RQ2 variable)
DEFAULT_PROMPT_TOKENS = 128           # fixed prompt (input) length; num_ctx varies separately
DEFAULT_REPS = 10
DEFAULT_NUM_PREDICT = 128             # generated tokens (fixed, for a fair comparison)
DEFAULT_CONCURRENCY = [1, 2, 4, 8]
GPU_SAMPLE_MS = 100                   # nvidia-smi sampling interval (~100 ms)

# Filler words used to build prompts of a controlled length.
# The real prompt_eval_count is reported by Ollama; this only pads the text.
_FILLER = (
    "The measurement laboratory records temperature humidity voltage current power "
    "and energy across many sensors while the system evaluates each token in sequence "
).split()


# ----------------------------------------------------------------------------
# nvidia-smi sampler (background thread)
# ----------------------------------------------------------------------------

@dataclass
class GpuSample:
    t: float          # time.time() at the moment of the reading
    mem_used: float   # MB
    mem_total: float  # MB
    util: float       # %
    temp: float       # °C
    power: float      # W  (nan if N/A)


class GpuSampler:
    """Runs `nvidia-smi ... -lms N` as a subprocess and reads its lines continuously."""

    QUERY = "memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"

    def __init__(self, gpu_index: int = 0, sample_ms: int = GPU_SAMPLE_MS):
        self.gpu_index = gpu_index
        self.sample_ms = sample_ms
        self.samples: list[GpuSample] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.power_supported = True
        self.available = shutil.which("nvidia-smi") is not None

    def _reader(self):
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 5:
                continue
            t = time.time()

            def _f(x):
                try:
                    return float(x)
                except ValueError:
                    return float("nan")
            power = _f(parts[4])
            if power != power:  # nan → power.draw unsupported on this card
                self.power_supported = False
            self.samples.append(GpuSample(
                t=t, mem_used=_f(parts[0]), mem_total=_f(parts[1]),
                util=_f(parts[2]), temp=_f(parts[3]), power=power,
            ))

    def start(self):
        if not self.available:
            print("[gpu] nvidia-smi not found — GPU sampling disabled.", file=sys.stderr)
            return
        cmd = [
            "nvidia-smi", f"--id={self.gpu_index}",
            f"--query-gpu={self.QUERY}",
            "--format=csv,noheader,nounits",
            f"-lms={self.sample_ms}",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)

    def window(self, t0: float, t1: float) -> list[GpuSample]:
        return [s for s in self.samples if t0 <= s.t <= t1]

    def idle_power(self, seconds: float = 5.0) -> float:
        """Measure idle power (no load) → mean W. nan if power is unavailable."""
        print(f"[gpu] measuring idle baseline for {seconds:.0f}s ...", file=sys.stderr)
        t0 = time.time()
        time.sleep(seconds)
        w = self.window(t0, time.time())
        vals = [s.power for s in w if s.power == s.power]
        return statistics.mean(vals) if vals else float("nan")


def trapezoid_energy(samples: list[GpuSample], idle_w: float = 0.0) -> float:
    """Energy (J) = ∫ P dt by the trapezoidal rule, minus the idle baseline."""
    pts = [(s.t, s.power) for s in samples if s.power == s.power]
    if len(pts) < 2:
        return float("nan")
    e = 0.0
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        p0i = max(p0 - idle_w, 0.0)
        p1i = max(p1 - idle_w, 0.0)
        e += (p0i + p1i) / 2.0 * (t1 - t0)
    return e


# ----------------------------------------------------------------------------
# Ollama client
# ----------------------------------------------------------------------------

@dataclass
class GenResult:
    model: str
    target_ctx: int          # num_ctx i alokuar
    rep: int
    ok: bool = True
    error: str = ""
    ttft_ms: float = float("nan")
    total_ms: float = float("nan")
    prompt_eval_count: int = 0
    prompt_eval_ms: float = float("nan")
    eval_count: int = 0
    eval_ms: float = float("nan")
    load_ms: float = float("nan")
    gen_tps: float = float("nan")        # tokens/s gjenerimi
    prompt_tps: float = float("nan")     # tokens/s prompt eval
    vram_peak_mb: float = float("nan")
    power_avg_w: float = float("nan")
    power_max_w: float = float("nan")
    energy_j: float = float("nan")       # GPU, inkremental (− idle)
    j_per_token: float = float("nan")
    gpu_offload_pct: float = float("nan")
    t_start_unix: float = float("nan")   # epoch, for joining with the host power log
    t_end_unix: float = float("nan")


def build_prompt(target_tokens: int) -> str:
    # ~0.75 words/token as an approximation; the real length comes from prompt_eval_count.
    n_words = max(4, int(target_tokens * 0.75))
    words = (_FILLER * (n_words // len(_FILLER) + 1))[:n_words]
    return "Summarize the following log verbatim then continue: " + " ".join(words)


def ollama_generate(host: str, model: str, prompt: str, num_predict: int,
                    num_ctx: int | None = None,
                    timeout: float = 600.0) -> tuple[dict, float, float, float]:
    """Kërkesë streaming. Kthen (final_json, t_start, t_first, t_end)."""
    url = f"{host.rstrip('/')}/api/generate"
    options = {"num_predict": num_predict, "temperature": 0.0, "seed": 42}
    if num_ctx:
        options["num_ctx"] = num_ctx      # dritarja e alokuar KV — kontrollon VRAM/ofloadim
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": options,
        "keep_alive": "5m",
    }
    data = json.dumps(payload).encode()
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"})
    t_start = time.time()
    t_first = float("nan")
    final: dict = {}
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            obj = json.loads(raw)
            if t_first != t_first and obj.get("response"):
                t_first = time.time()
            if obj.get("done"):
                final = obj
    t_end = time.time()
    return final, t_start, t_first, t_end


def ollama_unload(host: str, model: str):
    """Unload the model from VRAM (keep_alive=0) for cold-start measurements."""
    url = f"{host.rstrip('/')}/api/generate"
    payload = {"model": model, "keep_alive": 0}
    try:
        req = urlrequest.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
        urlrequest.urlopen(req, timeout=30).read()
    except URLError:
        pass


def gpu_offload_pct(model: str) -> float:
    """Lexon split-in CPU/GPU nga `ollama ps` (kolona PROCESSOR). Kthen % në GPU."""
    if shutil.which("ollama") is None:
        return float("nan")
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return float("nan")
    import re
    for line in out.splitlines():
        if line.startswith(model.split(":")[0]):
            low = line.lower()
            if "100% gpu" in low:
                return 100.0
            if "100% cpu" in low:
                return 0.0
            m = re.search(r"(\d+)%\s*/\s*(\d+)%\s*cpu/gpu", low)
            if m:
                return float(m.group(2))
            m = re.search(r"(\d+)%\s*gpu", low)
            if m:
                return float(m.group(1))
    return float("nan")


# ----------------------------------------------------------------------------
# Executing a single request + joining it with the GPU samples
# ----------------------------------------------------------------------------

def run_one(host: str, sampler: GpuSampler, idle_w: float, model: str,
            target_ctx: int, rep: int, num_predict: int,
            prompt_tokens: int) -> GenResult:
    r = GenResult(model=model, target_ctx=target_ctx, rep=rep)
    prompt = build_prompt(prompt_tokens)
    try:
        final, t0, tf, t1 = ollama_generate(host, model, prompt, num_predict, num_ctx=target_ctx)
    except Exception as e:  # noqa: BLE001
        r.ok = False
        r.error = f"{type(e).__name__}: {e}"
        return r

    r.total_ms = (t1 - t0) * 1000.0
    r.t_start_unix = t0
    r.t_end_unix = t1
    r.ttft_ms = (tf - t0) * 1000.0 if tf == tf else float("nan")
    r.prompt_eval_count = int(final.get("prompt_eval_count", 0) or 0)
    r.eval_count = int(final.get("eval_count", 0) or 0)
    ped = final.get("prompt_eval_duration", 0) or 0
    ed = final.get("eval_duration", 0) or 0
    ld = final.get("load_duration", 0) or 0
    r.prompt_eval_ms = ped / 1e6
    r.eval_ms = ed / 1e6
    r.load_ms = ld / 1e6
    if ed > 0 and r.eval_count:
        r.gen_tps = r.eval_count / (ed / 1e9)
    if ped > 0 and r.prompt_eval_count:
        r.prompt_tps = r.prompt_eval_count / (ped / 1e9)

    # GPU window
    win = sampler.window(t0, t1)
    if win:
        r.vram_peak_mb = max(s.mem_used for s in win)
        powers = [s.power for s in win if s.power == s.power]
        if powers:
            r.power_avg_w = statistics.mean(powers)
            r.power_max_w = max(powers)
            e = trapezoid_energy(win, idle_w=idle_w if idle_w == idle_w else 0.0)
            r.energy_j = e
            if e == e and r.eval_count:
                r.j_per_token = e / r.eval_count
    r.gpu_offload_pct = gpu_offload_pct(model)
    return r


def run_concurrency(host: str, sampler: GpuSampler, model: str, target_ctx: int,
                    num_predict: int, n_parallel: int, prompt_tokens: int) -> dict:
    """N parallel requests → aggregate & per-user throughput."""
    prompt = build_prompt(prompt_tokens)
    results: list[dict] = []
    lock = threading.Lock()

    def worker(idx: int):
        try:
            final, t0, tf, t1 = ollama_generate(host, model, prompt, num_predict, num_ctx=target_ctx)
            ec = int(final.get("eval_count", 0) or 0)
            ed = final.get("eval_duration", 0) or 0
            tps = ec / (ed / 1e9) if ed > 0 and ec else float("nan")
            with lock:
                results.append({"tps": tps, "eval_count": ec,
                                "ttft_ms": (tf - t0) * 1000.0 if tf == tf else float("nan"),
                                "wall_s": t1 - t0})
        except Exception as e:  # noqa: BLE001
            with lock:
                results.append({"error": str(e)})

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_parallel)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0

    ok = [x for x in results if "tps" in x]
    per_user = statistics.mean([x["tps"] for x in ok if x["tps"] == x["tps"]]) if ok else float("nan")
    total_tokens = sum(x["eval_count"] for x in ok)
    aggregate_tps = total_tokens / wall if wall > 0 else float("nan")
    win = sampler.window(t0, t0 + wall)
    vram = max((s.mem_used for s in win), default=float("nan"))
    return {
        "model": model, "target_ctx": target_ctx, "n_parallel": n_parallel,
        "n_ok": len(ok), "wall_s": round(wall, 3),
        "aggregate_tps": round(aggregate_tps, 2),
        "per_user_tps": round(per_user, 2) if per_user == per_user else "",
        "vram_peak_mb": vram,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def median_iqr(xs: list[float]) -> tuple[float, float, float]:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return (float("nan"),) * 3
    med = statistics.median(xs)
    if len(xs) >= 4:
        q1 = statistics.median(xs[: len(xs) // 2])
        q3 = statistics.median(xs[(len(xs) + 1) // 2:])
    else:
        q1 = q3 = med
    return med, q1, q3


def main():
    ap = argparse.ArgumentParser(description="Benchmark LLM lokale (Ollama) në GPU modeste.")
    ap.add_argument("--host", default="http://127.0.0.1:11434",
                    help="Ollama API base URL (default localhost; e.g. http://10.0.0.5:11434)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--ctx", nargs="+", type=int, default=DEFAULT_CTX_LENS,
                    help="num_ctx: allocated KV windows (the RQ2 variable)")
    ap.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS,
                    help="fixed prompt (input) length")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    ap.add_argument("--concurrency", nargs="+", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--no-gpu", action="store_true", help="Disable nvidia-smi (remote measurement).")
    ap.add_argument("--no-concurrency", action="store_true")
    ap.add_argument("--cold-start", action="store_true",
                    help="Unload the model before each first batch (measures load_duration).")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--label", default="", help="Label (e.g. 16gb).")
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.label}" if args.label else ""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    req_csv = outdir / f"requests{tag}_{ts}.csv"
    gpu_csv = outdir / f"gpu_samples{tag}_{ts}.csv"
    sum_csv = outdir / f"summary{tag}_{ts}.csv"
    con_csv = outdir / f"concurrency{tag}_{ts}.csv"
    meta_json = outdir / f"meta{tag}_{ts}.json"

    print(f"# Benchmark start {ts}  host={args.host}  label={args.label or '-'}")
    sampler = GpuSampler(gpu_index=args.gpu_index)
    if not args.no_gpu:
        sampler.start()
        time.sleep(1.0)  # lëri sampler-it kohë të nisë

    idle_w = float("nan")
    if not args.no_gpu and sampler.available:
        idle_w = sampler.idle_power(5.0)
        print(f"# idle GPU power = {idle_w:.2f} W" if idle_w == idle_w
              else "# power.draw = N/A (T400 mund të mos e mbështesë)")

    all_results: list[GenResult] = []
    con_rows: list[dict] = []
    try:
        for model in args.models:
            for ctx in args.ctx:
                if args.cold_start:
                    ollama_unload(args.host, model)
                    time.sleep(2)
                else:
                    # warm-up (not counted) — loads the model into VRAM, may take 10-40 s
                    print(f"[warm-up] loading {model} (num_ctx={ctx}) — please wait, not frozen ...",
                          flush=True)
                    try:
                        ollama_generate(args.host, model, build_prompt(args.prompt_tokens),
                                        8, num_ctx=ctx)
                    except Exception as e:  # noqa: BLE001
                        print(f"[warn] warm-up dështoi {model}/{ctx}: {e}", file=sys.stderr)
                for rep in range(1, args.reps + 1):
                    r = run_one(args.host, sampler, idle_w, model, ctx, rep,
                                args.num_predict, args.prompt_tokens)
                    all_results.append(r)
                    status = "ok" if r.ok else f"ERR {r.error}"
                    print(f"  {model:20s} num_ctx={ctx:5d} rep={rep:2d}  "
                          f"gen={r.gen_tps:6.1f} tps  ttft={r.ttft_ms:7.1f}ms  "
                          f"vram={r.vram_peak_mb:6.0f}MB  gpu={r.gpu_offload_pct:3.0f}%  {status}")
    finally:
        # Shkruaj gjithnjë çka u mblodh
        _write_requests(req_csv, all_results)
        _write_summary(sum_csv, all_results)
        if not args.no_gpu:
            _write_gpu(gpu_csv, sampler.samples)

        # Concurrency
        if not args.no_concurrency and all_results:
            print("# Test konkurrence ...")
            cm = args.models[0]
            cc = args.ctx[0]
            for n in args.concurrency:
                row = run_concurrency(args.host, sampler, cm, cc, args.num_predict, n,
                                      args.prompt_tokens)
                con_rows.append(row)
                print(f"  n={n}: agg={row['aggregate_tps']} tps  "
                      f"per_user={row['per_user_tps']} tps  vram={row['vram_peak_mb']}MB")
            _write_concurrency(con_csv, con_rows)

        if not args.no_gpu:
            sampler.stop()

        meta = {
            "timestamp_utc": ts, "host": args.host, "label": args.label,
            "models": args.models, "num_ctx": args.ctx, "prompt_tokens": args.prompt_tokens,
            "reps": args.reps,
            "num_predict": args.num_predict, "concurrency": args.concurrency,
            "idle_power_w": idle_w, "power_supported": sampler.power_supported,
            "n_requests": len(all_results),
        }
        meta_json.write_text(json.dumps(meta, indent=2))
        print(f"\n# Skedarë:\n  {req_csv}\n  {sum_csv}"
              + (f"\n  {gpu_csv}" if not args.no_gpu else "")
              + (f"\n  {con_csv}" if con_rows else "")
              + f"\n  {meta_json}")
        if not sampler.power_supported and not args.no_gpu:
            print("\n⚠️  power.draw returned N/A on this GPU — GPU energy (J/token) was not computed.\n"
                  "    Fall back to BMC/IPMI or a smart plug for system energy (see the energy section).")


def _write_requests(path: Path, results: list[GenResult]):
    if not results:
        return
    fields = list(asdict(results[0]).keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def _write_gpu(path: Path, samples: list[GpuSample]):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_unix", "iso", "mem_used_mb", "mem_total_mb", "util_pct", "temp_c", "power_w"])
        for s in samples:
            w.writerow([f"{s.t:.3f}", datetime.fromtimestamp(s.t, timezone.utc).isoformat(),
                        s.mem_used, s.mem_total, s.util, s.temp, s.power])


def _write_summary(path: Path, results: list[GenResult]):
    groups: dict[tuple, list[GenResult]] = {}
    for r in results:
        if r.ok:
            groups.setdefault((r.model, r.target_ctx), []).append(r)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "num_ctx", "n", "prompt_tokens_median",
                    "gen_tps_median", "gen_tps_q1", "gen_tps_q3",
                    "ttft_ms_median", "ttft_ms_q1", "ttft_ms_q3",
                    "vram_peak_mb_median", "power_avg_w_median",
                    "j_per_token_median", "gpu_offload_pct_median"])
        for (model, ctx), rs in sorted(groups.items()):
            gm, gq1, gq3 = median_iqr([r.gen_tps for r in rs])
            tm, tq1, tq3 = median_iqr([r.ttft_ms for r in rs])
            vram = median_iqr([r.vram_peak_mb for r in rs])[0]
            pw = median_iqr([r.power_avg_w for r in rs])[0]
            jt = median_iqr([r.j_per_token for r in rs])[0]
            off = median_iqr([r.gpu_offload_pct for r in rs])[0]
            ptok = median_iqr([float(r.prompt_eval_count) for r in rs])[0]
            w.writerow([model, ctx, len(rs), f"{ptok:.0f}",
                        f"{gm:.2f}", f"{gq1:.2f}", f"{gq3:.2f}",
                        f"{tm:.1f}", f"{tq1:.1f}", f"{tq3:.1f}",
                        f"{vram:.0f}", f"{pw:.2f}", f"{jt:.4f}", f"{off:.0f}"])


def _write_concurrency(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
