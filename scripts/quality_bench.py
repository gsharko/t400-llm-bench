#!/usr/bin/env python3
"""
quality_bench.py — Task-quality benchmark across quantization levels (Ollama, NVIDIA T400 4 GB)

Covers the quality dimension of RQ1: how much accuracy is sacrificed by the quantization
needed to fit into 4 GB of VRAM? It compares quantization tags of the same model
(e.g. q4_K_M vs q8_0) on two benchmarks:

  mmlu      — MMLU subset (multiple choice, A/B/C/D), accuracy %  [general-purpose models]
  humaneval — HumanEval pass@1 (generation + test execution)      [code models]

For every configuration it also records peak VRAM, offload percentage and throughput, so
that quality can be related directly to "does it fit in 4 GB" (the frontier of the paper).

The design follows bench_llm.py: standard library only, streaming API, temperature=0 and
seed=42, CSV output with label+timestamp under results/. Runs INSIDE the GPU-equipped VM
(local Ollama + local nvidia-smi); for remote use pass --no-gpu.

Datasets are downloaded automatically into --data-dir (default: data/):
  MMLU:      https://people.eecs.berkeley.edu/~hendrycks/data.tar (~160 MB, CSV)
  HumanEval: https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz

⚠️ HumanEval EXECUTES model-generated code (subprocess with a timeout).
   Run it only inside a disposable laboratory VM, never on a production host.

Typical usage (see PROTOCOL-Quality.md):
  python3 quality_bench.py mmlu --models qwen2.5:1.5b-instruct-q4_K_M qwen2.5:1.5b-instruct-q8_0 \
      --label 16gb
  python3 quality_bench.py humaneval --models qwen2.5-coder:7b --label 16gb

Autor: AI-LAB / FIE Measurement Lab
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import statistics
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlrequest

# Rimarrim sampler-in dhe leximin e ofloadimit nga benchmark-u ekzistues
try:
    from bench_llm import GpuSampler, gpu_offload_pct, median_iqr
except ImportError:
    print("Error: quality_bench.py must live in the same directory as bench_llm.py",
          file=sys.stderr)
    raise

# ----------------------------------------------------------------------------
# Default configuration
# ----------------------------------------------------------------------------

MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

# MMLU subset: STEM subjects close to an engineering curriculum, plus one non-STEM for balance
DEFAULT_SUBJECTS = [
    "electrical_engineering",
    "college_computer_science",
    "college_physics",
    "college_mathematics",
    "formal_logic",
    "high_school_statistics",
]
DEFAULT_LIMIT_PER_SUBJECT = 100   # question cap per subject (bounds runtime on offloaded models)
DEFAULT_NUM_CTX = 2048            # enough for 0-5 shot; fixed so VRAM stays comparable
DEFAULT_HE_NUM_PREDICT = 512      # max tokens for a HumanEval solution
EXEC_TIMEOUT_S = 15               # execution timeout for a HumanEval test

LETTERS = ["A", "B", "C", "D"]


# ----------------------------------------------------------------------------
# Klient Ollama (streaming, si te bench_llm.py)
# ----------------------------------------------------------------------------

def ollama_generate(host: str, model: str, prompt: str, num_predict: int,
                    num_ctx: int, timeout: float = 600.0) -> tuple[str, dict, float, float]:
    """Kthen (tekst, final_json, t_start, t_end)."""
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": num_predict, "temperature": 0.0, "seed": 42,
                    "num_ctx": num_ctx},
        "keep_alive": "10m",
    }
    req = urlrequest.Request(url, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
    t0 = time.time()
    chunks: list[str] = []
    final: dict = {}
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            obj = json.loads(raw)
            if obj.get("response"):
                chunks.append(obj["response"])
            if obj.get("done"):
                final = obj
    return "".join(chunks), final, t0, time.time()


def _tps(final: dict) -> tuple[float, float]:
    """(gen_tps, prompt_tps) nga fushat e Ollama-s."""
    ec, ed = final.get("eval_count", 0) or 0, final.get("eval_duration", 0) or 0
    pc, pd = final.get("prompt_eval_count", 0) or 0, final.get("prompt_eval_duration", 0) or 0
    gen = ec / (ed / 1e9) if ed and ec else float("nan")
    ptps = pc / (pd / 1e9) if pd and pc else float("nan")
    return gen, ptps


# ----------------------------------------------------------------------------
# Shkarkim i të dhënave
# ----------------------------------------------------------------------------

def _download(url: str, dest: Path):
    """Shkarkim atomik: .part + rename, që një shkarkim i ndërprerë të mos
    lërë skedar gjysmak që helmimon run-in tjetër."""
    print(f"[data] shkarkoj {url} → {dest} ...", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with urlrequest.urlopen(url, timeout=300) as r, part.open("wb") as f:
        total = int(r.headers.get("Content-Length", 0) or 0)
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total:
                print(f"\r[data] {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", file=sys.stderr)
    print("", file=sys.stderr)
    if total and done < total:
        part.unlink(missing_ok=True)
        raise IOError(f"shkarkim i paplotë ({done}/{total} byte) — provo sërish")
    part.rename(dest)


def ensure_mmlu(data_dir: Path, url: str) -> Path:
    """Siguron folderin data/mmlu/{dev,test}/*.csv. Kthen rrugën bazë."""
    base = data_dir / "mmlu"
    marker = base / ".complete"
    if marker.exists():
        return base
    tar_path = data_dir / "mmlu_data.tar"
    if not tar_path.exists():
        _download(url, tar_path)
    print("[data] ekstraktoj MMLU ...", file=sys.stderr)
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            # tar-i ka strukturën data/{dev,test,val,...}/<subject>_<split>.csv
            parts = Path(m.name).parts
            if len(parts) >= 3 and parts[1] in ("dev", "test") and m.isfile():
                target = base / parts[1] / parts[2]
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src:
                    target.write_bytes(src.read())
    marker.write_text("ok")
    return base


def load_mmlu_subject(base: Path, subject: str, split: str) -> list[dict]:
    """CSV pa header: question, A, B, C, D, answer."""
    path = base / split / f"{subject}_{split}.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) != 6:
                continue
            rows.append({"q": row[0], "choices": row[1:5], "ans": row[5].strip().upper()})
    return rows


def ensure_humaneval(data_dir: Path, url: str) -> list[dict]:
    path = data_dir / "HumanEval.jsonl.gz"
    if not path.exists():
        _download(url, path)
    problems = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


# ----------------------------------------------------------------------------
# MMLU
# ----------------------------------------------------------------------------

def mmlu_prompt(item: dict, subject: str, fewshot: list[dict]) -> str:
    subj = subject.replace("_", " ")
    parts = [f"The following is a multiple choice question about {subj}.",
             "Respond with ONLY the letter of the correct choice (A, B, C, or D).",
             "Do not explain. Do not repeat the choices. Output a single letter.", ""]
    for ex in fewshot:
        parts.append(ex["q"])
        for letter, choice in zip(LETTERS, ex["choices"]):
            parts.append(f"{letter}. {choice}")
        parts.append(f"Answer: {ex['ans']}")
        parts.append("")
    parts.append(item["q"])
    for letter, choice in zip(LETTERS, item["choices"]):
        parts.append(f"{letter}. {choice}")
    parts.append("Answer:")
    return "\n".join(parts)


# Order matters: explicit phrase > letter at the start > first letter anywhere
_ANS_PATTERNS = [
    re.compile(r"ANSWER\s+IS\s*:?\s*\(?([ABCD])\b"),
    re.compile(r"ANSWER\s*:?\s*\(?([ABCD])\b"),
    re.compile(r"^\s*\(?([ABCD])\)?\s*[.:,]?\s*$"),   # line containing only the letter
    re.compile(r"^\s*\(?([ABCD])\)?[.:,\s]"),          # starts with the letter
    re.compile(r"\b([ABCD])\b"),
]


def extract_letter(text: str) -> str:
    t = text.strip().upper()
    for pat in _ANS_PATTERNS:
        m = pat.search(t)
        if m:
            return m.group(1)
    return ""


@dataclass
class MmluRow:
    model: str
    subject: str
    idx: int
    gold: str
    pred: str
    correct: int
    raw: str
    gen_tps: float
    prompt_tps: float
    prompt_tokens: int
    total_ms: float
    error: str = ""


def run_mmlu(args, sampler: GpuSampler, outdir: Path, ts: str, tag: str):
    base = ensure_mmlu(Path(args.data_dir), args.mmlu_url)
    rows_csv = outdir / f"mmlu_rows{tag}_{ts}.csv"
    sum_csv = outdir / f"mmlu_summary{tag}_{ts}.csv"

    all_rows: list[MmluRow] = []
    # (model, subject) → [correct...]; (model) → metrika perf
    perf: dict[str, dict] = {}

    try:
        for model in args.models:
            print(f"\n### MMLU: {model} (num_ctx={args.num_ctx}, fewshot={args.fewshot})")
            # warm-up: loads the model so the performance metrics are measured warm
            try:
                ollama_generate(args.host, model, "Answer: A or B? Answer:", 4, args.num_ctx)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] warm-up dështoi {model}: {e}", file=sys.stderr)
            t_cfg0 = time.time()
            gen_l, ptps_l = [], []
            for subject in args.subjects:
                test = load_mmlu_subject(base, subject, "test")[: args.limit_per_subject]
                fewshot = (load_mmlu_subject(base, subject, "dev")[: args.fewshot]
                           if args.fewshot else [])
                n_ok = 0
                for i, item in enumerate(test):
                    r = MmluRow(model=model, subject=subject, idx=i, gold=item["ans"],
                                pred="", correct=0, raw="", gen_tps=float("nan"),
                                prompt_tps=float("nan"), prompt_tokens=0,
                                total_ms=float("nan"))
                    try:
                        text, final, t0, t1 = ollama_generate(
                            args.host, model, mmlu_prompt(item, subject, fewshot),
                            num_predict=48, num_ctx=args.num_ctx)
                        r.raw = text.strip()[:80]
                        r.pred = extract_letter(text)
                        r.correct = int(r.pred == item["ans"])
                        r.gen_tps, r.prompt_tps = _tps(final)
                        r.prompt_tokens = int(final.get("prompt_eval_count", 0) or 0)
                        r.total_ms = (t1 - t0) * 1000.0
                        n_ok += r.correct
                        gen_l.append(r.gen_tps)
                        ptps_l.append(r.prompt_tps)
                    except Exception as e:  # noqa: BLE001
                        r.error = f"{type(e).__name__}: {e}"
                    all_rows.append(r)
                acc = 100.0 * n_ok / len(test) if test else float("nan")
                print(f"  {subject:28s} n={len(test):3d}  acc={acc:5.1f}%")
            # performance/VRAM metrics for this configuration
            win = sampler.window(t_cfg0, time.time())
            perf[model] = {
                "vram_peak_mb": max((s.mem_used for s in win), default=float("nan")),
                "gpu_offload_pct": gpu_offload_pct(model),
                "prompt_tps_median": median_iqr(ptps_l)[0],
                "gen_tps_median": median_iqr(gen_l)[0],
                "wall_min": (time.time() - t_cfg0) / 60.0,
            }
    finally:
        _write_dataclass_csv(rows_csv, all_rows)
        _write_mmlu_summary(sum_csv, all_rows, perf, args)
        print(f"\n# Skedarë:\n  {rows_csv}\n  {sum_csv}")


def _write_mmlu_summary(path: Path, rows: list[MmluRow], perf: dict, args):
    groups: dict[tuple, list[MmluRow]] = {}
    for r in rows:
        if not r.error:
            groups.setdefault((r.model, r.subject), []).append(r)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subject", "n", "accuracy_pct", "fewshot", "num_ctx",
                    "vram_peak_mb", "gpu_offload_pct", "prompt_tps_median",
                    "gen_tps_median", "wall_min"])
        models = sorted({m for m, _ in groups})
        for model in models:
            subj_accs = []
            total_n = total_c = 0
            for (m, subject), rs in sorted(groups.items()):
                if m != model:
                    continue
                acc = 100.0 * sum(r.correct for r in rs) / len(rs)
                subj_accs.append(acc)
                total_n += len(rs)
                total_c += sum(r.correct for r in rs)
                w.writerow([model, subject, len(rs), f"{acc:.1f}", args.fewshot,
                            args.num_ctx, "", "", "", "", ""])
            p = perf.get(model, {})
            micro = 100.0 * total_c / total_n if total_n else float("nan")
            macro = statistics.mean(subj_accs) if subj_accs else float("nan")
            w.writerow([model, "OVERALL_micro", total_n, f"{micro:.1f}", args.fewshot,
                        args.num_ctx, f"{p.get('vram_peak_mb', float('nan')):.0f}",
                        f"{p.get('gpu_offload_pct', float('nan')):.0f}",
                        f"{p.get('prompt_tps_median', float('nan')):.1f}",
                        f"{p.get('gen_tps_median', float('nan')):.1f}",
                        f"{p.get('wall_min', float('nan')):.1f}"])
            w.writerow([model, "OVERALL_macro", total_n, f"{macro:.1f}", args.fewshot,
                        args.num_ctx, "", "", "", "", ""])


# ----------------------------------------------------------------------------
# HumanEval
# ----------------------------------------------------------------------------

HE_PROMPT = """You are an expert Python programmer. Complete the following function.
Output ONLY the complete function definition (repeat the signature shown), inside a single
```python code block. No explanations.

```python
{prompt}
```
"""

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_PRELUDE = ("from typing import *\nimport math\nimport re\nimport collections\n"
            "import itertools\nimport heapq\nimport string\n")


def extract_code(text: str, problem: dict) -> str:
    """Extract the code: the ```python``` block if present, otherwise the whole text.
    If it lacks def <entry_point>, it is treated as a prompt continuation (completion-style)."""
    m = _CODE_BLOCK_RE.search(text)
    code = m.group(1) if m else text
    if f"def {problem['entry_point']}" not in code:
        code = problem["prompt"] + code
    return code


def run_test(code: str, problem: dict, timeout: int = EXEC_TIMEOUT_S) -> tuple[bool, str]:
    """Execute the code + official tests in a subprocess with a timeout. Returns (pass, reason)."""
    program = (_PRELUDE + code + "\n\n" + problem["test"] +
               f"\n\ncheck({problem['entry_point']})\n")
    try:
        p = subprocess.run([sys.executable, "-c", program],
                           capture_output=True, text=True, timeout=timeout)
        if p.returncode == 0:
            return True, ""
        return False, (p.stderr or "").strip().splitlines()[-1][:120] if p.stderr else "nonzero"
    except subprocess.TimeoutExpired:
        return False, f"timeout>{timeout}s"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


@dataclass
class HeRow:
    model: str
    task_id: str
    passed: int
    fail_reason: str
    gen_tps: float
    eval_count: int
    total_ms: float
    error: str = ""


def run_humaneval(args, sampler: GpuSampler, outdir: Path, ts: str, tag: str):
    problems = ensure_humaneval(Path(args.data_dir), args.humaneval_url)
    if args.limit:
        problems = problems[: args.limit]
    rows_csv = outdir / f"humaneval_rows{tag}_{ts}.csv"
    sum_csv = outdir / f"humaneval_summary{tag}_{ts}.csv"

    all_rows: list[HeRow] = []
    perf: dict[str, dict] = {}
    try:
        for model in args.models:
            print(f"\n### HumanEval: {model} (n={len(problems)}, "
                  f"num_predict={args.num_predict}, num_ctx={args.num_ctx})")
            try:
                ollama_generate(args.host, model, "print('hi')", 4, args.num_ctx)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] warm-up dështoi {model}: {e}", file=sys.stderr)
            t_cfg0 = time.time()
            gen_l = []
            n_pass = 0
            for i, prob in enumerate(problems, 1):
                r = HeRow(model=model, task_id=prob["task_id"], passed=0, fail_reason="",
                          gen_tps=float("nan"), eval_count=0, total_ms=float("nan"))
                try:
                    text, final, t0, t1 = ollama_generate(
                        args.host, model, HE_PROMPT.format(prompt=prob["prompt"]),
                        num_predict=args.num_predict, num_ctx=args.num_ctx)
                    code = extract_code(text, prob)
                    ok, reason = run_test(code, prob)
                    r.passed, r.fail_reason = int(ok), reason
                    r.gen_tps, _ = _tps(final)
                    r.eval_count = int(final.get("eval_count", 0) or 0)
                    r.total_ms = (t1 - t0) * 1000.0
                    n_pass += r.passed
                    gen_l.append(r.gen_tps)
                except Exception as e:  # noqa: BLE001
                    r.error = f"{type(e).__name__}: {e}"
                all_rows.append(r)
                mark = "✓" if r.passed else "✗"
                print(f"  [{i:3d}/{len(problems)}] {prob['task_id']:14s} {mark} "
                      f"({r.gen_tps:5.1f} tps)  pass@1 deri tani: "
                      f"{100.0 * n_pass / i:.1f}%", flush=True)
            win = sampler.window(t_cfg0, time.time())
            perf[model] = {
                "vram_peak_mb": max((s.mem_used for s in win), default=float("nan")),
                "gpu_offload_pct": gpu_offload_pct(model),
                "gen_tps_median": median_iqr(gen_l)[0],
                "wall_min": (time.time() - t_cfg0) / 60.0,
            }
    finally:
        _write_dataclass_csv(rows_csv, all_rows)
        with sum_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "n", "pass_at_1_pct", "num_predict", "num_ctx",
                        "vram_peak_mb", "gpu_offload_pct", "gen_tps_median", "wall_min"])
            by_model: dict[str, list[HeRow]] = {}
            for r in all_rows:
                if not r.error:
                    by_model.setdefault(r.model, []).append(r)
            for model, rs in sorted(by_model.items()):
                p = perf.get(model, {})
                w.writerow([model, len(rs),
                            f"{100.0 * sum(r.passed for r in rs) / len(rs):.1f}",
                            args.num_predict, args.num_ctx,
                            f"{p.get('vram_peak_mb', float('nan')):.0f}",
                            f"{p.get('gpu_offload_pct', float('nan')):.0f}",
                            f"{p.get('gen_tps_median', float('nan')):.1f}",
                            f"{p.get('wall_min', float('nan')):.1f}"])
        print(f"\n# Skedarë:\n  {rows_csv}\n  {sum_csv}")


# ----------------------------------------------------------------------------
# Util & main
# ----------------------------------------------------------------------------

def _write_dataclass_csv(path: Path, rows):
    if not rows:
        return
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main():
    ap = argparse.ArgumentParser(description="Benchmark cilësie kuantizimi (Ollama).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--host", default="http://127.0.0.1:11434")
        p.add_argument("--models", nargs="+", required=True,
                       help="Tags të plota kuantizimi, p.sh. qwen2.5:1.5b-instruct-q4_K_M")
        p.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
        p.add_argument("--data-dir", default="data")
        p.add_argument("--outdir", default="results")
        p.add_argument("--label", default="", help="Etiketë (p.sh. vm105-16gb)")
        p.add_argument("--no-gpu", action="store_true")
        p.add_argument("--gpu-index", type=int, default=0)

    pm = sub.add_parser("mmlu", help="Nën-set MMLU (saktësi në %%)")
    common(pm)
    pm.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    pm.add_argument("--limit-per-subject", type=int, default=DEFAULT_LIMIT_PER_SUBJECT)
    pm.add_argument("--fewshot", type=int, default=0, choices=range(0, 6),
                    help="Shembuj few-shot nga dev split (default 0-shot)")
    pm.add_argument("--mmlu-url", default=MMLU_URL)

    ph = sub.add_parser("humaneval", help="HumanEval pass@1 (coder)")
    common(ph)
    ph.add_argument("--num-predict", type=int, default=DEFAULT_HE_NUM_PREDICT)
    ph.add_argument("--limit", type=int, default=0, help="Limit to the first N problems (smoke test)")
    ph.add_argument("--humaneval-url", default=HUMANEVAL_URL)

    args = ap.parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.label}" if args.label else ""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"# quality_bench {args.cmd} start {ts}  host={args.host}  label={args.label or '-'}")
    sampler = GpuSampler(gpu_index=args.gpu_index, sample_ms=500)  # 500ms mjafton (s'duam per-token)
    if not args.no_gpu:
        sampler.start()
        time.sleep(1.0)
    try:
        if args.cmd == "mmlu":
            run_mmlu(args, sampler, outdir, ts, tag)
        else:
            run_humaneval(args, sampler, outdir, ts, tag)
    finally:
        if not args.no_gpu:
            sampler.stop()


if __name__ == "__main__":
    main()
