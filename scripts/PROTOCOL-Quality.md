# Protocol — Quantization quality benchmark (RQ1, quality dimension)

**Purpose:** measure how much task accuracy is sacrificed by the quantization needed to
fit into 4 GB of VRAM, and answer the practical question facing a laboratory with a T400:
**a larger model at Q4, or a smaller model at Q8/FP16?** Script: `quality_bench.py`
(keep it in the same directory as `bench_llm.py`, which it imports `GpuSampler` from).

**Where it runs:** inside the GPU-equipped benchmark VM — local Ollama endpoint plus a
local `nvidia-smi`. Keep the host quiet during measurement (see the methodology section
of the paper).

---

## 1. Configuration matrix

### MMLU (general-purpose models) — 6 STEM subjects × 100 questions = 600 questions, 0-shot

| Tag | Size on disk | Prediction @num_ctx=2048 | Rationale |
|---|---|---|---|
| `qwen2.5:1.5b-instruct-q4_K_M` | ~1.0 GB | 100% GPU | baseline (same quant as the performance runs) |
| `qwen2.5:1.5b-instruct-q8_0` | ~1.9 GB | 100% GPU | higher precision that still fits |
| `qwen2.5:1.5b-instruct-fp16` | ~3.1 GB | boundary / light offloading | quality ceiling of the 1.5B model |
| `phi3.5:3.8b-mini-instruct-q4_K_M` | ~2.4 GB | ~55% GPU | largest model at Q4 — the key comparison |
| `phi3.5:3.8b-mini-instruct-q8_0` | ~4.1 GB | heavy offloading | higher precision that does NOT fit |

Central comparison: **1.5B@FP16 (~3.1 GB) vs 3.8B@Q4 (~2.4 GB)** — two different ways of
spending the same VRAM budget.

### HumanEval (code models) — 164 problems, pass@1, greedy

| Tag | Size on disk | Prediction | Rationale |
|---|---|---|---|
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | ~1.0 GB | 100% GPU | code model that fits entirely |
| `qwen2.5-coder:1.5b-instruct-q8_0` | ~1.9 GB | 100% GPU | effect of quantization on code |
| `qwen2.5-coder:7b` (= 7b-instruct-q4_K_M) | ~4.7 GB | ~47% GPU | same model as the performance runs |
| `qwen2.5-coder:7b-instruct-q8_0` *(optional)* | ~8.1 GB | heavy offloading | only if spare overnight time is available |

> ⚠️ Sizes are approximate — verify with `ollama list` after pulling.
> If a tag no longer exists in the registry, check `ollama.com/library/<model>/tags`.

## 2. Preparation (once)

```bash
cd /path/to/benchmark        # directory holding bench_llm.py + quality_bench.py

# Pull tags (~12 GB total excluding 7b-q8)
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull qwen2.5:1.5b-instruct-q8_0
ollama pull qwen2.5:1.5b-instruct-fp16
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
ollama pull phi3.5:3.8b-mini-instruct-q8_0
ollama pull qwen2.5-coder:1.5b-instruct-q4_K_M
ollama pull qwen2.5-coder:1.5b-instruct-q8_0
# optional (8.1 GB): ollama pull qwen2.5-coder:7b-instruct-q8_0
```

The datasets (MMLU ~160 MB tar + HumanEval ~50 KB) are downloaded automatically into
`data/` on the first run.

**Smoke test (~2 min)** before the full run:

```bash
python3 quality_bench.py mmlu --models qwen2.5:1.5b-instruct-q4_K_M \
    --subjects formal_logic --limit-per-subject 5 --label smoke
python3 quality_bench.py humaneval --models qwen2.5-coder:1.5b-instruct-q4_K_M \
    --limit 3 --label smoke
```

Check that: `pred` is non-empty in `mmlu_rows` (letter extraction works), at least one
problem passes in HumanEval, and the vram/offload columns are populated in the summary.

## 3. Full runs

```bash
# A. MMLU (~4-6 h total; phi3.5-q8 is the slow part) — run under nohup/tmux
nohup python3 quality_bench.py mmlu \
    --models qwen2.5:1.5b-instruct-q4_K_M qwen2.5:1.5b-instruct-q8_0 \
             qwen2.5:1.5b-instruct-fp16 \
             phi3.5:3.8b-mini-instruct-q4_K_M phi3.5:3.8b-mini-instruct-q8_0 \
    --label 16gb > mmlu_run.log 2>&1 &

# B. HumanEval, small code models (~1-1.5 h)
nohup python3 quality_bench.py humaneval \
    --models qwen2.5-coder:1.5b-instruct-q4_K_M qwen2.5-coder:1.5b-instruct-q8_0 \
    --label 16gb > he_small_run.log 2>&1 &

# C. HumanEval 7b @Q4 (~4-6 h, overnight)
nohup python3 quality_bench.py humaneval --models qwen2.5-coder:7b \
    --label 16gb > he_7b_run.log 2>&1 &

# D. (optional) 7b @Q8 (~7-10 h, overnight) — only for a fourth point on the curve
```

Run **A, B and C in series**, not in parallel — concurrency corrupts the performance and
VRAM measurements. The durations are estimates derived from the measured throughputs;
phi3.5-q8 and fp16 have no prior measurements and may deviate.

⚠️ **Safety:** `humaneval` executes model-generated code (subprocess, 15 s timeout, no
additional sandbox). Run it only inside a disposable VM — never directly on the
hypervisor host.

## 4. Output for analysis

From `results/`:

- `mmlu_summary_<label>_<ts>.csv` — per-subject accuracy plus `OVERALL_micro/macro`,
  with VRAM/offload/tps per tag
- `mmlu_rows_*.csv` — per-question records (audit trail for answer extraction)
- `humaneval_summary_*.csv` — pass@1 plus VRAM/offload/tps per tag
- `humaneval_rows_*.csv` — per-problem records (failure reasons)

These feed the quantization-quality section of the paper and the corresponding
quality-versus-footprint figures.

## 5. Methodological decisions (as reported in the paper's Methodology)

- **0-shot, not 5-shot:** the models are instruction-tuned; 0-shot cuts runtime ~3× on
  offloaded models and the comparison across quantizations remains internally
  consistent (identical protocol for every tag). `--fewshot 5` is available as an
  option if a reviewer requests it.
- **STEM subset:** 6 subjects close to the target audience (an engineering laboratory)
  rather than all 57 — reported as "MMLU-STEM subset", with the subject list stated in
  the paper.
- **Greedy (temperature=0, seed=42):** deterministic, pass@1 without sampling — the same
  principle as the performance measurements.
- **Fixed num_ctx=2048:** an identical KV budget for every tag, so that differences in
  VRAM come only from the weight precision.
- **Stated limitations:** the MMLU subset is not full MMLU; HumanEval uses a
  standard-imports prelude (typing/math/re/…) and code-block regex extraction.
