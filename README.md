# t400-llm-bench

Benchmark suite and measurement dataset for **local LLM inference on an
entry-level 4 GB GPU (NVIDIA T400)** in a virtualized laboratory server
(Proxmox VE, VFIO PCIe passthrough, Ollama/llama.cpp).

Companion artifact for the paper:

> *Benchmarking Local LLM Inference on Entry-Level 4 GB GPUs: Throughput,
> Energy, and Quantization Trade-offs on the NVIDIA T400 for
> Resource-Constrained Laboratories* — Measurement Laboratory, Faculty of
> Electrical Engineering, Polytechnic University of Tirana. (Under review.)

## Contents

| Path | Description |
|---|---|
| `scripts/bench_llm.py` | Performance benchmark: throughput, prompt rate, TTFT, latency, peak VRAM (100 ms `nvidia-smi` sampler), GPU/CPU layer split, concurrency (1/2/4/8 parallel). |
| `scripts/power_logger.py` | System power logger via BMC (IPMI DCMI, ~1 Hz), epoch-timestamped. |
| `scripts/join_energy.py` | Offline alignment of power traces with requests; trapezoidal integration, idle-baseline subtraction → J/token. |
| `scripts/quality_bench.py` | Quantization quality: MMLU-STEM subset (600 questions, 0-shot) and HumanEval (164 problems, pass@1, sandboxed subprocess execution). Downloads datasets automatically. |
| `scripts/PROTOCOL-Quality.md` | Measurement protocol for the quality campaign (model tags, run plan, methodological decisions). |
| `results/*.csv` | Raw measurement summaries: per-configuration performance (VM105 16 GB / 8 GB, VM100), concurrency, system energy, MMLU and HumanEval rows + summaries. |
| `figures/*.png` | Figures 1–10 as used in the paper. |
| `figure-scripts/*.py` | Matplotlib scripts that regenerate every figure from `results/`. |

## Requirements

- Python ≥ 3.10, **standard library only** for all benchmark scripts
  (matplotlib + numpy needed only for figure regeneration).
- An [Ollama](https://ollama.com) host (any GPU); `nvidia-smi` locally for
  VRAM sampling; `ipmitool` on the hypervisor for system power (optional).

## Quick start

```bash
# Performance benchmark (run inside the GPU VM)
python3 scripts/bench_llm.py --models qwen2.5:1.5b phi3.5 --ctx 512 2048 8192 --label myhost

# Quality benchmark (quantization variants)
python3 scripts/quality_bench.py mmlu --models qwen2.5:1.5b-instruct-q4_K_M --label myhost
python3 scripts/quality_bench.py humaneval --models qwen2.5-coder:1.5b-instruct-q4_K_M --label myhost

# Regenerate all figures
cd figure-scripts && for f in make_*.py; do python3 "$f"; done
```

⚠️ `quality_bench.py humaneval` **executes model-generated code** (subprocess,
15 s timeout). Run it only inside a disposable VM.

## Measurement conditions

NVIDIA T400 4 GB (Turing, 31 W), VFIO passthrough into Ubuntu 24.04 VMs on
Proxmox VE (Dell T420 / R730xd); Ollama native; greedy decoding
(temperature 0, seed 42); 10 repetitions per configuration, medians + IQR;
quiet host (no co-scheduled workloads, backups disabled). System energy via
iDRAC DCMI, idle baseline 120 W subtracted, trapezoidal integration. Full
methodology in the paper.

## Citation

See `CITATION.cff`. Please cite the paper (and this archive's DOI) if you use
the scripts or the dataset.

## License

MIT — see `LICENSE`.
