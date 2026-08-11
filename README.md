# t400-llm-bench

Benchmark suite and measurement dataset for **local LLM inference on an
entry-level 4 GB GPU (NVIDIA T400)** in a virtualized laboratory server
(Proxmox VE, VFIO PCIe passthrough, Ollama/llama.cpp).

Companion artifact for the paper:

> *Parameters Beat Precision: A Measured Residency Frontier for Local Large
> Language Model Inference Under a 4 GB Memory Budget* — Measurement
> Laboratory, Faculty of Electrical Engineering, Polytechnic University of
> Tirana. (In preparation for submission to The Journal of Supercomputing.)

## Contents

| Path | Description |
|---|---|
| `scripts/bench_llm.py` | Performance benchmark: throughput, prompt rate, TTFT, latency, peak VRAM (100 ms `nvidia-smi` sampler), GPU/CPU layer split, concurrency (1/2/4/8 parallel). |
| `scripts/power_logger.py` | System power logger via BMC (IPMI DCMI, ~1 Hz), epoch-timestamped. |
| `scripts/join_energy.py` | Offline alignment of power traces with requests; trapezoidal integration, idle-baseline subtraction → J/token. |
| `scripts/quality_bench.py` | Quantization quality: MMLU-STEM subset (600 questions, 0-shot) and HumanEval (164 problems, pass@1, sandboxed subprocess execution). Downloads datasets automatically. |
| `scripts/PROTOCOL-Quality.md` | Measurement protocol for the quality campaign (model tags, run plan, methodological decisions). |
| `results/*.csv` | Raw measurement summaries: per-configuration performance (VM105 16 GB / 8 GB, VM100), concurrency, system energy, MMLU and HumanEval rows + summaries. |
| `figures/*.png` | All 13 figures as used in the paper (see the figure map below). |
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

## Figure map (paper number → file)

| Paper | File | Content |
|---|---|---|
| Fig. 1 | `fig0_architecture.png` | Measurement architecture (VM, sampling, BMC energy path) |
| Fig. 2 | `fig_frontier.png` | Usability frontier: GPU residency × throughput over model size × context |
| Fig. 3 | `fig1_throughput_vs_ctx.png` | Throughput vs. context window |
| Fig. 4 | `fig2_throughput_vs_size.png` | Throughput vs. model size (offloading cliff) |
| Fig. 5 | `fig3_offload_vs_ctx.png` | GPU residency vs. context window |
| Fig. 6 | `fig4_concurrency.png` | Concurrency: per-user vs. aggregate |
| Fig. 7 | `fig5_vm100_vs_vm105.png` | Cross-host CPU effect (8 vs. 4 vCPU) |
| Fig. 8 | `fig6_ram_isolated.png` | Isolated RAM effect (8 vs. 16 GB, same VM) |
| Fig. 9 | `fig7_energy.png` | System energy per token |
| Fig. 10 | `fig8_cost.png` | Cost and wall-time per 1M tokens (local vs. cloud) |
| Fig. 11 | `fig9_quality_mmlu.png` | MMLU-STEM accuracy vs. quantization |
| Fig. 12 | `fig10_quality_humaneval.png` | HumanEval pass@1 vs. quantization and size |
| Fig. 13 | `fig_qualfront.png` | Quality–footprint frontier vs. the ~2.6 GB usable-VRAM ceiling |

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
