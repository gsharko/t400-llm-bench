# Protokoll — Benchmark cilësie kuantizimi (RQ1, dimensioni i cilësisë)

**Qëllimi:** të matet sa saktësi sakrifikohet nga kuantizimi për të hyrë në 4GB VRAM,
dhe të përgjigjemi pyetjes praktike të një lab-i me T400: **model më i madh @Q4 apo
model më i vogël @Q8/FP16?** Script: `quality_bench.py` (rri në të njëjtin folder me
`bench_llm.py`, se importon `GpuSampler`).

**Ku xhirohet:** VM105 (`lab@192.168.20.70`), si benchmark-et e tjera — localhost Ollama +
nvidia-smi lokal. Host i qetë gjatë matjes (higjiena e §Metodologji).

---

## 1. Matrica e konfigurimeve

### MMLU (modele të përgjithshme) — 6 lëndë STEM × 100 pyetje = 600 pyetje, 0-shot

| Tag | Madhësi disk | Parashikim @num_ctx=2048 | Pse |
|---|---|---|---|
| `qwen2.5:1.5b-instruct-q4_K_M` | ~1.0GB | 100% GPU | baseline (i njëjti quant si matjet perf) |
| `qwen2.5:1.5b-instruct-q8_0` | ~1.9GB | 100% GPU | quant i lartë që ende hyn |
| `qwen2.5:1.5b-instruct-fp16` | ~3.1GB | kufi/ofloadim i lehtë | tavani i cilësisë së 1.5B |
| `phi3.5:3.8b-mini-instruct-q4_K_M` | ~2.4GB | ~55% GPU | modeli më i madh @Q4 — krahasimi kyç |
| `phi3.5:3.8b-mini-instruct-q8_0` | ~4.1GB | ofloadim i fortë | quant i lartë që NUK hyn |

Krahasimi qendror: **1.5B@fp16 (~3.1GB) vs 3.8B@Q4 (~2.4GB)** — dy rrugë të ndryshme
për të shpenzuar të njëjtin buxhet VRAM.

### HumanEval (coder) — 164 probleme, pass@1, greedy

| Tag | Madhësi disk | Parashikim | Pse |
|---|---|---|---|
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | ~1.0GB | 100% GPU | coder-i që hyn plotësisht |
| `qwen2.5-coder:1.5b-instruct-q8_0` | ~1.9GB | 100% GPU | efekti i quant-it te kodi |
| `qwen2.5-coder:7b` (= 7b-instruct-q4_K_M) | ~4.7GB | ~47% GPU | i njëjti model si matjet perf |
| `qwen2.5-coder:7b-instruct-q8_0` *(ops.)* | ~8.1GB | ofloadim i fortë | vetëm nëse ka natë të lirë |

> ⚠️ Madhësitë janë të përafërta — verifiko me `ollama list` pas pull.
> Nëse ndonjë tag s'ekziston më në regjistër, shih `ollama.com/library/<model>/tags`.

## 2. Përgatitja (një herë)

```bash
ssh lab@192.168.20.70
cd ~/benchmark-t400        # ku janë bench_llm.py + quality_bench.py

# Pull tags (~12GB total pa 7b-q8; disku i VM105 = 82GB, ok)
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull qwen2.5:1.5b-instruct-q8_0
ollama pull qwen2.5:1.5b-instruct-fp16
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
ollama pull phi3.5:3.8b-mini-instruct-q8_0
ollama pull qwen2.5-coder:1.5b-instruct-q4_K_M
ollama pull qwen2.5-coder:1.5b-instruct-q8_0
# opsionale (8.1GB): ollama pull qwen2.5-coder:7b-instruct-q8_0
```

Të dhënat (MMLU ~160MB tar + HumanEval ~50KB) shkarkohen vetë te `data/` në run-in e parë.

**Smoke test (~2 min)** para run-it të plotë:

```bash
python3 quality_bench.py mmlu --models qwen2.5:1.5b-instruct-q4_K_M \
    --subjects formal_logic --limit-per-subject 5 --label smoke
python3 quality_bench.py humaneval --models qwen2.5-coder:1.5b-instruct-q4_K_M \
    --limit 3 --label smoke
```

Kontrollo: `pred` jo bosh te mmlu_rows (nxjerrja e shkronjës punon), të paktën 1 pass
te humaneval, kolonat vram/offload jo bosh te summary.

## 3. Run-et e plota

```bash
# A. MMLU (~4–6 orë total; phi3.5-q8 është pjesa e ngadaltë) — xhiro me nohup/tmux
nohup python3 quality_bench.py mmlu \
    --models qwen2.5:1.5b-instruct-q4_K_M qwen2.5:1.5b-instruct-q8_0 \
             qwen2.5:1.5b-instruct-fp16 \
             phi3.5:3.8b-mini-instruct-q4_K_M phi3.5:3.8b-mini-instruct-q8_0 \
    --label vm105-16gb > mmlu_run.log 2>&1 &

# B. HumanEval coder të vegjël (~1–1.5 orë)
nohup python3 quality_bench.py humaneval \
    --models qwen2.5-coder:1.5b-instruct-q4_K_M qwen2.5-coder:1.5b-instruct-q8_0 \
    --label vm105-16gb > he_small_run.log 2>&1 &

# C. HumanEval 7b @Q4 (~4–6 orë, natën)
nohup python3 quality_bench.py humaneval --models qwen2.5-coder:7b \
    --label vm105-16gb > he_7b_run.log 2>&1 &

# D. (ops.) 7b @Q8 (~7–10 orë, natën) — vetëm nëse duam pikën e katërt të kurbës
```

Xhiroji **A, B, C në seri** (jo paralel — konkurrenca prish matjet perf/VRAM).
Kohët janë estimime nga throughput-et e matura; phi3.5-q8 dhe fp16 s'kanë matje
paraprake, mund të devijojnë.

⚠️ **Siguria:** `humaneval` ekzekuton kod të gjeneruar nga modeli (subprocess,
timeout 15s, pa sandbox shtesë). VM105 është ambient i pranueshëm; mos e xhiro
kurrë në pve2 direkt.

## 4. Çfarë kthehet për analizë

Nga `results/`:

- `mmlu_summary_vm105-16gb_<ts>.csv` — saktësi për lëndë + `OVERALL_micro/macro`
  + VRAM/offload/tps për tag
- `mmlu_rows_*.csv` — për-pyetje (audit i nxjerrjes së përgjigjes)
- `humaneval_summary_*.csv` — pass@1 + VRAM/offload/tps për tag
- `humaneval_rows_*.csv` — për-problem (fail reasons)

Pastaj (sesioni tjetër me Claude): §4.9 te `Results-draft.md` + **Fig. 9**
(saktësi vs VRAM buxhet — kurba cilësi/madhësi/quant) + **Fig. 10** (pass@1 vs quant),
dhe integrimi te RQ1.

## 5. Vendime metodologjike (për Methodology të artikullit)

- **0-shot, jo 5-shot:** modelet janë instruct; 0-shot ul kohën ~3× në modelet e
  ofloaduara dhe krahasimi mes quant-eve mbetet i brendshëm (i njëjti protokoll për
  të gjithë). `--fewshot 5` ekziston si opsion nëse reviewer-i e kërkon.
- **Nën-set STEM:** 6 lëndë afër audiencës (lab inxhinierik), jo 57 lëndët e plota —
  raportohet si "MMLU-STEM subset", me listën e lëndëve në appendix.
- **Greedy (temperature=0, seed=42):** deterministik, pass@1 pa sampling — i njëjti
  parim si matjet perf.
- **num_ctx=2048 fiks:** i njëjti buxhet KV për çdo tag, që dallimi në VRAM të vijë
  vetëm nga pesha e quant-it.
- **Kufizim për §Limitations:** MMLU subset ≠ MMLU i plotë; HumanEval me prelude
  importesh standarde (typing/math/re/…); ekstraktim me code-block regex.
