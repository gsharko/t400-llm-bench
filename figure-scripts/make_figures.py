import csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

rows=[]
with open("results/summary_vm105-16gb.csv") as f:
    for r in csv.DictReader(f):
        rows.append(r)

# params (B) and quant for labels
PARAMS={"qwen2.5:1.5b":1.5,"phi3.5":3.8,"qwen2.5-coder:7b":7.6,"phi4":14.7}
QUANT={"qwen2.5:1.5b":"Q4_K_M","phi3.5":"Q4_0","qwen2.5-coder:7b":"Q4_K_M","phi4":"Q4_K_M"}
ORDER=["qwen2.5:1.5b","phi3.5","qwen2.5-coder:7b","phi4"]
COLORS={"qwen2.5:1.5b":"#2ca02c","phi3.5":"#1f77b4","qwen2.5-coder:7b":"#ff7f0e","phi4":"#d62728"}
CTX=[512,2048,8192]

data=defaultdict(dict)
for r in rows:
    data[r["model"]][int(r["num_ctx"])]={
        "tps":float(r["gen_tps_median"]),
        "q1":float(r["gen_tps_q1"]),"q3":float(r["gen_tps_q3"]),
        "vram":float(r["vram_peak_mb_median"]),
        "off":float(r["gpu_offload_pct_median"]),
        "ttft":float(r["ttft_ms_median"]),
    }

plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":300})

# ---- Fig 1: throughput vs num_ctx ----
fig,ax=plt.subplots(figsize=(7,4.5))
for m in ORDER:
    xs=CTX; ys=[data[m][c]["tps"] for c in CTX]
    lo=[data[m][c]["tps"]-data[m][c]["q1"] for c in CTX]
    hi=[data[m][c]["q3"]-data[m][c]["tps"] for c in CTX]
    ax.errorbar(xs,ys,yerr=[lo,hi],marker="o",capsize=3,color=COLORS[m],
                label=f"{m} ({PARAMS[m]}B)")
ax.set_xscale("log",base=2); ax.set_xticks(CTX); ax.set_xticklabels(CTX)
ax.set_xlabel("Context window num_ctx (tokens, log2)")
ax.set_ylabel("Generation throughput (tokens/s)")
ax.axhline(10,ls="--",color="gray",lw=1)
ax.text(512,11,"~10 tps: practical usability threshold",fontsize=8,color="gray")
ax.set_title("Throughput vs context length — NVIDIA T400 4GB")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig1_throughput_vs_ctx.png"); plt.close()

# ---- Fig 2: throughput vs model size @512, colored by GPU% ----
fig,ax=plt.subplots(figsize=(7,4.5))
xs=[PARAMS[m] for m in ORDER]
ys=[data[m][512]["tps"] for m in ORDER]
offs=[data[m][512]["off"] for m in ORDER]
bars=ax.bar([f"{m.split(':')[0]}\n{PARAMS[m]}B {QUANT[m]}" for m in ORDER],ys,
            color=[COLORS[m] for m in ORDER])
for b,m in zip(bars,ORDER):
    d=data[m][512]
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.6,
            f"{d['tps']:.1f} tps\n{d['off']:.0f}% GPU",ha="center",fontsize=8)
ax.set_ylabel("Generation throughput (tokens/s)")
ax.set_title("Throughput vs model size @ num_ctx=512 — T400 4GB")
ax.set_ylim(0,48)
fig.tight_layout(); fig.savefig("figures/fig2_throughput_vs_size.png"); plt.close()

# ---- Fig 3: GPU offload % vs num_ctx ----
fig,ax=plt.subplots(figsize=(7,4.5))
for m in ORDER:
    ys=[data[m][c]["off"] for c in CTX]
    ax.plot(CTX,ys,marker="s",color=COLORS[m],label=f"{m} ({PARAMS[m]}B)")
ax.set_xscale("log",base=2); ax.set_xticks(CTX); ax.set_xticklabels(CTX)
ax.set_xlabel("Context window num_ctx (tokens, log2)")
ax.set_ylabel("GPU residency (%)  — 100% = no offloading")
ax.set_ylim(0,105)
ax.axhline(100,ls=":",color="green",lw=1)
ax.set_title("GPU→RAM offloading grows with context — T400 4GB")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig3_offload_vs_ctx.png"); plt.close()

print("Figurat u ruajtën:")
import os
for f in sorted(os.listdir("figures")):
    print("  figures/"+f, os.path.getsize("figures/"+f),"B")
