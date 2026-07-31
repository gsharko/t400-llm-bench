import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans","figure.dpi":300})
fig,ax=plt.subplots(figsize=(9,5.6)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")

def box(x,y,w,h,txt,fc,ec="#333",fs=9,bold=False,style="round,pad=0.02",lw=1.2,tc="#111"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=style,fc=fc,ec=ec,lw=lw,mutation_scale=8))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,
            fontweight="bold" if bold else "normal",color=tc,zorder=5)
def arrow(x1,y1,x2,y2,txt="",ls="-",color="#444",fs=8,rad=0.0,off=(0,2)):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,
                 lw=1.3,color=color,linestyle=ls,connectionstyle=f"arc3,rad={rad}",zorder=4))
    if txt: ax.text((x1+x2)/2+off[0],(y1+y2)/2+off[1],txt,ha="center",va="center",
                    fontsize=fs,color=color,style="italic")

# Physical host container
box(3,18,60,74,"",fc="#f4f6f9",ec="#8aa0b8",lw=1.6,style="round,pad=0.02")
ax.text(6,88,"Physical host — Dell PowerEdge (R730xd / T420)  •  Proxmox VE hypervisor",
        ha="left",va="center",fontsize=9.5,fontweight="bold",color="#33475b")

# Benchmark VM
box(7,40,42,42,"",fc="#e8f0fb",ec="#3d6fb4",lw=1.5)
ax.text(9,78,"Benchmark VM — Ubuntu 24.04 (N vCPU, RAM)",ha="left",va="center",
        fontsize=8.8,fontweight="bold",color="#274a7a")
box(10,63,16,10,"Ollama\n(llama.cpp)",fc="#ffffff",ec="#3d6fb4",fs=8.5)
box(30,63,16,10,"Models\nQ4/Q8/FP16",fc="#ffffff",ec="#3d6fb4",fs=8)
box(10,46,16,11,"bench_llm.py\nAPI + TTFT +\nnvidia-smi",fc="#fff7e6",ec="#c88a2a",fs=8)
box(30,46,16,11,"nvidia-smi\nsampler ~100ms\nVRAM/offload",fc="#fff7e6",ec="#c88a2a",fs=7.8)
arrow(26,68,30,68,"")                       # ollama<->models
arrow(18,63,18,57,"HTTP",fs=7,off=(-4,0))   # bench->ollama (up)

# GPU passthrough
box(14,23,28,11,"NVIDIA T400 4 GB  (Turing, 31 W)\nVFIO / PCIe passthrough",
    fc="#eafaf0",ec="#2e9e5b",fs=8.3,bold=False)
arrow(28,46,28,34,"passthrough",fs=7,off=(9,0))

# BMC / iDRAC path
box(52,55,9.5,22,"BMC /\niDRAC\n(PMBus)",fc="#f0e8fb",ec="#7a4fb0",fs=8)

# power logger (host, outside VM but inside host)
box(52,30,9.5,16,"power_\nlogger.py\n(host)",fc="#fdeaea",ec="#c0504d",fs=7.8)
arrow(56.7,55,56.7,46,"IPMI DCMI\n~1 Hz",fs=7,off=(11,0))

# join + output (right, outside host)
box(70,52,26,16,"join_energy.py\nepoch-aligned merge\n(VM requests + host power)",
    fc="#ffffff",ec="#555",fs=8.2)
arrow(49,74,70,64,"request logs (epoch ts)",fs=7,rad=-0.28,off=(-3,4))
arrow(61.5,38,70,55,"power trace\n(epoch ts)",fs=7,rad=0.15,off=(6,0))

box(70,33,26,12,"System J/token,  throughput,\nVRAM, offload, quality",
    fc="#eafaf0",ec="#2e9e5b",fs=8.2,bold=True)
arrow(83,52,83,45,"")

# Cloud comparison (dashed, external)
box(70,14,26,12,"Cloud API\n(€/1M token reference)",fc="#f5f5f5",ec="#999",fs=8.2,style="round,pad=0.02")
arrow(83,33,83,26,"cost / energy\ncomparison",ls="--",color="#888",fs=7,off=(11,0))

ax.text(50,7,"Figure 1. Measurement architecture: virtualized 4 GB-GPU benchmark VM, in-guest performance sampling, "
        "and telemetry-free\nsystem-energy measurement via the host BMC (IPMI DCMI), aligned by epoch timestamp.",
        ha="center",va="center",fontsize=8.2,color="#333")
fig.tight_layout()
fig.savefig("figures/fig0_architecture.png",bbox_inches="tight")
print("saved figures/fig0_architecture.png")
