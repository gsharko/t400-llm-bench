import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
# sys J/token @512, throughput @512 (vm105-16gb)
J={"qwen2.5:1.5b":0.676,"phi3.5":1.441,"qwen2.5-coder:7b":7.483,"phi4":17.541}
TPS={"qwen2.5:1.5b":41.7,"phi3.5":20.2,"qwen2.5-coder:7b":3.30,"phi4":2.00}
# cloud reference €/1M output token (serverless, ~2026, të përafërta, USD→EUR 0.92)
CLOUD={"qwen2.5:1.5b":0.09,"phi3.5":0.14,"qwen2.5-coder:7b":0.18,"phi4":0.28}
ELEC=0.10  # €/kWh (Shqipëri, amvisëri ~ALL 10.2)
ORDER=["qwen2.5:1.5b","phi3.5","qwen2.5-coder:7b","phi4"]
PAR={"qwen2.5:1.5b":"1.5B","phi3.5":"3.8B","qwen2.5-coder:7b":"7.6B","phi4":"14.7B"}
def eur_per_M(m): return J[m]*1e6/3.6e6*ELEC   # kWh/1M × €/kWh
def hours_per_M(m): return 1e6/TPS[m]/3600.0
plt.rcParams.update({"font.size":11,"figure.dpi":300})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.8))
x=np.arange(len(ORDER)); w=0.38
loc=[eur_per_M(m) for m in ORDER]; cld=[CLOUD[m] for m in ORDER]
b1=ax1.bar(x-w/2,loc,w,label="Local (marginal energy, T400)",color="#2ca02c")
b2=ax1.bar(x+w/2,cld,w,label="Cloud API (reference, ~2026)",color="#7f7f7f")
ax1.set_yscale("log")
for bar in list(b1)+list(b2): ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.08,f"€{bar.get_height():.2f}",ha="center",fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels([f"{m.split(':')[0]}\n{PAR[m]}" for m in ORDER])
ax1.set_ylabel("€ / 1M output tokens (log)")
ax1.set_title(f"(a) Cost: local (energy) vs cloud  —  electricity €{ELEC:.2f}/kWh")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3,axis="y",which="both")
# (b) hours to 1M tokens
h=[hours_per_M(m) for m in ORDER]
b=ax2.bar([f"{m.split(':')[0]}\n{PAR[m]}" for m in ORDER],h,color=["#2ca02c","#1f77b4","#ff7f0e","#d62728"])
ax2.set_yscale("log")
for bar,m in zip(b,ORDER):
    hh=hours_per_M(m); lab=f"{hh:.1f}h" if hh<48 else f"{hh/24:.1f}d"
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.08,lab,ha="center",fontsize=9)
ax2.set_ylabel("Hours per 1M tokens (single stream, log)")
ax2.set_title("(b) Wall time per 1M tokens @42→2 tps")
ax2.grid(alpha=0.3,axis="y",which="both")
fig.suptitle("RQ3 — local vs cloud cost/energy (VM105/T400 4GB)",fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig8_cost.png"); plt.close()
print("fig8 saved")
print(f"{'model':18s}{'J/tok':>8s}{'kWh/1M':>9s}{'€/1M loc':>10s}{'€/1M cloud':>12s}{'h/1M':>8s}")
for m in ORDER:
    print(f"{m:18s}{J[m]:8.2f}{J[m]*1e6/3.6e6:9.3f}{eur_per_M(m):10.3f}{CLOUD[m]:12.2f}{hours_per_M(m):8.1f}")
