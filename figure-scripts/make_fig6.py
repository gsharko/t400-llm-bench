import csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
def load(p):
    d={}
    for r in csv.DictReader(open(p)):
        d[(r["model"],int(r["num_ctx"]))]=float(r["gen_tps_median"])
    return d
g8=load("results/summary_vm105-8gb.csv")
g16=load("results/summary_vm105-16gb.csv")
ORDER=["qwen2.5:1.5b","phi3.5","qwen2.5-coder:7b","phi4"]
PAR={"qwen2.5:1.5b":"1.5B","phi3.5":"3.8B","qwen2.5-coder:7b":"7.6B","phi4":"14.7B"}
plt.rcParams.update({"font.size":11,"figure.dpi":300})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.8))
# (a) all models @512 8 vs 16
x=np.arange(len(ORDER)); w=0.38
y8=[g8[(m,512)] for m in ORDER]; y16=[g16[(m,512)] for m in ORDER]
b1=ax1.bar(x-w/2,y8,w,label="VM105 @ 8GB",color="#9467bd")
b2=ax1.bar(x+w/2,y16,w,label="VM105 @ 16GB",color="#1f77b4")
for b in list(b1)+list(b2): ax1.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,f"{b.get_height():.1f}",ha="center",fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels([f"{m.split(':')[0]}\n{PAR[m]}" for m in ORDER])
ax1.set_ylabel("Throughput (tokens/s)"); ax1.set_ylim(0,48)
ax1.set_title("(a) @num_ctx=512 — RAM has no effect (same VM)")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3,axis="y")
# (b) phi4 across ctx, 8 vs 16 (log y) — the cliff
CTX=[512,2048,8192]
ax2.plot(CTX,[g8[("phi4",c)] for c in CTX],marker="o",color="#9467bd",label="8GB")
ax2.plot(CTX,[g16[("phi4",c)] for c in CTX],marker="o",color="#1f77b4",label="16GB")
ax2.set_yscale("log"); ax2.set_xscale("log",base=2); ax2.set_xticks(CTX); ax2.set_xticklabels(CTX)
ax2.set_xlabel("num_ctx (tokens, log2)"); ax2.set_ylabel("Throughput tokens/s (log)")
ax2.annotate("footprint 11GB > 8GB RAM\n→ disk-thrash: 0.02 tps\n(TTFT 34s)",xy=(8192,0.02),xytext=(1100,0.06),
             fontsize=8,color="#9467bd",arrowprops=dict(arrowstyle="->",color="#9467bd"))
ax2.set_title("(b) phi4 (14.7B) — the RAM cliff @8192")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3,which="both")
fig.suptitle("RQ4 isolated: same VM105, only RAM changes (8 vs 16 GB)",fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig6_ram_isolated.png"); plt.close()
print("fig6 saved")
