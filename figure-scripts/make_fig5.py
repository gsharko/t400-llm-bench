import csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
def load(p):
    d={}
    for r in csv.DictReader(open(p)):
        d[(r["model"],int(r["num_ctx"]))]={"tps":float(r["gen_tps_median"]),"off":float(r["gpu_offload_pct_median"])}
    return d
v105=load("results/summary_vm105-16gb.csv")
v100=load("results/summary_vm100-10gb.csv")
ORDER=["qwen2.5:1.5b","phi3.5","qwen2.5-coder:7b","phi4"]
PAR={"qwen2.5:1.5b":"1.5B","phi3.5":"3.8B","qwen2.5-coder:7b":"7.6B","phi4":"14.7B"}
plt.rcParams.update({"font.size":11,"figure.dpi":300})

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.8))

# Panel A: gen_tps @512, VM100 vs VM105
labels=[f"{m.split(':')[0]}\n{PAR[m]}" for m in ORDER]
x=np.arange(len(ORDER)); w=0.38
y100=[v100[(m,512)]["tps"] for m in ORDER]
y105=[v105[(m,512)]["tps"] for m in ORDER]
b1=ax1.bar(x-w/2,y100,w,label="VM100 (10GB RAM, 8 cores, R730xd)",color="#8c564b")
b2=ax1.bar(x+w/2,y105,w,label="VM105 (16GB RAM, 4 cores, T420)",color="#1f77b4")
for b in list(b1)+list(b2):
    ax1.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,f"{b.get_height():.1f}",ha="center",fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_ylabel("Generation throughput (tokens/s)"); ax1.set_ylim(0,50)
ax1.set_title("(a) Throughput @ num_ctx=512"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3,axis="y")

# Panel B: phi3.5 across contexts, both VMs (the divergent case)
CTX=[512,2048,8192]
ax2.plot(CTX,[v100[("phi3.5",c)]["tps"] for c in CTX],marker="o",color="#8c564b",label="VM100 (8 cores)")
ax2.plot(CTX,[v105[("phi3.5",c)]["tps"] for c in CTX],marker="o",color="#1f77b4",label="VM105 (4 cores)")
for c in CTX:
    ax2.annotate(f"{v100[('phi3.5',c)]['off']:.0f}%",(c,v100[('phi3.5',c)]["tps"]),textcoords="offset points",xytext=(0,8),fontsize=8,color="#8c564b",ha="center")
    ax2.annotate(f"{v105[('phi3.5',c)]['off']:.0f}%",(c,v105[('phi3.5',c)]["tps"]),textcoords="offset points",xytext=(0,-14),fontsize=8,color="#1f77b4",ha="center")
ax2.set_xscale("log",base=2); ax2.set_xticks(CTX); ax2.set_xticklabels(CTX)
ax2.set_xlabel("num_ctx (tokens, log2)"); ax2.set_ylabel("Throughput (tokens/s)")
ax2.set_title("(b) phi3.5 — % = GPU residency"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
fig.suptitle("VM100 vs VM105 — confounded comparison (RAM + cores + host + driver all differ)",fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig5_vm100_vs_vm105.png"); plt.close()
print("fig5 saved")
# print comparison table
print(f"{'model':18s}{'ctx':>6s}{'VM100':>8s}{'VM105':>8s}{'Δ%':>8s}{'off100':>8s}{'off105':>8s}")
for m in ORDER:
    for c in CTX:
        a=v100[(m,c)]["tps"]; b=v105[(m,c)]["tps"]
        print(f"{m:18s}{c:6d}{a:8.1f}{b:8.1f}{100*(a-b)/b:+7.0f}%{v100[(m,c)]['off']:7.0f}%{v105[(m,c)]['off']:7.0f}%")
