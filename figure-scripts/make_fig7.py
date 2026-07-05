import csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
E={}
for r in csv.DictReader(open("results/energy_vm105.csv")):
    if int(r["num_ctx"])==512:
        E[r["model"]]=(float(r["sys_j_per_token_median"]),float(r["sys_avg_w_median"]))
TPS={"qwen2.5:1.5b":41.7,"phi3.5":20.2,"qwen2.5-coder:7b":3.30,"phi4":2.00}
ORDER=["qwen2.5:1.5b","phi3.5","qwen2.5-coder:7b","phi4"]
PAR={"qwen2.5:1.5b":"1.5B","phi3.5":"3.8B","qwen2.5-coder:7b":"7.6B","phi4":"14.7B"}
COL={"qwen2.5:1.5b":"#2ca02c","phi3.5":"#1f77b4","qwen2.5-coder:7b":"#ff7f0e","phi4":"#d62728"}
plt.rcParams.update({"font.size":11,"figure.dpi":300})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.8))
# (a) sys J/token bar (log)
y=[E[m][0] for m in ORDER]
b=ax1.bar([f"{m.split(':')[0]}\n{PAR[m]}" for m in ORDER],y,color=[COL[m] for m in ORDER])
ax1.set_yscale("log")
for bar,m in zip(b,ORDER):
    ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.08,f"{E[m][0]:.2f}",ha="center",fontsize=9)
ax1.set_ylabel("System energy per token (J/token, log)")
ax1.set_title("(a) System J/token @512 (120 W idle subtracted)")
ax1.grid(alpha=0.3,axis="y",which="both")
# (b) J/token vs throughput (inverse)
for m in ORDER:
    ax2.scatter(TPS[m],E[m][0],s=90,color=COL[m],label=f"{m.split(':')[0]} ({PAR[m]})",zorder=3)
xs=np.linspace(1.5,45,100); ax2.plot(xs,33.0/xs,ls="--",color="gray",lw=1,label="≈ 33W / throughput")
ax2.set_xlabel("Throughput (tokens/s)"); ax2.set_ylabel("Sys J/token")
ax2.set_title("(b) J/token ∝ 1/throughput (~constant 33 W power)")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig.suptitle("System energy (RQ3) — VM105/T400, iDRAC DCMI measurement",fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig7_energy.png"); plt.close()
print("fig7 saved")
