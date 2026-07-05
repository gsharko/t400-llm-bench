import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
# Fig 10 — RQ1 cilësi kodi: HumanEval pass@1 (164 probleme, greedy)
# Burimi: results/humaneval_summary_vm105-16gb_20260703T{134501,212528}Z.csv
# (emri, pass@1 %, gen tps, gpu %, madhësi GB)
D=[("coder 1.5B\nQ4_K_M",65.9,41.4,100,0.99),
   ("coder 1.5B\nQ8_0",  68.9,18.1, 54,1.6),
   ("coder 7.6B\nQ4_K_M",85.4, 3.5, 47,4.7)]
plt.rcParams.update({"font.size":11,"figure.dpi":300})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.8))
# (a) pass@1 bar
names=[d[0] for d in D]; p1=[d[1] for d in D]
cols=["#1f77b4","#1f77b4","#ff7f0e"]
b=ax1.bar(names,p1,color=cols,width=0.55)
for bar,(nm,p,tps,gpu,gb) in zip(b,D):
    ax1.text(bar.get_x()+bar.get_width()/2,p+1,f"{p:.1f}%",ha="center",fontsize=10,fontweight="bold")
    ax1.text(bar.get_x()+bar.get_width()/2,4,f"{tps:.1f} tps\n{gpu}% GPU",ha="center",fontsize=8,color="white")
ax1.set_ylabel("HumanEval pass@1 (%)")
ax1.set_ylim(0,100); ax1.grid(alpha=0.3,axis="y")
ax1.set_title("(a) pass@1: 7.6B@Q4 dominates, but at 3.5 tps")
# (b) pass@1 vs tps — Pareto i cilësi/shpejtësisë
for nm,p,tps,gpu,gb in D:
    c="#ff7f0e" if "7.6" in nm else "#1f77b4"
    ax2.scatter(tps,p,s=130,color=c,facecolor=c if gpu==100 else "white",edgecolor=c,linewidth=2,zorder=3)
    ax2.annotate(nm.replace("\n"," "),(tps,p),textcoords="offset points",xytext=(7,4),fontsize=9)
ax2.set_xlabel("Generation throughput (tokens/s, median)")
ax2.set_ylabel("HumanEval pass@1 (%)")
ax2.set_title("(b) Code quality vs speed (open circle = offloaded)")
ax2.set_xlim(0,46); ax2.set_ylim(60,90); ax2.grid(alpha=0.3)
fig.suptitle("RQ1 — quantization vs code quality (VM105/T400 4GB, num_ctx=2048, num_predict=512)",fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig10_quality_humaneval.png"); plt.close()
print("fig10 saved")
