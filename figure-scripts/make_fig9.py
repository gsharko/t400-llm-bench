import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Fig 9 — RQ1 cilësi: MMLU-STEM (600 pyetje, 0-shot) vs footprint & shpejtësi
# Burimi: results/mmlu_summary_vm105-16gb_20260703T113632Z.csv (OVERALL_micro)
# (emri, madhësi peshash GB, acc %, gen tps, gpu %, familja)
D=[("1.5B Q4_K_M",0.99,43.5,72.7,100,"qwen"),
   ("1.5B Q8_0",  1.6, 46.5,39.0, 54,"qwen"),
   ("1.5B FP16",  3.1, 47.0,25.4, 51,"qwen"),
   ("3.8B Q4_K_M",2.4, 52.7,30.7, 52,"phi"),
   ("3.8B Q8_0",  4.1, 51.8,13.0, 38,"phi")]
COL={"qwen":"#1f77b4","phi":"#d62728"}
plt.rcParams.update({"font.size":11,"figure.dpi":300})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.8))
# (a) acc vs footprint peshash
for name,gb,acc,tps,gpu,fam in D:
    filled = gpu==100
    ax1.scatter(gb,acc,s=110,color=COL[fam],
                edgecolor=COL[fam],facecolor=COL[fam] if filled else "white",
                linewidth=2,zorder=3)
    ax1.annotate(name,(gb,acc),textcoords="offset points",xytext=(8,-4),fontsize=9)
ax1.axhline(25,color="gray",ls=":",lw=1); ax1.text(4.05,25.5,"chance (25%)",fontsize=8,color="gray",ha="right")
ax1.set_xlabel("Weight size on disk (GB)")
ax1.set_ylabel("MMLU-STEM accuracy (%)  —  600 questions, 0-shot")
ax1.set_title("(a) Quality vs footprint: parameters beat precision")
ax1.set_xlim(0.5,4.6); ax1.set_ylim(20,60); ax1.grid(alpha=0.3)
# legjendë e improvizuar
ax1.scatter([],[],s=110,color="#1f77b4",label="qwen2.5 1.5B")
ax1.scatter([],[],s=110,color="#d62728",label="phi3.5 3.8B")
ax1.scatter([],[],s=110,facecolor="white",edgecolor="gray",linewidth=2,label="offloaded (<100% GPU)")
ax1.legend(fontsize=8,loc="lower right")
# (b) acc vs throughput (tradeoff cilësi–shpejtësi)
for name,gb,acc,tps,gpu,fam in D:
    ax2.scatter(tps,acc,s=110,color=COL[fam],
                facecolor=COL[fam] if gpu==100 else "white",edgecolor=COL[fam],linewidth=2,zorder=3)
    ax2.annotate(name,(tps,acc),textcoords="offset points",xytext=(6,5),fontsize=9)
ax2.set_xlabel("Generation throughput (tokens/s, median — auxiliary)")
ax2.set_ylabel("MMLU-STEM accuracy (%)")
ax2.set_title("(b) Quality–speed trade-off at 4 GB")
ax2.set_xlim(0,80); ax2.set_ylim(40,56); ax2.grid(alpha=0.3)
fig.suptitle("RQ1 — quantization vs quality (VM105/T400 4GB, num_ctx=2048)",fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig9_quality_mmlu.png"); plt.close()
print("fig9 saved")
