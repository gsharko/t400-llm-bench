import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Fig — Quality vs footprint frontier (§4.8): consolidates MMLU-STEM + HumanEval
# onto one plane and overlays the ~2.6 GB usable-VRAM ceiling from the residency
# frontier (Fig. 2). Sweet spot = high score, left of the wall (100% GPU-resident).
# x = weight footprint on disk (GB); y = task score (%); fill = 100% GPU-resident.
USABLE=2.6
# (label, GB, score%, gpu%, family)
MMLU=[("Qwen 1.5B Q4",0.99,43.5,100),("Qwen 1.5B Q8",1.6,46.5,54),
      ("Qwen 1.5B FP16",3.1,47.0,51),("Phi 3.8B Q4",2.4,52.7,52),("Phi 3.8B Q8",4.1,51.8,38)]
HEVAL=[("Coder 1.5B Q4",0.99,65.9,100),("Coder 1.5B Q8",1.6,68.9,54),("Coder 7.6B Q4",4.7,85.4,47)]
plt.rcParams.update({"font.size":11,"figure.dpi":300})
fig,ax=plt.subplots(figsize=(8.4,5.6))
# usable-VRAM band
ax.axvspan(USABLE,5.0,color="#d62728",alpha=0.06,zorder=0)
ax.axvline(USABLE,color="#d62728",ls="--",lw=1.4,zorder=1)
ax.text(USABLE+0.05,90,"~2.6 GB usable-VRAM ceiling\n(right of line ⇒ offloads)",
        fontsize=8.5,color="#b01818",va="top")
ax.text(0.6,90,"GPU-resident\n(interactive)",fontsize=8.5,color="#2a7a2a",va="top",fontweight="bold")
def plot(series,color,marker,lbl):
    for nm,gb,sc,gpu in series:
        f= gpu==100
        ax.scatter(gb,sc,s=140,marker=marker,color=color,
                   facecolor=color if f else "white",edgecolor=color,linewidth=2,zorder=3)
        ax.annotate(nm,(gb,sc),textcoords="offset points",xytext=(8,-3),fontsize=8.3)
    ax.scatter([],[],s=140,marker=marker,color=color,label=lbl)
plot(MMLU,"#1f77b4","o","MMLU-STEM accuracy (reasoning)")
plot(HEVAL,"#ff7f0e","^","HumanEval pass@1 (code)")
ax.scatter([],[],s=140,marker="s",facecolor="gray",edgecolor="gray",label="filled = 100% GPU-resident")
ax.scatter([],[],s=140,marker="s",facecolor="white",edgecolor="gray",linewidth=2,label="open = offloaded (<100% GPU)")
ax.set_xlabel("Weight footprint on disk (GB)")
ax.set_ylabel("Task score (%)")
ax.set_title("Quality–footprint frontier on a 4 GB GPU: parameters beat precision,\nbut only Q4 stays inside the interactive residency budget",fontsize=10.5)
ax.set_xlim(0.5,5.0); ax.set_ylim(35,95); ax.grid(alpha=0.3)
ax.legend(fontsize=8.2,loc="lower right",framealpha=0.95)
fig.tight_layout(); fig.savefig("figures/fig_qualfront.png"); plt.close()
print("fig_qualfront saved")
