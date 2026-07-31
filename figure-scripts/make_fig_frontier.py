import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.size":10,"figure.dpi":300})

models=["Qwen2.5 1.5B","Phi-3.5 3.8B","Qwen2.5-Coder 7.6B","Phi-4 14.7B"]  # bottom->top
ctx=["512","2048","8192"]
GPU=np.array([[100,100,100],[100,55,41],[49,47,46],[28,28,25]],float)      # rows=models
TPS=np.array([[41.7,40.8,40.7],[20.2,12.7,3.8],[3.30,3.21,3.21],[2.00,2.00,1.93]])

fig,ax=plt.subplots(figsize=(7.2,5.0))
im=ax.imshow(GPU,cmap="RdYlGn",vmin=0,vmax=100,aspect="auto",origin="lower")
ax.set_xticks(range(3)); ax.set_xticklabels(ctx)
ax.set_yticks(range(4)); ax.set_yticklabels(models)
ax.set_xlabel("Context window  num_ctx (tokens)")
ax.set_ylabel("Model (parameters)")
ax.set_title("Usability frontier on a 4 GB GPU:\nGPU residency and throughput vs. size × context")
for i in range(4):
    for j in range(3):
        ax.text(j,i,f"{GPU[i,j]:.0f}% GPU\n{TPS[i,j]:.1f} tok/s",ha="center",va="center",
                fontsize=8.5,color=("#111" if GPU[i,j]>45 else "white"),fontweight="bold")
# outline resident cells (100% GPU) as the frontier
for i in range(4):
    for j in range(3):
        if GPU[i,j]>=100:
            ax.add_patch(Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor="#0a3",lw=3,zorder=5))
ax.text(0.02,0.98,"green outline = 100% GPU-resident\n(interactive regime)",transform=ax.transAxes,
        fontsize=8,va="top",ha="left",bbox=dict(fc="white",ec="#0a3",alpha=0.9,boxstyle="round,pad=0.3"))
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03); cb.set_label("GPU residency (%)  —  100 = no offload")
fig.tight_layout(); fig.savefig("figures/fig_frontier.png",bbox_inches="tight")
print("saved figures/fig_frontier.png")
