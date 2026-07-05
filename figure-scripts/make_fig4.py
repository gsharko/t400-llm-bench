import csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
n=[];agg=[];pu=[];wall=[]
for r in csv.DictReader(open("results/concurrency_vm105-16gb.csv")):
    n.append(int(r["n_parallel"]));agg.append(float(r["aggregate_tps"]))
    pu.append(float(r["per_user_tps"]));wall.append(float(r["wall_s"]))
plt.rcParams.update({"font.size":11,"figure.dpi":300})
fig,ax1=plt.subplots(figsize=(7,4.5))
ax1.plot(n,pu,marker="o",color="#2ca02c",label="per-user tps")
ax1.plot(n,agg,marker="s",color="#1f77b4",label="aggregate tps")
ax1.set_xlabel("Parallel requests (N)"); ax1.set_ylabel("Throughput (tokens/s)")
ax1.set_xticks(n); ax1.set_ylim(0,48); ax1.grid(alpha=0.3)
ax1.annotate("n=1: cold start\n(model load)",xy=(1,8.22),xytext=(1.6,15),
             fontsize=8,color="gray",arrowprops=dict(arrowstyle="->",color="gray"))
ax2=ax1.twinx()
ax2.plot(n,wall,marker="^",ls="--",color="#d62728",label="wall-time (s)")
ax2.set_ylabel("Total wall time (s)",color="#d62728")
ax2.tick_params(axis="y",labelcolor="#d62728")
l1,la=ax1.get_legend_handles_labels(); l2,lb=ax2.get_legend_handles_labels()
ax1.legend(l1+l2,la+lb,fontsize=9,loc="center left")
ax1.set_title("Concurrency (qwen2.5:1.5b, num_ctx=512) — T400 4GB")
fig.tight_layout(); fig.savefig("figures/fig4_concurrency.png"); plt.close()
print("fig4 saved")
