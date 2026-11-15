import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pd.read_csv("data/students.csv")

mpl.rcParams.update({
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 0.9,
    "axes.axisbelow": True, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9,
})

BLUE, ORANGE, TEAL, RED = "#2563eb", "#f97316", "#0d9488", "#dc2626"

# Figure 1: Distributions
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
axes[0].hist(df.screen_on_time_h, bins=20, color=BLUE, edgecolor="white", linewidth=0.6)
axes[0].set_title("Screen-On Time (Hours)", loc="left", pad=8)
axes[0].set_xlabel("Hours")
axes[0].set_ylabel("Devices")

axes[1].hist(df.battery_drain_rate_pct_h, bins=20, color=TEAL, edgecolor="white", linewidth=0.6)
axes[1].set_title("Battery Drain Rate (%/h)", loc="left", pad=8)
axes[1].set_xlabel("Drain Rate (%/h)")

axes[2].hist(df.max_temp_c, bins=20, color=ORANGE, edgecolor="white", linewidth=0.6)
axes[2].set_title("Peak Temperature (°C)", loc="left", pad=8)
axes[2].set_xlabel("Temperature (°C)")

fig.savefig("img/f1_profile.png")
plt.close(fig)

# Figure 2: Correlation Matrix
cols = ["screen_on_time_h", "gaming_hours_h", "brightness_pct", "bg_apps_count", "max_temp_c", "battery_drain_rate_pct_h", "performance_index"]
corr = df[cols].corr()

fig, ax = plt.subplots(figsize=(6.2, 4.3))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(["SOT", "Gaming", "Brightness", "BG Apps", "Temp", "Drain Rate", "Perf Index"], rotation=42, ha="right")
ax.set_yticks(range(len(cols)))
ax.set_yticklabels(["SOT", "Gaming", "Brightness", "BG Apps", "Temp", "Drain Rate", "Perf Index"])
ax.grid(False)

for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(corr.iloc[i, j]) > 0.55 else "#334155", fontsize=8)

ax.set_title("Correlation Matrix", loc="left", pad=10)
fig.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
fig.savefig("img/f2_corr.png")
plt.close(fig)

# Figure 3: Scatter Plot
fig, ax = plt.subplots(figsize=(8.0, 4.4))
sc = ax.scatter(df.gaming_hours_h, df.battery_drain_rate_pct_h, c=df.max_temp_c, cmap="YlOrRd", s=30, alpha=0.8, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Gaming Hours (h)")
ax.set_ylabel("Battery Drain Rate (%/h)")
ax.set_title("Gaming Hours vs Battery Drain Rate", loc="left", pad=10)
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Max Temp (°C)")
fig.savefig("img/f3_quadrant.png")
plt.close(fig)

# Figure 4: Temp Bands
df["temp_band"] = pd.cut(df.max_temp_c, bins=[0, 36, 40, 44, 60], labels=["Normal (<36°C)", "Warm (36-40°C)", "Hot (40-44°C)", "Severe (>44°C)"])
temp_grp = df.groupby("temp_band", observed=True).agg(perf=("performance_index", "mean"), throttled=("throttled", "mean"))

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
axes[0].bar(temp_grp.index, temp_grp.perf, color=[BLUE, TEAL, ORANGE, RED], width=0.55)
axes[0].set_title("Mean Performance Index", loc="left", pad=8)
axes[0].set_ylabel("Performance Index")
axes[0].tick_params(axis='x', rotation=15)

axes[1].bar(temp_grp.index, temp_grp.throttled * 100, color=[BLUE, TEAL, ORANGE, RED], width=0.55)
axes[1].set_title("Throttling Rate (%)", loc="left", pad=8)
axes[1].set_ylabel("Throttled (%)")
axes[1].tick_params(axis='x', rotation=15)

fig.savefig("img/f4_stress.png")
plt.close(fig)

# Figure 5: Dual Axis Plot
bright_bins = pd.cut(df.brightness_pct, bins=[0, 40, 60, 80, 100], labels=["Low (<40%)", "Medium (40-60%)", "High (60-80%)", "Max (>80%)"])
b_grp = df.groupby(bright_bins, observed=True).agg(drain=("battery_drain_rate_pct_h", "mean"), temp=("max_temp_c", "mean"))

fig, ax1 = plt.subplots(figsize=(7.5, 3.5))
ax1.bar(b_grp.index, b_grp.drain, width=0.4, color=TEAL, label="Drain Rate")
ax1.set_ylabel("Drain Rate (%/h)", color=TEAL)
ax1.tick_params(axis='y', labelcolor=TEAL)
ax1.set_xlabel("Screen Brightness Level")
ax1.set_title("Brightness vs Drain and Temp", loc="left", pad=10)

ax2 = ax1.twinx()
ax2.plot(b_grp.index, b_grp.temp, "o-", color=RED, linewidth=2, label="Temp")
ax2.set_ylabel("Temp (°C)", color=RED)
ax2.tick_params(axis='y', labelcolor=RED)
ax2.grid(False)

fig.savefig("img/f5_levers.png")
plt.close(fig)

print("All 5 figures generated successfully in img/ folder!")