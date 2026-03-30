#!/usr/bin/env python3
"""Generate figures for AutoResearch-NG README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ── Data from experiment_report.md ──

# All KEEP experiments (chronological)
keep_ids =   [0,   3,    9,    11,   12,   13,   16,   25,   26,   36,   37,   44,   50,   65,   66,   67,   83,   98]
keep_bpb =   [0.9949,0.9942,0.9894,0.9859,0.9857,0.9855,0.9833,0.9828,0.9806,0.9806,0.9801,0.9798,0.9793,0.9782,0.9771,0.9771,0.9769,0.9768]

# All experiments val_bpb (KEEP and REJECT)
all_ids = list(range(0, 103))
all_bpb = [
    0.9949, 1.0132, 1.0034, 0.9942, 0.9944, 0.9967, 1.0035, 0.9974, 1.0233, 0.9894,
    0.9978, 0.9859, 0.9857, 0.9855, 1.0026, 0.9861, 0.9833, 0.9874, 0.9851, 0.9853,
    0.9885, 0.9880, 0.9856, 0.9852, 0.9878, 0.9828, 0.9806, 0.9820, 0.9836, 0.9819,
    0.9813, 0.9876, 0.9825, 0.9815, 0.9822, 1.4331, 0.9806, 0.9801, 0.9808, 0.9811,
    0.9819, 0.9807, 0.9809, 0.9817, 0.9798, 0.9807, 0.9925, 0.9804, 0.9810, 0.9798,
    0.9793, 0.9808, 0.9808, 0.9805, 0.9807, 0.9800, 0.9805, 0.9797, 0.9813, 0.9813,
    1.0191, 0.9830, 0.9849, 0.9863, 0.9814, 0.9782, 0.9771, 0.9771, 0.9786, 0.9782,
    0.9784, 0.9783, 0.9920, 0.9827, 0.9929, 0.9818, 1.3153, 0.9802, None,  0.9787,
    0.9889, 0.9790, 0.9819, 0.9769, 0.9774, 0.9774, 0.9919, 0.9808, 0.9782, None,
    0.9777, 0.9776, 0.9815, 0.9869, 0.9772, 0.9810, 0.9778, 0.9864, 0.9768, 0.9777,
    0.9778, 0.9796, 0.9779,
]
all_results = [
    "BASE","REJ","REJ","KEEP","REJ","REJ","REJ","REJ","REJ","KEEP",
    "REJ","KEEP","KEEP","KEEP","REJ","REJ","KEEP","REJ","REJ","REJ",
    "REJ","REJ","REJ","REJ","REJ","KEEP","KEEP","REJ","REJ","REJ",
    "REJ","REJ","REJ","REJ","REJ","REJ","KEEP","KEEP","REJ","REJ",
    "REJ","REJ","REJ","REJ","KEEP","REJ","REJ","REJ","REJ","REJ",
    "KEEP","REJ","REJ","REJ","REJ","REJ","REJ","REJ","REJ","REJ",
    "REJ","REJ","REJ","REJ","REJ","KEEP","KEEP","KEEP","REJ","REJ",
    "REJ","REJ","REJ","REJ","REJ","REJ","REJ","REJ","CRASH","REJ",
    "REJ","REJ","REJ","KEEP","REJ","REJ","REJ","REJ","REJ","CRASH",
    "REJ","REJ","REJ","REJ","REJ","REJ","REJ","REJ","KEEP","REJ",
    "REJ","REJ","REJ",
]

# ── Figure 1: val_bpb Trajectory ──
fig, ax = plt.subplots(figsize=(12, 5))

# Plot all experiments as scatter (clip outliers for readability)
valid_ids = [i for i, b in zip(all_ids, all_bpb) if b is not None and b < 1.05]
valid_bpb = [b for b in all_bpb if b is not None and b < 1.05]
valid_res = [r for r, b in zip(all_results, all_bpb) if b is not None and b < 1.05]

colors = ["#2ecc71" if r == "KEEP" else "#e74c3c" if r == "REJ" else "#3498db" for r in valid_res]
ax.scatter(valid_ids, valid_bpb, c=colors, s=18, alpha=0.6, zorder=2)

# Plot best-so-far line
ax.step(keep_ids, keep_bpb, where="post", color="#2c3e50", linewidth=2.2, label="Best val_bpb", zorder=3)
ax.scatter(keep_ids, keep_bpb, c="#2ecc71", s=50, edgecolors="#2c3e50", linewidths=1.2, zorder=4)

# Phase boundaries
ax.axvline(x=20, color="#9b59b6", linestyle="--", alpha=0.5, linewidth=1)
ax.axvline(x=60, color="#9b59b6", linestyle="--", alpha=0.5, linewidth=1)
ax.text(10, 1.048, "EXPLORE", ha="center", fontsize=10, color="#9b59b6", fontweight="bold")
ax.text(40, 1.048, "NARROW", ha="center", fontsize=10, color="#9b59b6", fontweight="bold")
ax.text(80, 1.048, "REFINE", ha="center", fontsize=10, color="#9b59b6", fontweight="bold")

# Baseline
ax.axhline(y=0.9949, color="#e67e22", linestyle=":", alpha=0.6, linewidth=1.2)
ax.text(102, 0.9953, "baseline", fontsize=9, color="#e67e22", ha="right")

ax.set_xlabel("Experiment #")
ax.set_ylabel("val_bpb (lower is better)")
ax.set_title("AutoResearch-NG: val_bpb Optimization Trajectory (~102 experiments, single H100)")
ax.set_ylim(0.974, 1.05)
ax.set_xlim(-2, 105)

keep_patch = mpatches.Patch(color="#2ecc71", label="KEEP")
rej_patch = mpatches.Patch(color="#e74c3c", label="REJECT")
ax.legend(handles=[keep_patch, rej_patch], loc="upper right")

fig.tight_layout()
fig.savefig("figures/fig1_trajectory.png", dpi=150)
plt.close()

# ── Figure 2: Improvement Breakdown by Category ──
fig, ax = plt.subplots(figsize=(8, 5))

categories = ["Batch size\n& windows", "Regularization\n(WD, softcap)", "Architecture\n(QK-norm, MLP, VE)", "Optimizer\n(LR, momentum)"]
contributions = [41.4, 27.6, 24.3, 6.6]
bar_colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

bars = ax.barh(categories, contributions, color=bar_colors, height=0.55, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, contributions):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2, f"{val}%",
            va="center", fontweight="bold", fontsize=11)

ax.set_xlabel("Contribution to total improvement (%)")
ax.set_title("Improvement Sources by Category")
ax.set_xlim(0, 52)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig("figures/fig2_categories.png", dpi=150)
plt.close()

# ── Figure 3: Phase Comparison ──
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

# Phase data
phases = ["EXPLORE\n(#1-19)", "NARROW\n(#20-67)", "REFINE\n(#68-102)"]
keep_rates = [31.6, 16.7, 8.6]
bpb_gains = [0.0116, 0.0062, 0.0003]
experiments = [19, 48, 35]

# Keep rate
ax = axes[0]
bars = ax.bar(phases, keep_rates, color=["#3498db", "#2ecc71", "#e74c3c"], width=0.5, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, keep_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val}%",
            ha="center", fontweight="bold", fontsize=11)
ax.set_ylabel("Keep Rate (%)")
ax.set_title("Keep Rate by Phase")
ax.set_ylim(0, 42)

# val_bpb gain
ax = axes[1]
bars = ax.bar(phases, bpb_gains, color=["#3498db", "#2ecc71", "#e74c3c"], width=0.5, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, bpb_gains):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003, f"{val:.4f}",
            ha="center", fontweight="bold", fontsize=10)
ax.set_ylabel("val_bpb improvement")
ax.set_title("Absolute Gain by Phase")
ax.set_ylim(0, 0.015)

# Contribution %
ax = axes[2]
contribution_pct = [64.1, 34.3, 1.7]
bars = ax.bar(phases, contribution_pct, color=["#3498db", "#2ecc71", "#e74c3c"], width=0.5, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, contribution_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val}%",
            ha="center", fontweight="bold", fontsize=11)
ax.set_ylabel("Share of total gain (%)")
ax.set_title("Contribution by Phase")
ax.set_ylim(0, 80)

fig.suptitle("Three-Phase Search Strategy Performance", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("figures/fig3_phases.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 4: Original vs NG Comparison (Architecture Diagram) ──
fig, ax = plt.subplots(figsize=(10, 6))

features = [
    "Acceptance\nStrategy",
    "Evaluation\nMetrics",
    "Experiment\nMemory",
    "Stagnation\nDetection",
    "Safety\nGates",
    "Comparison\nBaseline",
]

# Qualitative scores (0-5 scale)
original_scores = [2, 1, 0, 0, 1, 1]  # greedy, single metric, none, none, basic, previous only
ng_scores = [4, 5, 5, 4, 4, 5]        # annealing, multi-obj, JSONL+summaries, meta-opt, staged, 3-tier

x = np.arange(len(features))
width = 0.32

bars1 = ax.bar(x - width/2, original_scores, width, label="Original autoresearch",
               color="#bdc3c7", edgecolor="white", linewidth=1.5)
bars2 = ax.bar(x + width/2, ng_scores, width, label="AutoResearch-NG",
               color="#3498db", edgecolor="white", linewidth=1.5)

ax.set_ylabel("Capability Score (0-5)")
ax.set_title("Feature Comparison: Original vs AutoResearch-NG")
ax.set_xticks(x)
ax.set_xticklabels(features, fontsize=10)
ax.legend(loc="upper left")
ax.set_ylim(0, 6)

fig.tight_layout()
fig.savefig("figures/fig4_comparison.png", dpi=150)
plt.close()

# ── Figure 5: Top 5 Improvements Waterfall ──
fig, ax = plt.subplots(figsize=(10, 5))

labels = ["Baseline", "Batch 2^18\n(#9)", "WD=0.05\n(#11-12)", "Window //8\n(#25-26)",
          "QK-norm\nscale (#16)", "MLP 8x\n(#65-67)", "Other\n(7 exps)", "Final"]
start_val = 0.9949
improvements = [0, -0.0047, -0.0038, -0.0027, -0.0022, -0.0022, -0.0025, 0]
cumulative = [start_val]
for imp in improvements[1:-1]:
    cumulative.append(cumulative[-1] + imp)
cumulative.append(cumulative[-1])  # final = same as last

# Waterfall bars
bar_colors = []
bottoms = []
heights = []
for i in range(len(labels)):
    if i == 0:
        bottoms.append(0)
        heights.append(cumulative[0])
        bar_colors.append("#95a5a6")
    elif i == len(labels) - 1:
        bottoms.append(0)
        heights.append(cumulative[-1])
        bar_colors.append("#2ecc71")
    else:
        bottoms.append(cumulative[i])
        heights.append(-improvements[i])
        bar_colors.append("#3498db")

bars = ax.bar(labels, heights, bottom=bottoms, color=bar_colors, width=0.55,
              edgecolor="white", linewidth=1.5)

# Connectors
for i in range(len(labels) - 2):
    if i == 0:
        y = cumulative[0]
    else:
        y = cumulative[i]
    ax.plot([i + 0.3, i + 0.7], [cumulative[i+1] + abs(improvements[i+1]), cumulative[i+1] + abs(improvements[i+1])],
            color="#7f8c8d", linewidth=0.8, linestyle="--")

# Value labels
for i, bar in enumerate(bars):
    if i == 0 or i == len(labels) - 1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                f"{cumulative[i]:.4f}", ha="center", fontweight="bold", fontsize=9)
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bottoms[i] + heights[i] + 0.0003,
                f"{improvements[i]:+.4f}", ha="center", fontsize=9, color="#2c3e50")

ax.set_ylabel("val_bpb")
ax.set_title("Waterfall: Top Improvements from Baseline to Final")
ax.set_ylim(0.970, 1.000)

fig.tight_layout()
fig.savefig("figures/fig5_waterfall.png", dpi=150)
plt.close()

print("All figures generated in figures/")
