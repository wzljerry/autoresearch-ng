#!/usr/bin/env python3
"""AutoResearch-NG real-time monitoring — run anytime to check experiment status."""

import json
import sys
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_experience():
    path = Path("experience.jsonl")
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().strip().split("\n"):
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_baseline():
    path = Path("baseline_metrics.json")
    if path.exists():
        data = json.loads(path.read_text())
        return data.get("metrics", data)
    return None


def print_status(exp):
    if not exp:
        print("  No experiment data yet. The loop may not have started.")
        return

    total = len(exp)
    keeps = [e for e in exp if e.get("result") == "KEEP"]
    weak_keeps = [e for e in exp if e.get("result") == "WEAK_KEEP"]
    anneal = [e for e in exp if e.get("result") == "ANNEAL_ACCEPT"]
    rejects = [e for e in exp if e.get("result") == "REJECT"]
    crashes = [e for e in exp if e.get("result") == "CRASH"]

    accepted = len(keeps) + len(weak_keeps) + len(anneal)
    rate = accepted / total * 100 if total > 0 else 0

    # Current phase
    if total <= 20:
        phase = "EXPLORE (1-20)"
    elif total <= 60:
        phase = "NARROW (21-60)"
    else:
        phase = "REFINE (61+)"

    # Best val_bpb
    best_bpb = float("inf")
    for e in exp:
        bpb = e.get("metrics", {}).get("val_bpb", float("inf"))
        if e.get("result") in ("KEEP", "WEAK_KEEP", "ANNEAL_ACCEPT") and bpb < best_bpb:
            best_bpb = bpb

    # Baseline
    baseline = load_baseline()
    baseline_bpb = baseline.get("val_bpb", "?") if baseline else "?"

    # Latest experiment
    last = exp[-1]
    last_bpb = last.get("metrics", {}).get("val_bpb", "?")
    last_result = last.get("result", "?")
    last_desc = last.get("description", last.get("hypothesis", "?"))
    last_time = last.get("timestamp", "?")

    # Consecutive rejections
    consecutive_rejects = 0
    for e in reversed(exp):
        if e.get("result") == "REJECT":
            consecutive_rejects += 1
        else:
            break

    sep = "=" * 60
    best_bpb_str = f"{best_bpb:.6f}" if best_bpb != float("inf") else "N/A"
    if isinstance(baseline_bpb, float) and best_bpb != float("inf"):
        improve_str = f"{((baseline_bpb - best_bpb) / baseline_bpb * 100):.2f}% ({baseline_bpb:.6f} -> {best_bpb:.6f})"
    else:
        improve_str = "N/A"
    stall_warn = " WARNING: may be stagnating" if consecutive_rejects >= 5 else ""
    desc_str = last_desc[:70] if isinstance(last_desc, str) else str(last_desc)

    print(f"""
{sep}
  AutoResearch-NG Experiment Monitor
{sep}

  Total experiments: {total}
  Current phase:     {phase}
  Accept rate:       {accepted}/{total} ({rate:.1f}%)
    - KEEP:          {len(keeps)}
    - WEAK_KEEP:     {len(weak_keeps)}
    - ANNEAL_ACCEPT: {len(anneal)}
    - REJECT:        {len(rejects)}
    - CRASH:         {len(crashes)}

  Baseline val_bpb:  {baseline_bpb}
  Best val_bpb:      {best_bpb_str}
  Improvement:       {improve_str}

  Consecutive rejects: {consecutive_rejects}{stall_warn}

  Latest experiment (#{last.get("id", total)}):
    Result:  {last_result}
    val_bpb: {last_bpb}
    Description: {desc_str}
    Time:    {last_time}
{sep}""")

    # Statistics by category
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"keep": 0, "total": 0})
    for e in exp:
        cat = e.get("change_category", "unknown")
        cat_stats[cat]["total"] += 1
        if e.get("result") in ("KEEP", "WEAK_KEEP"):
            cat_stats[cat]["keep"] += 1

    if cat_stats:
        print("\n  Success rate by direction:")
        sorted_cats = sorted(cat_stats.items(),
                            key=lambda x: x[1]["keep"] / max(x[1]["total"], 1),
                            reverse=True)
        for cat, s in sorted_cats[:10]:
            r = s["keep"] / s["total"] * 100 if s["total"] > 0 else 0
            bar = "█" * int(r / 5) + "░" * (20 - int(r / 5))
            print(f"    {cat:25s} {s['keep']:2d}/{s['total']:2d} {bar} {r:.0f}%")

    # Last 5 experiments
    print(f"\n  Last 5 experiments:")
    for e in exp[-5:]:
        eid = e.get("id", "?")
        res = e.get("result", "?")
        bpb = e.get("metrics", {}).get("val_bpb", "?")
        desc = e.get("description", e.get("hypothesis", "?"))
        if isinstance(desc, str) and len(desc) > 50:
            desc = desc[:50] + "..."
        marker = {"KEEP": "+", "WEAK_KEEP": "~", "ANNEAL_ACCEPT": "~",
                  "REJECT": "x", "CRASH": "!"}.get(res, "?")
        bpb_str = f"{bpb:.6f}" if isinstance(bpb, float) else str(bpb)
        print(f"    {marker} #{eid:3} | {res:14s} | bpb={bpb_str} | {desc}")


def plot_progress(exp, output="progress.png"):
    if not HAS_MPL:
        print("\n  matplotlib not available, skipping plot.")
        return
    if len(exp) < 2:
        print("\n  Not enough data, need at least 2 experiments to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # --- Top: val_bpb curve ---
    ax1 = axes[0]
    ids = []
    bpbs = []
    colors = []
    best_ids = []
    best_bpbs = []
    running_best = float("inf")

    color_map = {
        "KEEP": "#2ecc71",
        "WEAK_KEEP": "#f1c40f",
        "ANNEAL_ACCEPT": "#e67e22",
        "REJECT": "#e74c3c",
        "CRASH": "#95a5a6",
    }

    for e in exp:
        eid = e.get("id", len(ids) + 1)
        bpb = e.get("metrics", {}).get("val_bpb")
        result = e.get("result", "REJECT")
        if bpb is None:
            continue
        ids.append(eid)
        bpbs.append(bpb)
        colors.append(color_map.get(result, "#95a5a6"))

        if result in ("KEEP", "WEAK_KEEP", "ANNEAL_ACCEPT") and bpb < running_best:
            running_best = bpb
        best_ids.append(eid)
        best_bpbs.append(running_best if running_best != float("inf") else bpb)

    ax1.scatter(ids, bpbs, c=colors, s=40, zorder=3, edgecolors="white", linewidth=0.5)
    ax1.plot(best_ids, best_bpbs, color="#3498db", linewidth=2, label="Best val_bpb", zorder=2)

    baseline = load_baseline()
    if baseline:
        ax1.axhline(y=baseline.get("val_bpb", 0), color="#95a5a6", linestyle="--",
                    linewidth=1, label=f"Baseline ({baseline.get('val_bpb', 0):.4f})")

    ax1.set_ylabel("val_bpb (lower is better)")
    ax1.set_title("AutoResearch-NG Experiment Progress")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=8, label="KEEP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f1c40f", markersize=8, label="WEAK_KEEP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markersize=8, label="ANNEAL_ACCEPT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=8, label="REJECT"),
    ]
    ax1.legend(handles=legend_items + ax1.get_legend_handles_labels()[0][-2:],
              loc="upper right", fontsize=8)

    # --- Bottom: sliding accept rate ---
    ax2 = axes[1]
    window = 10
    if len(exp) >= window:
        accept_rates = []
        rate_ids = []
        for i in range(window, len(exp) + 1):
            chunk = exp[i - window:i]
            accepted = sum(1 for e in chunk if e.get("result") in ("KEEP", "WEAK_KEEP", "ANNEAL_ACCEPT"))
            accept_rates.append(accepted / window * 100)
            rate_ids.append(chunk[-1].get("id", i))
        ax2.fill_between(rate_ids, accept_rates, alpha=0.3, color="#3498db")
        ax2.plot(rate_ids, accept_rates, color="#3498db", linewidth=1.5)
    ax2.set_ylabel("Accept Rate (%)")
    ax2.set_xlabel("Experiment #")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {output}")


if __name__ == "__main__":
    print()
    exp = load_experience()
    print_status(exp)

    if "--plot" in sys.argv or "-p" in sys.argv:
        output = "progress.png"
        for i, arg in enumerate(sys.argv):
            if arg in ("--output", "-o") and i + 1 < len(sys.argv):
                output = sys.argv[i + 1]
        plot_progress(exp, output)

    print()
