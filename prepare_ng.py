# prepare_ng.py — AutoResearch-NG extensions for prepare.py
#
# This file does not replace the original prepare.py; it is imported as a supplement.
# The original prepare.py's data loading, tokenizer, evaluate_bpb, etc. remain unchanged.
# This file adds: multi-metric collection, constraint checking, experience logging,
# baseline management, and stage summaries.
#
# Usage: import prepare_ng in the autoresearch-ng loop script

import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
#  Multi-Metric Collection
# ═══════════════════════════════════════════════════════════════

def collect_all_metrics(training_output: str, train_py_path: str = "train.py") -> dict:
    """
    Parse all metrics from training output.

    The original autoresearch only checks val_bpb.
    The NG version collects a complete metrics dictionary.

    Args:
        training_output: stdout/stderr output from the training script
        train_py_path: path to train.py (used for code complexity metrics)

    Returns:
        dict with all metrics, or None if parsing failed
    """
    metrics = {}

    # 1. val_bpb — primary metric (parsed from training output)
    for line in training_output.strip().split("\n"):
        if "val_bpb" in line:
            try:
                # Supports multiple output formats
                if "val_bpb:" in line:
                    metrics["val_bpb"] = float(line.split("val_bpb:")[-1].strip().split()[0])
                elif "val_bpb=" in line:
                    metrics["val_bpb"] = float(line.split("val_bpb=")[-1].strip().split()[0])
                elif "val_bpb " in line:
                    parts = line.split()
                    idx = parts.index("val_bpb")
                    metrics["val_bpb"] = float(parts[idx + 1])
            except (ValueError, IndexError):
                continue

    if "val_bpb" not in metrics:
        return None  # Cannot parse primary metric = experiment failed

    # 2. peak_memory_mb — from torch report or nvidia-smi
    try:
        import torch
        if torch.cuda.is_available():
            metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            metrics["peak_memory_mb"] = 0
    except ImportError:
        metrics["peak_memory_mb"] = _parse_memory_from_output(training_output)

    # 3. throughput_tokens_per_sec — parsed from training output
    metrics["throughput_tokens_per_sec"] = _parse_throughput(training_output)

    # 4. parameter_count — parsed from training output
    metrics["parameter_count"] = _parse_param_count(training_output)

    # 5. training_steps — parsed from training output
    metrics["training_steps"] = _parse_training_steps(training_output)

    # 6. code_lines — line count of train.py
    metrics["code_lines"] = _count_code_lines(train_py_path)

    # 7. code_complexity — simplified cyclomatic complexity
    metrics["code_complexity"] = _measure_complexity(train_py_path)

    return metrics


def _parse_memory_from_output(output: str) -> float:
    """Parse memory usage from training output"""
    for line in reversed(output.strip().split("\n")):
        line_lower = line.lower()
        if "memory" in line_lower and ("mb" in line_lower or "gb" in line_lower):
            import re
            numbers = re.findall(r"[\d.]+", line)
            if numbers:
                val = float(numbers[-1])
                if "gb" in line_lower:
                    val *= 1024
                return val
    return 0.0


def _parse_throughput(output: str) -> float:
    """Parse throughput from training output"""
    for line in reversed(output.strip().split("\n")):
        if "tok/s" in line or "tokens/sec" in line or "throughput" in line.lower():
            import re
            numbers = re.findall(r"[\d,]+\.?\d*", line)
            if numbers:
                return float(numbers[-1].replace(",", ""))
    return 0.0


def _parse_param_count(output: str) -> int:
    """Parse parameter count from training output"""
    for line in output.strip().split("\n"):
        line_lower = line.lower()
        if "param" in line_lower and ("total" in line_lower or "count" in line_lower or "num" in line_lower):
            import re
            numbers = re.findall(r"[\d,]+", line)
            if numbers:
                return int(numbers[-1].replace(",", ""))
    return 0


def _parse_training_steps(output: str) -> int:
    """Parse training step count from training output"""
    max_step = 0
    for line in output.strip().split("\n"):
        if "step" in line.lower():
            import re
            numbers = re.findall(r"\d+", line)
            for n in numbers:
                val = int(n)
                if val > max_step and val < 1000000:  # reasonable range
                    max_step = val
    return max_step


def _count_code_lines(filepath: str) -> int:
    """Count code lines (excluding blank lines and comments)"""
    try:
        lines = Path(filepath).read_text().strip().split("\n")
        return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
    except FileNotFoundError:
        return 0


def _measure_complexity(filepath: str) -> int:
    """Simplified cyclomatic complexity metric"""
    try:
        code = Path(filepath).read_text()
        keywords = ["if ", "elif ", "for ", "while ", "except ", " and ", " or ",
                     "try:", "with ", "assert "]
        return sum(code.count(k) for k in keywords)
    except FileNotFoundError:
        return 0


# ═══════════════════════════════════════════════════════════════
#  Constraint Checking
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONSTRAINTS = {
    "peak_memory_mb": ("<=", 80000),     # H100 80GB
    "training_steps": (">=", 50),
    "code_lines": ("<=", 1200),
}

def check_constraints(metrics: dict, constraints: dict = None) -> tuple:
    """
    Hard constraint checking.

    Returns:
        (passed: bool, violations: list[str])
    """
    if constraints is None:
        constraints = DEFAULT_CONSTRAINTS

    violations = []

    for metric_name, (operator, threshold) in constraints.items():
        value = metrics.get(metric_name)
        if value is None:
            continue  # skip missing metrics

        if operator == "<=" and value > threshold:
            violations.append(f"{metric_name}: {value} > {threshold}")
        elif operator == ">=" and value < threshold:
            violations.append(f"{metric_name}: {value} < {threshold}")
        elif operator == "==" and value != threshold:
            violations.append(f"{metric_name}: {value} != {threshold}")

    return len(violations) == 0, violations


def check_secondary_degradation(metrics: dict, best_metrics: dict,
                                 max_degradation: float = 0.15) -> tuple:
    """
    Check whether secondary metrics have degraded beyond the threshold.

    Returns:
        (ok: bool, degraded_metrics: list[str])
    """
    secondary = {
        "throughput_tokens_per_sec": "higher",
        "parameter_count": "lower",
        "training_steps": "higher",
    }

    degraded = []
    for name, direction in secondary.items():
        current = metrics.get(name, 0)
        best = best_metrics.get(name, 0)

        if best == 0:
            continue

        if direction == "higher":
            change = (current - best) / abs(best)
            if change < -max_degradation:
                degraded.append(f"{name}: degraded {abs(change)*100:.1f}% (threshold {max_degradation*100}%)")
        else:  # lower is better
            change = (current - best) / abs(best)
            if change > max_degradation:
                degraded.append(f"{name}: increased {change*100:.1f}% (threshold {max_degradation*100}%)")

    return len(degraded) == 0, degraded


# ═══════════════════════════════════════════════════════════════
#  Annealing Acceptance Decision
# ═══════════════════════════════════════════════════════════════

def annealing_decision(metrics: dict, best_metrics: dict,
                        temperature: float, constraints: dict = None) -> str:
    """
    Multi-objective + probabilistic annealing acceptance decision.

    Returns:
        "KEEP" | "WEAK_KEEP" | "ANNEAL_ACCEPT" | "REJECT"
    """
    import random

    # 1. Hard constraint check
    passed, violations = check_constraints(metrics, constraints)
    if not passed:
        return "REJECT"

    # 2. Primary metric comparison
    current_bpb = metrics.get("val_bpb", float("inf"))
    best_bpb = best_metrics.get("val_bpb", float("inf"))

    if current_bpb < best_bpb:
        # Primary metric improved — check secondary metrics
        sec_ok, _ = check_secondary_degradation(metrics, best_metrics)
        return "KEEP" if sec_ok else "WEAK_KEEP"

    # 3. Primary metric did not improve — probabilistic annealing
    if best_bpb == 0:
        return "REJECT"

    delta = (current_bpb - best_bpb) / abs(best_bpb)

    # Degradation exceeds 10% — reject outright (even at high temperature)
    if delta > 0.10:
        return "REJECT"

    # Annealing acceptance
    T_min = 0.001
    if temperature > T_min and delta > 0:
        accept_prob = math.exp(-delta / temperature)
        if random.random() < accept_prob:
            return "ANNEAL_ACCEPT"

    return "REJECT"


# ═══════════════════════════════════════════════════════════════
#  Experience Logging
# ═══════════════════════════════════════════════════════════════

EXPERIENCE_FILE = "experience.jsonl"

def append_experience(entry: dict):
    """Append one experience record"""
    entry["timestamp"] = datetime.now().isoformat()
    with open(EXPERIENCE_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_experience(last_n: int = None) -> list:
    """Load experience records"""
    entries = []
    if Path(EXPERIENCE_FILE).exists():
        with open(EXPERIENCE_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    if last_n:
        return entries[-last_n:]
    return entries


def get_exhausted_directions(experience: list, threshold: int = 3) -> list:
    """Identify exhausted directions (categories with N+ consecutive failures)"""
    consecutive_fails = {}

    for entry in reversed(experience):
        category = entry.get("change_category", "unknown")
        if entry.get("result") == "REJECT":
            consecutive_fails[category] = consecutive_fails.get(category, 0) + 1
        else:
            # A success resets the counter for that category
            if category in consecutive_fails:
                del consecutive_fails[category]

    return [cat for cat, count in consecutive_fails.items() if count >= threshold]


def get_promising_directions(experience: list, min_experiments: int = 2,
                              min_success_rate: float = 0.3) -> list:
    """Identify promising directions"""
    from collections import defaultdict
    stats = defaultdict(lambda: {"keep": 0, "total": 0})

    # Only look at last 30 experiments
    recent = experience[-30:] if len(experience) > 30 else experience

    for entry in recent:
        category = entry.get("change_category", "unknown")
        stats[category]["total"] += 1
        if entry.get("result") in ("KEEP", "WEAK_KEEP"):
            stats[category]["keep"] += 1

    return [
        {"category": cat, "success_rate": s["keep"] / s["total"],
         "total": s["total"], "kept": s["keep"]}
        for cat, s in stats.items()
        if s["total"] >= min_experiments and s["keep"] / s["total"] >= min_success_rate
    ]


# ═══════════════════════════════════════════════════════════════
#  Baseline Management
# ═══════════════════════════════════════════════════════════════

BASELINE_FILE = "baseline_metrics.json"

def save_baseline(metrics: dict):
    """Save baseline metrics (called on first run)"""
    data = {
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    Path(BASELINE_FILE).write_text(json.dumps(data, indent=2))


def load_baseline() -> dict:
    """Load baseline metrics"""
    if Path(BASELINE_FILE).exists():
        data = json.loads(Path(BASELINE_FILE).read_text())
        return data.get("metrics", data)
    return None


def baseline_regression_report(current_best: dict) -> str:
    """Generate a baseline regression report"""
    baseline = load_baseline()
    if not baseline:
        return "WARNING: No baseline saved"

    lines = ["=== Baseline Regression Report ==="]
    any_warning = False

    for key in baseline:
        if key in current_best:
            base_val = baseline[key]
            curr_val = current_best[key]
            if base_val != 0:
                drift_pct = (curr_val - base_val) / abs(base_val) * 100
                status = ""
                if key == "val_bpb" and abs(drift_pct) < 0.5:
                    status = " WARNING: Search may be stagnating"
                    any_warning = True
                lines.append(f"  {key}: baseline={base_val:.4f}, "
                           f"current={curr_val:.4f}, drift={drift_pct:+.2f}%{status}")

    if not any_warning:
        lines.append("  All metrics show healthy progress from baseline")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Pareto Front Management
# ═══════════════════════════════════════════════════════════════

PARETO_FILE = "pareto_front.json"

def load_pareto_front() -> list:
    if Path(PARETO_FILE).exists():
        return json.loads(Path(PARETO_FILE).read_text())
    return []


def save_pareto_front(front: list):
    Path(PARETO_FILE).write_text(json.dumps(front, indent=2))


def dominates(a: dict, b: dict) -> bool:
    """Check whether a Pareto-dominates b"""
    objectives = [
        ("val_bpb", "lower"),
        ("throughput_tokens_per_sec", "higher"),
        ("peak_memory_mb", "lower"),
    ]

    better_in_any = False
    for metric, direction in objectives:
        va = a.get(metric, float("inf") if direction == "lower" else 0)
        vb = b.get(metric, float("inf") if direction == "lower" else 0)

        if direction == "lower":
            if va > vb: return False
            if va < vb: better_in_any = True
        else:
            if va < vb: return False
            if va > vb: better_in_any = True

    return better_in_any


def update_pareto_front(front: list, new_metrics: dict, experiment_id: int) -> list:
    """Update the Pareto front and return the updated front"""
    new_point = {
        "experiment_id": experiment_id,
        "metrics": new_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    # Remove old points dominated by the new point
    front = [p for p in front if not dominates(new_metrics, p.get("metrics", p))]

    # Add new point if it is not dominated by any existing point
    existing_metrics = [p.get("metrics", p) for p in front]
    if not any(dominates(em, new_metrics) for em in existing_metrics):
        front.append(new_point)

    save_pareto_front(front)
    return front


# ═══════════════════════════════════════════════════════════════
#  Meta-Optimization Trigger Detection
# ═══════════════════════════════════════════════════════════════

def should_trigger_meta_review(experience: list) -> tuple:
    """
    Detect whether a meta-optimization review should be triggered.

    Returns:
        (should_trigger: bool, reason: str)
    """
    if len(experience) < 10:
        return False, ""

    # 1. 5+ consecutive rejections
    consecutive_rejects = 0
    for e in reversed(experience):
        if e.get("result") == "REJECT":
            consecutive_rejects += 1
        else:
            break
    if consecutive_rejects >= 5:
        return True, f"{consecutive_rejects} consecutive REJECTs"

    # 2. Success rate < 10% over last 20 experiments
    recent_20 = experience[-20:]
    if len(recent_20) >= 20:
        keeps = sum(1 for e in recent_20
                    if e.get("result") in ("KEEP", "WEAK_KEEP"))
        rate = keeps / len(recent_20)
        if rate < 0.1:
            return True, f"Success rate only {rate*100:.0f}% over last 20 experiments"

    # 3. Last 10 experiments all in the same category
    recent_10 = experience[-10:]
    if len(recent_10) >= 10:
        categories = [e.get("change_category", "?") for e in recent_10]
        if len(set(categories)) == 1:
            return True, f"Tunnel vision: 10 consecutive experiments in '{categories[0]}'"

    # 4. No improvement in 15 rounds
    if len(experience) >= 15:
        recent_15 = experience[-15:]
        any_keep = any(e.get("result") in ("KEEP", "WEAK_KEEP") for e in recent_15)
        if not any_keep:
            return True, "No improvement in 15 rounds"

    # 5. Diminishing returns
    recent_keeps = [e for e in experience[-10:]
                    if e.get("result") in ("KEEP", "WEAK_KEEP")]
    if len(recent_keeps) >= 5:
        deltas = [abs(e.get("delta_vs_best", {}).get("val_bpb", 0))
                  for e in recent_keeps[-5:]]
        if all(d < 0.001 for d in deltas):
            return True, "Diminishing returns: last 5 improvements all < 0.001"

    return False, ""


# ═══════════════════════════════════════════════════════════════
#  Stage Summary Generation
# ═══════════════════════════════════════════════════════════════

def generate_stage_summary(experience: list, stage_start: int, stage_end: int,
                            baseline_metrics: dict, best_metrics: dict) -> str:
    """Generate a stage summary in Markdown format"""
    stage_exp = [e for e in experience if stage_start <= e.get("id", 0) <= stage_end]

    total = len(stage_exp)
    kept = sum(1 for e in stage_exp if e.get("result") == "KEEP")
    weak = sum(1 for e in stage_exp if e.get("result") == "WEAK_KEEP")
    anneal = sum(1 for e in stage_exp if e.get("result") == "ANNEAL_ACCEPT")
    crashes = sum(1 for e in stage_exp if e.get("result") == "CRASH")

    # Statistics by category
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"keep": 0, "total": 0})
    for e in stage_exp:
        cat = e.get("change_category", "unknown")
        cat_stats[cat]["total"] += 1
        if e.get("result") in ("KEEP", "WEAK_KEEP"):
            cat_stats[cat]["keep"] += 1

    best_bpb = min(
        (e.get("metrics", {}).get("val_bpb", float("inf")) for e in stage_exp
         if e.get("result") in ("KEEP", "WEAK_KEEP", "ANNEAL_ACCEPT")),
        default=float("inf")
    )
    start_bpb = stage_exp[0].get("metrics", {}).get("val_bpb", "?") if stage_exp else "?"
    baseline_bpb = baseline_metrics.get("val_bpb", "?") if baseline_metrics else "?"

    successful = sorted(
        [(cat, s) for cat, s in cat_stats.items() if s["keep"] > 0],
        key=lambda x: x[1]["keep"] / x[1]["total"],
        reverse=True
    )

    exhausted = [(cat, s) for cat, s in cat_stats.items()
                 if s["total"] >= 2 and s["keep"] == 0]

    lines = [
        f"# Stage Summary: Experiments {stage_start} to {stage_end}",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Results",
        f"- Total experiments: {total}",
        f"- Kept: {kept} ({kept/total*100:.0f}%)" if total > 0 else "- Kept: 0",
        f"- Weak keep: {weak}",
        f"- Anneal accepted: {anneal}",
        f"- Crashes: {crashes}",
        f"- Best val_bpb this stage: {best_bpb:.6f}" if best_bpb != float("inf") else "- No improvements",
        f"- Stage start val_bpb: {start_bpb}",
        f"- Baseline val_bpb: {baseline_bpb}",
        "",
        "## Successful directions",
    ]

    for i, (cat, s) in enumerate(successful[:5], 1):
        rate = s["keep"] / s["total"] * 100
        lines.append(f"{i}. **{cat}**: {s['keep']}/{s['total']} kept ({rate:.0f}%)")

    if not successful:
        lines.append("(none this stage)")

    lines.extend(["", "## Exhausted directions"])
    for i, (cat, s) in enumerate(exhausted[:5], 1):
        lines.append(f"{i}. **{cat}**: 0/{s['total']} kept")

    if not exhausted:
        lines.append("(none)")

    # Collect recent insights
    recent_insights = [e.get("insight", "") for e in stage_exp if e.get("insight")]
    lines.extend([
        "",
        "## Key insights from this stage",
    ])
    for insight in recent_insights[-3:]:
        lines.append(f"- {insight}")

    return "\n".join(lines)


def save_stage_summary(summary_text: str, stage_number: int):
    """Save a stage summary"""
    Path("stage_summaries").mkdir(exist_ok=True)
    filename = f"stage_summaries/stage_{stage_number:03d}.md"
    Path(filename).write_text(summary_text)
    return filename


# ═══════════════════════════════════════════════════════════════
#  Extended results.tsv
# ═══════════════════════════════════════════════════════════════

RESULTS_FILE = "results.tsv"
RESULTS_HEADER = (
    "experiment_id\tcommit_hash\tresult\tval_bpb\tpeak_memory_mb\t"
    "throughput_tps\tparameter_count\ttraining_steps\t"
    "temperature\tphase\tdescription\n"
)

def init_results():
    """Initialize results.tsv (if it does not exist)"""
    if not Path(RESULTS_FILE).exists():
        Path(RESULTS_FILE).write_text(RESULTS_HEADER)


def append_result(experiment_id: int, commit_hash: str, result: str,
                  metrics: dict, temperature: float, phase: str,
                  description: str):
    """Append one row to results.tsv"""
    row = (
        f"{experiment_id}\t{commit_hash}\t{result}\t"
        f"{metrics.get('val_bpb', '')}\t{metrics.get('peak_memory_mb', '')}\t"
        f"{metrics.get('throughput_tokens_per_sec', '')}\t"
        f"{metrics.get('parameter_count', '')}\t"
        f"{metrics.get('training_steps', '')}\t"
        f"{temperature:.6f}\t{phase}\t{description}\n"
    )
    with open(RESULTS_FILE, "a") as f:
        f.write(row)


# ═══════════════════════════════════════════════════════════════
#  Initialization
# ═══════════════════════════════════════════════════════════════

def init_ng():
    """Initialize all files and directories required by AutoResearch-NG"""
    init_results()
    Path("stage_summaries").mkdir(exist_ok=True)

    # Create experience.jsonl if it does not exist
    if not Path(EXPERIENCE_FILE).exists():
        Path(EXPERIENCE_FILE).touch()

    # Create pareto_front.json if it does not exist
    if not Path(PARETO_FILE).exists():
        save_pareto_front([])

    print("AutoResearch-NG initialized:")
    print(f"  - {RESULTS_FILE}")
    print(f"  - {EXPERIENCE_FILE}")
    print(f"  - {PARETO_FILE}")
    print(f"  - stage_summaries/")

    baseline = load_baseline()
    if baseline:
        print(f"  - Baseline loaded: val_bpb = {baseline.get('val_bpb', '?')}")
    else:
        print("  - WARNING: No baseline yet. Run baseline experiment first.")


if __name__ == "__main__":
    init_ng()
