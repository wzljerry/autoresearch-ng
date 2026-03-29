# prepare_ng.py — AutoResearch-NG extensions for prepare.py
# 
# 这个文件不替代原版 prepare.py，而是作为补丁导入。
# 原版 prepare.py 的数据加载、tokenizer、evaluate_bpb 等功能保持不变。
# 本文件新增：多指标收集、约束检查、经验记录、基线管理、阶段总结。
#
# 使用方式：在 autoresearch-ng 循环脚本中 import prepare_ng

import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
#  多指标收集
# ═══════════════════════════════════════════════════════════════

def collect_all_metrics(training_output: str, train_py_path: str = "train.py") -> dict:
    """
    从训练输出中解析所有指标。
    
    原版 autoresearch 只看 val_bpb。
    NG 版本收集完整的指标字典。
    
    Args:
        training_output: 训练脚本的 stdout/stderr 输出
        train_py_path: train.py 的路径（用于代码复杂度度量）
    
    Returns:
        dict with all metrics, or None if parsing failed
    """
    metrics = {}
    
    # 1. val_bpb — 主指标（从训练输出中解析）
    for line in training_output.strip().split("\n"):
        if "val_bpb" in line:
            try:
                # 适配多种输出格式
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
        return None  # 无法解析主指标 = 实验失败
    
    # 2. peak_memory_mb — 从 torch 报告或 nvidia-smi 获取
    try:
        import torch
        if torch.cuda.is_available():
            metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            metrics["peak_memory_mb"] = 0
    except ImportError:
        metrics["peak_memory_mb"] = _parse_memory_from_output(training_output)
    
    # 3. throughput_tokens_per_sec — 从训练输出解析
    metrics["throughput_tokens_per_sec"] = _parse_throughput(training_output)
    
    # 4. parameter_count — 从训练输出解析
    metrics["parameter_count"] = _parse_param_count(training_output)
    
    # 5. training_steps — 从训练输出解析
    metrics["training_steps"] = _parse_training_steps(training_output)
    
    # 6. code_lines — train.py 的行数
    metrics["code_lines"] = _count_code_lines(train_py_path)
    
    # 7. code_complexity — 简化的圈复杂度
    metrics["code_complexity"] = _measure_complexity(train_py_path)
    
    return metrics


def _parse_memory_from_output(output: str) -> float:
    """从训练输出中解析内存使用"""
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
    """从训练输出中解析吞吐量"""
    for line in reversed(output.strip().split("\n")):
        if "tok/s" in line or "tokens/sec" in line or "throughput" in line.lower():
            import re
            numbers = re.findall(r"[\d,]+\.?\d*", line)
            if numbers:
                return float(numbers[-1].replace(",", ""))
    return 0.0


def _parse_param_count(output: str) -> int:
    """从训练输出中解析参数数量"""
    for line in output.strip().split("\n"):
        line_lower = line.lower()
        if "param" in line_lower and ("total" in line_lower or "count" in line_lower or "num" in line_lower):
            import re
            numbers = re.findall(r"[\d,]+", line)
            if numbers:
                return int(numbers[-1].replace(",", ""))
    return 0


def _parse_training_steps(output: str) -> int:
    """从训练输出中解析训练步数"""
    max_step = 0
    for line in output.strip().split("\n"):
        if "step" in line.lower():
            import re
            numbers = re.findall(r"\d+", line)
            for n in numbers:
                val = int(n)
                if val > max_step and val < 1000000:  # 合理范围
                    max_step = val
    return max_step


def _count_code_lines(filepath: str) -> int:
    """计算代码行数（排除空行和注释）"""
    try:
        lines = Path(filepath).read_text().strip().split("\n")
        return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
    except FileNotFoundError:
        return 0


def _measure_complexity(filepath: str) -> int:
    """简化的圈复杂度度量"""
    try:
        code = Path(filepath).read_text()
        keywords = ["if ", "elif ", "for ", "while ", "except ", " and ", " or ",
                     "try:", "with ", "assert "]
        return sum(code.count(k) for k in keywords)
    except FileNotFoundError:
        return 0


# ═══════════════════════════════════════════════════════════════
#  约束检查
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONSTRAINTS = {
    "peak_memory_mb": ("<=", 80000),     # H100 80GB
    "training_steps": (">=", 50),
    "code_lines": ("<=", 1200),
}

def check_constraints(metrics: dict, constraints: dict = None) -> tuple:
    """
    硬约束检查。
    
    Returns:
        (passed: bool, violations: list[str])
    """
    if constraints is None:
        constraints = DEFAULT_CONSTRAINTS
    
    violations = []
    
    for metric_name, (operator, threshold) in constraints.items():
        value = metrics.get(metric_name)
        if value is None:
            continue  # 跳过缺失指标
        
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
    检查次要指标是否退化超过阈值。
    
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
                degraded.append(f"{name}: 退化 {abs(change)*100:.1f}% (阈值 {max_degradation*100}%)")
        else:  # lower is better
            change = (current - best) / abs(best)
            if change > max_degradation:
                degraded.append(f"{name}: 增长 {change*100:.1f}% (阈值 {max_degradation*100}%)")
    
    return len(degraded) == 0, degraded


# ═══════════════════════════════════════════════════════════════
#  退火接受决策
# ═══════════════════════════════════════════════════════════════

def annealing_decision(metrics: dict, best_metrics: dict,
                        temperature: float, constraints: dict = None) -> str:
    """
    多目标 + 概率退火接受决策。
    
    Returns:
        "KEEP" | "WEAK_KEEP" | "ANNEAL_ACCEPT" | "REJECT"
    """
    import random
    
    # 1. 硬约束检查
    passed, violations = check_constraints(metrics, constraints)
    if not passed:
        return "REJECT"
    
    # 2. 主指标比较
    current_bpb = metrics.get("val_bpb", float("inf"))
    best_bpb = best_metrics.get("val_bpb", float("inf"))
    
    if current_bpb < best_bpb:
        # 主指标改善 — 检查次要指标
        sec_ok, _ = check_secondary_degradation(metrics, best_metrics)
        return "KEEP" if sec_ok else "WEAK_KEEP"
    
    # 3. 主指标未改善 — 概率退火
    if best_bpb == 0:
        return "REJECT"
    
    delta = (current_bpb - best_bpb) / abs(best_bpb)
    
    # 退化超过 10% — 直接拒绝（即使温度很高）
    if delta > 0.10:
        return "REJECT"
    
    # 退火接受
    T_min = 0.001
    if temperature > T_min and delta > 0:
        accept_prob = math.exp(-delta / temperature)
        if random.random() < accept_prob:
            return "ANNEAL_ACCEPT"
    
    return "REJECT"


# ═══════════════════════════════════════════════════════════════
#  经验记录
# ═══════════════════════════════════════════════════════════════

EXPERIENCE_FILE = "experience.jsonl"

def append_experience(entry: dict):
    """追加一条经验记录"""
    entry["timestamp"] = datetime.now().isoformat()
    with open(EXPERIENCE_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_experience(last_n: int = None) -> list:
    """加载经验记录"""
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
    """识别已穷尽的方向（连续 N 次失败的类别）"""
    consecutive_fails = {}
    
    for entry in reversed(experience):
        category = entry.get("change_category", "unknown")
        if entry.get("result") == "REJECT":
            consecutive_fails[category] = consecutive_fails.get(category, 0) + 1
        else:
            # 遇到一个成功就重置该类别的计数
            if category in consecutive_fails:
                del consecutive_fails[category]
    
    return [cat for cat, count in consecutive_fails.items() if count >= threshold]


def get_promising_directions(experience: list, min_experiments: int = 2,
                              min_success_rate: float = 0.3) -> list:
    """识别有希望的方向"""
    from collections import defaultdict
    stats = defaultdict(lambda: {"keep": 0, "total": 0})
    
    # 只看最近 30 次
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
#  基线管理
# ═══════════════════════════════════════════════════════════════

BASELINE_FILE = "baseline_metrics.json"

def save_baseline(metrics: dict):
    """保存基线指标（首次运行时调用）"""
    data = {
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    Path(BASELINE_FILE).write_text(json.dumps(data, indent=2))


def load_baseline() -> dict:
    """加载基线指标"""
    if Path(BASELINE_FILE).exists():
        data = json.loads(Path(BASELINE_FILE).read_text())
        return data.get("metrics", data)
    return None


def baseline_regression_report(current_best: dict) -> str:
    """生成基线回归报告"""
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
                    status = " ⚠ 搜索可能停滞"
                    any_warning = True
                lines.append(f"  {key}: baseline={base_val:.4f}, "
                           f"current={curr_val:.4f}, drift={drift_pct:+.2f}%{status}")
    
    if not any_warning:
        lines.append("  ✓ All metrics show healthy progress from baseline")
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Pareto 前沿管理
# ═══════════════════════════════════════════════════════════════

PARETO_FILE = "pareto_front.json"

def load_pareto_front() -> list:
    if Path(PARETO_FILE).exists():
        return json.loads(Path(PARETO_FILE).read_text())
    return []


def save_pareto_front(front: list):
    Path(PARETO_FILE).write_text(json.dumps(front, indent=2))


def dominates(a: dict, b: dict) -> bool:
    """a 是否 Pareto 支配 b"""
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
    """更新 Pareto 前沿，返回更新后的前沿"""
    new_point = {
        "experiment_id": experiment_id,
        "metrics": new_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    # 移除被新点支配的旧点
    front = [p for p in front if not dominates(new_metrics, p.get("metrics", p))]
    
    # 如果新点不被任何旧点支配，加入前沿
    existing_metrics = [p.get("metrics", p) for p in front]
    if not any(dominates(em, new_metrics) for em in existing_metrics):
        front.append(new_point)
    
    save_pareto_front(front)
    return front


# ═══════════════════════════════════════════════════════════════
#  元优化触发检测
# ═══════════════════════════════════════════════════════════════

def should_trigger_meta_review(experience: list) -> tuple:
    """
    检测是否应该触发元优化。
    
    Returns:
        (should_trigger: bool, reason: str)
    """
    if len(experience) < 10:
        return False, ""
    
    # 1. 连续 5+ 次拒绝
    consecutive_rejects = 0
    for e in reversed(experience):
        if e.get("result") == "REJECT":
            consecutive_rejects += 1
        else:
            break
    if consecutive_rejects >= 5:
        return True, f"连续 {consecutive_rejects} 次 REJECT"
    
    # 2. 最近 20 次成功率 < 10%
    recent_20 = experience[-20:]
    if len(recent_20) >= 20:
        keeps = sum(1 for e in recent_20
                    if e.get("result") in ("KEEP", "WEAK_KEEP"))
        rate = keeps / len(recent_20)
        if rate < 0.1:
            return True, f"最近20次成功率仅 {rate*100:.0f}%"
    
    # 3. 最近 10 次都是同一个类别
    recent_10 = experience[-10:]
    if len(recent_10) >= 10:
        categories = [e.get("change_category", "?") for e in recent_10]
        if len(set(categories)) == 1:
            return True, f"隧道视野: 连续10次都是 '{categories[0]}'"
    
    # 4. 15 轮无改善
    if len(experience) >= 15:
        recent_15 = experience[-15:]
        any_keep = any(e.get("result") in ("KEEP", "WEAK_KEEP") for e in recent_15)
        if not any_keep:
            return True, "15 轮内无任何改善"
    
    # 5. 递减收益
    recent_keeps = [e for e in experience[-10:]
                    if e.get("result") in ("KEEP", "WEAK_KEEP")]
    if len(recent_keeps) >= 5:
        deltas = [abs(e.get("delta_vs_best", {}).get("val_bpb", 0))
                  for e in recent_keeps[-5:]]
        if all(d < 0.001 for d in deltas):
            return True, "递减收益: 最近5次改善都 < 0.001"
    
    return False, ""


# ═══════════════════════════════════════════════════════════════
#  阶段总结生成
# ═══════════════════════════════════════════════════════════════

def generate_stage_summary(experience: list, stage_start: int, stage_end: int,
                            baseline_metrics: dict, best_metrics: dict) -> str:
    """生成阶段总结的 Markdown 文本"""
    stage_exp = [e for e in experience if stage_start <= e.get("id", 0) <= stage_end]
    
    total = len(stage_exp)
    kept = sum(1 for e in stage_exp if e.get("result") == "KEEP")
    weak = sum(1 for e in stage_exp if e.get("result") == "WEAK_KEEP")
    anneal = sum(1 for e in stage_exp if e.get("result") == "ANNEAL_ACCEPT")
    crashes = sum(1 for e in stage_exp if e.get("result") == "CRASH")
    
    # 按类别统计
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
    
    # 收集最近的 insights
    recent_insights = [e.get("insight", "") for e in stage_exp if e.get("insight")]
    lines.extend([
        "",
        "## Key insights from this stage",
    ])
    for insight in recent_insights[-3:]:
        lines.append(f"- {insight}")
    
    return "\n".join(lines)


def save_stage_summary(summary_text: str, stage_number: int):
    """保存阶段总结"""
    Path("stage_summaries").mkdir(exist_ok=True)
    filename = f"stage_summaries/stage_{stage_number:03d}.md"
    Path(filename).write_text(summary_text)
    return filename


# ═══════════════════════════════════════════════════════════════
#  扩展版 results.tsv
# ═══════════════════════════════════════════════════════════════

RESULTS_FILE = "results.tsv"
RESULTS_HEADER = (
    "experiment_id\tcommit_hash\tresult\tval_bpb\tpeak_memory_mb\t"
    "throughput_tps\tparameter_count\ttraining_steps\t"
    "temperature\tphase\tdescription\n"
)

def init_results():
    """初始化 results.tsv（如果不存在）"""
    if not Path(RESULTS_FILE).exists():
        Path(RESULTS_FILE).write_text(RESULTS_HEADER)


def append_result(experiment_id: int, commit_hash: str, result: str,
                  metrics: dict, temperature: float, phase: str,
                  description: str):
    """追加一行到 results.tsv"""
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
#  初始化
# ═══════════════════════════════════════════════════════════════

def init_ng():
    """初始化 AutoResearch-NG 所需的所有文件和目录"""
    init_results()
    Path("stage_summaries").mkdir(exist_ok=True)
    
    # 如果 experience.jsonl 不存在则创建
    if not Path(EXPERIENCE_FILE).exists():
        Path(EXPERIENCE_FILE).touch()
    
    # 如果 pareto_front.json 不存在则创建
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
        print("  - ⚠ No baseline yet. Run baseline experiment first.")


if __name__ == "__main__":
    init_ng()
