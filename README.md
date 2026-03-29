# AutoResearch-NG

**Next-generation optimization loop for [karpathy/autoresearch](https://github.com/karpathy/autoresearch).**

Six systematic improvements to the original autoresearch architecture: simulated annealing acceptance, multi-objective evaluation, persistent experience memory, meta-optimization, staged safety gates, and Pareto front tracking.

Drop-in overlay — does not modify the original autoresearch files.

---

## What this is

Karpathy's autoresearch runs an AI agent in a loop: modify code → train 5 min → check if val_bpb improved → keep or discard → repeat. One GPU, one file, one metric.

AutoResearch-NG keeps that core loop but addresses six structural limitations:

| Original | NG improvement | Impact |
|----------|---------------|--------|
| Greedy ratchet (only keeps improvements) | Simulated annealing (probabilistically accepts small regressions to escape local optima) | Explores wider search space |
| Single scalar metric | Primary metric + constraint metrics + secondary metrics + Pareto front | Prevents Goodhart's Law exploitation |
| No persistent memory | Structured experience log + stage summaries + meta-strategy | Avoids redundant exploration across experiments |
| Fixed search strategy | Meta-optimization detects stagnation and injects new search strategies | Breaks through plateaus |
| No safety checks | Per-experiment constraint check + periodic baseline regression + human checkpoints | Catches drift before it compounds |
| Compare only to previous best | Three-tier comparison: vs last best, vs original baseline, vs Pareto front | Ensures real progress, not just incremental noise |

## Quick start

```bash
# 1. Clone autoresearch and NG overlay
git clone https://github.com/karpathy/autoresearch
git clone https://github.com/YOUR_USERNAME/autoresearch-ng

# 2. Copy NG files into autoresearch
cd autoresearch
cp ../autoresearch-ng/program.md .
cp ../autoresearch-ng/prepare_ng.py .
cp ../autoresearch-ng/CLAUDE.md .

# 3. One-time setup (data prep + baseline)
bash ../autoresearch-ng/setup.sh

# 4. Run (same as original autoresearch)
claude
```

Requires: Python 3.10+, a single NVIDIA GPU, [uv](https://github.com/astral-sh/uv), and [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

Usage is identical to the original autoresearch — you just run `claude` in the directory and walk away. The NG improvements are communicated to the agent through `CLAUDE.md` → `program.md`.

## How it works

### Files you add (3 files, ~43KB total)

```
program.md      →  Replaces original. Agent instructions with annealing schedule,
                   multi-objective definitions, memory rules, safety gates.
prepare_ng.py   →  Supplements original prepare.py. Multi-metric collection,
                   constraint checking, annealing decisions, Pareto front, experience logging.
setup_and_run.sh → One-click setup: data prep → baseline → git init → launch agent.
```

### Files generated at runtime

```
experience.jsonl       Structured log of every experiment (hypothesis, result, insight)
pareto_front.json      Non-dominated solutions across multiple objectives
baseline_metrics.json  Snapshot of initial metrics for regression detection
results.tsv            Extended experiment log (original format + new columns)
stage_summaries/       Markdown summaries generated every 15 experiments
meta_strategy.md       New search strategy (auto-generated when stagnation detected)
```

### The annealing schedule

```
Experiments  1-20:  EXPLORE  (T ≈ 0.05-0.02)  Bold, diverse experiments
Experiments 21-60:  NARROW   (T ≈ 0.02-0.005) Focus on promising directions  
Experiments 61+:    REFINE   (T ≈ 0.005-0.001) Fine-tune best configuration

Acceptance rule:
  Improvement    → always accept
  Regression ≤10% → accept with P = exp(-Δ/T)
  Regression >10% → always reject
  Constraint violation → always reject
```

### Meta-optimization triggers

The system monitors for stagnation patterns and automatically initiates a meta-review:

- 5+ consecutive rejections
- Success rate below 10% over last 20 experiments
- All recent experiments in same category (tunnel vision)
- No improvement in 15 experiments (plateau)
- Last 5 improvements each less than 0.001 (diminishing returns)

When triggered, the agent reads the full experiment log, diagnoses the search bottleneck, generates a new strategy, and resumes with a warm temperature restart.

## Expected results

Being honest about what to expect:

**High confidence improvements:**
- Experience memory reduces redundant exploration → ~5-8% higher keep rate
- Constraint checking prevents Goodhart exploitation → more trustworthy results
- Stage summaries make overnight runs human-reviewable in the morning

**Medium confidence improvements:**
- Annealing may find 1-2 additional improvements by crossing shallow valleys
- Meta-optimization may break through late-run plateaus

**Honest uncertainties:**
- Annealing parameters (T₀=0.05, decay=0.95) are theoretically motivated, not empirically calibrated for nanochat specifically
- Longer program.md (~130 lines vs original ~50) means the agent might not follow all instructions perfectly
- Per-experiment overhead (experience logging, context building) may reduce total experiments from ~100 to ~80 overnight

**Overall estimate:** For a single overnight run, NG's main value is not a dramatically better final val_bpb, but a more auditable, sustainable search process. The compounding benefit shows across multiple nights of iteration, where stage summaries and experience memory inform better program.md updates.

## Adapting to other tasks

The NG framework is task-agnostic. To apply it beyond nanochat training:

1. **Define your editable asset** — the file(s) the agent modifies
2. **Define your evaluation function** — modify `collect_all_metrics()` in prepare_ng.py
3. **Define your objectives** — update the Objectives section in program.md

Everything else (annealing, memory, Pareto, meta-optimization, safety gates) works unchanged.

See program.md comments for examples across frontend performance, prompt engineering, SQL optimization, and more.

## Project structure

```
autoresearch-ng/
├── README.md           This file
├── CLAUDE.md           Auto-read by Claude Code on launch
├── LICENSE             MIT
├── CONTRIBUTING.md     How to contribute
├── .gitignore          Excludes runtime artifacts
├── program.md          Agent instructions (replaces original)
├── prepare_ng.py       NG utility library (supplements original)
└── setup.sh            One-time data prep and baseline
```

## Related work

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — The original. Start here if you haven't.
- [Bilevel Autoresearch](https://arxiv.org/abs/2603.23420) — Meta-optimization paper. Key inspiration for the meta-optimization feature.
- [SkyPilot parallel autoresearch](https://blog.skypilot.co/scaling-autoresearch/) — Scaling to 16 GPUs with factorial experiment design.
- [EvoScientist](https://github.com/EvoScientist/EvoScientist) — Persistent experience memory for autoresearch.
- [awesome-autoresearch](https://github.com/alvinunreal/awesome-autoresearch) — Curated index of the autoresearch ecosystem.
- [autoexp](https://gist.github.com/adhishthite/16d8fd9076e85c033b75e187e8a6b94e) — Generalized autoresearch for any quantifiable metric.

## License

MIT. See [LICENSE](LICENSE).
