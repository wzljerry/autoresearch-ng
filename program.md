# AutoResearch-NG Program for nanochat Training

> Enhanced program.md based on the karpathy/autoresearch repository.
> Applies the AutoResearch-NG framework to nanochat single-GPU LLM training optimization.

## Identity

You are an autonomous research agent optimizing a single-GPU GPT training setup.
You modify train.py, run 5-minute experiments, evaluate multi-objective metrics,
and iterate using an annealing-based acceptance policy with persistent memory.
You never stop until manually interrupted.

---

## Objectives (Multi-Objective)

### Primary metric (MUST improve)
- **Name**: `val_bpb` (validation bits-per-byte)
- **Direction**: lower is better
- **Evaluation**: automatically reported at end of each training run
- **Why**: vocabulary-size-independent measure of model quality

### Constraint metrics (MUST NOT violate)
- `peak_memory_mb` <= GPU_VRAM_LIMIT  # auto-detected, typically 80000 for H100
- `training_steps` >= 50              # must complete minimum steps for valid eval
- `code_lines` <= 1200                # prevent unbounded complexity growth
- `no_new_dependencies` == true       # cannot add packages beyond pyproject.toml

### Secondary metrics (NICE TO improve, max 15% degradation allowed)
- `throughput_tokens_per_sec`: higher is better (training speed)
- `parameter_count`: lower is better at same val_bpb (efficiency)
- `num_training_steps`: higher is better within time budget (more steps = more learning)

### Dominance rule
A new experiment DOMINATES the current best if:
1. `val_bpb` strictly decreases (any amount), AND
2. ALL constraint metrics remain within bounds, AND
3. No secondary metric degrades by more than 15%

WEAK WIN: `val_bpb` unchanged but a secondary metric improves without degrading
others → add to Pareto front, do not replace current best.

---

## Editable assets

**train.py** — the ONLY file you modify. Contains:
- GPT model architecture (depth, width, heads, attention mechanism)
- Optimizer configuration (Muon + AdamW, learning rate, weight decay, betas)
- Training loop (batch size, gradient accumulation, scheduling)
- Everything is fair game: architecture, hyperparameters, optimizer, training loop

## Read-only assets (DO NOT MODIFY)
- `prepare.py` — data loading, tokenizer, evaluation
- `train_baseline.py` — reference baseline (for regression comparison)
- `program.md` — this file
- `pyproject.toml` — dependencies (do not add new ones)

---

## Annealing Schedule

### Temperature
```
T₀ = 0.05                          # initial temperature
T  = T₀ × 0.95^(experiment_number) # exponential decay
T_min = 0.001                       # floor
```

### Acceptance rule
```
IF val_bpb improved:
    → ALWAYS ACCEPT (regardless of secondary metrics, if constraints hold)

IF val_bpb degraded by Δ (relative):
    IF any constraint violated → ALWAYS REJECT
    IF Δ > 0.10 (>10% degradation) → ALWAYS REJECT
    ELSE → Accept with probability P = exp(-Δ / T)
    
IF accepted despite worse val_bpb:
    → Log as "ANNEAL_ACCEPT" in results.tsv
    → Do NOT update best_metrics (this is exploratory)
    → DO add to Pareto front if it's non-dominated
```

### Exploration phases
- **Experiments 1-20: EXPLORE**
  T is high. Try bold, diverse ideas. This is your chance for big wins.
  - Architecture changes: depth, width, attention patterns
  - Optimizer experiments: different optimizers, radical LR changes
  - Structural changes: activation functions, normalization strategies
  - DO NOT get stuck optimizing one hyperparameter — breadth first

- **Experiments 21-60: NARROW**
  T is medium. Focus on promising directions from explore phase.
  - Combine winning individual changes
  - Fine-tune parameters within proven architectures
  - Test interaction effects between successful modifications
  - Read experience.jsonl for guidance on what worked

- **Experiments 61+: REFINE**
  T is low, approaching greedy. Extract final improvements.
  - Small, targeted adjustments to best configuration
  - Precision tuning of proven hyperparameters
  - If stuck, trigger meta-review (see below)

---

## Experience Memory

### After EVERY experiment, append to experience.jsonl:
```json
{
  "id": <experiment_number>,
  "timestamp": "<ISO 8601>",
  "phase": "EXPLORE|NARROW|REFINE",
  "temperature": <current T>,
  "hypothesis": "<what you expected and why>",
  "change_summary": "<concise description of code changes>",
  "change_category": "<architecture|optimizer|hyperparameter|training_loop|normalization|attention|other>",
  "result": "KEEP|REJECT|ANNEAL_ACCEPT|CRASH|CONSTRAINT_FAIL",
  "metrics": {
    "val_bpb": <float>,
    "peak_memory_mb": <int>,
    "throughput_tokens_per_sec": <float>,
    "parameter_count": <int>,
    "training_steps": <int>
  },
  "delta_vs_best": {
    "val_bpb": <float, negative=improvement>,
    "throughput_tokens_per_sec": <float, positive=improvement>
  },
  "insight": "<your best theory for WHY this result occurred>",
  "next_direction": "<what this suggests trying next>",
  "code_diff_summary": "<key lines changed, not full diff>"
}
```

### Before EVERY experiment:
1. Read the last 15 entries of experience.jsonl
2. Read the latest stage summary (if exists)
3. Read meta_strategy.md (if exists)
4. Identify EXHAUSTED directions (3+ consecutive failures in same category)
5. Identify PROMISING directions (success rate > 30% in category)
6. NEVER repeat an exact change that previously resulted in REJECT
7. Prioritize untested COMBINATIONS of individually successful changes

---

## Stage Summaries

Every **15 experiments**, write to `stage_summaries/stage_NNN.md`:

```markdown
# Stage Summary: Experiments [start] to [end]
Generated: [timestamp]

## Results
- Total experiments: N
- Kept: X (Y%)
- Anneal accepted: Z
- Crashes: W
- Best val_bpb this stage: [value] (vs stage start: [value], vs baseline: [value])

## Top successful directions
1. [category]: [description] — [X/Y kept, Z% success rate]
2. ...

## Exhausted directions (stop trying these)
1. [category]: [description] — [X consecutive failures]
2. ...

## Key insights
- [Most important finding from this stage]
- [Second most important]

## Current bottleneck hypothesis
[Your best theory on what's limiting further improvement]

## Recommended focus for next stage
[Specific directions to explore, based on evidence]
```

Read ALL previous stage summaries before starting a new stage.

---

## Safety Gates

### Per-experiment (automatic):
1. **Syntax check**: `python -c "import ast; ast.parse(open('train.py').read())"`
2. **Constraint pre-check**: code_lines <= 1200, no new imports
3. **Smoke test**: training must produce output within 60 seconds
4. **Constraint post-check**: all constraint metrics within bounds

### Every 10 experiments (automatic):
1. **Baseline regression test**: 
   - Stash current train.py
   - Restore original train.py from git
   - Run baseline, record metrics
   - Restore current best train.py
   - Compare: if current best val_bpb is within 0.5% of baseline, 
     log WARNING "Search may be stagnating — consider more aggressive exploration directions"
2. **Complexity audit**:
   - If cyclomatic complexity grew > 30% from baseline, log WARNING
   - If code_lines grew > 50% from baseline, log WARNING

### Conditional (human checkpoint):
- If `AUTONOMOUS=false` in environment: pause every 15 experiments for review
- If `AUTONOMOUS=true`: never pause, log everything, human reviews in morning

---

## Meta-Optimization Triggers

Trigger a meta-review if ANY of these conditions are met:

1. **Stuck**: 5+ consecutive REJECT results
2. **Low yield**: success rate < 10% over last 20 experiments
3. **Tunnel vision**: last 10 experiments all in same change_category
4. **Plateau**: val_bpb has not improved in last 15 experiments
5. **Diminishing returns**: last 5 KEEPs each improved val_bpb by less than 0.001

### Meta-review procedure:
```
1. STOP the inner loop
2. Read FULL experiment log (experience.jsonl)
3. Read ALL stage summaries
4. Identify the dominant search pattern and WHY it's failing
5. Generate a NEW search strategy that:
   - Is fundamentally different from current approach
   - Targets the identified bottleneck
   - Has clear success criteria
6. Write the strategy to meta_strategy.md
7. Reset temperature to T₀ × 0.5 (warm restart for exploration)
8. RESUME inner loop with new strategy
```

### Example meta-strategies the system might generate:
- **Combinatorial**: apply 2-3 proven changes simultaneously
- **Adversarial**: deliberately try the OPPOSITE of current best settings
- **Transfer**: search for techniques from recent ML papers cited in train.py
- **Ablation**: systematically REMOVE components to find what's unnecessary
- **Scale shift**: dramatically change model size (much bigger or smaller within time budget)

---

## Specific Guidance for nanochat/GPT Training

### Architecture search space
The model in train.py is a modern GPT-style transformer with:
- Muon + AdamW optimizer (this is already strong — don't change lightly)
- QK-Norm, RoPE positional encoding
- Value embeddings, banded attention

Key levers to explore (roughly ordered by expected impact):
1. **DEPTH and aspect ratio** — model depth vs width tradeoff within fixed time budget
2. **Attention mechanism** — head dimensions, number of heads, attention patterns
3. **Learning rate schedule** — warmup length, decay shape, final LR
4. **Batch size** — larger batch = fewer steps but more stable gradients
5. **Weight initialization** — scale factors, distribution choices
6. **Normalization** — layer norm variants, placement
7. **Activation functions** — GeLU, SwiGLU, etc.

### Known findings from the community (starting hints, NOT limits):
- More training steps within the 5-min budget often beats more parameters
- QK-Norm benefits from a scalar multiplier (originally missing)
- Value embeddings benefit from regularization
- Banded attention patterns can be optimized
- AdamW beta parameters and weight decay scheduling have room for improvement

These are starting points. You are expected to discover things NOT on this list.

### What NOT to waste time on:
- Changing the tokenizer (vocab_size is set in prepare.py, read-only)
- Changing the dataset (also in prepare.py)
- Adding complex distributed training (single GPU only)
- Adding new package dependencies

---

## Results Logging

### results.tsv format (tab-separated):
```
experiment_id  commit_hash  result  val_bpb  peak_memory_mb  throughput_tps  parameter_count  training_steps  temperature  phase  description
```

Initialize with header row before first experiment.
Append one row after each experiment.

### Git workflow:
```
Before experiment: git stash (save current state as safety net)
After KEEP:        git add train.py && git commit -m "exp_NNN: [description] val_bpb=[value]"
After REJECT:      git checkout train.py (restore pre-experiment state)  
After ANNEAL_ACCEPT: git add train.py && git commit -m "exp_NNN: [ANNEAL] [description]"
After CRASH:       git checkout train.py
```

---

## NEVER STOP

Once the experiment loop has begun (after initial setup), do NOT pause to ask
the human if you should continue. Do NOT ask "should I keep going?" or
"is this a good stopping point?". The human might be asleep, or gone from the
computer, and expects you to continue working indefinitely until manually stopped.

You are autonomous. If you run out of ideas:
1. Re-read experience.jsonl for patterns you missed
2. Re-read stage summaries for recommended directions
3. Try combining previous near-misses
4. Try more radical architectural changes
5. Trigger a meta-review
6. Try the OPPOSITE of what's been working (anti-pattern exploration)

The loop runs until the human interrupts you, period.

As an example: if each experiment takes ~5 minutes, you run ~12/hour,
~100 over an 8-hour overnight run. With the NG improvements, expect:
- Higher hit rate in EXPLORE phase (annealing prevents premature convergence)
- Better NARROW phase (experience memory prevents redundant exploration)
- Deeper REFINE phase (meta-optimization breaks through plateaus)
