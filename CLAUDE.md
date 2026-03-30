# CLAUDE.md

Read `program.md` for your full instructions. It contains objectives, constraints,
annealing schedule, experience memory rules, safety gates, and meta-optimization triggers.

`prepare_ng.py` has utility functions you can use: `collect_all_metrics()`,
`annealing_decision()`, `append_experience()`, `check_constraints()`,
`update_pareto_front()`, `should_trigger_meta_review()`, `generate_stage_summary()`.
Import with `from prepare_ng import *`. These are optional helpers — you can
implement the same logic inline if you prefer.

`prepare.py` provides the data pipeline: `Tokenizer`, `make_dataloader()`,
`evaluate_bpb()`. Do NOT modify this file.

`train_baseline.py` is the reference baseline. Do NOT modify it.
`train.py` is your working copy — this is the ONLY file you edit.

Start by reading `program.md` completely, then begin the experiment loop.
