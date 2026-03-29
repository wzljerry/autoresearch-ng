# CLAUDE.md

Read `program.md` for your full instructions. It contains objectives, constraints,
annealing schedule, experience memory rules, safety gates, and meta-optimization triggers.

`prepare_ng.py` has utility functions you can use: `collect_all_metrics()`,
`annealing_decision()`, `append_experience()`, `check_constraints()`,
`update_pareto_front()`, `should_trigger_meta_review()`, `generate_stage_summary()`.
Import with `from prepare_ng import *`. These are optional helpers — you can
implement the same logic inline if you prefer.

Start by reading `program.md` completely, then begin the experiment loop.
