# Contributing to AutoResearch-NG

Thank you for your interest in improving AutoResearch-NG!

## How to contribute

### Report results
The most valuable contribution right now is **running the NG loop and sharing your results**. Open an issue or discussion with:
- Hardware (GPU model, VRAM)
- Number of experiments run
- Original vs NG val_bpb comparison
- Which NG features worked well / didn't work
- Your experience.jsonl and stage summaries (if willing to share)

### Improve the framework
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-improvement`)
3. Make your changes
4. Test by running at least 20 experiments with the NG loop
5. Open a pull request with your results

### Areas we especially need help with
- **Annealing parameter calibration**: Running controlled experiments to find optimal T₀ and decay rate for different tasks
- **Meta-optimization strategies**: Improving the meta-review prompt and strategy generation
- **Domain adaptations**: Porting NG to non-ML tasks (frontend perf, SQL, prompt engineering) with working eval functions
- **Agent compatibility**: Testing with different coding agents (Claude Code, Codex, Cursor, etc.)

## Design principles
When contributing, keep these in mind:
- **No modifications to original autoresearch files**: NG works as a drop-in overlay, not a fork
- **Additive, not replacing**: prepare_ng.py supplements prepare.py, it doesn't replace it
- **Incremental adoption**: Each NG feature should work independently; users can adopt one improvement at a time
- **Measurable claims**: If you claim an improvement, include the experiment data
