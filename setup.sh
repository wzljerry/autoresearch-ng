#!/bin/bash
# setup.sh — One-time setup, then launch with 'claude'
set -e

echo "AutoResearch-NG setup"
echo ""

if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found. Run this inside the autoresearch directory."
    exit 1
fi

# Data prep (same as original)
if [ ! -d "$HOME/.cache/autoresearch" ]; then
    echo "Preparing data..."
    uv run prepare.py
fi

# Initialize NG files
mkdir -p stage_summaries
touch experience.jsonl
echo '[]' > pareto_front.json

# Run baseline and save metrics
echo "Running baseline..."
uv run train.py 2>&1 | tee baseline_output.log
python prepare_ng.py  # saves baseline_metrics.json from last run

echo ""
echo "Setup complete. Start the loop with:"
echo "  claude"
