#!/bin/bash
# setup.sh — One-time setup, then launch with 'claude'
set -e

echo "AutoResearch-NG setup"
echo ""

# Copy baseline as the starting train.py
if [ ! -f "train.py" ]; then
    echo "Creating train.py from train_baseline.py..."
    cp train_baseline.py train.py
fi

# Data prep
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
uv run prepare_ng.py

echo ""
echo "Setup complete. Start the loop with:"
echo "  claude"
