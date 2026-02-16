#!/bin/bash
# Quick test training with Trackio monitoring

set -e

echo "Starting FunctionGemma quick test training with Trackio..."
echo "Dataset: monday8am/cycling-copilot"
echo "Examples: 50"
echo "Epochs: 1"
echo "Hardware: t4-small (~$0.05, ~5 minutes)"
echo ""

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set"
    echo "Please run: export HF_TOKEN=hf_..."
    exit 1
fi

# Run training job
hf jobs uv run \
  --flavor t4-small \
  --timeout 30m \
  --secrets HF_TOKEN \
  train_functiongemma.py \
  -- \
    --dataset monday8am/cycling-copilot \
    --tools-url "https://raw.githubusercontent.com/monday8am/riding-copilot-files/main/cycling-copilot-tools.json" \
    --output-repo monday8am/cycling-copilot-functiongemma-test \
    --epochs 1 \
    --max-examples 50 \
    --trackio-project cycling-copilot-test

echo ""
echo "Training job submitted!"
echo "Monitor at: https://huggingface.co/spaces/monday8am/trackio"
echo "Or check logs with: hf jobs logs <JOB_ID>"
