# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cycling Copilot is an ML pipeline for fine-tuning **FunctionGemma** (Google's 270M parameter on-device function-calling model) as a tool-calling router for a cycling mobile app. The model receives natural language from a cyclist mid-ride, selects exactly ONE tool to call with the correct parameter, then a separate model (Gemma 3 1B locally or a remote LLM) generates the final response.

**Base model**: `google/functiongemma-270m-it` — 58% zero-shot accuracy, 70.1% after 3-epoch fine-tuning (target: 85%).

## Architecture

```
Voice/Chat Input → FunctionGemma 270M (tool selection) → Tool Execution → Gemma 1B (local) or Remote LLM → Output
```

FunctionGemma selects from 6 tools, each taking a single `query` string parameter:
- `get_ride_status` — speed, distance, elevation, battery, time
- `get_segment_ahead` — 5km, 10km, 20km, next_section
- `get_weather_forecast` — current, ahead, wind
- `find_nearby_poi` — water, cafe, food, bike_shop, rest_area, shelter, any
- `get_route_alternatives` — flatter, shorter, paved, scenic, sheltered, avoid_main_roads
- `get_rider_profile` — fitness, preferences, all

Tool schemas are in `cycling-copilot-tools.json`. Every tool call maps to exactly ONE tool with ONE `query` argument.

## Commands

All scripts use **uv** with inline PEP 723 dependencies (no requirements.txt). Run locally or on HF Jobs.

```bash
# Validate dataset before training (local CSV)
uv run validate_functiongemma_dataset.py --dataset cycling-copilot-dataset-merged.csv --tools cycling-copilot-tools.json

# Train on HF Jobs (production - uses dataset from HF Hub)
hf jobs run --flavor t4-small --timeout 1h \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run train_functiongemma.py \
    --dataset monday8am/cycling-copilot \
    --tools cycling-copilot-tools.json \
    --output-repo monday8am/cycling-copilot-functiongemma \
    --epochs 3

# Quick test run (50 examples, 1 epoch)
hf jobs run --flavor t4-small --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run train_functiongemma.py \
    --dataset monday8am/cycling-copilot \
    --tools cycling-copilot-tools.json \
    --output-repo monday8am/cycling-copilot-functiongemma-test \
    --epochs 1 --max-examples 50

# Evaluate
uv run evaluate_functiongemma.py \
  --model monday8am/cycling-copilot-functiongemma \
  --dataset monday8am/cycling-copilot \
  --tools cycling-copilot-tools.json

# Export to LiteRT-LM for Android (adds .litertlm to same repo)
hf jobs run --flavor t4-small --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run export_litertlm.py \
    --model monday8am/cycling-copilot-functiongemma

# Generate training data (HF Inference API)
uv run generate_dataset.py

# Generate training data (local Ollama)
uv run generate_dataset-local-ollama.py

# Merge and validate datasets
python3 merge_and_validate.py
```

## Key Files

| File | Purpose |
|------|---------|
| `train_functiongemma.py` | SFT training with LoRA (rank 16, alpha 32, max_length 1280, targets q/k/v/o_proj) |
| `TRAINING_RESULTS.md` | Complete training run results, metrics, and analysis |
| `evaluate_functiongemma.py` | Reports tool selection, argument, and combined accuracy per tool |
| `validate_functiongemma_dataset.py` | Checks CSV format, JSON validity, tool coverage before training |
| `export_litertlm.py` | Merges LoRA → builds PyTorch model → converts to `.litertlm` for Android |
| `generate_dataset.py` | Expands seed examples via HF Inference API (Llama 3 8B) |
| `generate_dataset-local-ollama.py` | Same expansion via local Ollama (qwen2.5:7b) |
| `cycling-copilot-tools.json` | OpenAI-compatible tool schema definitions (6 tools) |
| `cycling-copilot-seeds.csv` | 25 seed examples for dataset expansion |
| `dataset-expansion-prompt.md` | Prompt template for generating training data variations |
| `SKILL.md` | Full FunctionGemma trainer skill documentation |

## Dataset

**HF Hub**: https://huggingface.co/datasets/monday8am/cycling-copilot

The production dataset `cycling-copilot-dataset-merged.csv` contains:
- **942 validated examples** (all 6 tools covered)
- **Tool distribution**: get_ride_status (253), get_route_alternatives (185), find_nearby_poi (176), get_segment_ahead (171), get_weather_forecast (126), get_rider_profile (31)
- **Variations**: Spanish, voice-style commands, conversational, indirect intent, standard natural language

### Dataset Format

CSV with two columns: `user_message` and `tool_calls`. Every row maps to exactly ONE tool call.

```csv
user_message,tool_calls
"Where can I get water?","[{""name"": ""find_nearby_poi"", ""args"": {""query"": ""water""}}]"
```

## FunctionGemma Prompt Format

```
<start_of_turn>system
You are a helpful assistant with access to the following tools:
[TOOL_SCHEMAS_JSON]
When a tool is needed, respond ONLY with the tool call in this format:
[{"name": "function_name", "args": {"query": "value"}}]
<end_of_turn>
<start_of_turn>user
USER_MESSAGE
<end_of_turn>
<start_of_turn>model
TOOL_CALLS_JSON
<end_of_turn>
```

## Training Results (February 2026)

**Current Model**: https://huggingface.co/monday8am/cycling-copilot-functiongemma

### Metrics (3 epochs, 942 examples)
- **Combined Accuracy**: 70.1% (target: 85%)
- **Tool Selection**: 78.1%
- **Argument Accuracy**: 71.1%
- **Training Time**: ~21 minutes on t4-small (~$0.14)

### Per-Tool Performance
| Tool | Accuracy | Examples |
|------|----------|----------|
| get_segment_ahead | 83.0% | 171 |
| get_ride_status | 81.8% | 253 |
| find_nearby_poi | 64.8% | 176 |
| get_weather_forecast | 62.7% | 126 |
| get_rider_profile | 58.1% | 31 |
| get_route_alternatives | 54.1% | 185 |

**LiteRT-LM Export**: Available in the same repo
- File: `cycling-copilot_q8_ekv1024.litertlm` (284 MB)
- Download: https://huggingface.co/monday8am/cycling-copilot-functiongemma/resolve/main/cycling-copilot_q8_ekv1024.litertlm
- Ready for Android deployment

### Critical Fix: Prompt Truncation
The initial training used `max_length=512`, which truncated all training examples (tool schemas alone = ~825 tokens). This caused fake "100% training accuracy" but 0% eval accuracy. Fix: increased to `max_length=1280`.

See `TRAINING_RESULTS.md` for complete analysis and improvement recommendations.

---

## Making the SKILL Available in Claude Code

The `SKILL.md` file defines a Claude Code skill for training FunctionGemma. To make it available to Claude Code users:

### Option 1: HuggingFace Skills Collection (Recommended)

1. **Create a HuggingFace Space or Model repo** with the skill files:
   - Navigate to https://huggingface.co/new-space or https://huggingface.co/new
   - Choose "Skills" or "Model" type
   - Upload `SKILL.md` and all referenced scripts

2. **Structure for HF Skills**:
   ```
   monday8am/functiongemma-trainer/
   ├── README.md                   # Overview and usage
   ├── SKILL.md                    # Skill metadata (frontmatter + docs)
   ├── train_functiongemma.py
   ├── evaluate_functiongemma.py
   ├── validate_functiongemma_dataset.py
   ├── export_litertlm.py
   ├── generate_dataset.py
   ├── generate_dataset-local-ollama.py
   └── cycling-copilot-tools.json
   ```

3. **Users can install via Claude Code Settings**:
   - In Claude Code, go to Skills settings
   - Add skill from HuggingFace: `monday8am/functiongemma-trainer`
   - Or add via command: `/skill add monday8am/functiongemma-trainer`

### Option 2: Local Skill Installation

Users can install directly from the GitHub repository:

```bash
# In Claude Code
/skill add https://github.com/monday8am/riding-copilot-files
```

Or manually copy to Claude Code skills directory:
```bash
# Clone the repo
git clone https://github.com/monday8am/riding-copilot-files.git

# Copy to Claude Code skills directory
cp -r riding-copilot-files ~/.claude/skills/functiongemma-trainer
```

### Skill Invocation

Once installed, users invoke the skill in Claude Code:
```
/functiongemma-trainer <args>
```

The skill name is defined in the `SKILL.md` frontmatter:
```yaml
---
name: functiongemma-trainer
description: Fine-tune FunctionGemma for on-device function calling using SFT on Hugging Face Jobs...
---
```

Refer to `SKILL.md` for complete documentation on skill usage and parameters.

## Environment Variables

- `HF_TOKEN` — Hugging Face token (required for training/evaluation on Hub)
- `OLLAMA_REMOTE_HOST` — Ollama server IP for local dataset generation (default: `192.168.0.33`)

## Repository Structure

- **GitHub**: https://github.com/monday8am/riding-copilot-files
- **HF Dataset**: https://huggingface.co/datasets/monday8am/cycling-copilot
- **Model output**: `monday8am/cycling-copilot-functiongemma` (includes both LoRA adapter and .litertlm export)
