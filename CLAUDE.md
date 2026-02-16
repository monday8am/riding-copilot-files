# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cycling Copilot is an ML pipeline for fine-tuning **FunctionGemma** (Google's 270M parameter on-device function-calling model) as a tool-calling router for a cycling mobile app. The model receives natural language from a cyclist mid-ride, selects exactly ONE tool to call with the correct parameter, then a separate model (Gemma 3 1B locally or a remote LLM) generates the final response.

**Base model**: `google/functiongemma-270m-it` — 58% zero-shot accuracy, 85%+ after fine-tuning.

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
# Validate dataset before training
uv run validate_functiongemma_dataset.py --dataset USER/DATASET --tools cycling-copilot-tools.json

# Train (locally or via HF Jobs with --flavor t4-small)
uv run train_functiongemma.py --dataset USER/DATASET --tools cycling-copilot-tools.json --output-repo USER/MODEL --epochs 3

# Quick test run (fewer examples, 1 epoch)
uv run train_functiongemma.py --dataset USER/DATASET --tools cycling-copilot-tools.json --output-repo USER/MODEL-test --epochs 1 --max-examples 50

# Evaluate
uv run evaluate_functiongemma.py --model USER/MODEL --dataset USER/DATASET --tools cycling-copilot-tools.json

# Export to LiteRT-LM for Android
uv run export_litertlm.py --model USER/MODEL --output-repo USER/MODEL-litertlm

# Generate training data (HF Inference API)
uv run generate_dataset.py

# Generate training data (local Ollama)
uv run generate_dataset-local-ollama.py
```

For GPU training on HF Jobs, prefix with:
```bash
hf jobs run --flavor t4-small --timeout 1h --secrets HF_TOKEN=$HF_TOKEN -- uv run ...
```

## Key Files

| File | Purpose |
|------|---------|
| `train_functiongemma.py` | SFT training with LoRA (rank 16, alpha 32, targets q/k/v/o_proj) |
| `evaluate_functiongemma.py` | Reports tool selection, argument, and combined accuracy per tool |
| `validate_functiongemma_dataset.py` | Checks CSV format, JSON validity, tool coverage before training |
| `export_litertlm.py` | Merges LoRA → builds PyTorch model → converts to `.litertlm` for Android |
| `generate_dataset.py` | Expands seed examples via HF Inference API (Llama 3 8B) |
| `generate_dataset-local-ollama.py` | Same expansion via local Ollama (qwen2.5:7b) |
| `cycling-copilot-tools.json` | OpenAI-compatible tool schema definitions (6 tools) |
| `cycling-copilot-seeds.csv` | 25 seed examples for dataset expansion |
| `dataset-expansion-prompt.md` | Prompt template for generating training data variations |
| `SKILL.md` | Full FunctionGemma trainer skill documentation |

## Dataset Format

CSV with two columns: `user_message` and `tool_calls`. Every row maps to exactly ONE tool call.

```csv
user_message,tool_calls
"Where can I get water?","[{""name"": ""find_nearby_poi"", ""args"": {""query"": ""water""}}]"
```

Target: 400-500 examples, minimum 30 per tool. Five variation types: standard (40%), voice-style (15%), indirect intent (15%), Spanish (15%), conversational (15%).

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

## Environment Variables

- `HF_TOKEN` — Hugging Face token (required for training/evaluation on Hub)
- `OLLAMA_REMOTE_HOST` — Ollama server IP for local dataset generation (default: `192.168.0.33`)
