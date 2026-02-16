---
name: functiongemma-trainer
description: Fine-tune FunctionGemma for on-device function calling using SFT on Hugging Face Jobs. Handles FunctionGemma-specific prompt formatting, CSV dataset validation, training on HF GPUs, evaluation of function-calling accuracy, and LiteRT-LM export for Android deployment. Use this skill when training FunctionGemma models for custom tool schemas.
---

# FunctionGemma Trainer

Fine-tune Google's FunctionGemma (270M) for custom on-device function calling. This skill handles the full pipeline: dataset validation, FunctionGemma prompt formatting, SFT training on HF Jobs, evaluation, and LiteRT-LM export for Android.

## Overview

FunctionGemma is a 270M parameter model built on Gemma 3, fine-tuned for function calling. It achieves 58% accuracy zero-shot and 85% after fine-tuning on domain-specific tools. This skill automates the fine-tuning process for your custom tool schemas.

**Key facts:**
- Base model: `google/functiongemma-270m-it`
- Size: 288 MB (dynamic int8 quantization)
- Latency: 0.3s time-to-first-token on Samsung S25 Ultra
- Fine-tuning is NOT optional — 58% base accuracy is not production-ready
- Export format: LiteRT-LM (`.litertlm`) for Android, not GGUF

## Prerequisites

- Hugging Face Pro account (for Jobs access)
- Write-access HF token: `hf auth login` or `export HF_TOKEN=hf_xxx`
- Accept the FunctionGemma license at https://huggingface.co/google/functiongemma-270m-it
- Dataset uploaded to HF Hub in the expected CSV format

## Dataset Format

The training dataset must be a CSV on HF Hub with exactly two columns:

```csv
user_message,tool_calls
"What's the weather?","[{""name"": ""get_weather"", ""args"": {""query"": ""current""}}]"
"Find a coffee shop","[{""name"": ""find_poi"", ""args"": {""query"": ""cafe""}}]"
```

**Rules:**
- `user_message`: Natural language input, quoted
- `tool_calls`: JSON array with exactly ONE tool call object containing `name` and `args`
- Each tool should have a single string parameter called `query` for maximum compatibility
- No empty tool calls (`[]`) — every row must map to a tool
- Recommended: 400-500 examples, ~10 per tool per variation type

### Validate a dataset before training

Use the validation script to check format before spending GPU time:

```bash
uv run scripts/validate_functiongemma_dataset.py \
  --dataset USERNAME/DATASET_NAME \
  --tools path/to/tools.json \
  --split train
```

## Tool Schema Format

Tools must follow this JSON format. Every tool takes a single `query` string parameter:

```json
[
  {
    "type": "function",
    "function": {
      "name": "tool_name",
      "description": "What this tool does and when to call it.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Parameter description with valid values"
          }
        },
        "required": ["query"]
      },
      "return": {
        "type": "string"
      }
    }
  }
]
```

## FunctionGemma Prompt Format

FunctionGemma uses a specific prompt structure during training and inference. The skill handles this automatically, but for reference:

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

The training script formats each CSV row into this structure automatically.

## Training

### Hardware Selection

FunctionGemma is 270M parameters. Use the cheapest GPU available:

| Hardware | Hourly Cost | Training Time (~500 examples) | Total Cost |
|----------|------------|-------------------------------|------------|
| t4-small | $0.40/hr | ~20-30 min | ~$0.15-0.20 |
| l4-small | $0.80/hr | ~10-15 min | ~$0.15-0.20 |

**Always use `t4-small` unless you have a reason not to.** FunctionGemma is tiny and trains fast.

### Run Training

```bash
hf jobs run \
  --flavor t4-small \
  --timeout 1h \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run scripts/train_functiongemma.py \
    --dataset USERNAME/DATASET_NAME \
    --tools path/to/tools.json \
    --output-repo USERNAME/MODEL_NAME \
    --epochs 3 \
    --lr 2e-4 \
    --batch-size 8
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | required | HF dataset repo with CSV |
| `--tools` | required | Path to tool schemas JSON |
| `--output-repo` | required | HF repo for the fine-tuned model |
| `--epochs` | 3 | Number of training epochs |
| `--lr` | 2e-4 | Learning rate |
| `--batch-size` | 8 | Training batch size |
| `--max-seq-length` | 512 | Max sequence length (512 is enough for function calling) |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha |
| `--test-split` | 0.1 | Fraction of data for evaluation |
| `--trackio-project` | none | Optional Trackio project name for monitoring |

### Quick Test Run

Always test before a full run:

```bash
hf jobs run \
  --flavor t4-small \
  --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run scripts/train_functiongemma.py \
    --dataset USERNAME/DATASET_NAME \
    --tools path/to/tools.json \
    --output-repo USERNAME/MODEL_NAME-test \
    --epochs 1 \
    --max-examples 50
```

### Monitor Training

Check job status:
```bash
hf jobs logs JOB_ID
```

If Trackio is configured, view real-time metrics at:
```
https://huggingface.co/spaces/USERNAME/trackio
```

## Evaluation

After training, evaluate function-calling accuracy:

```bash
uv run scripts/evaluate_functiongemma.py \
  --model USERNAME/MODEL_NAME \
  --dataset USERNAME/DATASET_NAME \
  --tools path/to/tools.json \
  --split test
```

This reports:
- **Tool selection accuracy**: Did the model pick the right tool?
- **Argument accuracy**: Did the model pass the correct query value?
- **Combined accuracy**: Both tool and argument correct
- **Per-tool breakdown**: Accuracy for each individual tool

Target: 85%+ combined accuracy after fine-tuning (vs 58% base).

## LiteRT-LM Export

Convert the fine-tuned model to `.litertlm` for Android deployment. This uses Google's `ai-edge-torch` library, which:
1. Merges LoRA adapters (if applicable)
2. Builds the PyTorch model via `gemma3.build_model_270m()`
3. Converts to `.litertlm` in a single call via `converter.convert_to_litert()`

The conversion writes a FunctionGemma-specific metadata textproto that configures stop tokens (`<end_of_turn>` and `<start_function_response>`) and model type (`function_gemma`).

**Important**: This requires `ai-edge-torch-nightly` and `ai-edge-litert-nightly`. The checkpoint directory must contain a `tokenizer.model` file (SentencePiece format).

```bash
hf jobs run \
  --flavor t4-small \
  --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run scripts/export_litertlm.py \
    --model USERNAME/MODEL_NAME \
    --output-repo USERNAME/MODEL_NAME-litertlm \
    --output-name-prefix cycling-copilot
```

The exported model can be loaded with the LiteRT-LM Android SDK or deployed via the Google AI Edge Gallery app.

### Export Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | required | HF repo of the fine-tuned model |
| `--output-repo` | required | HF repo for the LiteRT-LM export |
| `--output-name-prefix` | cycling-copilot | Prefix for output files |
| `--prefill-seq-len` | 256 | Prefill sequence length |
| `--kv-cache-max-len` | 1024 | KV cache max length (context window) |
| `--quantize` | dynamic_int8 | Quantization method |

## Full Pipeline Example

```bash
# 1. Validate dataset
uv run scripts/validate_functiongemma_dataset.py \
  --dataset monday8am/cycling-copilot-dataset \
  --tools tools.json

# 2. Quick test run
hf jobs run --flavor t4-small --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run scripts/train_functiongemma.py \
    --dataset monday8am/cycling-copilot-dataset \
    --tools tools.json \
    --output-repo monday8am/cycling-copilot-functiongemma-test \
    --epochs 1 --max-examples 50

# 3. Full training
hf jobs run --flavor t4-small --timeout 1h \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run scripts/train_functiongemma.py \
    --dataset monday8am/cycling-copilot-dataset \
    --tools tools.json \
    --output-repo monday8am/cycling-copilot-functiongemma \
    --epochs 3

# 4. Evaluate
uv run scripts/evaluate_functiongemma.py \
  --model monday8am/cycling-copilot-functiongemma \
  --dataset monday8am/cycling-copilot-dataset \
  --tools tools.json

# 5. Export for Android
hf jobs run --flavor t4-small --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  -- uv run scripts/export_litertlm.py \
    --model monday8am/cycling-copilot-functiongemma \
    --output-repo monday8am/cycling-copilot-functiongemma-litertlm \
    --output-name-prefix cycling-copilot
```

## Troubleshooting

**"Model not found" error**: Make sure you accepted the FunctionGemma license at https://huggingface.co/google/functiongemma-270m-it

**Low accuracy after training**: Check that your dataset has enough examples per tool (minimum 30-40 each). Verify tool descriptions are clear and non-overlapping. Try increasing epochs to 5.

**Out of memory**: Unlikely with 270M params on T4, but reduce batch size to 4 if it happens.

**LiteRT-LM export fails**: Ensure `ai-edge-torch-nightly` and `ai-edge-litert-nightly` are installed. Check that `tokenizer.model` exists in the checkpoint directory. The conversion requires a GPU runtime.

**Missing tokenizer.model**: FunctionGemma checkpoints from HF Hub should include `tokenizer.model`. If your fine-tuned model doesn't have it, copy it from the base `google/functiongemma-270m-it` checkpoint.

## References

- [FunctionGemma overview](https://ai.google.dev/gemma/docs/functiongemma)
- [FunctionGemma model card](https://ai.google.dev/gemma/docs/functiongemma/model_card)
- [Official fine-tuning + export notebook](https://github.com/google-gemini/gemma-cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb)
- [HF-to-MediaPipe conversion guide](https://ai.google.dev/gemma/docs/conversions/hf-to-mediapipe-task)
- [LiteRT-LM documentation](https://github.com/google-ai-edge/LiteRT-LM)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [HF Jobs documentation](https://huggingface.co/docs/hub/jobs-overview)
