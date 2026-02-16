# FunctionGemma Training Results

## Training Pipeline Execution

**Date**: February 16, 2026
**Base Model**: `google/functiongemma-270m-it`
**Dataset**: https://huggingface.co/datasets/monday8am/cycling-copilot (942 examples)

---

## Key Issue Discovered & Fixed

### Root Cause: Prompt Truncation
The initial training with `max_length=512` was catastrophically truncating all training examples:
- Tool schemas JSON: ~825 tokens
- Full prompt (system + tools + user + response): ~925+ tokens
- At `max_length=512`, the model response was always cut off
- This caused fake "100% training accuracy" (model only learned the system prompt prefix) but 0% eval accuracy

### Solution
Changed training parameters:
- `max_length`: 512 → **1280**
- `batch_size`: 2 → **1** (to fit longer sequences in GPU memory)
- `gradient_accumulation_steps`: 4 → **8** (to maintain effective batch size)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Learning Rate | 2e-4 |
| Batch Size | 1 |
| Gradient Accumulation | 8 steps |
| Max Length | 1280 tokens |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Hardware | t4-small (HF Jobs) |
| Training Time | ~21 minutes |

---

## Training Metrics

| Metric | Value |
|--------|-------|
| Training Loss | 0.153 |
| Eval Loss | 0.024 |
| Eval Token Accuracy | 99.46% |
| Training Examples | 847 |
| Eval Examples | 95 |

**HF Hub**: https://huggingface.co/monday8am/cycling-copilot-functiongemma

---

## Evaluation Results

**Job ID**: `699333e3c97e6acf8876accd`
**Test Set**: 942 examples (full dataset)

### Overall Accuracy

| Metric | Score | Notes |
|--------|-------|-------|
| **Combined Accuracy** | **70.1%** | Tool + arguments both correct |
| Tool Selection | 78.1% | Correct tool chosen |
| Argument Accuracy | 71.1% | Correct `query` parameter |
| Parse Failures | 101/942 | Model outputs empty `<end_of_turn>` |

**Target**: 85% (NOT MET)

### Per-Tool Breakdown

| Tool | Accuracy | Examples | Notes |
|------|----------|----------|-------|
| `get_segment_ahead` | 83.0% | 171 | ✅ Best performer |
| `get_ride_status` | 81.8% | 253 | ✅ Strong |
| `find_nearby_poi` | 64.8% | 176 | ⚠️ Below target |
| `get_weather_forecast` | 62.7% | 126 | ⚠️ Below target |
| `get_rider_profile` | 58.1% | 31 | ❌ Weakest (low data) |
| `get_route_alternatives` | 54.1% | 185 | ❌ Weakest |

---

## LiteRT-LM Export

**Job ID**: `699338ebc97e6acf8876acd1`
**Export Time**: ~12 minutes

### Output
- **File**: `cycling-copilot_q8_ekv1024.litertlm`
- **Size**: 284 MB (dynamic int8 quantization)
- **KV Cache**: 1024 tokens
- **Prefill Length**: 256 tokens
- **Download**: https://huggingface.co/monday8am/cycling-copilot-functiongemma/resolve/main/cycling-copilot_q8_ekv1024.litertlm

### Export Configuration
```python
prefill_seq_len = 256
kv_cache_max_len = 1024
quantize = "dynamic_int8"
output_format = "litertlm"
```

### FunctionGemma Metadata
```protobuf
start_token: { token_ids: { ids: [ 2 ] } }
stop_tokens: { token_str: "<end_of_turn>" }
stop_tokens: { token_str: "<start_function_response>" }
llm_model_type: { function_gemma: {} }
```

---

## Model Behavior Analysis

### What Changed After Fix
- **Before fix** (max_length=512): 0% accuracy, generated natural language responses
- **After fix** (max_length=1280): 70.1% accuracy, generates structured tool calls

### Example Outputs

#### ✅ Successful Tool Calls
```
Input: "What's my current speed?"
Output: [{"name": "get_ride_status", "args": {"query": "speed"}}]
```

```
Input: "Where can I get water?"
Output: [{"name": "find_nearby_poi", "args": {"query": "water"}}]
```

#### ❌ Parse Failures (101 cases)
```
Input: "All status?"
Output: <end_of_turn>
```

```
Input: "Next segment?"
Output: <end_of_turn>
```

---

## Improvement Recommendations

### To Reach 85% Target

1. **Increase Training Epochs**
   - Current: 3 epochs → Try 5-10 epochs
   - Expected gain: +5-10% accuracy

2. **Augment Weak Tools**
   - `get_rider_profile`: Only 31 examples → Add 100+ more
   - `get_route_alternatives`: 185 examples but 54.1% accuracy → Review data quality
   - Use the dataset expansion prompts to generate more variations

3. **Increase LoRA Rank** (if overfitting isn't an issue)
   - Current: rank=16 → Try rank=32 or 64
   - More parameters = better capacity for tool-specific patterns

4. **Learning Rate Schedule**
   - Current: Cosine with warmup_ratio=0.1
   - Try lower LR (1e-4) for more epochs to fine-tune

5. **Fix Parse Failures**
   - 101 cases where model outputs only `<end_of_turn>`
   - May need to adjust generation params (temperature, top_p) or add more training data with very short queries

---

## Repository Structure

```
riding-copilot/
├── train_functiongemma.py           # Training script (fixed max_length=1280)
├── evaluate_functiongemma.py         # Evaluation script (fixed tokenizer loading)
├── export_litertlm.py                # LiteRT-LM export script
├── validate_functiongemma_dataset.py # Dataset validation
├── generate_dataset.py               # Dataset expansion (HF API)
├── generate_dataset-local-ollama.py  # Dataset expansion (local)
├── cycling-copilot-tools.json        # Tool schemas (6 tools)
├── cycling-copilot-seeds.csv         # 25 seed examples
├── cycling-copilot-dataset-merged.csv # 942 examples (production dataset)
├── CLAUDE.md                         # Project instructions
├── SKILL.md                          # FunctionGemma trainer skill
└── TRAINING_RESULTS.md               # This file
```

---

## Commands Reference

### Train (Production)
```bash
hf jobs uv run \
  --flavor t4-small \
  --timeout 1h \
  --secrets HF_TOKEN=$HF_TOKEN \
  train_functiongemma.py \
  -- \
    --dataset monday8am/cycling-copilot \
    --tools cycling-copilot-tools.json \
    --output-repo monday8am/cycling-copilot-functiongemma \
    --epochs 3
```

### Evaluate
```bash
uv run evaluate_functiongemma.py \
  --model monday8am/cycling-copilot-functiongemma \
  --dataset monday8am/cycling-copilot \
  --tools cycling-copilot-tools.json \
  --split train
```

### Export to LiteRT-LM
```bash
hf jobs uv run \
  --flavor t4-small \
  --timeout 30m \
  --secrets HF_TOKEN=$HF_TOKEN \
  export_litertlm.py \
  -- \
    --model monday8am/cycling-copilot-functiongemma \
    --output-repo monday8am/cycling-copilot-functiongemma
```

---

## Next Steps

1. **Deploy to Android** using the exported `.litertlm` file
2. **Collect real-world usage data** to identify failure modes
3. **Iterate on training** with more epochs or augmented data to reach 85%+
4. **A/B test** against base model (58% zero-shot) to validate improvement

---

## Cost Summary

| Task | Hardware | Time | Cost |
|------|----------|------|------|
| Training (3 epochs, 942 examples) | t4-small | 21 min | ~$0.14 |
| Evaluation (942 examples) | t4-small | 5 min | ~$0.03 |
| LiteRT-LM Export | t4-small | 12 min | ~$0.08 |
| **Total** | | **38 min** | **~$0.25** |

---

## Technical Notes

### Critical Fixes Applied
1. **Training**: `max_length=512` → `1280` (prevents truncation)
2. **Evaluation**: Load tokenizer from base model to avoid embedding resize
3. **Model Loading**: Use `PeftModel.from_pretrained()` + `merge_and_unload()` instead of `AutoPeftModelForCausalLM`

### Lessons Learned
- Always validate prompt token length against `max_length` before training
- "100% training accuracy" with 0% eval accuracy = likely data truncation
- FunctionGemma tool schemas can be large (~825 tokens for 6 tools)
- GitHub CDN caches raw URLs — use commit-pinned URLs for HF Jobs
- cpu-basic is too slow for eval (30+ min) — use t4-small instead (~5 min)

---

Generated: 2026-02-16
