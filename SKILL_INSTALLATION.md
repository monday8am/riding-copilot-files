# Making the FunctionGemma Trainer Skill Available

This guide explains how to make the FunctionGemma trainer skill available for Claude Code users.

## Current Status

✅ **Repository**: https://github.com/monday8am/riding-copilot-files
✅ **Dataset**: https://huggingface.co/datasets/monday8am/cycling-copilot
✅ **Skill Definition**: `SKILL.md` with proper frontmatter
📦 **Next Step**: Publish to HuggingFace Hub as a skill

## Skill Metadata

The skill is defined in `SKILL.md` with the following frontmatter:

```yaml
---
name: functiongemma-trainer
description: Fine-tune FunctionGemma for on-device function calling using SFT on Hugging Face Jobs. Handles FunctionGemma-specific prompt formatting, CSV dataset validation, training on HF GPUs, evaluation of function-calling accuracy, and LiteRT-LM export for Android deployment. Use this skill when training FunctionGemma models for custom tool schemas.
---
```

## Option 1: Publish to HuggingFace Hub (Recommended)

### Step 1: Create a HuggingFace Skills Collection

1. **Navigate to HuggingFace Hub**: https://huggingface.co/new-space
2. **Choose Space type**: Select "Gradio" or "Static" (Static is simpler for skill hosting)
3. **Repository details**:
   - **Owner**: `monday8am`
   - **Space name**: `functiongemma-trainer-skill`
   - **Visibility**: Public
   - **License**: MIT

### Step 2: Upload Skill Files

Upload these files to the HF Space:

**Required files**:
- `SKILL.md` - Skill definition and documentation
- `README.md` - Overview and quick start
- `train_functiongemma.py`
- `evaluate_functiongemma.py`
- `validate_functiongemma_dataset.py`
- `export_litertlm.py`
- `generate_dataset.py`
- `generate_dataset-local-ollama.py`
- `cycling-copilot-tools.json`
- `dataset-expansion-prompt.md`

**Optional files**:
- `merge_and_validate.py`
- `list_models.py`
- Architecture diagrams (`.mermaid` files)

### Step 3: Add README.md

Create a `README.md` in the Space with:

```markdown
---
title: FunctionGemma Trainer Skill
emoji: 🚴
colorFrom: blue
colorTo: green
sdk: static
pinned: false
tags:
  - claude-code
  - skill
  - functiongemma
  - huggingface
  - on-device-ai
---

# FunctionGemma Trainer Skill

Fine-tune Google's FunctionGemma (270M) for custom on-device function calling.

## Installation

In Claude Code, add this skill:

```
/skill add monday8am/functiongemma-trainer-skill
```

## Usage

See [SKILL.md](./SKILL.md) for complete documentation.

## Links

- **Dataset**: https://huggingface.co/datasets/monday8am/cycling-copilot
- **GitHub**: https://github.com/monday8am/riding-copilot-files
```

### Step 4: Tag and Publish

1. Add tags in Space settings: `claude-code`, `skill`, `functiongemma`
2. Make the Space public
3. Share the URL: `https://huggingface.co/spaces/monday8am/functiongemma-trainer-skill`

### Step 5: Users Install the Skill

Users can now install via Claude Code:

```
/skill add monday8am/functiongemma-trainer-skill
```

Or in Claude Code settings:
1. Open Skills panel
2. Click "Add Skill"
3. Enter: `monday8am/functiongemma-trainer-skill`

## Option 2: Direct GitHub Installation

Users can install directly from GitHub (simpler but less discoverable):

```bash
# In Claude Code
/skill add https://github.com/monday8am/riding-copilot-files
```

## Option 3: Submit to HuggingFace Skills Plugin Registry

Once the skill is tested and stable, submit it to the official Claude Code skills registry:

1. **Create a PR** to https://github.com/huggingface/huggingface-skills
2. **Add your skill** to the registry with metadata:
   ```yaml
   - name: functiongemma-trainer
     repo: monday8am/functiongemma-trainer-skill
     description: Fine-tune FunctionGemma for on-device function calling
     tags: [ml, training, on-device, android]
     author: monday8am
   ```

3. Once merged, the skill becomes available in the official catalog

## Testing the Skill

Before publishing, test locally:

1. **Copy skill to Claude Code directory**:
   ```bash
   mkdir -p ~/.claude/skills/functiongemma-trainer
   cp SKILL.md *.py *.json ~/.claude/skills/functiongemma-trainer/
   ```

2. **Invoke in Claude Code**:
   ```
   /functiongemma-trainer
   ```

3. **Verify the skill loads** and shows documentation

## Skill Commands

Once installed, users can invoke:

```
/functiongemma-trainer
```

This will show the skill documentation from `SKILL.md` and allow users to request training, evaluation, validation, or export operations.

## Next Steps

1. ✅ Skill metadata defined in SKILL.md
2. ✅ All scripts use uv with inline dependencies
3. ✅ Dataset published to HF Hub
4. ✅ GitHub repository created
5. 🔄 **TODO**: Create HF Space for skill distribution
6. 🔄 **TODO**: Test skill installation in Claude Code
7. 🔄 **TODO**: Submit to HuggingFace Skills registry

## Support

- **Issues**: https://github.com/monday8am/riding-copilot-files/issues
- **Dataset**: https://huggingface.co/datasets/monday8am/cycling-copilot
- **Documentation**: See [SKILL.md](./SKILL.md)
