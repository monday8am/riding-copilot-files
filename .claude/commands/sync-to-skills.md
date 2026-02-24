---
description: Commit any pending changes in this repo, then sync the trainer scripts to the monday8am/skills public repo and push. Run this after modifying train/evaluate/validate/export scripts.
argument-hint: [optional commit message]
allowed-tools: Bash, Read, Glob
---

Sync this project's trainer scripts to the public `monday8am/skills` repo. Follow every step below in order.

## Context

Current state of this repo:
- Git status: !`git -C /Users/anton/Projects/riding-copilot status --short`
- Last commit: !`git -C /Users/anton/Projects/riding-copilot log --oneline -1`

Skills repo present: !`ls /Users/anton/Projects/skills 2>/dev/null && echo "YES — $(git -C /Users/anton/Projects/skills log --oneline -1)" || echo "NO — needs cloning"`

Files that will be synced (script diffs vs skills repo):
- train_functiongemma.py: !`diff /Users/anton/Projects/riding-copilot/train_functiongemma.py /Users/anton/Projects/skills/skills/functiongemma-trainer/scripts/train_functiongemma.py >/dev/null 2>&1 && echo "in sync" || echo "DIFFERS"`
- evaluate_functiongemma.py: !`diff /Users/anton/Projects/riding-copilot/evaluate_functiongemma.py /Users/anton/Projects/skills/skills/functiongemma-trainer/scripts/evaluate_functiongemma.py >/dev/null 2>&1 && echo "in sync" || echo "DIFFERS"`
- validate_functiongemma_dataset.py: !`diff /Users/anton/Projects/riding-copilot/validate_functiongemma_dataset.py /Users/anton/Projects/skills/skills/functiongemma-trainer/scripts/validate_functiongemma_dataset.py >/dev/null 2>&1 && echo "in sync" || echo "DIFFERS"`
- export_litertlm.py: !`diff /Users/anton/Projects/riding-copilot/export_litertlm.py /Users/anton/Projects/skills/skills/functiongemma-trainer/scripts/export_litertlm.py >/dev/null 2>&1 && echo "in sync" || echo "DIFFERS"`
- cycling-copilot-tools.json: !`diff /Users/anton/Projects/riding-copilot/cycling-copilot-tools.json /Users/anton/Projects/skills/skills/functiongemma-trainer/references/cycling-copilot-tools.json >/dev/null 2>&1 && echo "in sync" || echo "DIFFERS"`

---

## Step 1 — Commit this repo (riding-copilot)

Check the git status shown above. If there are uncommitted changes:
- Use the `commit-commands:commit` skill to create a commit.
- If `$ARGUMENTS` was provided, use it as context for the commit message.
- If the working tree is already clean, skip this step and say so.

## Step 2 — Ensure the skills repo is available

If the skills repo context above says **NO**, clone it:
```bash
git clone git@github.com:monday8am/skills.git /Users/anton/Projects/skills
```

Then verify the SSH remote is set (fix it if not):
```bash
git -C /Users/anton/Projects/skills remote set-url origin git@github.com:monday8am/skills.git
```

## Step 3 — Copy files to the skills repo

Copy all five files unconditionally (cp is idempotent):
```bash
cp /Users/anton/Projects/riding-copilot/train_functiongemma.py \
   /Users/anton/Projects/skills/skills/functiongemma-trainer/scripts/

cp /Users/anton/Projects/riding-copilot/evaluate_functiongemma.py \
   /Users/anton/Projects/skills/skills/functiongemma-trainer/scripts/

cp /Users/anton/Projects/riding-copilot/validate_functiongemma_dataset.py \
   /Users/anton/Projects/skills/skills/functiongemma-trainer/scripts/

cp /Users/anton/Projects/riding-copilot/export_litertlm.py \
   /Users/anton/Projects/skills/skills/functiongemma-trainer/scripts/

cp /Users/anton/Projects/riding-copilot/cycling-copilot-tools.json \
   /Users/anton/Projects/skills/skills/functiongemma-trainer/references/
```

## Step 4 — Commit and push the skills repo

Check for changes:
```bash
git -C /Users/anton/Projects/skills status --short
```

If there are changes:
1. Stage the skill files: `git -C /Users/anton/Projects/skills add skills/functiongemma-trainer/`
2. Write a concise commit message that lists which files changed and why (infer from the diff context above or from `$ARGUMENTS` if provided).
3. Commit and push:
```bash
git -C /Users/anton/Projects/skills commit -m "<message>"
git -C /Users/anton/Projects/skills push
```

If there are no changes (all files were already in sync), say so — no commit needed.

## Step 5 — Report

Summarise what was done:
- Files synced (or already in sync)
- Commit hash + message for this repo (if committed)
- Commit hash + message for skills repo (if committed)
- Link: https://github.com/monday8am/skills/tree/main/skills/functiongemma-trainer/scripts
