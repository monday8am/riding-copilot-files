# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=4.57.0",
#   "trl>=0.25.0",
#   "datasets",
#   "peft",
#   "bitsandbytes",
#   "huggingface_hub",
#   "trackio",
#   "pandas",
# ]
# ///
"""
FunctionGemma Fine-Tuning Script
================================
Fine-tunes google/functiongemma-270m-it on a CSV dataset of
user_message -> tool_calls mappings using SFT with LoRA.

Designed to run as a HF Job via:
  hf jobs run --flavor t4-small -- uv run train_functiongemma.py ...
"""

import argparse
import json
import os
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune FunctionGemma")
    parser.add_argument("--dataset", required=True, help="HF dataset repo (CSV format)")
    parser.add_argument("--tools", required=True, help="Path to tool schemas JSON file")
    parser.add_argument("--output-repo", required=True, help="HF repo for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples for test runs")
    parser.add_argument("--trackio-project", type=str, default=None)
    return parser.parse_args()


def load_tools(tools_path: str) -> str:
    """Load tool schemas and return as formatted JSON string."""
    with open(tools_path, "r") as f:
        tools = json.load(f)
    return json.dumps(tools, indent=2)


def format_functiongemma_prompt(user_message: str, tool_calls: str, tools_json: str) -> str:
    """Format a single example into FunctionGemma's expected prompt structure."""
    return (
        f"<start_of_turn>system\n"
        f"You are a helpful assistant with access to the following tools:\n\n"
        f"{tools_json}\n\n"
        f"When a tool is needed, respond ONLY with the tool call in this format:\n"
        f'[{{"name": "function_name", "args": {{"query": "value"}}}}]\n'
        f"<end_of_turn>\n"
        f"<start_of_turn>user\n"
        f"{user_message}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{tool_calls}\n"
        f"<end_of_turn>"
    )


def prepare_dataset(dataset_repo: str, tools_json: str, test_split: float, max_examples: int = None):
    """Load CSV dataset and format into FunctionGemma prompts."""
    ds = load_dataset(dataset_repo, split="train")

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    formatted = []
    for row in ds:
        text = format_functiongemma_prompt(
            user_message=row["user_message"],
            tool_calls=row["tool_calls"],
            tools_json=tools_json,
        )
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)

    if test_split > 0:
        split = dataset.train_test_split(test_size=test_split, seed=42)
        return split["train"], split["test"]
    return dataset, None


def main():
    args = parse_args()

    # Authenticate
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load tools
    tools_json = load_tools(args.tools)
    print(f"Loaded tool schemas from {args.tools}")

    # Prepare dataset
    print(f"Loading dataset from {args.dataset}...")
    train_dataset, eval_dataset = prepare_dataset(
        args.dataset, tools_json, args.test_split, args.max_examples
    )
    print(f"Training examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation examples: {len(eval_dataset)}")

    # Load model and tokenizer
    model_name = "google/functiongemma-270m-it"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Training config
    training_args = SFTConfig(
        output_dir="/tmp/functiongemma-output",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        max_seq_length=args.max_seq_length,
        push_to_hub=True,
        hub_model_id=args.output_repo,
        hub_token=hf_token,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # Trackio integration
    if args.trackio_project:
        try:
            import trackio
            trackio.init(project=args.trackio_project)
            training_args.report_to = "trackio"
            print(f"Trackio enabled: {args.trackio_project}")
        except ImportError:
            print("Trackio not available, skipping monitoring")

    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=lora_config,
    )

    print("Starting training...")
    trainer.train()

    # Save and push
    print(f"Pushing model to {args.output_repo}...")
    trainer.push_to_hub()

    print("Training complete!")


if __name__ == "__main__":
    main()
