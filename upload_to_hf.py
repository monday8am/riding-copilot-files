#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub",
# ]
# ///
"""
Upload cycling copilot dataset to HF Hub.
Replaces function_gemma_train_safety_agent.csv with cycling-copilot-dataset-merged.csv
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("Error: huggingface_hub not found")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


def main():
    # Configuration
    repo_id = "monday8am/cycling-copilot"
    csv_file = "cycling-copilot-dataset-merged.csv"
    old_file = "function_gemma_train_safety_agent.csv"

    # Check file exists
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found")
        sys.exit(1)

    # Authenticate
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        print("Please set it with: export HF_TOKEN=hf_...")
        print("Or login with: huggingface-cli login")
        sys.exit(1)

    try:
        login(token=token)
        print(f"✓ Authenticated to Hugging Face")
    except Exception as e:
        print(f"Error authenticating: {e}")
        sys.exit(1)

    api = HfApi()

    # Check if repo exists
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"✓ Found repository: {repo_id}")
    except Exception as e:
        print(f"Error: Repository not found or not accessible: {e}")
        sys.exit(1)

    # Delete old file if it exists
    try:
        print(f"\nDeleting old file: {old_file}...")
        api.delete_file(
            path_in_repo=old_file,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"✓ Deleted {old_file}")
    except Exception as e:
        print(f"  (Note: {old_file} may not exist, skipping deletion)")

    # Upload new file
    try:
        print(f"\nUploading: {csv_file}...")
        file_size_mb = Path(csv_file).stat().st_size / (1024 ** 2)
        print(f"  File size: {file_size_mb:.2f} MB")

        api.upload_file(
            path_or_fileobj=csv_file,
            path_in_repo=csv_file,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Upload merged dataset with 942 validated examples\n\n"
                          f"Replaces {old_file} with deduplicated, validated dataset.\n"
                          f"All 6 tools covered with minimum 30 examples each.\n"
                          f"Ready for FunctionGemma fine-tuning.",
        )
        print(f"✓ Uploaded {csv_file}")
    except Exception as e:
        print(f"Error uploading file: {e}")
        sys.exit(1)

    # Success
    print(f"\n{'='*60}")
    print("SUCCESS!")
    print(f"{'='*60}")
    print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
    print(f"File: {csv_file}")
    print(f"Examples: 942")
    print(f"\nYou can now use it in training with:")
    print(f"  --dataset {repo_id}")


if __name__ == "__main__":
    main()
