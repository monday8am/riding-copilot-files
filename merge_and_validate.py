#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""
Merge and validate all cycling copilot CSV datasets.
"""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

def load_tools(tools_path):
    """Load tool schemas."""
    with open(tools_path) as f:
        tools = json.load(f)
    return {t["function"]["name"] for t in tools}

def validate_row(row_num, row, valid_tools):
    """Validate a single CSV row."""
    errors = []

    # Check user_message
    if not row.get("user_message", "").strip():
        errors.append(f"Row {row_num}: Empty user_message")
        return errors, None, None

    # Check tool_calls JSON
    try:
        calls = json.loads(row["tool_calls"])
    except json.JSONDecodeError as e:
        errors.append(f"Row {row_num}: Invalid JSON: {e}")
        return errors, None, None

    # Check array format
    if not isinstance(calls, list) or len(calls) != 1:
        errors.append(f"Row {row_num}: Expected exactly 1 tool call, got {len(calls) if isinstance(calls, list) else 'non-list'}")
        return errors, None, None

    call = calls[0]

    # Check structure
    if "name" not in call or "args" not in call:
        errors.append(f"Row {row_num}: Missing 'name' or 'args' field")
        return errors, None, None

    # Check tool name
    tool_name = call["name"]
    if tool_name not in valid_tools:
        errors.append(f"Row {row_num}: Unknown tool '{tool_name}'")
        return errors, None, None

    # Check args
    if "query" not in call["args"]:
        errors.append(f"Row {row_num}: Missing 'query' in args")
        return errors, None, None

    return errors, row["user_message"], tool_name

def process_csv(csv_path, valid_tools):
    """Process a single CSV file."""
    print(f"\n{'='*60}")
    print(f"Processing: {csv_path.name}")
    print(f"{'='*60}")

    rows = []
    errors = []
    tool_counts = defaultdict(int)

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Check columns
            if "user_message" not in reader.fieldnames or "tool_calls" not in reader.fieldnames:
                print(f"FAIL: Missing required columns. Got: {reader.fieldnames}")
                return rows, errors, tool_counts

            for i, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                row_errors, message, tool = validate_row(i, row, valid_tools)
                errors.extend(row_errors)

                if not row_errors:
                    rows.append(row)
                    tool_counts[tool] += 1

    except Exception as e:
        print(f"ERROR reading file: {e}")
        return rows, errors, tool_counts

    # Report
    print(f"Total rows: {len(rows)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nFirst 5 errors:")
        for e in errors[:5]:
            print(f"  {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    print(f"\nTool distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100 if len(rows) > 0 else 0
        bar = "#" * min(50, int(pct))
        print(f"  {tool:30s} {count:4d} ({pct:5.1f}%) {bar}")

    return rows, errors, tool_counts

def main():
    # Load tools
    tools_path = Path("cycling-copilot-tools.json")
    valid_tools = load_tools(tools_path)
    print(f"Valid tools: {', '.join(sorted(valid_tools))}")

    # CSV files to process
    csv_files = [
        "cycling-copilot-seeds.csv",
        "cycling-copilot-dataset-expanded.csv",
        "cycling-copilot-dataset-hf.csv",
        "cycling-copilot-dataset-expanded_old.csv"
    ]

    all_rows = []
    all_errors = []
    total_tool_counts = defaultdict(int)
    seen_messages = set()
    duplicates = 0

    # Process each file
    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"\nWARNING: {csv_file} not found, skipping...")
            continue

        rows, errors, tool_counts = process_csv(csv_path, valid_tools)
        all_errors.extend(errors)

        # Deduplicate by user_message
        for row in rows:
            msg = row["user_message"].strip().lower()
            if msg not in seen_messages:
                seen_messages.add(msg)
                all_rows.append(row)

                # Parse tool name for stats
                try:
                    calls = json.loads(row["tool_calls"])
                    tool_name = calls[0]["name"]
                    total_tool_counts[tool_name] += 1
                except Exception:
                    pass
            else:
                duplicates += 1

    # Final report
    print(f"\n{'='*60}")
    print("MERGED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total unique entries: {len(all_rows)}")
    print(f"Duplicates removed: {duplicates}")
    print(f"Total errors: {len(all_errors)}")

    print(f"\nFinal tool distribution:")
    for tool, count in sorted(total_tool_counts.items()):
        pct = count / len(all_rows) * 100 if len(all_rows) > 0 else 0
        bar = "#" * min(50, int(pct / 2))
        print(f"  {tool:30s} {count:4d} ({pct:5.1f}%) {bar}")

    # Check minimum coverage
    low_count_tools = [t for t, c in total_tool_counts.items() if c < 30]
    if low_count_tools:
        print(f"\nWARNING: Tools with <30 examples (may underfit):")
        for t in low_count_tools:
            print(f"  {t}: {total_tool_counts[t]} examples")

    # Write merged file
    if all_rows:
        output_path = Path("cycling-copilot-dataset-merged.csv")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["user_message", "tool_calls"], quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"\n✓ Merged dataset saved to: {output_path}")
        print(f"  Ready for training with {len(all_rows)} unique examples")

    if all_errors:
        print(f"\n✗ Dataset has {len(all_errors)} errors")
        return 1
    else:
        print(f"\n✓ Dataset validation passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
