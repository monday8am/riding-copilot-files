# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai",
# ]
# ///
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' library not found.")
    print("Please install it using: pip install openai")
    sys.exit(1)

# --- Configuration ---

OUTPUT_FILE = "cycling-copilot-dataset-expanded.csv"

# Ollama Configuration
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_HOST = os.environ.get("OLLAMA_REMOTE_HOST", "192.168.0.33")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:11434/v1"

# Context for the model
TOOLS_CONTEXT = """
You are generating dataset entries for a cycling copilot.
The available tools are:
1. get_ride_status(query): [all, speed, distance, elevation, battery, time]
2. get_segment_ahead(query): [5km, 10km, 20km, next_section]
3. get_weather_forecast(query): [current, ahead, wind]
4. find_nearby_poi(query): [water, cafe, food, bike_shop, rest_area, shelter, any]
5. get_route_alternatives(query): [flatter, shorter, paved, scenic, sheltered, avoid_main_roads]
6. get_rider_profile(query): [fitness, preferences, all]
"""

SEEDS = [
    ("How am I doing?", "get_ride_status", "all"),
    ("What's my current speed?", "get_ride_status", "speed"),
    ("How far have I gone?", "get_ride_status", "distance"),
    ("How much battery do I have left?", "get_ride_status", "battery"),
    ("What's the elevation here?", "get_ride_status", "elevation"),
    ("How long have I been riding?", "get_ride_status", "time"),
    ("What's ahead in the next 5 kilometers?", "get_segment_ahead", "5km"),
    ("Are there any big climbs coming up?", "get_segment_ahead", "next_section"),
    ("How hard is the next section?", "get_segment_ahead", "next_section"),
    ("What does the next 10k look like?", "get_segment_ahead", "10km"),
    ("Is it going to rain?", "get_weather_forecast", "ahead"),
    ("How's the wind right now?", "get_weather_forecast", "wind"),
    ("What's the temperature?", "get_weather_forecast", "current"),
    ("Where can I get water?", "find_nearby_poi", "water"),
    ("I need a coffee stop", "find_nearby_poi", "cafe"),
    ("Is there a bike shop nearby? I have a flat", "find_nearby_poi", "bike_shop"),
    ("Where can I take a break?", "find_nearby_poi", "rest_area"),
    ("The weather looks bad, where can I shelter?", "find_nearby_poi", "shelter"),
    ("Find me a flatter route", "get_route_alternatives", "flatter"),
    ("I want to avoid the main roads", "get_route_alternatives", "avoid_main_roads"),
    ("Is there a shorter way to get there?", "get_route_alternatives", "shorter"),
    ("Find me a more scenic route", "get_route_alternatives", "scenic"),
    ("I'm exhausted, what's the easiest way to finish?", "get_route_alternatives", "flatter"),
    ("It's getting windy, should I change my route?", "get_route_alternatives", "sheltered"),
    ("How am I doing compared to my usual pace?", "get_rider_profile", "fitness"),
]


@dataclass
class DatasetEntry:
    user_message: str
    tool_name: str
    tool_args: Dict[str, str]

    def to_csv_row(self) -> List[str]:
        return [
            self.user_message,
            json.dumps([{"name": self.tool_name, "args": self.tool_args}]),
        ]


class LLMGenerator:
    """Generates dataset entries using Ollama's OpenAI-compatible API."""

    def __init__(self):
        print(f"Connecting to Ollama at {OLLAMA_BASE_URL} [{OLLAMA_MODEL}]...")

        # Verify Ollama is reachable
        try:
            self.client = OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama",  # Ollama ignores the key, but the client requires one
                timeout=120.0,
            )
            # Quick connectivity check
            self.client.models.list()
            print("Connected successfully.")
        except Exception as e:
            print(f"Error: Could not connect to Ollama at {OLLAMA_BASE_URL}")
            print(f"Details: {e}")
            print(f"\nMake sure Ollama is running on {OLLAMA_HOST} with:")
            print(f"  export OLLAMA_HOST=0.0.0.0")
            print(f"  ollama serve")
            sys.exit(1)

    def generate_batch(
        self, tool_name: str, query_arg: str, count: int, style: str
    ) -> List[str]:

        prompt = f"""
{TOOLS_CONTEXT}

TASK: Generate {count} unique, natural language user messages that would trigger the following tool call:
Tool: {tool_name}
Argument: query="{query_arg}"

STYLE: {style}

OUTPUT FORMAT:
Return ONLY a raw JSON array of strings. No markdown, no explanation, no extra text.
Example: ["message 1", "message 2"]
"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=OLLAMA_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful data generator. Always output valid JSON arrays only. No markdown, no explanation.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.8,
                    max_tokens=800,
                )

                content = response.choices[0].message.content.strip()

                # Clean potential markdown wrappers
                content = content.replace("```json", "").replace("```", "").strip()

                # Try to extract JSON array if model added extra text
                start = content.find("[")
                end = content.rfind("]")
                if start != -1 and end != -1:
                    content = content[start : end + 1]

                messages = json.loads(content)

                if not isinstance(messages, list):
                    print(
                        f" (Warning: non-list for {tool_name}:{query_arg})",
                        end="",
                        flush=True,
                    )
                    return []

                messages = [str(m).strip() for m in messages if isinstance(m, str) and m.strip()]
                print(".", end="", flush=True)
                return messages[:count]

            except json.JSONDecodeError as e:
                print(f" (JSON error: {e}, retry {attempt+1}/{max_retries})", end="", flush=True)
                time.sleep(5)
            except Exception as e:
                print(f" (Error: {e}, retry {attempt+1}/{max_retries})", end="", flush=True)
                time.sleep(10)

        # Fallback: return generic messages if all retries fail
        print(" (using fallback)", end="", flush=True)
        return [
            f"Requesting {tool_name} with {query_arg} ({style} variant {i})"
            for i in range(count)
        ]


class DatasetGenerator:
    def __init__(self, generator: LLMGenerator):
        self.generator = generator
        self.entries: List[DatasetEntry] = []
        self.generated_messages: set = set()

    def add_entry(self, messages: List[str], tool: str, query: str):
        for msg in messages:
            if msg not in self.generated_messages:
                self.generated_messages.add(msg)
                self.entries.append(DatasetEntry(msg, tool, {"query": query}))

    def run(self):
        total = len(SEEDS)
        print(f"Starting generation for {total} seeds...")
        print(f"Estimated time: ~{total * 5} minutes on slow hardware\n")

        for seed_idx, (seed_msg, tool, query) in enumerate(SEEDS):
            print(
                f"\n[{seed_idx+1}/{total}] Expanding: {tool}({query})", end=" ", flush=True
            )

            # 1. Standard (4 instead of 6 — smaller batches for reliability)
            msgs = self.generator.generate_batch(
                tool, query, 4, "Standard natural language questions. Varied phrasing."
            )
            self.add_entry(msgs, tool, query)

            # 2. Voice (2)
            msgs = self.generator.generate_batch(
                tool,
                query,
                2,
                "Short, terse, voice-command style. (e.g. 'Status?', 'Check battery')",
            )
            self.add_entry(msgs, tool, query)

            # 3. Indirect (2)
            msgs = self.generator.generate_batch(
                tool,
                query,
                2,
                "Indirect intent. Implied need without naming the tool explicitly.",
            )
            self.add_entry(msgs, tool, query)

            # 4. Spanish (3)
            msgs = self.generator.generate_batch(
                tool, query, 3, "Natural Spanish language (Spain/International)."
            )
            self.add_entry(msgs, tool, query)

            # 5. Conversational (2)
            msgs = self.generator.generate_batch(
                tool,
                query,
                2,
                "Conversational, polite, or verbose phrasing.",
            )
            self.add_entry(msgs, tool, query)

        return self.entries


def main():
    print("=" * 50)
    print("Cycling Copilot Dataset Generator (Ollama)")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Server: {OLLAMA_BASE_URL}")
    print("=" * 50)

    llm = LLMGenerator()
    generator = DatasetGenerator(llm)

    start_time = time.time()

    try:
        dataset = generator.run()
    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Saving {len(generator.entries)} entries collected so far...")
        dataset = generator.entries
    except Exception as e:
        print(f"\nCritical Error: {e}")
        print(f"Saving {len(generator.entries)} entries collected so far...")
        dataset = generator.entries

    elapsed = time.time() - start_time
    print(f"\n\nGenerated {len(dataset)} unique entries in {elapsed/60:.1f} minutes.")

    if dataset:
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["user_message", "tool_calls"])
            for entry in dataset:
                writer.writerow(entry.to_csv_row())

        print(f"Saved to {OUTPUT_FILE}")
    else:
        print("No entries to save.")


if __name__ == "__main__":
    main()
