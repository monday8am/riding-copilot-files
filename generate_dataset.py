# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub",
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
    from huggingface_hub import InferenceClient
except ImportError:
    print("Error: 'huggingface_hub' library not found.")
    print("Please install it using: pip install huggingface_hub")
    sys.exit(1)

# --- Configuration ---

OUTPUT_FILE = "cycling-copilot-dataset-hf.csv"

# Configuration for Hugging Face
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

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
    ("How am I doing compared to my usual pace?", "get_rider_profile", "fitness")
]

@dataclass
class DatasetEntry:
    user_message: str
    tool_name: str
    tool_args: Dict[str, str]

    def to_csv_row(self) -> List[str]:
        return [self.user_message, json.dumps([{"name": self.tool_name, "args": self.tool_args}])]

class LLMGenerator:
    """Generates dataset entries using Hugging Face InferenceClient."""
    
    def __init__(self, api_key: Optional[str] = None):
        # If no API key provided, we trust the environment (logged in via hf cli)
        token_arg = api_key if api_key else True
            
        print(f"Connecting to Hugging Face API [{HF_MODEL}]...")
        try:
            self.client = InferenceClient(model=HF_MODEL, token=token_arg)
        except Exception as e:
            print(f"Error initializing Client: {e}")
            if not api_key:
                print("Tip: If not logged in, export HF_TOKEN or run 'huggingface-cli login'")
            sys.exit(1)
        
    def generate_batch(self, tool_name: str, query_arg: str, count: int, 
                       style: str) -> List[str]:
        
        prompt = f"""{TOOLS_CONTEXT}

TASK: Generate {count} unique, natural language user messages that would trigger the following tool call:
Tool: {tool_name}
Argument: query="{query_arg}"

STYLE: {style}

OUTPUT FORMAT:
Return ONLY a raw JSON array of strings. No markdown formatting. Do not include "Here is the JSON" or other chatter.
Example: ["message 1", "message 2", ...]"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use serverless inference (non-streaming for stability)
                response = self.client.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a helpful data generator helper. Always output valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                )
                
                if not response.choices:
                    print(f" (Empty response, retrying {attempt+1}/{max_retries})", end="", flush=True)
                    continue

                content = response.choices[0].message.content.strip()
                # Clean potential markdown
                content = content.replace("```json", "").replace("```", "").strip()
                
                try:
                    messages = json.loads(content)
                except json.JSONDecodeError:
                    # sometimes models add extra text
                    start = content.find("[")
                    end = content.rfind("]")
                    if start != -1 and end != -1:
                        content = content[start:end+1]
                        messages = json.loads(content)
                    else:
                        raise ValueError("Could not find JSON array")

                if not isinstance(messages, list):
                    print(f"Warning: Model returned non-list for {tool_name}:{query_arg}")
                    return []
                    
                messages = [str(m) for m in messages if isinstance(m, str)]
                print(".", end="", flush=True) # Progress dot
                return messages[:count]
                
            except Exception as e:
                print(f" (Error: {e}, retrying {attempt+1}/{max_retries})", end="", flush=True)
                time.sleep(2)
                    
        return [f"Requesting {tool_name} with {query_arg} ({style} variant {i})" for i in range(count)]

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
        print(f"Starting generation...")
        
        for seed_idx, (seed_msg, tool, query) in enumerate(SEEDS):
            print(f"\n[{seed_idx+1}/{len(SEEDS)}] Expanding: {tool}({query})", end=" ")
            
            # 1. Standard (6)
            msgs = self.generator.generate_batch(tool, query, 6, "Standard natural language questions. Varied phrasing.")
            self.add_entry(msgs, tool, query)
            
            # 2. Voice (2)
            msgs = self.generator.generate_batch(tool, query, 2, "Short, terse, voice-command style. (e.g. 'Status?', 'Check battery')")
            self.add_entry(msgs, tool, query)
            
            # 3. Indirect (2)
            msgs = self.generator.generate_batch(tool, query, 2, "Indirect intent. Implied need without naming the tool explicitly.")
            self.add_entry(msgs, tool, query)
            
            # 4. Spanish (3)
            msgs = self.generator.generate_batch(tool, query, 3, "Natural Spanish language (Spain/International).")
            self.add_entry(msgs, tool, query)
            
            # 5. Conversational (2)
            msgs = self.generator.generate_batch(tool, query, 2, "Conversational, polite, or verbose phrasing.")
            self.add_entry(msgs, tool, query)
            
        return self.entries

def main():
    api_key = os.environ.get("HF_TOKEN")
    
    llm = LLMGenerator(api_key)
    generator = DatasetGenerator(llm)
    
    try:
        dataset = generator.run()
    except Exception as e:
        print(f"\nCritical Error: {e}")
        return

    print(f"\nGenerated {len(dataset)} unique entries.")
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["user_message", "tool_calls"])
        for entry in dataset:
            writer.writerow(entry.to_csv_row())
            
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
