import os
import sys
from google import genai

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not set")
    sys.exit(1)

client = genai.Client(api_key=api_key)

print("Listing available models:")
try:
    for m in client.models.list():
        # Check if it supports generation
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
