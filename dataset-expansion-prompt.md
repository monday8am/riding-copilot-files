# FunctionGemma Cycling Copilot — Dataset Expansion Prompt

## Context

You are helping generate training data for fine-tuning FunctionGemma (a 270M parameter on-device model) as a tool-calling router for a cycling copilot app. The model's ONLY job is to receive natural language input from a cyclist mid-ride and select exactly ONE tool to call with the correct parameter. It does NOT generate the final answer — a separate model handles that.

Important rules:
- Every example maps to exactly ONE tool call
- Never output empty tool calls or multiple tools
- The model picks the single MOST relevant tool for the user's intent

## Tool Definitions

The model has access to 6 tools. Every tool takes a single string parameter called `query`:

1. **get_ride_status** — Returns current ride metrics (speed, distance, elevation, battery, time)
   - query values: `all`, `speed`, `distance`, `elevation`, `battery`, `time`

2. **get_segment_ahead** — Returns info about upcoming route section (elevation profile, surface, difficulty)
   - query values: `5km`, `10km`, `20km`, `next_section`

3. **get_weather_forecast** — Returns weather at current location and along route (temp, wind, rain)
   - query values: `current`, `ahead`, `wind`

4. **find_nearby_poi** — Returns nearby points of interest (cafes, water, bike shops, rest areas)
   - query values: `water`, `cafe`, `food`, `bike_shop`, `rest_area`, `shelter`, `any`

5. **get_route_alternatives** — Returns 1-3 alternative routes with constraints (requires network)
   - query values: `flatter`, `shorter`, `paved`, `scenic`, `sheltered`, `avoid_main_roads`

6. **get_rider_profile** — Returns rider's historical baselines and preferences
   - query values: `fitness`, `preferences`, `all`

## Output Format

Generate a CSV with two columns: `user_message` and `tool_calls`.

- `user_message`: A natural language message from a cyclist mid-ride, in quotes.
- `tool_calls`: A JSON array with exactly ONE tool call object containing `name` and `args`.

### Example:
```
"Where can I fill up my water bottle?","[{""name"": ""find_nearby_poi"", ""args"": {""query"": ""water""}}]"
```

## Seed Scenarios (25 examples)

### get_ride_status (6 seeds)
- "How am I doing?" → get_ride_status(all)
- "What's my current speed?" → get_ride_status(speed)
- "How far have I gone?" → get_ride_status(distance)
- "How much battery do I have left?" → get_ride_status(battery)
- "What's the elevation here?" → get_ride_status(elevation)
- "How long have I been riding?" → get_ride_status(time)

### get_segment_ahead (4 seeds)
- "What's ahead in the next 5 kilometers?" → get_segment_ahead(5km)
- "Are there any big climbs coming up?" → get_segment_ahead(next_section)
- "How hard is the next section?" → get_segment_ahead(next_section)
- "What does the next 10k look like?" → get_segment_ahead(10km)

### get_weather_forecast (3 seeds)
- "Is it going to rain?" → get_weather_forecast(ahead)
- "How's the wind right now?" → get_weather_forecast(wind)
- "What's the temperature?" → get_weather_forecast(current)

### find_nearby_poi (5 seeds)
- "Where can I get water?" → find_nearby_poi(water)
- "I need a coffee stop" → find_nearby_poi(cafe)
- "Is there a bike shop nearby? I have a flat" → find_nearby_poi(bike_shop)
- "Where can I take a break?" → find_nearby_poi(rest_area)
- "The weather looks bad, where can I shelter?" → find_nearby_poi(shelter)

### get_route_alternatives (5 seeds)
- "Find me a flatter route" → get_route_alternatives(flatter)
- "I want to avoid the main roads" → get_route_alternatives(avoid_main_roads)
- "Is there a shorter way to get there?" → get_route_alternatives(shorter)
- "I'm exhausted, what's the easiest way to finish?" → get_route_alternatives(flatter)
- "It's getting windy, should I change my route?" → get_route_alternatives(sheltered)

### get_rider_profile (2 seeds)
- "How am I doing compared to my usual pace?" → get_rider_profile(fitness)
- "Find me a more scenic route" → get_route_alternatives(scenic)

## Expansion Instructions

Generate 60 NEW examples. Each must map to exactly ONE tool call. Distribute roughly evenly across all 6 tools (10 per tool). Each example must have a unique, realistic user message.

### Variation types to include across all tools:

**Voice-style commands (15%)** — Short, terse inputs like a cyclist would say mid-ride:
- "speed?" → get_ride_status(speed)
- "water nearby" → find_nearby_poi(water)
- "wind?" → get_weather_forecast(wind)

**Indirect / implied intent (15%)** — The user doesn't name the tool but implies it:
- "My legs are heavy" → get_ride_status(all)
- "This road is terrible" → get_route_alternatives(paved)
- "I'm starving" → find_nearby_poi(food)

**Spanish language (15%)** — Natural Spanish inputs:
- "¿Hay alguna fuente cerca?" → find_nearby_poi(water)
- "¿Cómo voy de tiempo?" → get_ride_status(time)
- "¿Va a llover?" → get_weather_forecast(ahead)

**Conversational / verbose (15%)** — Longer, chattier phrasing:
- "I was wondering if there are any tough hills in the next few kilometers" → get_segment_ahead(next_section)
- "Do you think I should look for a different route? The wind is really bad" → get_route_alternatives(sheltered)

**Standard natural language (40%)** — Regular mid-ride questions:
- "Show me the elevation for the next 20km" → get_segment_ahead(20km)
- "Any cafes nearby?" → find_nearby_poi(cafe)

## Quality Rules

1. Every example maps to exactly ONE tool call — never zero, never multiple
2. Never repeat a user message — every input must be unique
3. Tool calls must use ONLY the parameter values listed above
4. Do not include any explanation — output ONLY the CSV rows
5. Start with the CSV header: user_message,tool_calls
