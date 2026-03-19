import json

input_file = "trials_filtered.json"
output_file = "trials_us_only.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = {}

for key, trial in data.items():
    countries = trial.get("Countries", [])

    if any(c.get("Country") == "United States of America" for c in countries):
        filtered[key] = trial

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2)

print(f"Filtered {len(filtered)} entries with United States of America")
