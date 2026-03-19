import json

# Input and output file paths
input_path = "./trials_filtered.json"
output_path = "./trials_pruned.json"

# Fields to retain
fields_to_keep = {
    "Title",
    "RCT ID",
    "Public Data URL",
}

# Load original JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Prune each entry
pruned_data = {}
for key, value in data.items():
    pruned_data[key] = {
        field: value.get(field)
        for field in fields_to_keep
        if field in value
    }

# Save pruned JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(pruned_data, f, indent=2, ensure_ascii=False)

print(f"Pruned file saved to: {output_path}")
