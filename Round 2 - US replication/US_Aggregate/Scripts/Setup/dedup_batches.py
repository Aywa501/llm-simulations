import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
batch_dir = SCRIPT_DIR.parents[1] / "Data" / "Simulation" / "Batch_Input"

# Find all configs (no_reasoning, reasoning_low, etc) to ensure we de-dup across the entire sequence for that config
configs = set()
for p in batch_dir.glob("batch_*.jsonl"):
    m = re.match(r"batch_(.+?)_\d+_of_", p.name)
    if m:
        configs.add(m.group(1))

print("Starting native deduplication patch...")

for cfg in configs:
    seen_ids = set()
    files = sorted(batch_dir.glob(f"batch_{cfg}_*.jsonl"))
    
    total_fixed_for_cfg = 0
    for filename in files:
        records = []
        changes = 0
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cid = r["custom_id"]
                if cid in seen_ids:
                    # Find a unique dup suffix to securely bypass OpenAI rejection
                    parts = cid.split("__")
                    out_id_orig = parts[-2]
                    dup_idx = 1
                    while True:
                        parts[-2] = f"{out_id_orig}_dup{dup_idx}"
                        new_cid = "__".join(parts)
                        if new_cid not in seen_ids:
                            cid = new_cid
                            r["custom_id"] = cid
                            changes += 1
                            break
                        dup_idx += 1
                seen_ids.add(cid)
                records.append(r)
        
        if changes > 0:
            total_fixed_for_cfg += changes
            print(f"[{cfg}] Fixed {changes} identical duplicate custom_ids in {filename.name}")
            with open(filename, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    
    print(f"Finished {cfg}: structurally deduplicated {total_fixed_for_cfg} prompts across all chunks!")
