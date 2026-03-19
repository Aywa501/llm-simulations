# prepare_batch.py
import json
from config import make_prompt, get_groups

MODEL = "gpt-5-mini"
N_PER_GROUP = 500
TASKS_FILE = "batch_tasks.jsonl"

with open(TASKS_FILE, "w", encoding="utf-8") as f:
    for group in get_groups():
        for i in range(1, N_PER_GROUP+1):
            task = {
                "custom_id": f"{group}_{i:04d}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL,
                    "input": make_prompt(group),
                    "reasoning": {"effort": "low"}
                }
            }
            f.write(json.dumps(task)+"\n")

print(f"✅ Wrote {len(get_groups())*N_PER_GROUP} tasks to {TASKS_FILE}")

