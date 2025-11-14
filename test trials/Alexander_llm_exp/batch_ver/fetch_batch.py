# fetch_batch.py
import time, json
from openai import OpenAI

client = OpenAI()
BATCH_ID = input("Enter your batch ID: ").strip()

# Poll until done
while True:
    job = client.batches.retrieve(BATCH_ID)
    print(f"Status: {job.status} | completed: {job.request_counts.completed}/{job.request_counts.total}")
    if job.status == "completed":
        break
    if job.status == "failed":
        raise RuntimeError("Batch job failed:", job)
    time.sleep(30)

# Download results
output_file_id = job.output_file_id
content = client.files.content(output_file_id).content.decode("utf-8")

results_path = f"batch_results_{BATCH_ID}.jsonl"
with open(results_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"✅ Results saved to {results_path}")

