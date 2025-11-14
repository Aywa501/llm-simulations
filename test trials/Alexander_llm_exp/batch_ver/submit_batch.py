# submit_batch.py
from openai import OpenAI

client = OpenAI()
TASKS_FILE = "batch_tasks.jsonl"

# Upload file
file_obj = client.files.create(file=open(TASKS_FILE, "rb"), purpose="batch")
print("Uploaded file_id:", file_obj.id)

# Create batch job
batch = client.batches.create(
    input_file_id=file_obj.id,
    endpoint="/v1/responses",
    completion_window="24h"
)

print(f"✅ Batch created!\nBatch ID: {batch.id}\nStatus: {batch.status}")
print("You can safely close your laptop. Run fetch_batch.py tomorrow to download results.")

